# 开发文档

## 安装

```bash
# 注意要用虚拟环境，不要覆盖主机的原版 deep_ep
source env3.12/bin/activate

rm -rf build dist *.egg-info

# nvshmem 已随 paddle 自动安装, 无需指定路径
python setup.py bdist_wheel

python -m pip install --force-reinstall dist/deep_ep-*.whl
```


## 单测

使用 mpirun 在集群的 node 0 和 node 1 上运行

当前集群为共享开发机，性能会有波动，当前开发阶段以正确性为主

```bash
source env3.12/bin/activate
cd tests_overlap
mpirun python run.py 0,1 test_internode_baseline.py 
```


## 开发进展

dispatch + unzip 融合的第一步（不带信号）已完成，在 internode dispatch 的 NVL receiver 上加多播。

- 实现：`csrc/kernels/internode.cu` 的 `kNVLReceivers` 分支，token 的 TMA 数据已在 smem，多播只是对每个命中的本地专家再发一次 `tma_store_1d`；槽位用 `atomicAdd` 抢，专家段基址是 CPU 侧算好的对齐前缀和
- 开关：`buffer.dispatch(..., unzip_alignment=128)`，kernel 内部靠 `unzipped_x` 指针是否为空判断，接口复用，`unzip_alignment=0` 时行为与原版完全一致
- 单测：`tests_overlap/test_internode_fused_unzip.py`，把 token 全局 id 编码进 x 的前三列，按 id 排序后与 `moe_permute` 做逐行精确比对（因为段内是 atomic 序，只能做集合比对）
- 正确性：PASS
- 性能（SEQLEN=16384, H=4096, TOPK=8, E=16, 16 ranks, 48 SM）：
  - dispatch only: 1.96 ms
  - dispatch + moe_permute: 2.87 ms
  - dispatch 融合 unzip: 2.44 ms

即融合本身给 dispatch 增加了约 0.48 ms（+24%），但比现在 dispatch + 独立 unzip 的两段式快约 15%。

把多播的 TMA store 临时注释掉再测，可以把这 0.48 ms 拆开：

- dispatch only: 2.00 ms
- 只留 atomic + prob store，不发 TMA: 2.08 ms
- 完整融合: 2.43 ms

即**写带宽占 82%（约 0.35 ms），atomic 和循环开销只占 18%（约 0.077 ms）**。所以融合的代价基本就是多写的那 1.23 份 token，不是同步开销。

随后做了两个 atomic 侧的优化（都已合入）：

- 每个命中的 lane 并行抢自己的槽位，一个 token 只付一次 atomic 往返，替代原来 lane 0 串行循环
- per-expert counter 按 `NUM_UNZIP_COUNTER_STRIDE`(=32 ints, 128B) 跨步，避免 16 个 counter 挤在一两条 cache line 上被 384 个 receiver warp 争用

这两个改动的收益上限就是上面那 0.077 ms，当时机器整体变慢了约 2 倍（连没碰过的 `moe_permute` 都慢了 2.9 倍），没能量出来，留待机器安静时复测。

待办：padding 行目前是未初始化的脏数据；`atomic_to_zip` / `zip_to_atomic` / `task_queue` 等信号还没做。


### chunk 计数 + task_queue 入队 + 两张映射表

同一个 chunk 的多个槽位是由不同 channel / 不同 SM 的 receiver warp 写的，所以"抢到最后一个槽位"不等于"整个 chunk 写完"。做法是把认领和完成分开：

- 认领：`atomicAdd(unzipped_expert_counter[e])` 拿槽位，决定它属于哪个 chunk
- 完成：写完数据后 `atomicAdd(unzip_chunk_done[chunk_idx])`，返回值等于 `m_size - 1` 的那个 warp 是最后一个完成的，由它入队
- 入队：`atomicAdd(task_queue_counter)` 拿 FIFO 下标，写完 `[expert_idx, m_start, m_size]` 后再 release 写 `ready`

per-expert 元数据合成一张 `[num_local_experts, 3]` 的表 `{base, count, chunk_base}`，仍然用一个 lane 一个专家的寄存器缓存 + `__shfl` 读取。

TMA 的可见性问题：原来的 `tma_store_wait` 用的是 `cp.async.bulk.wait_group.read`，只保证源 smem 可复用，**不保证数据在全局内存可见**。原版 DeepEP 靠 kernel 结束的隐式 fence 兜底就够了，但我们的消费者是另一条 stream 上并发的 kernel，所以新增了 `tma_store_wait_complete`（`cp.async.bulk.wait_group` + `fence.proxy.async.global`），并改成每个命中的 lane 自己发 TMA、自己等自己那份完成，这样"写完"和"报完成"在同一个线程里，不需要跨 lane 的顺序保证。

单测新增 `check_mapping_tables` 和 `check_task_queue`：
- padding 和无效 topk 位置必须保持 -1
- `unzipped_tokens[a] == recv_x[atomic_to_zip[a]]`
- 两张表互为逆映射，且 `zip_to_atomic` 指向的槽位落在正确的专家段内
- 每个 chunk 恰好入队一次、全部 ready、ready 从 0 开始密集填充（FIFO）、描述符集合与预期分段一致

正确性：三项检查在两个 rank 上都 PASS（38 chunks）。

性能（机器状态与前面同一水平，dispatch only 1.96 / moe_permute 2.90 可对齐）：

- dispatch only: 1.96 ms
- dispatch + moe_permute: 2.90 ms
- dispatch 融合 unzip + 信号: 3.22 ms

即信号机制又加了约 0.78 ms，融合版现在反而比两段式慢了。主要嫌疑是每个 (token, expert) 对都要付一次**真完成等待** + 一次 `__threadfence()`，把原来靠 TMA 异步流水掩盖掉的写延迟重新暴露了出来。下一步优化方向：把 `__threadfence()` + relaxed `atomicAdd` 换成 release 语义的 atomic（`atom.add.release.gpu`），以及不要每个 token 都等完成、改成攒几个 token 再等一次。


### FP8 dispatch

融合 unzip 现在跟着 `x` 的类型走：`dispatch((x_fp8, x_scale), unzip_alignment=...)` 时返回的 `unzipped_x` 也是 `(unzipped_tokens, unzipped_scale)` 元组，bf16 下仍然是单个 tensor，接口向后兼容。

kernel 侧改动只有多播那一段：receiver 的 TMA load 本来就把 `hidden + scale` 一起搬进 smem（`scale_bytes % 16 == 0` 时），所以每个命中的 lane 在发 token 的 TMA store 之后再发一次 scale 的 TMA store 就行，两次 store 由同一个 lane 的 `tma_store_wait_complete` 一起等，chunk 入队的 release 语义不用动。scale_bytes 不对齐时退化成该 lane 自己从 NVL buffer 逐个 int 拷贝（此时通道 head 还没推进，源数据仍有效）。

`unzipped_scale` 的 layout 和 `recv_x_scales` 一致，是**非 transpose** 的连续 `[num_unzipped_tokens, num_scales]`（DeepEP 不支持发 transpose 的 scale），需要 transpose 的话由计算侧自己 `.T.contiguous().T`，和 baseline 里 `moe_permute` 之后那一步一样。

单测 `tests_overlap/test_internode_forward_fp8.py`：dispatch 的输出和 baseline（原版 dispatch + `moe_permute`）逐字节比对，`recv_x` 直接比，`unzipped_x` 按 `atomic_to_zip` 排回 DeepEP 序再比。注意 `moe_permute` 输出的 scale 底层是 transpose 的，必须先整体 `contiguous()` 再切片，否则切片出来的东西不能直接 `view`。fp8 也没有 `gather` kernel，重排前要先转成 uint8 视图。

正确性：16 个 rank 上 token 和 scale 的 diff 全为 0。


## 后续优化思路（暂不做，等 overlap 跑通）

1. **融合模式下考虑不写 `recv_x`**。当前 receiver 的写入量是原版的 2.23 倍（recv_x 1.0 + 多播 1.23）。每个收到的 token 至少命中一个本地专家，内容已完整存在于 `unzipped_x`，而计算读 `unzipped_tokens`、combine 读 `combine_input` 和 handle 里的 `recv_src_meta`，都不碰 `recv_x`。如果确认没有消费者（需要先查反向 cached dispatch 是否依赖），写入量可降到 1.23 倍，约省 0.35 ms，融合后的 dispatch 有可能比原版纯 dispatch 还快。

2. **在这台共享机上做性能对比要在同一进程内 A/B**。跨时间比较不可靠（同一份代码前后能差 2 倍）。建议把 counter stride 之类的开关做成 kernel 参数，一次运行内交替测量。

3. TOPK 越大融合的相对开销越高（TOPK=8 约 +24%，TOPK=6 约 +5%），这个非线性说明 receiver 存在从"有余量"到"成为关键路径"的跨越点；每 token 的多播份数本身只从 1.24 降到 1.17，解释不了这个差距，值得在 receiver 成为瓶颈前后各做一次 ncu。

4. **攒批发布（signal 开销的主要优化方向）**。信号部分实测加了约 0.78 ms，其中 `tma_store_wait_complete` 只占 +0.086 ms，剩下 0.58~0.67 ms 全在"让写入在设备范围可见"这一步。已验证的负面结论：把 `__threadfence()` + relaxed `atomicAdd` 换成 `atom.release.gpu.global.add.s32` **没有任何改善**（3.20 ms vs 3.22 ms），所以这不是 fence 种类的问题，也不是带宽问题（额外原子流量只有约 520 KB，占 0.05%）。真正的成本是每个 token 都要暴露一次 L2 往返延迟（约 1.7 µs × 每 warp 约 340 次 ≈ 0.58 ms），并通过 NVL ring 反压转化为吞吐损失。
   - 方向：一个 warp 攒 N 个 token 再统一 `tma_store_wait_complete` + release，让每个 warp 有多份 in-flight 的写入互相掩盖延迟，把每 token 一次的暴露延迟摊薄成每 N 个 token 一次。
   - 风险：需要额外寄存器保存 N 组 `slot / unzipped_idx / local_expert_idx / prob`，而 kernel 是按 `--register-usage-level=10` 编的；smem 已被 TMA buffer 占满，不能拿来做暂存。所以 N 要小步试（先 2、4），并盯住 occupancy 是否掉。
