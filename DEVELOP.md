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


### zip 算子

新增 `csrc/kernels/zip.cu`，persistent kernel，`num_ctas`（默认 4）个 CTA、每个 512 线程。

接口是**独立的 python api** `deep_ep.zip(o3, zip_to_atomic, recv_topk_idx, zip_task_queue, zip_done, num_ctas)`，返回 `combine_input`。C++ 侧是 `deep_ep` 命名空间下的自由函数 `zip_tokens`（pybind 暴露名为 `zip`，因为 `deep_ep::zip` 这个名字被 kernel 的命名空间占了），不挂在 `Buffer` 上——它不需要任何 buffer 状态，这样单测才能单卡跑、不用初始化 NVSHMEM。用 `getCurrentCUDAStream()`，也就是默认计算流，符合"用户在 dispatch(async=True) 之后单独调用 zip"的设计。

任务分配：**一个 CTA 负责一整个 token**，按下标取模认领，`for (task_idx = cta_id; task_idx < num_recv_tokens; task_idx += num_ctas)`，不需要抢任务。最初实现是 4 个 CTA 切 hidden 维度共同处理一个 token（想的是延迟优先），但实测更差：hidden 最大也就 7168 这个量级，一个 CTA 已经足够喂满自己的线程，切开反而让 release 次数变成 4 倍。改成一 CTA 一 token 后 `zip_done` 也就回到了 0/1 语义（**这一点要同步给 combine**，之前一度约定成 0~4，作废）。

累加顺序：按本地专家下标**升序**，fp32 累加后转 bf16，和 paddle 原版对齐，所以单测是精确相等而非容差比较。`recv_topk_idx` 的一行是乱序的（DeepEP 只是按 lane 原样搬运发送侧的 topk_idx，不排序），需要自己排。没有用双调排序：同一 token 不会重复选同一专家，专家下标互不相等，所以直接算 rank 就是全序——每个 lane 和全 warp 做 32 次 `__shfl_sync` 比较，`rank = #{j : expert_j < expert_i}`，然后 `smem_slots[rank] = slot`，无效 lane 取 `INT_MAX` 因此永不被计入。32 次 shuffle 比 15 级双调排序更短也更好读。

有效性判断**只信 `zip_to_atomic`，不数 `recv_topk_idx` 的非 -1 个数**。这是可见性分析的直接结果：`recv_topk_idx` 的空槽位是 `torch::empty` 的未初始化显存、receiver 从未为它做 release，运行中读出来可能是垃圾；而 `zip_to_atomic` 是 host 侧 memset 成 -1 的，可靠。只有 `slot >= 0` 时才去读同一位置的 `recv_topk_idx` 取排序键，那个值由同一个 lane 写、被它自己的 chunk 发布，是安全的。也因此 zip 完全不需要 `num_valid_topk`，那张表是给计算侧用的。

发布 `zip_done` 的正确写法是 `__syncthreads()` + 一次 release store（`st_na_release`）。因为一个 CTA 的 token 是 512 个线程一起写的，而 release 只覆盖"happens-before 它"的写入——没有那道 barrier 的话，thread 0 的 release 只覆盖它自己写的那一小段 hidden，其余对 combine 不保证可见，和 chunk 入队那里是同一个坑。barrier 建立 happens-before 边，release 的**累积性**负责把整条边上的写入一起发布，所以中间不需要额外的 `__threadfence()`（一开始加了，后来确认冗余、删掉）。

超时检测照抄 DeepEP 的模式：`clock64()` + `NUM_TIMEOUT_CYCLES`，超时打印 CTA / task 下标和总任务数后 `trap()`。

单测 `tests_overlap/test_zip.py`，**单卡直接 `python test_zip.py`**，不用 run.py。用 `moe_permute` / `moe_unpermute` 造参考值，`recv_topk_idx` 的无效槽位**故意填垃圾**，专门用来验证上面那条有效性判断。两个 case：队列预先填好（纯功能）、先 launch zip 再从另一条流把队列 assign 进去（验证 persistent kernel 真的在自旋等待，而不是靠启动时队列已就绪侥幸通过）。两者都 `diff: 0`。

性能：`kNumThreads=512` 实测最优。目前 zip 不是瓶颈，但按相同 SM 数折算，带宽利用率只有 paddle 原版的一半左右，说明每 token 的两道 sync + 一次暴露的 acquire 延迟没被掩盖（只有 1 个 block/SM，没有别的 block 能换进来）。


### FP8 dispatch

融合 unzip 现在跟着 `x` 的类型走：`dispatch((x_fp8, x_scale), unzip_alignment=...)` 时返回的 `unzipped_x` 也是 `(unzipped_tokens, unzipped_scale)` 元组，bf16 下仍然是单个 tensor，接口向后兼容。

kernel 侧改动只有多播那一段：receiver 的 TMA load 本来就把 `hidden + scale` 一起搬进 smem（`scale_bytes % 16 == 0` 时），所以每个命中的 lane 在发 token 的 TMA store 之后再发一次 scale 的 TMA store 就行，两次 store 由同一个 lane 的 `tma_store_wait_complete` 一起等，chunk 入队的 release 语义不用动。scale_bytes 不对齐时退化成该 lane 自己从 NVL buffer 逐个 int 拷贝（此时通道 head 还没推进，源数据仍有效）。

`unzipped_scale` 的 layout 和 `recv_x_scales` 一致，是**非 transpose** 的连续 `[num_unzipped_tokens, num_scales]`（DeepEP 不支持发 transpose 的 scale），需要 transpose 的话由计算侧自己 `.T.contiguous().T`，和 baseline 里 `moe_permute` 之后那一步一样。

单测 `tests_overlap/test_internode_forward_fp8.py`：dispatch 的输出和 baseline（原版 dispatch + `moe_permute`）逐字节比对，`recv_x` 直接比，`unzipped_x` 按 `atomic_to_zip` 排回 DeepEP 序再比。注意 `moe_permute` 输出的 scale 底层是 transpose 的，必须先整体 `contiguous()` 再切片，否则切片出来的东西不能直接 `view`。fp8 也没有 `gather` kernel，重排前要先转成 uint8 视图。

正确性：16 个 rank 上 token 和 scale 的 diff 全为 0。


## 后续优化思路（暂不做，等 overlap 跑通）

1. **融合模式下考虑不写 `recv_x`**。当前 receiver 的写入量是原版的 2.23 倍（recv_x 1.0 + 多播 1.23）。每个收到的 token 至少命中一个本地专家，内容已完整存在于 `unzipped_x`，而计算读 `unzipped_tokens`、zip 写 `combine_input`、combine 读 `combine_input` 和 handle 里的 `recv_src_meta`，都不碰 `recv_x`。如果确认没有消费者（需要先查反向 cached dispatch 是否依赖），写入量可降到 1.23 倍，约省 0.35 ms，融合后的 dispatch 有可能比原版纯 dispatch 还快。

2. **在这台共享机上做性能对比要在同一进程内 A/B**。跨时间比较不可靠（同一份代码前后能差 2 倍）。建议把 counter stride 之类的开关做成 kernel 参数，一次运行内交替测量。

3. TOPK 越大融合的相对开销越高（TOPK=8 约 +24%，TOPK=6 约 +5%），这个非线性说明 receiver 存在从"有余量"到"成为关键路径"的跨越点；每 token 的多播份数本身只从 1.24 降到 1.17，解释不了这个差距，值得在 receiver 成为瓶颈前后各做一次 ncu。

4. **攒批发布（signal 开销的主要优化方向）**。信号部分实测加了约 0.78 ms，其中 `tma_store_wait_complete` 只占 +0.086 ms，剩下 0.58~0.67 ms 全在"让写入在设备范围可见"这一步。已验证的负面结论：把 `__threadfence()` + relaxed `atomicAdd` 换成 `atom.release.gpu.global.add.s32` **没有任何改善**（3.20 ms vs 3.22 ms），所以这不是 fence 种类的问题，也不是带宽问题（额外原子流量只有约 520 KB，占 0.05%）。真正的成本是每个 token 都要暴露一次 L2 往返延迟（约 1.7 µs × 每 warp 约 340 次 ≈ 0.58 ms），并通过 NVL ring 反压转化为吞吐损失。
   - 方向：一个 warp 攒 N 个 token 再统一 `tma_store_wait_complete` + release，让每个 warp 有多份 in-flight 的写入互相掩盖延迟，把每 token 一次的暴露延迟摊薄成每 N 个 token 一次。
   - 风险：需要额外寄存器保存 N 组 `slot / unzipped_idx / local_expert_idx / prob`，而 kernel 是按 `--register-usage-level=10` 编的；smem 已被 TMA buffer 占满，不能拿来做暂存。所以 N 要小步试（先 2、4），并盯住 occupancy 是否掉。

5. **zip 改成一个 warp 一个 token（结构性最大的一项）**。`task_idx = global_warp_id + i * num_warps_total`，16 warp × 4 CTA = 64 条独立流水线。两道 `__syncthreads` 全部消失（排序那 32 次 shuffle 每个 warp 自己做一遍，也不用 smem），且一个 warp 在等 queue 时其他 15 个 warp 在跑访存，acquire 延迟被天然掩盖。注意发布：warp 内 32 个 lane 都写了 token 的一部分，release 前必须 `__syncwarp()`，靠的还是"happens-before 边 + release 累积性"，只是 warp 级这条边在文档里不如 `__syncthreads` 明确，要在注释里写清楚。

6. **zip 的 slot 循环按固定上限展开**。现在 `for (s = 0; s < num_slots; ++s)` 上界是运行时值，编译器基本无法展开，k 次 16B load 很可能串着发，MLP 极差。改成对编译期上限 `#pragma unroll` + 谓词跳过，让 k 条 load 背靠背发出。改动很小。

7. **zip 加 `num_slots == 1` 快路径**。真实分布下这一项比单测看起来重要得多：`num_experts = world_size × E = 256`、本地只占 1/16，实测 pair/token = 130703/106626 ≈ 1.23，即**约 80% 的 token 只命中一个专家**，此时 zip 就是一次纯拷贝，不需要 bf16→fp32→bf16 往返和累加指令。

8. **zip 攒批发布**。连做 M 个 token 再一次 `__threadfence()` + M 次 relaxed 置位，fence 次数降到 1/M。代价是每个 token 的可见时刻被推后，而 combine 是顺序等待不能跳过的，会增加 combine 被阻塞的概率——要和 combine 一起端到端测才有意义，放在 5~7 之后。

9. 上面几项动手前先用 ncu 确认一次 zip 的 stall 构成：如果主要在 barrier，第 5 项对症；如果主要在 `long_scoreboard` 且 MLP 低，第 6 项优先。避免再出现一次像 release atomic 那样"理论上该快、实测没动"的情况。
