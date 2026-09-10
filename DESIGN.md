# 基于跨Stream信号的细粒度计算-通信Overlap实现（通信部分）

## 项目背景

这是我们 Paddle 预训练团队的一个探索性研究项目，本人负责通信部分的实现。

当前我们在 EP 场景下的 MoE 执行方法如下：
1. 调用 DeepEP dispatch 进行 all2all 通信
2. 调用 unzip 算子将收到的 token 分发到各个专家的输入 buffer
3. 调用 DeepGEMM 的 gemm/group_gemm（当token数量大时普通gemm更快）计算专家的 FFN
4. 调用 zip 算子聚合（求和）各个专家的输出 buffer，得到无重复的 tokens
5. 调用 DeepEP combine 进行 all2all 通信

上述方案作为 baseline 性能尚可，但在通信时间较长的场景下，由于计算和通信无法 overlap，导致计算或通信带宽的浪费。

现在我们提出一个新方案，就是通过多流同时 launch 计算和通信，并通过跨 Stream 信号进行同步，让计算和通信能有效 Overlap 起来，该方案简述如下：

首先说明一个 chunk 的概念，这是本方案中最重要的概念之一：
  * 我们知道，dispatch 本来收到 token 的顺序是不固定的，在通信底层看来，可能先收到这个专家或那个专家，可能先收到这个 channel 或那个 channel
  * 但是，站在专家消费者的角度看，其实 token 到达就像“细水长流”，当前专家的 buffer 不停地有 token 进来，专家只要不断“消费”它们就好了
  * 因此，我们可以把 unzip 融合到 dispatch 里，dispatch 一边收 token 一边转发（append）到各个专家的 buffer，每个专家凑够 chunk 个 token 后，就可以开始一次 FFN 计算
  * chunk 的意义在于保证计算密度，如果每来一个 token 就计算，矩阵乘很难打满；根据矩阵乘硬件要求，chunk 应当是 128 的倍数，根据之前调试 subbatch 的经验，chunk 至少为 2048-4096 才是足够的
  * 当前我们使用的的 DeepGEMM 提供了一个很好的特性，就是 token 的位置无关性，token 不管在 buffer 的哪里，FFN 算出来的结果都是一样的，因此专家 buffer 不需要保持和 DeepEP 一样的顺序，而是可以 atomic 地先到先放置，只要最后 combine 前再恢复到 DeepEP 的顺序即可

新方案的一次前向过程即可描述如下：

1. 进入 MoE 模块后，CPU 同时 launch 计算流和通信流的算子

2. dispatch 与 unzip 进行融合，执行如下过程：
  * dispatch 一边接收 token，一边把 token 转发到各个专家的 buffer，并更新计数器
  * token 在 buffer 里使用压实的放置，也就是先到先放，不像 DeepEP 自己的 buffer 是离散放置，这样是为了让矩阵乘可以读到连续的输入
  * 当一个专家每凑满 chunk 个 token，就可以入队一个计算任务，交给计算部分

3. 计算部分，使用 DeepGEMM 的 bf16 矩阵乘，在另一个流上并发进行：
  * 计算 kernel 每次从任务队列拿出一个任务，进行计算
  * 每组算子完成任务后，往一个表记录每个 token 的完成情况（已经被几个专家完成）
  * 计算部分正在由另一组同事开发中，目前虽然不能运行，但是已经约定好双方的数据交换格式与信号逻辑，见后

4. zip 与 combine 部分，zip 是一个单独的 persistent kernel，combine 也使用 DeepEP，增加信号等待逻辑：
  * zip 通过读计算完成队列，对于一个不重复 token，当它 topk 的所有专家都计算完就对其进行求和操作，写到 combine 的输入 buffer
  * combine 和 dispatch 同一个流，所以一定在 dispatch 完成后才启动
  * comine 同样增加一个等待逻辑，等输入 buffer 里的一个 token 就绪之后才发出


## 通信部分设计细节

下面的细节可以讨论

### kernel选型

首先说明，我们只在 internode 上支持，不需要改 intranode，因为我们这种 overlap 做法只在更大规模的 EP 上有优势，没必要做 intranode；我们目前也只做 bf16，因为 fp8 实际上对性能提升不明确，quant 很难做融合，因此下面全部基于 bf16 进行

dispatch+unzip：我已经看过 DeepEP 的代码，也看过其他一些竞品的实现，基本就是在 receiver 这个角色上增加一个多播，就是 receiver 本来就要把收到的 token 写到一个不重复的 recv_tokens buffer 里，我们只要在这里再加一些写逻辑，把 token 顺便写到各专家的 buffer，改动应该很小；当然，这样改可能会对性能有影响，但也不确定，需要先改了才知道，但根据竞品的测试，影响不大，因为 dispatch 瓶颈在跨机带宽

zip：需要新实现一个 persistent 风格的 kernel，读计算完成的信号，然后进行累加，再更新 combine 的输入信号

combine：同样改动很小，就是直接在原有 combine 的 sender 上增加一个等待逻辑，本来 sender 是把输入的 token 无条件地搬运到 nvl buffer 上，现在要加一个等待，必须等 token 就绪才搬；这个等待是按顺序等待，不能跳过 token，这样可能会施加比较强的阻塞条件，但可能还好，我们之前有同事做过模拟实验，combine 和计算的 overlap 率能达到 50%，再怎么阻塞也比完全不 overlap 赚了

我和计算那边也对 SM 资源进行了协商，约定 dispatch/combine 使用 48 SM，zip 使用 4 SM，其他的是计算的 persistent kernel。另外，所有 kernel 统一使用 2-CTA，避免 launch 时出现不同 CTA 冲突的情况。


### buffer设计

由于很多变量命名比较混乱，先去歧义一些定义：
`seq_len`：dispatch 前每个 rank 的 token 数量，也就是写在训练 recipe 里那个值
`num_recv_tokens`：dispatch 后当前 rank 收到的**不重复 token** 的数量，每轮 microbatch 都不同
`num_experts`：每个 rank 上的专家数（即本地专家数）
`tokens_per_expert[num_experts]`：dispatch 后当前 rank 上每个 expert 收到的 token 数量，是一个数组
`num_unzipped_tokens`：sum((n + 127) // 128 * 128 for n in tokens_per_expert)，也就是向 128 对齐后的展开的总 token 数，向 128 对齐是 GEMM 的固有要求

对 token 在 buffer 中的顺序消歧义如下：
`sequence序`：dispatch 之前的 token 顺序，也就是 hidden_states 中的顺序，确定性
`DeepEP序`：原版 DeepEP dispatch 收到不重复 token 的顺序，combine 的输入也用该顺序，确定性
`前向atomic序`：前向 dispatch + unzip 后得到的顺序，由于使用了 atomic 先到先放，非确定
`反向atomic序`：反向 dispatch + unzip 后得到的顺序，同样非确定，且和前向的顺序不一样，因此在一次前反向中其实存在两种不同的随机顺序

dispatch 原有的主要输出为：
`recv_tokens`[num_recv_tokens, hidden_size] bf16
`recv_token_indices`[num_recv_tokens, topk] int64: 收到的 token 属于本地哪些专家（最少1个，最多topk个，使用本地专家下标，无效部分用-1填充，未排序）
`recv_token_probs`[num_recv_tokens, topk] float32: 和 indices 一一对应的每个 token 对于每个专家的权重

融合了 unzip 后，dispatch 给计算的则是 unzip 的输出，里面每个专家的 token 连续排列，即前 tokens_per_expert[0] 个 token 是专家0的，**向128对齐后**，接下来的 tokens_per_expert[1] 个 token 是专家1的，以此类推
但是与原有 unzip 不同的是，原来 unzip 输出的 token 顺序是确定性的，而现在新的是非确定的，每个专家的段内是 atomic 先到先放，往前压实
* `unzipped_tokens`[num_unzipped_tokens, hidden_size] bf16
* `unzipped_probs`[num_unzipped_tokens] fp32

注：融合 unzip 之后，dispatch 原有的 recv_tokens 依然是要写的，反向要用

之后，上述两个变量就会给到计算部分进行计算，计算过程的 buffer 如下，在各 buffer 中 token 的位置都是不变的，都向 unzipped_tokens 对齐，其实 o1/o2 我们不用管，只看 o3 就行
* `o1`[num_unzipped_tokens, 2*intermediate_size] bf16 : gateup 的输出
* `o2`[num_unzipped_tokens, intermediate_size] bf16 : weighted_swiglu 的输出，注意 unzipped_probs 是在这里乘寄去而不是到后面 zip 的时候才乘，后面 zip 只有加法
* `o3`[num_unzipped_tokens, hidden_size] bf16 : down 的输出

zip 需要监控 o3 的完成情况，当一个 token 在 o3 里面已经被所有它所属的专家完成，就可以进行累加，累加结果输出到一个新 buffer，作为 combine 的输入
该 buffer 的顺序恢复到原来确定性的 DeepEP 的顺序，因此 zip 需要负责从 atomic 序到 DeepEP 序的转换
* `combine_input`[num_recv_tokens, hidden_size] bf16


### 信号设计

由于目前通信和计算在并行开发，目前已经约定了一部分 buffer 和信号的规范，剩下还没约定的可以先不纠结，会走一步看一步

dispatch 需要给到计算的除了 unzipped_tokens 和 unzipped_probs，还有一个任务队列：
`task_queue`[num_chunks, 4] int32 : 记录每个 chunk 的描述符和就绪信号，格式为 [expert_idx, m_start, m_size, ready]，使用 atomic 竞争下标并依次写入（随机顺序）
* num_chunks 是可以提前算出来的，因为 dispatch 开始前就已经知道 tokens_per_expert，则 num_chunks = sum(ceil(n / chunk) for n in tokens_per_expert)
* expert_idx 是该 chunk 属于哪个本地专家
* m_start 是该 chunk 在 unzipped_tokens 里的偏移 token 数
* m_size 是 chunk 包含的 token 数，一般为 chunk 大小，但对于一个专家的余数部分是可以小于 chunk 大小的，计算侧已经做了动态 M 的适配
* 该 buffer 初始化为全 0，使用 push 的形式，每有一个专家凑够一个 chunk 就往里 push，通信侧需要自己记录 push 到哪个下标了，但不需要让计算知道，因为计算 kernel 是靠 ready 信号判断的，不需要知道你 push 到第几个下标了

另外，需要两个映射表用于前向 atomic 序和 DeepEP 序之间的转换，这两张表都是随着 receiver 更新，收到 token 后才赋值，当然信号机制会保证消费者在读到 ready 信号时才会访问对应的值：
`atomic_to_zip`[num_unzipped_tokens] int32 : 使用 atomic 序，记录一个 token 指向 DeepEP 序里的哪个下标，padding 位置填 -1
`zip_to_atomic`[num_recv_tokens, topk] int32 : 使用 DeepEP 序，位置和 recv_token_indices 一一对应，记录对于每个有效的 token，zip 的时候应该去 o3 的哪个下标读，无效位置填 -1

为了让计算侧知道每个 token 的有效 topk 数，dispatch 还会给一张计数表，同样是随 ready 动态更新的：
`num_valid_topk`[num_recv_tokens] int32 : 使用 DeepEP 序，记录每个 token 在本地有几个专家，其值等价于**通信完成后** recv_token_indices 里面每行非 -1 的和，但是在运行时不相等，因为 recv_token_indices 的一行是分专家更新的，一个 chunk 就绪时只保证这个 chunk 所属专家在 recv_token_indices 里面的槽位就绪，不保证这一行所有专家都就绪，导致数少了；num_valid_topk 则是通过冗余更新解决这个问题，一个 token 的每个专家的 chunk 发布时都会重新写一次 topk 值；保险起见，num_valid_topk 初始化为 0，如果计算侧读到 0 则认为出错

计算那边给到我们的则是一个 token 完成队列，记录可以进行 zip 的 token 下标：
`zip_task_queue`[num_recv_tokens] int32 : 和 task_queue 类似使用 atomic 竞争入队，内容为计算完的 token 在 DeepEP 序中的下标；计算侧会计数每个 token 被多少个专家完成了，当完成的次数等于 num_valid_topk 里的值时，该 token 就会被 push 进来；该队列初始化为全 -1，这样当读到非 -1 时就知道就绪了，不需要额外 ready 信号；zip 从头开始连续扫描，遇到 -1 则等待，不可跳过

说明：计算侧使用了非常暴力的同步保证，计算侧对每个 chunk 都 launch 了一个独立的 gemm kernel，gemm 完成后也不由自己更新 zip_task_queue，而是又启动一个 kernel 来更新计数器和入队 zip_task_queue，因此 zip 读到任务时 token 一定已经写入 o3

zip 和 combine 之间还有一个信号，记录每个 token 是否可以被 combine：
`zip_done`[num_recv_tokens] int32 : 使用 DeepEP 序，仅使用 0/1 值表示即可；combine 时每个 channel 的 sender 仍然按原来的顺序进行发送，但是当一个 token 未就绪时需要等待，不可跳过


### 任务入队机制

这里需要单独用一节说明 chunk 任务入队的逻辑，这对正确性和性能都非常关键，开发者必须充分理解

首先明确一点，<b>一个 chunk 的最后一个下标的 token 完成写入</b> 不等于 <b>这个 chunk 里所有 token 都已写入</b>，因为 chunk 里的 token 是由很多的 receiver warp 并发写的，一个 warp 恰好负责最后一个下标的 token 只是说明它在 atomic 时抢到了这个 slot 下标，不代表它前面的 token 一定先于它写入

因此，写入下标和写完成必须分开计数，用`unzipped_expert_counter [num_experts] int32`来抢 slot 下标，用`unzip_chunk_done [num_chunks] int32`来记录写入完成，后者是 chunk 级别的计数，记录当前 chunk 中已经**完成写入**多少个 token，当写入数量恰好等于这个 chunk 的预期大小时，就可以入队这个 chunk

然而，这里就有个一个性能瓶颈，因为 unzip_chunk_done 必须等 token 写入并同步（release）才能更新，这个 release 信号的延迟很高，要等到一个 token（例如 8KB 级别）传到 L2 同步点才能完成，导致 receiver 等待时间变得很长；注意这里不是说 release 导致了更高的带宽占用，只是增加了 receiver 延迟，导致整个通信链条被阻塞

目前暂时保证了功能正确，后续性能优化的方向就是怎么让它隐藏这个延迟，例如使用更粗粒度的 release，写多个 token 才同步一次，这个可以到功能全部跑通后再做


### FP8支持

FP8 相对 BF16 改动很少，主要改动在计算那边，通信只需要将 dispatch 替换为 fp8 并多输出一个 scale 即可，combine 仍然维持 bf16
