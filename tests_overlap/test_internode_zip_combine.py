import time

import paddle
import paddle.nn.functional as F
import paddle.distributed as dist

paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)
paddle.enable_compat(scope={"deep_ep"})

import deep_ep
print("deep_ep:", deep_ep.__file__)

from utils import initialize_fleet, configure_buffer, get_buffer

E = 16
H = 4096
SEQLEN = 16384
TOPK = 8

NUM_SMS = 48
NUM_CHUNKS = 8  # 分块延迟置位的块数


def prepare_case_inputs(group):
    num_experts = group.world_size * E

    x = paddle.randn([SEQLEN, H], "bfloat16")
    scores = paddle.randn([SEQLEN, num_experts])
    scores += paddle.randn([num_experts]) * 0.1

    topk_weights, topk_idx = scores.topk(TOPK)
    topk_weights = F.sigmoid(topk_weights)
    topk_weights /= topk_weights.sum(axis=-1, keepdim=True)

    return x, topk_weights, topk_idx


def run_dispatch(group, buffer, x, token_probs, token_indices):
    """原版 dispatch, 只为了拿到 handle 和 recv_x"""
    num_experts = group.world_size * E

    layout = buffer.get_dispatch_layout(
        token_indices, num_experts, async_finish=False, allocate_on_comm_stream=False
    )
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = layout

    recv_x, _, _, _, handle, _ = buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    return recv_x, handle


def check(name, out, ref):
    diff = paddle.abs(out.float() - ref.float())
    count = int((diff != 0).sum())
    print(f"{name}: diff {count} avg {float(diff.mean()):.6f} max {float(diff.max()):.6f}",
          "PASS" if count == 0 else "FAIL")
    return count == 0


def main():
    group = initialize_fleet()
    configure_buffer(NUM_SMS)
    buffer = get_buffer(group, H * 2)
    x, token_probs, token_indices = prepare_case_inputs(group)

    recv_x, handle = run_dispatch(group, buffer, x, token_probs, token_indices)
    num_tokens = len(recv_x)
    print("num_recv_tokens:", num_tokens)

    # 假装这是 zip 的输出, 内容无所谓, 只要两次 combine 用同一份
    zipped = recv_x.clone()

    # 基准: 不传 zip_done, 走原来的无条件发送
    ref, _, _ = buffer.combine(zipped, handle, async_finish=False, allocate_on_comm_stream=False)
    ref = ref.clone()

    ############################ CASE 1: 全部已就绪 ############################

    zip_done = paddle.ones([num_tokens], "int32")
    paddle.base.core.nvprof_nvtx_push("combine_ready")
    out, _, _ = buffer.combine(zipped, handle, async_finish=False, allocate_on_comm_stream=False,
                               zip_done=zip_done)
    paddle.base.core.nvprof_nvtx_pop()
    ok = check("[all ready]", out, ref)

    ####################### CASE 2: 逆序分块延迟置位 #########################
    # 输入一开始全是垃圾值且没有一个 token 就绪, 生产者在计算流上逆序地一块块写入真实
    # 数据并置位。如果 sender 不等信号, 发出去的就是垃圾值; 如果 sender 跳过未就绪的
    # token, 结果也会错

    staging = paddle.full([num_tokens, H], float("nan"), "bfloat16")
    zip_done = paddle.zeros([num_tokens], "int32")
    ones = paddle.ones_like(zip_done)

    dist.barrier()
    paddle.base.core.nvprof_start()
    dist.all_reduce(paddle.empty([1])) # 使用 soft sync 尽可能同步集群但不阻塞 cpu

    paddle.base.core.nvprof_nvtx_push("combine_wait")
    out, _, event = buffer.combine(staging, handle, async_finish=True,
                                   allocate_on_comm_stream=False, zip_done=zip_done)
    paddle.base.core.nvprof_nvtx_pop()

    # 等前面的 all_reduce 完成 (即 combine 启动) 才开始更新 done 信号
    paddle.device.current_stream().synchronize()
    bounds = [num_tokens * i // NUM_CHUNKS for i in range(NUM_CHUNKS + 1)]

    paddle.base.core.nvprof_nvtx_push("zip_done")
    for i in reversed(range(NUM_CHUNKS)):
        time.sleep(0.0001)
        lo, hi = bounds[i], bounds[i + 1]
        # 使用平行拷贝 (底层对应 cudaMemcpyAsync), 避免触发 broadcast 等算子导致死锁
        paddle.assign(zipped[lo:hi], staging[lo:hi])
        paddle.assign(ones[lo:hi], zip_done[lo:hi])
    paddle.base.core.nvprof_nvtx_pop()

    event.current_stream_wait()
    ok = check("[delayed reverse]", out, ref) and ok
    print("RESULT:", "PASS" if ok else "FAIL")

    ############################ PERFORMANCE ############################

    for i in range(5):
        dist.all_reduce(paddle.empty([1]))
        paddle.base.core.nvprof_nvtx_push("combine_baseline")
        out, _, event = buffer.combine(zipped, handle, async_finish=False,
                                       allocate_on_comm_stream=False)
        paddle.base.core.nvprof_nvtx_pop()

    for i in range(5):
        dist.all_reduce(paddle.empty([1]))
        paddle.base.core.nvprof_nvtx_push("combine_zip_done")
        out, _, event = buffer.combine(zipped, handle, async_finish=False,
                                       allocate_on_comm_stream=False, zip_done=zip_done)
        paddle.base.core.nvprof_nvtx_pop()

    dist.barrier()
    paddle.base.core.nvprof_stop()


if __name__ == "__main__":
    main()
