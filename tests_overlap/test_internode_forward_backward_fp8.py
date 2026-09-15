import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
import paddle.nn.functional as F

paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)
paddle.enable_compat(scope={"deep_ep"})

import deep_ep
print("deep_ep:", deep_ep.__file__)

import deep_gemm
print("deep_gemm:", deep_gemm.__file__)

from utils import (
    initialize_fleet, configure_buffer, get_buffer, quant_input, quant_weight, dequant
)

# 使用特别编译的注释掉 deep_ep/deep_gemm 的版本, 否则会头文件冲突
import paddlefleet_ops
assert not paddlefleet_ops._DEEP_EP_AVAILABLE
assert not paddlefleet_ops._DEEP_GEMM_AVAILABLE

E = 8
H = 4096
I = 2048
SEQLEN = 16384
TOPK = 8

COMM_NUM_SMS = 48
CALC_NUM_SMS = 100

ALIGNMENT = 128
CHUNK = 4096
USE_UE8M0 = True  # 当前只支持 ue8m0


def prepare_case_inputs(group):
    # E 是本地专家数, num_experts 是全局专家数
    num_experts = group.world_size * E

    x = paddle.randn([SEQLEN, H], "bfloat16")

    scores = paddle.randn([SEQLEN, num_experts])
    # 模拟给专家选择增加一定的系统不均衡
    # scores += paddle.randn([num_experts]) * 0.1

    topk_weights, topk_idx = scores.topk(TOPK)
    topk_weights = F.sigmoid(topk_weights)
    topk_weights /= topk_weights.sum(axis=-1, keepdim=True)

    w_gateup = (paddle.randn([E, H, 2 * I]) * 0.02).cast("bfloat16")
    w_down = (paddle.randn([E, I, H]) * 0.02).cast("bfloat16")

    return x, topk_weights, topk_idx, w_gateup, w_down


def run_baseline_forward(group, buffer, hidden_states, token_probs, token_indices,
                         w_gateup_t, w_down_t):
    num_experts = group.world_size * E
    deep_gemm.set_num_sms(0)
    paddle.zeros([1])

    ################################# DISPATCH #################################
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )

    (
        recv_x, recv_token_indices, recv_token_probs,
        num_recv_tokens_per_expert_list, handle, event,
    ) = buffer.dispatch(
        hidden_states,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        async_finish=False,
        allocate_on_comm_stream=False,
    )

    tokens_per_expert = num_recv_tokens_per_expert_list
    recv_token_indices = recv_token_indices.cast("int32")
    print("tokens_per_expert:", tokens_per_expert)

    ################################## UNZIP ###################################

    recv_x_fp8, recv_scale = recv_x

    (
        unzipped_tokens,
        zipped_expertwise_rowmap,
        unzipped_probs,
        unzipped_scale,
    ) = paddle.nn.functional.moe_permute(
        recv_x_fp8,
        recv_scale,
        recv_token_indices,
        recv_token_probs,
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
        using_ue8m0_scale=USE_UE8M0,
    )

    unzipped_x = (unzipped_tokens, unzipped_scale)

    ################################### GEMM ###################################

    o1 = paddle.empty([len(unzipped_tokens), 2 * I], dtype="bfloat16")
    m_indices = paddle.concat(
        [
            paddle.full([(n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT], i, "int32")
            for i, n in enumerate(tokens_per_expert)
        ]
    )

    # x_scale 也要求是 transpose 的, 但是 deep_ep 不支持发送 transpose 的 scale,
    # 因此前面通信仍然用 contiguous 的, 计算时再 transpose.
    # paddle 的部分 quant 函数输出底层已经是 transpose 的, 但是为了可读性和安全性,
    # 这里显式调用一下 transpose, 如果底层已经是 transpose 的下面就是空操作
    unzipped_scale = unzipped_scale.T.contiguous().T

    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (unzipped_tokens, unzipped_scale), w_gateup_t, o1, m_indices)

    o2_fp8, o2_scale = paddlefleet_ops.fuse_weighted_swiglu_fp8_quant(
        o1, unzipped_probs, using_pow2_scaling=True, use_ue8m0=USE_UE8M0)

    o2 = (o2_fp8, o2_scale)
    o2_scale = o2_scale.T.contiguous().T

    o3 = paddle.empty([len(unzipped_tokens), H], dtype="bfloat16")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((o2_fp8, o2_scale), w_down_t, o3, m_indices)

    ################################### ZIP ####################################

    zipped_tokens, zipped_probs = paddle.nn.functional.moe_unpermute(
        o3,
        zipped_expertwise_rowmap,
        recv_token_indices,
        unzipped_probs,
        total_zipped_tokens=len(recv_token_indices),
        num_experts=E,
    )

    ################################# COMBINE ##################################

    out, _, event = buffer.combine(zipped_tokens, handle, async_finish=False,
                                   allocate_on_comm_stream=False)

    dist.all_reduce(paddle.zeros([1]))
    return recv_x, unzipped_x, o1, o2, o3, zipped_tokens, out


def run_overlap_forward(group, buffer, hidden_states, token_probs, token_indices):
    num_experts = group.world_size * E
    deep_gemm.set_num_sms(CALC_NUM_SMS)
    paddle.zeros([1])

    ################################# DISPATCH #################################
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )

    (
        recv_x, recv_token_indices, recv_token_probs,
        num_recv_tokens_per_expert_list, handle, event,
        unzipped_x, unzipped_probs, atomic_to_zip, zip_to_atomic,
        num_valid_topk, task_queue
    ) = buffer.dispatch(
        hidden_states,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        async_finish=True,
        allocate_on_comm_stream=False,
        unzip_alignment=ALIGNMENT,
        unzip_chunk_size=CHUNK,
    )

    unzipped_tokens, unzipped_scale = unzipped_x

    tokens_per_expert = num_recv_tokens_per_expert_list
    num_unzipped_tokens = len(unzipped_tokens)
    num_tasks = len(task_queue)
    num_recv_tokens = len(recv_x[0])

    print("tokens_per_expert:", tokens_per_expert)
    print("num_tasks:", [(n + CHUNK - 1) // CHUNK for n in tokens_per_expert], "=", num_tasks)
    print("num_recv_tokens:", num_recv_tokens)

    event.current_stream_wait()

    return recv_x, unzipped_x, atomic_to_zip, tokens_per_expert


def as_bytes(x):
    return x.contiguous().view("uint8")


def count_diff_bytes(out, ref):
    return int((as_bytes(out) != as_bytes(ref)).sum())


def check_bytes(out, ref, perm, offset):
    # reorder out to deep_ep order
    out = paddle.gather(out, perm + offset)
    count = int((out != ref[offset : offset + out.shape[0]]).sum())
    return count == 0, f"{count}"


def main():
    group = initialize_fleet()
    configure_buffer(COMM_NUM_SMS)
    buffer = get_buffer(group, H * 2)
    x, token_probs, token_indices, w_gateup, w_down = prepare_case_inputs(group)

    x_quant = quant_input(x, USE_UE8M0)
    # fp8 的权重 layout 和 bf16 不同, 因为 fp8 的性能对 layout 敏感
    w_gateup_t_quant = quant_weight(w_gateup, transpose=True, use_ue8m0=USE_UE8M0)
    w_down_t_quant = quant_weight(w_down, transpose=True, use_ue8m0=USE_UE8M0)

    # warmup
    run_baseline_forward(
        group, buffer, x_quant, token_probs, token_indices, w_gateup_t_quant, w_down_t_quant)
    run_overlap_forward(group, buffer, x_quant, token_probs, token_indices)

    # validate
    refs = run_baseline_forward(
        group, buffer, x_quant, token_probs, token_indices, w_gateup_t_quant, w_down_t_quant)
    recv_x, unzipped_x, atomic_to_zip, tokens_per_expert = run_overlap_forward(
        group, buffer, x_quant, token_probs, token_indices)

    names = ["recv_x", "unzipped_x", "o1", "o2", "o3", "zipped_tokens", "out"]
    for name, ref in zip(names, refs):
        if isinstance(ref, tuple):
            ref = dequant(*ref, USE_UE8M0)
        print(name + ":", ref)

    recv_x_ref, unzipped_x_ref = refs[0], refs[1]

    print("[recv_x] token diff:", count_diff_bytes(recv_x[0], recv_x_ref[0]),
          "scale diff:", count_diff_bytes(recv_x[1], recv_x_ref[1]))

    # unzipped_x 的段内是 atomic 序, 先按 atomic_to_zip 排回 DeepEP 序再比
    outs = [as_bytes(t) for t in unzipped_x]
    refs_ = [as_bytes(t) for t in unzipped_x_ref]
    print("unzipped shapes:", [t.shape for t in outs], "vs", [t.shape for t in refs_])

    offset = 0
    for i, n in enumerate(tokens_per_expert):
        perm = atomic_to_zip[offset : offset + n].argsort()

        ok1, msg1 = check_bytes(outs[0], refs_[0], perm, offset)
        ok2, msg2 = check_bytes(outs[1], refs_[1], perm, offset)

        print(f"[expert {i}] token: {msg1} scale: {msg2}",
              *([] if (ok1 and ok2) else ["X" * 80]))

        offset += (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    # profile
    paddle.base.core.nvprof_start()
    dist.all_reduce(paddle.empty([1]))

    for i in range(5):
        paddle.base.core.nvprof_nvtx_push(f"baseline_{i}")
        run_baseline_forward(
            group, buffer, x_quant, token_probs, token_indices, w_gateup_t_quant, w_down_t_quant)
        paddle.base.core.nvprof_nvtx_pop()

    for i in range(5):
        paddle.base.core.nvprof_nvtx_push(f"overlap_{i}")
        run_overlap_forward(group, buffer, x_quant, token_probs, token_indices)
        paddle.base.core.nvprof_nvtx_pop()

    dist.barrier()
    paddle.base.core.nvprof_stop()


if __name__ == "__main__":
    main()
