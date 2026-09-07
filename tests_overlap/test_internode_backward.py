import os
import sys
from typing import NamedTuple

import paddle
from paddle import Tensor
import paddle.distributed as dist
from paddle.distributed import fleet
import paddle.nn.functional as F

paddle.empty([32, 1024, 1024, 1024], dtype="uint8")
paddle.set_printoptions(linewidth=200)
paddle.enable_compat(scope={"deep_ep"})

import deep_ep
print("deep_ep:", deep_ep.__file__)

import deep_gemm
print("deep_gemm:", deep_gemm.__file__)

from utils import initialize_fleet, configure_buffer, get_buffer, AsyncLoad

# 使用特别编译的注释掉所有第三方库的版本, 不然会和开发中的 deep_ep/deep_gemm 冲突
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
FUSE_SWIGLU = True
PRECISE_SWIGLU = False
OVERLAP_WGRAD = True


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


def align(n):
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def run_wgrad(tokens_per_expert, x, do1, w_gateup_grad, o2, do3, w_down_grad):
    ks_cpu = [align(n) for n in tokens_per_expert]
    async_load = AsyncLoad()
    grouped_layout = async_load(ks_cpu, dtype="int32")
    deep_gemm.set_num_sms(CALC_NUM_SMS) if OVERLAP_WGRAD else ()

    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        x, do1, w_gateup_grad, ks_cpu, grouped_layout, w_gateup_grad)
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        o2, do3, w_down_grad, ks_cpu, grouped_layout, w_down_grad)

    deep_gemm.set_num_sms(0)


def run_overlap(group, buffer, hidden_states, token_probs, token_indices, dout, w_gateup, w_down,
                w_gateup_grad, w_down_grad):
    num_experts = group.world_size * E
    deep_gemm.set_num_sms(CALC_NUM_SMS)
    paddle.base.core.nvprof_nvtx_push("forward")
    paddle.zeros([1])

    ############################# DISPATCH FORWARD #############################
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
        unzipped_tokens, unzipped_probs, atomic_to_zip, zip_to_atomic,
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

    tokens_per_expert = num_recv_tokens_per_expert_list
    num_unzipped_tokens = len(unzipped_tokens)
    num_tasks = len(task_queue)
    num_recv_tokens = len(recv_x)

    print("tokens_per_expert:", tokens_per_expert)
    print("num_tasks:", [(n + CHUNK - 1) // CHUNK for n in tokens_per_expert], "=", num_tasks)
    print("num_recv_tokens:", num_recv_tokens)

    ############################### GEMM FORWARD ###############################

    o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
    o2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
    o3 = paddle.empty_like(unzipped_tokens)
    zipped_out = paddle.empty_like(recv_x)

    token_done = paddle.zeros([num_recv_tokens], "int32")
    zip_task_queue = paddle.full([num_recv_tokens], -1, "int32")
    zip_queue_tail = paddle.zeros([1], "int32")
    zip_done = paddle.zeros([num_recv_tokens], "int32")

    full_compute = False
    combine_event = None

    for task_idx in range(num_tasks):
        paddle.base.core.nvprof_nvtx_push(f"task_{task_idx}")
        deep_gemm.bf16_chunk_gemm_nn(
            unzipped_tokens, w_gateup, o1, task_queue, task_idx,
            **(dict(o2=o2, probs=unzipped_probs) if FUSE_SWIGLU else {}))
        if not FUSE_SWIGLU:
            deep_gemm.chunk_weighted_swiglu(
                o1, unzipped_probs, o2, task_queue, task_idx, precise=PRECISE_SWIGLU)
        deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, task_queue, task_idx)
        deep_gemm.chunk_zip(o3, zipped_out, atomic_to_zip, zip_to_atomic, recv_token_indices,
                            num_valid_topk, token_done, zip_done, task_queue, task_idx, CHUNK)
        paddle.base.core.nvprof_nvtx_pop()

        if not full_compute and task_idx >= num_tasks * 0.2:
            print(f"switch to full compute at {task_idx}/{num_tasks}")
            full_compute = True
            deep_gemm.set_num_sms(CALC_NUM_SMS + COMM_NUM_SMS)

        if combine_event is None and task_idx >= num_tasks * 0.7:
            print(f"capture combine_event at {task_idx}/{num_tasks}")
            combine_event = deep_ep.Buffer.capture()
            deep_gemm.set_num_sms(CALC_NUM_SMS)

    ############################# COMBINE FORWARD ##############################

    out, _, event = buffer.combine(zipped_out, handle, async_finish=False,
                                   previous_event=combine_event, allocate_on_comm_stream=False,
                                   zip_done=zip_done)

    paddle.zeros([1])
    paddle.base.core.nvprof_nvtx_pop()
    dist.all_reduce(paddle.empty([1]))

    ############################# COMBINE BACKWARD #############################

    paddle.base.core.nvprof_nvtx_push("backward")
    paddle.zeros([1])

    # 本来 dispatch 反向应该用 cache_mode, 但目前 cache_mode 无法计算 atomic_to_zip/zip_to_atomic,
    # 因此只能像前向一样再跑一遍, 会浪费一定的通信带宽
    (
        _, _, _, _, handle, event, do3, _,
        atomic_to_zip_bwd, zip_to_atomic_bwd, _, task_queue_bwd
    ) = buffer.dispatch(
        dout,
        topk_idx=token_indices,
        topk_weights=token_probs,  # placeholder
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        async_finish=True,
        allocate_on_comm_stream=False,
        unzip_alignment=ALIGNMENT,
        unzip_chunk_size=CHUNK,
    )

    ############################## GEMM BACKWARD ###############################

    dx = paddle.empty_like(unzipped_tokens)
    do1 = paddle.empty_like(o1)
    do2 = paddle.empty_like(o2)
    drecv_x = paddle.empty_like(recv_x)
    drecv_probs = paddle.zeros_like(recv_token_probs)  # 无效位预先填0
    o2_bwd = paddle.empty_like(o2)

    token_done = paddle.zeros([len(recv_token_probs)], dtype="int32")
    zip_done = paddle.zeros([len(recv_token_probs)], dtype="int32")

    full_compute = False
    combine_event = None

    for task_idx in range(len(task_queue_bwd)):
        paddle.base.core.nvprof_nvtx_push(f"task_{task_idx}")
        deep_gemm.bf16_chunk_gemm_nt(do3, w_down, do2, task_queue_bwd, task_idx)
        deep_gemm.chunk_weighted_swiglu_grad(
            o1, unzipped_probs, do2, o2_bwd, do1, drecv_probs, atomic_to_zip_bwd, zip_to_atomic,
            recv_token_indices, task_queue_bwd, task_idx, precise=PRECISE_SWIGLU)
        deep_gemm.bf16_chunk_gemm_nt(do1, w_gateup, dx, task_queue_bwd, task_idx)
        deep_gemm.chunk_zip(dx, drecv_x, atomic_to_zip_bwd, zip_to_atomic_bwd, recv_token_indices,
                            num_valid_topk, token_done, zip_done, task_queue_bwd, task_idx, CHUNK)
        paddle.base.core.nvprof_nvtx_pop()

        if not full_compute and task_idx >= num_tasks * 0.2:
            print(f"switch to full compute at {task_idx}/{num_tasks}")
            full_compute = True
            deep_gemm.set_num_sms(CALC_NUM_SMS + COMM_NUM_SMS)

        if combine_event is None and task_idx >= num_tasks * 0.7:
            print(f"capture combine_event at {task_idx}/{num_tasks}")
            combine_event = deep_ep.Buffer.capture()
            deep_gemm.set_num_sms(CALC_NUM_SMS)

    ############################ DISPATCH BACKWARD #############################

    dhidden_states, dtoken_probs, event = buffer.combine(
        drecv_x, handle, drecv_probs, async_finish=OVERLAP_WGRAD, previous_event=combine_event,
        allocate_on_comm_stream=False, zip_done=zip_done)

    run_wgrad(tokens_per_expert, unzipped_tokens, do1, w_gateup_grad, o2_bwd, do3, w_down_grad)

    event.current_stream_wait() if OVERLAP_WGRAD else ()

    paddle.zeros([1])
    paddle.base.core.nvprof_nvtx_pop()
    dist.all_reduce(paddle.empty([1]))

    return out, do1, dhidden_states, dtoken_probs, tokens_per_expert, atomic_to_zip_bwd


def run_baseline(group, buffer, hidden_states, token_probs, token_indices, dout, w_gateup, w_down,
                 w_gateup_grad, w_down_grad):
    num_experts = group.world_size * E
    deep_gemm.set_num_sms(0)
    paddle.base.core.nvprof_nvtx_push("forward")
    paddle.zeros([1])

    ############################# DISPATCH FORWARD #############################
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

    m_indices = paddle.concat(
        [paddle.full([align(n)], i, "int32") for i, n in enumerate(tokens_per_expert)])

    ############################### GEMM FORWARD ###############################

    (
        unzipped_tokens,
        zipped_expertwise_rowmap,
        unzipped_probs,
        _,
    ) = paddle.nn.functional.moe_permute(
        recv_x,
        None,  # scale
        recv_token_indices,
        recv_token_probs,
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    o1 = paddle.empty([len(unzipped_tokens), 2 * I], dtype="bfloat16")
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(unzipped_tokens, w_gateup, o1, m_indices)

    o2 = paddlefleet_ops.fused_swiglu_scale(o1, unzipped_probs)

    o3 = paddle.empty_like(unzipped_tokens)
    deep_gemm.m_grouped_bf16_gemm_nn_contiguous(o2, w_down, o3, m_indices)

    zipped_tokens, zipped_probs = paddle.nn.functional.moe_unpermute(
        o3,
        zipped_expertwise_rowmap,
        recv_token_indices,
        unzipped_probs,
        total_zipped_tokens=len(recv_x),
        num_experts=E,
    )

    ############################# COMBINE FORWARD ##############################

    out, _, event = buffer.combine(zipped_tokens, handle, async_finish=False,
                                   allocate_on_comm_stream=False)

    paddle.zeros([1])
    paddle.base.core.nvprof_nvtx_pop()
    dist.all_reduce(paddle.empty(1))

    ############################# COMBINE BACKWARD #############################

    paddle.base.core.nvprof_nvtx_push("backward")
    paddle.zeros([1])

    drecv_out, _, _, _, _, event = buffer.dispatch(
        dout, handle=handle, async_finish=False, allocate_on_comm_stream=False)

    ############################## GEMM BACKWARD ###############################

    do3, _, _, _ = paddle.nn.functional.moe_permute(
        drecv_out,
        None,  # scale
        recv_token_indices,
        recv_token_probs, # placeholder
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    do2 = paddle.empty_like(o2)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(do3, w_down, do2, m_indices)

    do1, dprobs = paddlefleet_ops.fused_swiglu_scale_bwd(o1, unzipped_probs, do2)

    dx = paddle.empty_like(unzipped_tokens)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(do1, w_gateup, dx, m_indices)

    drecv_x, drecv_probs = paddle.nn.functional.moe_unpermute(
        dx,
        zipped_expertwise_rowmap,
        recv_token_indices,
        dprobs,
        total_zipped_tokens=len(recv_x),
        num_experts=E,
    )

    ############################ DISPATCH BACKWARD #############################

    event = deep_ep.Buffer.capture() if OVERLAP_WGRAD else None

    dhidden_states, dtoken_probs, event = buffer.combine(
        drecv_x, handle, drecv_probs, async_finish=OVERLAP_WGRAD, previous_event=event,
        allocate_on_comm_stream=False)

    run_wgrad(tokens_per_expert, unzipped_tokens, do1, w_gateup_grad, o2, do3, w_down_grad)

    event.current_stream_wait() if OVERLAP_WGRAD else ()

    paddle.zeros([1])
    paddle.base.core.nvprof_nvtx_pop()
    dist.all_reduce(paddle.empty([1]))

    return out, do1, dhidden_states, dtoken_probs


def check(out, ref, perm, offset):
    # reorder out to deep_ep order
    out = paddle.gather(out, perm + offset)
    diff = paddle.abs(
        out.float() - ref[offset : offset + out.shape[0]].float()
    )
    count = int((diff != 0).sum())
    avg, max_ = float(diff.mean()), float(diff.max())
    return count == 0, f"{count} (avg: {avg}, max: {max_})"


def interleave_gateup(gateup):
    gate, up = gateup.chunk(2, axis=-1)
    return paddle.concat(
        [gate.unsqueeze(-1), up.unsqueeze(-1)], axis=-1
    ).reshape(gateup.shape)


def check(x, y):
    diff = paddle.abs(x.float() - y.float())
    avg, max = float(diff.mean()), float(diff.max())
    banner = "" if (avg == 0 and max == 0) else (" " + "X" * 40)
    avg = "0.0" if avg == 0 else f"{avg:e}"
    max = "0.0" if max == 0 else f"{max:e}"
    return f"avg: {avg} max: {max}" + banner


def get_atomic_perm(tokens_per_expert, atomic_to_zip):
    """将 atomic 序的 o1/o2/o3 等转换为参考序的映射表, padding 保留原位."""
    perm = paddle.arange(sum(align(n) for n in tokens_per_expert))
    offset = 0
    for n in tokens_per_expert:
        perm[offset : offset + n] = atomic_to_zip[offset : offset + n].argsort() + offset
        offset += align(n)
    return perm


def main():
    group = initialize_fleet()
    configure_buffer(48)
    buffer = get_buffer(group, H * 2)
    x, token_probs, token_indices, w_gateup, w_down = prepare_case_inputs(group)
    dout = paddle.randn_like(x)
    w_gateup_interleaved = interleave_gateup(w_gateup)

    # wgrad 使用累加语义, 需要提前置 0
    w_gateup_grad = paddle.zeros(w_gateup.shape, dtype="float32")
    w_down_grad = paddle.zeros(w_down.shape, dtype="float32")
    w_gateup_grad_ref = paddle.zeros_like(w_gateup_grad)
    w_down_grad_ref = paddle.zeros_like(w_down_grad)

    configure_buffer(COMM_NUM_SMS)

    # validate
    out, do1, dx, dprobs, tokens_per_expert, atomic_to_zip = run_overlap(
        group, buffer, x, token_probs, token_indices, dout, w_gateup_interleaved, w_down,
        w_gateup_grad, w_down_grad)
    out_ref, do1_ref, dx_ref, dprobs_ref = run_baseline(
        group, buffer, x, token_probs, token_indices, dout, w_gateup, w_down, w_gateup_grad_ref,
        w_down_grad_ref)

    atomic_perm = get_atomic_perm(tokens_per_expert, atomic_to_zip)

    print("out:", check(out, out_ref))
    print("do1:", check(do1[atomic_perm], interleave_gateup(do1_ref)))
    print("dx:", check(dx, dx_ref))  # gateup interleave 累加顺序不同
    print("dprobs:", check(dprobs, dprobs_ref))
    print("w_gateup_grad:", check(w_gateup_grad, interleave_gateup(w_gateup_grad_ref)))
    print("w_down_grad:", check(w_down_grad, w_down_grad_ref))

    # profile
    paddle.base.core.nvprof_start()
    dist.all_reduce(paddle.empty([1]))

    for i in range(5):
        paddle.base.core.nvprof_nvtx_push(f"overlap_{i}")
        run_overlap(group, buffer, x, token_probs, token_indices, dout, w_gateup_interleaved,
                    w_down, w_gateup_grad, w_down_grad)
        paddle.base.core.nvprof_nvtx_pop()

    for i in range(5):
        paddle.base.core.nvprof_nvtx_push(f"baseline_{i}")
        run_baseline(group, buffer, x, token_probs, token_indices, dout, w_gateup, w_down,
                     w_gateup_grad_ref, w_down_grad_ref)
        paddle.base.core.nvprof_nvtx_pop()

    dist.barrier()
    paddle.base.core.nvprof_stop()


if __name__ == "__main__":
    main()
