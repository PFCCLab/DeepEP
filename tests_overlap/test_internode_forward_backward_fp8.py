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

from utils import initialize_fleet, configure_buffer, get_buffer, AsyncLoad, GroupedTaskLauncher

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
CHUNK = 4096
COMBINE_OVERLAP_RATIO = 0.3

ALIGNMENT = 128
USE_UE8M0 = True  # 当前只支持 ue8m0
QUANT_BLOCK_SIZE = 512

# DeepEP doesn't expose its comm stream, use this as a parallel stream to
# launch compute kernels and proxy DeepEP events
comm_stream = paddle.cuda.Stream()
comm_event = paddle.cuda.Event()
calc_stream = paddle.cuda.current_stream()


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


def quant_input(x):
    """对于 hidden_states, 在 hidden 维上使用 128 分块量化."""
    x_fp8, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        x,
        quant_method="1x128",
        output_scale_transpose=False,
        using_ue8m0_scale=USE_UE8M0,
    )
    assert x_fp8.shape == x.shape
    assert scale.shape == [x.shape[0], x.shape[1] // QUANT_BLOCK_SIZE]
    assert x_fp8.is_contiguous()
    assert scale.is_contiguous()
    return x_fp8, scale


def quant_weight(w, transpose=False):
    import paddlefleet_ops
    # quant 算子只接受 list 输入，这里手动切成列表
    expert_weight_list = list(w)
    if transpose:
        w_fp8, scale = paddlefleet_ops.fuse_stack_transpose_fp8_quant(
            expert_weight_list,
            using_pow2_scaling=False,
            using_ue8m0_scale=USE_UE8M0,
            output_scale_transpose=False,
        )
        assert w_fp8.shape == [w.shape[0] * w.shape[2], w.shape[1]]
        assert scale.shape == [w.shape[0] * w.shape[2], w.shape[1] // QUANT_BLOCK_SIZE]
        assert w_fp8.is_contiguous()
        assert scale.is_contiguous()
    else:
        w_fp8, scale = paddlefleet_ops.fuse_stack_fp8_quant(
            expert_weight_list,
            using_pow2_scaling=False,
            using_ue8m0_scale=USE_UE8M0,
            output_scale_transpose=False,
        )
        assert w_fp8.shape == [w.shape[0] * w.shape[1], w.shape[2]]
        assert scale.shape == [w.shape[0] * w.shape[1], w.shape[2] // QUANT_BLOCK_SIZE]
        assert w_fp8.is_contiguous()
        assert scale.is_contiguous()
    # quant 算子输出把专家维铺平了，需要重新展开
    w_fp8 = w_fp8.reshape([w.shape[0], -1, w_fp8.shape[1]])
    scale = scale.reshape([w.shape[0], -1, scale.shape[1]])
    # ue8m0 要求 scale 最后两维 transpose
    if USE_UE8M0:
        scale = scale.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
    return w_fp8, scale


def dequant(x, scale):
    if USE_UE8M0:
        scale = 2.0 ** (scale.contiguous().view("int8").cast("int32") - 127)
    return x.float() * scale.repeat_interleave(128, axis=-1)


def run_baseline(group, buffer, hidden_states, dout, token_probs, token_indices,
                 w_gateup, w_down, w_gateup_t, w_down_t, w_gateup_grad, w_down_grad):
    num_experts = group.world_size * E
    deep_gemm.set_num_sms(0)
    paddle.base.core.nvprof_nvtx_push("forward")

    ############################# DISPATCH FORWARD #############################

    hidden_states = quant_input(hidden_states)

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

    ############################### GEMM FORWARD ###############################

    o1 = paddle.empty([len(unzipped_tokens), 2 * I], dtype="bfloat16")
    m_indices = paddle.concat(
        [paddle.full([align(n)], i, "int32") for i, n in enumerate(tokens_per_expert)])

    # x_scale 也要求是 transpose 的, 但是 deep_ep 不支持发送 transpose 的 scale,
    # 因此前面通信仍然用 contiguous 的, 计算时再 transpose.
    # paddle 的一些 quant 函数输出底层已经是 transpose 的, 但是为了可读性和安全性,
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

    zipped_tokens, zipped_probs = paddle.nn.functional.moe_unpermute(
        o3,
        zipped_expertwise_rowmap,
        recv_token_indices,
        unzipped_probs,
        total_zipped_tokens=len(recv_token_indices),
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

    do3, _, _, _ = paddle.nn.functional.moe_permute(
        drecv_out,
        None,  # scale
        recv_token_indices,
        recv_token_probs, # placeholder
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )

    ############################## GEMM BACKWARD ###############################

    # 反向 dispatch 发的是 BF16, 由计算阶段自己 quant
    do3_fp8, do3_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        do3,
        output_scale_transpose=True,
        quant_method="1x128",
        input_transpose=False,
        using_ue8m0_scale=USE_UE8M0,
    )

    do2 = paddle.empty(o2_fp8.shape, dtype="bfloat16")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((do3_fp8, do3_scale.T), w_down, do2, m_indices)

    # o2_bwd 的精度和前向的 o2 不同，但这与主干梯度无关，o2_bwd 只供 wgrad 使用
    do1, dprobs, o2_bwd = paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
        o1, do2, unzipped_probs.unsqueeze(-1))

    do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        do1,
        output_scale_transpose=True,
        quant_method="1x128",
        input_transpose=False,
        using_ue8m0_scale=USE_UE8M0,
    )
    do1_quant = (do1_fp8, do1_scale.T)

    dx = paddle.empty(unzipped_tokens.shape, dtype="bfloat16")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((do1_fp8, do1_scale.T), w_gateup, dx, m_indices)

    drecv_x, drecv_probs = paddle.nn.functional.moe_unpermute(
        dx,
        zipped_expertwise_rowmap,
        recv_token_indices,
        dprobs,
        total_zipped_tokens=len(recv_token_indices),
        num_experts=E,
    )

    ############################ DISPATCH BACKWARD #############################

    dhidden_states, dtoken_probs, event = buffer.combine(
        drecv_x, handle, drecv_probs, async_finish=True, previous_event=deep_ep.Buffer.capture(),
        allocate_on_comm_stream=False)

    paddle.base.core.nvprof_nvtx_push("wgrad")
    deep_gemm.set_num_sms(CALC_NUM_SMS)

    ks_cpu = [align(n) for n in tokens_per_expert]
    async_load = AsyncLoad()
    grouped_layout = async_load(ks_cpu, dtype="int32")

    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        o2_bwd, do3, w_down_grad, ks_cpu, grouped_layout, w_down_grad)
    # 为了使用 bf16 的 wgrad，需要对 x_fp8 激活进行 dequant, 既没有真正提升精度还增加了计算量
    x_dequant = paddle.incubate.nn.functional.fused_act_dequant(unzipped_tokens, unzipped_scale)
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        x_dequant, do1, w_gateup_grad, ks_cpu, grouped_layout, w_gateup_grad)

    deep_gemm.set_num_sms(0)
    paddle.base.core.nvprof_nvtx_pop()

    event.current_stream_wait()
    paddle.zeros([1])
    paddle.base.core.nvprof_nvtx_pop()
    dist.all_reduce(paddle.empty([1]))

    return out, dhidden_states, dtoken_probs


def run_overlap(group, buffer, hidden_states, dout, token_probs, token_indices,
                w_gateup, w_down, w_gateup_t, w_down_t, w_gateup_grad, w_down_grad, logging=False):
    num_experts = group.world_size * E
    deep_gemm.set_num_sms(CALC_NUM_SMS)
    paddle.base.core.nvprof_nvtx_push("forward")

    ############################# DISPATCH FORWARD #############################

    hidden_states = quant_input(hidden_states)

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
    unzipped_x = (unzipped_tokens, unzipped_scale.T)

    tokens_per_expert = num_recv_tokens_per_expert_list
    num_unzipped_tokens = len(unzipped_tokens)
    num_tasks = len(task_queue)
    num_recv_tokens = len(recv_x[0])

    with paddle.device.stream_guard(comm_stream):
        event.current_stream_wait()
        comm_event.record()

    if logging:
        print("tokens_per_expert:", tokens_per_expert)
        print("num_tasks:", [(n + CHUNK - 1) // CHUNK for n in tokens_per_expert], "=", num_tasks)
        print("num_recv_tokens:", num_recv_tokens)

    ############################### GEMM FORWARD ###############################

    o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
    o2_fp8 = paddle.empty([num_unzipped_tokens, I], dtype="float8_e4m3fn")
    # 注意即使是中间变量的 o2_scale 也要转置
    o2_scale = paddle.empty([I // QUANT_BLOCK_SIZE, num_unzipped_tokens], dtype="int32").T
    o3 = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
    zipped_out = paddle.empty([num_recv_tokens, H], dtype="bfloat16")

    token_done = paddle.zeros([num_recv_tokens], dtype="int32")
    zip_done = paddle.zeros([num_recv_tokens], dtype="int32")

    funcs = [
        lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(
            unzipped_x, w_gateup_t, o1, task_queue, task_idx),
        lambda task_idx: deep_gemm.chunk_weighted_swiglu(
            o1, unzipped_probs, o2_fp8, task_queue, task_idx, CHUNK, o2_scales=o2_scale),
        lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(
            (o2_fp8, o2_scale), w_down_t, o3, task_queue, task_idx),
        lambda task_idx: deep_gemm.chunk_zip(
            o3, zipped_out, atomic_to_zip, zip_to_atomic, recv_token_indices, num_valid_topk,
            token_done, zip_done, task_queue, task_idx, CHUNK),
    ]

    task_launcher = GroupedTaskLauncher(
        funcs, num_tasks, calc_stream, comm_stream, comm_event, COMBINE_OVERLAP_RATIO)

    n = task_launcher.run_dispatch_overlap()
    print("[FW] dispatch->compute:", n) if logging else ()

    deep_gemm.set_num_sms(0)
    n = task_launcher.run_compute()
    deep_gemm.set_num_sms(CALC_NUM_SMS)
    print("[FW] compute->combine:", n) if logging else ()

    ############################# COMBINE FORWARD ##############################

    combine_event = deep_ep.Buffer.capture()

    out, _, event = buffer.combine(zipped_out, handle, async_finish=True,
                                   previous_event=combine_event, allocate_on_comm_stream=False,
                                   zip_done=zip_done)

    task_launcher.run_combine_overlap()

    event.current_stream_wait()
    paddle.zeros([1])
    paddle.base.core.nvprof_nvtx_pop()
    dist.all_reduce(paddle.empty([1]))

    ############################# COMBINE BACKWARD #############################

    paddle.base.core.nvprof_nvtx_push("backward")

    dout_quant = quant_input(dout)

    (
        _, _, _, _, handle, event, do3, _,
        atomic_to_zip_bwd, zip_to_atomic_bwd, _, task_queue_bwd
    ) = buffer.dispatch(
        dout_quant,
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

    do3_fp8, do3_scale = do3
    do3 = (do3_fp8, do3_scale.T)

    with paddle.device.stream_guard(comm_stream):
        event.current_stream_wait()
        comm_event.record()

    ############################## GEMM BACKWARD ###############################

    do2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
    dx = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
    do1_fp8 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="float8_e4m3fn")
    do1_scale = paddle.empty([2 * I // QUANT_BLOCK_SIZE, num_unzipped_tokens], dtype="int32").T
    do1 = (do1_fp8, do1_scale)
    o2_bwd_fp8 = paddle.empty([num_unzipped_tokens, I], dtype="float8_e4m3fn")
    o2_bwd_scale = paddle.empty([I // QUANT_BLOCK_SIZE, num_unzipped_tokens], dtype="int32").T
    o2_bwd = (o2_bwd_fp8, o2_bwd_scale)
    drecv_x = paddle.empty_like(zipped_out)
    drecv_probs = paddle.zeros_like(recv_token_probs)  # 无效位预先填0

    token_done = paddle.zeros([num_recv_tokens], dtype="int32")
    zip_done = paddle.zeros([num_recv_tokens], dtype="int32")

    funcs = [
        lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(do3, w_down, do2, task_queue_bwd, task_idx),
        lambda task_idx: deep_gemm.chunk_weighted_swiglu_grad(
            o1, unzipped_probs, do2, o2_bwd_fp8, do1_fp8, drecv_probs, atomic_to_zip_bwd,
            zip_to_atomic, recv_token_indices, task_queue_bwd, task_idx, CHUNK,
            o2_bwd_scales=o2_bwd_scale, do1_scales=do1_scale),
        lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(do1, w_gateup, dx, task_queue_bwd, task_idx),
        lambda task_idx: deep_gemm.chunk_zip(
            dx, drecv_x, atomic_to_zip_bwd, zip_to_atomic_bwd, recv_token_indices,
            num_valid_topk, token_done, zip_done, task_queue_bwd, task_idx, CHUNK),
    ]

    task_launcher = GroupedTaskLauncher(
        funcs, num_tasks, calc_stream, comm_stream, comm_event, COMBINE_OVERLAP_RATIO)

    n = task_launcher.run_dispatch_overlap()
    print("[BW] dispatch->compute:", n) if logging else ()

    deep_gemm.set_num_sms(0)
    n = task_launcher.run_compute()
    deep_gemm.set_num_sms(CALC_NUM_SMS)
    print("[BW] compute->combine:", n) if logging else ()

    ############################ DISPATCH BACKWARD #############################

    dhidden_states, dtoken_probs, event = buffer.combine(
        drecv_x, handle, drecv_probs, async_finish=True, previous_event=deep_ep.Buffer.capture(),
        allocate_on_comm_stream=False, zip_done=zip_done)

    task_launcher.run_combine_overlap()

    ################################## WGRAD ###################################

    paddle.base.core.nvprof_nvtx_push("wgrad")

    # 各专家 seq 维重新向 512 对齐
    ks_cpu, m_start, m_start_wgrad = [], [0], [0]
    for n in tokens_per_expert:
        ks_cpu.append((n + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE * QUANT_BLOCK_SIZE)
        m_start.append(m_start[-1] + align(n))
        m_start_wgrad.append(m_start_wgrad[-1] + ks_cpu[-1])

    async_load = AsyncLoad()
    t = async_load(ks_cpu + m_start + m_start_wgrad, dtype="int32")
    grouped_layout = t[:len(ks_cpu)]
    m_start_gpu = t[len(ks_cpu):-len(m_start_wgrad)]
    m_start_wgrad_gpu = t[-len(m_start_wgrad):]

    ordered_to_zip, ordered_to_atomic = deep_gemm.sort_map(
        zip_to_atomic_bwd, m_start_gpu, m_start_wgrad[-1], m_start_wgrad_gpu)

    # 只有 x 是从 zipped 的向量解压，其他都是原样大小重排
    x_w = deep_gemm.requant_wgrad_input(recv_x[0], recv_x[1].T.contiguous().T, ordered_to_zip)
    do1_w = deep_gemm.requant_wgrad_input(*do1, ordered_to_atomic)
    o2_w = deep_gemm.requant_wgrad_input(*o2_bwd, ordered_to_atomic)
    do3_w = deep_gemm.requant_wgrad_input(*do3, ordered_to_atomic)

    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
        x_w, do1_w, w_gateup_grad, ks_cpu, grouped_layout, w_gateup_grad)
    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
        o2_w, do3_w, w_down_grad, ks_cpu, grouped_layout, w_down_grad)

    paddle.base.core.nvprof_nvtx_pop()

    event.current_stream_wait()
    paddle.zeros([1])
    paddle.base.core.nvprof_nvtx_pop()
    dist.all_reduce(paddle.empty([1]))

    return out, dhidden_states, dtoken_probs


def check(x, y):
    diff = paddle.abs(x.float() - y.float())
    avg, max = float(diff.mean()), float(diff.max())
    fmt = lambda n: "0.0" if n == 0 else f"{n:e}"
    msg = f"avg: {fmt(avg)} max: {fmt(max)}"
    if max != 0:
        cos = F.cosine_similarity(x.flatten(), y.flatten(), axis=0, eps=0)
        msg += f" cos: {cos:.6f} " + "X" * 40
    return msg


def main():
    group = initialize_fleet()
    configure_buffer(COMM_NUM_SMS)
    buffer = get_buffer(group, H * 2)
    x, token_probs, token_indices, w_gateup, w_down = prepare_case_inputs(group)

    dout = (paddle.randn_like(x) * 0.02).cast("bfloat16")
    w_gateup_grad = paddle.empty(w_gateup.shape, dtype="float32")
    w_down_grad = paddle.empty(w_down.shape, dtype="float32")
    w_gateup_grad_ref = paddle.empty_like(w_gateup_grad)
    w_down_grad_ref = paddle.empty_like(w_down_grad)

    # x 的 quant 算在前反向时间里, 权重的 quant 是则预处理不计入
    # fp8 的权重需要维护两份 layout, 因为 fp8 的性能对 layout 敏感, 前反向需要用不同 layout
    w_gateup_quant = quant_weight(w_gateup, transpose=False)
    w_down_quant = quant_weight(w_down, transpose=False)
    w_gateup_t_quant = quant_weight(w_gateup, transpose=True)
    w_down_t_quant = quant_weight(w_down, transpose=True)

    ################################## WARMUP ##################################

    run_baseline(group, buffer, x, dout, token_probs, token_indices, w_gateup_quant, w_down_quant,
                 w_gateup_t_quant, w_down_t_quant, w_gateup_grad_ref, w_down_grad_ref)
    run_overlap(group, buffer, x, dout, token_probs, token_indices, w_gateup_quant, w_down_quant,
                w_gateup_t_quant, w_down_t_quant, w_gateup_grad, w_down_grad)

    w_gateup_grad.zero_()
    w_down_grad.zero_()
    w_gateup_grad_ref.zero_()
    w_down_grad_ref.zero_()
    dist.all_reduce(paddle.empty([1]))

    ################################# VALIDATE #################################

    out_ref, dx_ref, dprobs_ref = run_baseline(
        group, buffer, x, dout, token_probs, token_indices, w_gateup_quant, w_down_quant,
        w_gateup_t_quant, w_down_t_quant, w_gateup_grad_ref, w_down_grad_ref)
    out, dx, dprobs = run_overlap(
        group, buffer, x, dout, token_probs, token_indices, w_gateup_quant, w_down_quant,
        w_gateup_t_quant, w_down_t_quant, w_gateup_grad, w_down_grad, logging=True)

    print("out:", check(out, out_ref))
    print("dx:", check(dx, dx_ref))
    print("dprobs:", check(dprobs, dprobs_ref))
    print("w_gateup_grad:", check(w_gateup_grad, w_gateup_grad_ref))
    print("w_down_grad:", check(w_down_grad, w_down_grad_ref))

    del out_ref, dx_ref, dprobs_ref, out, dx, dprobs

    ################################# PROFILE ##################################

    paddle.base.core.nvprof_start()
    dist.all_reduce(paddle.empty([1]))

    for i in range(10):
        paddle.base.core.nvprof_nvtx_push(f"baseline_{i}")
        run_baseline(
            group, buffer, x, dout, token_probs, token_indices, w_gateup_quant, w_down_quant,
            w_gateup_t_quant, w_down_t_quant, w_gateup_grad_ref, w_down_grad_ref)
        paddle.base.core.nvprof_nvtx_pop()

    for i in range(10):
        paddle.base.core.nvprof_nvtx_push(f"overlap_{i}")
        run_overlap(
            group, buffer, x, dout, token_probs, token_indices, w_gateup_quant, w_down_quant,
            w_gateup_t_quant, w_down_t_quant, w_gateup_grad, w_down_grad)
        paddle.base.core.nvprof_nvtx_pop()

    dist.barrier()
    paddle.base.core.nvprof_stop()


if __name__ == "__main__":
    main()
