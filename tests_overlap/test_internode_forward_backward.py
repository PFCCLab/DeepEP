import os
import paddle
from paddle import Tensor
import paddle.distributed as dist
from paddle.distributed import fleet
import paddle.nn.functional as F

paddle.empty([32, 1024, 1024, 1024], dtype="uint8")
paddle.set_printoptions(linewidth=200)

from utils import (
    deep_ep, deep_gemm, initialize_fleet, configure_buffer, get_buffer, AsyncLoad, grouped_launch
)

import paddlefleet_ops

E = 8
H = 4096
I = 2048
SEQLEN = 16384
TOPK = 8

COMM_NUM_SMS = int(os.environ.get("COMM_NUM_SMS", "48"))   # dynamic-SM design: env-overridable
CALC_NUM_SMS = int(os.environ.get("CALC_NUM_SMS", "100"))
FWD_ONLY = False   # bench hook: when True, run_overlap/run_baseline return after the forward pass

ALIGNMENT = 128
CHUNK = 4096
COMBINE_OVERLAP_RATIO = 0.3   # max (balanced) ratio; adaptively scaled down under skew, see below

PRECISE_SWIGLU = True
INTERLEAVED = False
OVERLAP_WGRAD = True
ORDERED_WGRAD = True

# DeepEP doesn't expose its comm stream, use this as a parallel stream to
# launch compute kernels and proxy DeepEP events
comm_stream = paddle.cuda.Stream()


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


def adaptive_combine_ratio(tokens_per_expert):
    """Choose the combine-overlap ratio from this rank's measured load imbalance.

    Combine-overlap defers the tail `ratio * num_tasks` chunks to run on the reduced
    CALC_NUM_SMS while the collective `combine` is in flight. That is a win when the
    deferred compute hides under the comm. But under heavy expert skew the deferred
    tail chunks are the huge hot-expert GEMMs: running them on CALC_NUM_SMS instead of
    all SMs costs more wall-time than the combine they overlap, so overlap goes
    net-negative (empirically all in the backward). We scale the ratio down with the
    per-rank load imbalance so a skewed rank stops deferring (ratio -> 0, i.e. degrade
    to dispatch-only overlap, which is >= baseline), while a balanced rank keeps the
    full ratio and its overlap win.

    Signal = this rank's own received-token count vs the perfectly-balanced
    expectation (SEQLEN*TOPK). On the hottest rank this equals the global max/mean
    per-rank imbalance, and it is the rank that sets the makespan -- so a purely local
    decision (no extra all-reduce / device sync; tokens_per_expert is already a CPU
    list) is sufficient and correctness-safe (combine is collective, called once per
    rank regardless of `end`).
    """
    if os.environ.get("ADAPTIVE_OVERLAP", "1") != "1":
        return COMBINE_OVERLAP_RATIO
    balanced = SEQLEN * TOPK  # perfectly-balanced per-rank received tokens
    imb = sum(tokens_per_expert) / max(balanced, 1)
    lo = float(os.environ.get("OVERLAP_IMB_LO", "1.2"))
    hi = float(os.environ.get("OVERLAP_IMB_HI", "1.5"))
    if imb <= lo:
        return COMBINE_OVERLAP_RATIO
    if imb >= hi:
        return 0.0
    return COMBINE_OVERLAP_RATIO * (hi - imb) / (hi - lo)


def overlap_is_beneficial(group, token_indices, num_experts):
    """Baseline floor: predict whether the overlap schedule beats the monolithic baseline.

    run_overlap is a distinct chunk-GEMM code path (fused-unzip dispatch + per-chunk
    task-queue GEMMs in Stage A/B/C); run_baseline is a single m_grouped GEMM. Overlap
    only wins once (a) the comm fraction is large enough to hide the SM-steal -- which
    grows with EP scale (measured: EP16 net-loses, EP32 breaks even, EP64 +14% balanced)
    -- and (b) the load is balanced enough that no single hot-expert chunk dominates the
    critical path (under skew the backward-overlap net-loses a few %% at every scale).

    So we gate on world_size and a cheap pre-dispatch imbalance estimate (bincount over
    the routing indices, one all-reduce; decided BEFORE dispatch so there is no
    double-dispatch). When overlap is not predicted to win we return False and the caller
    runs the baseline schedule -> overlap is guaranteed >= baseline (gain never negative).

    Env-gated (OVERLAP_BASELINE_FLOOR, default off) so the correctness oracle and the
    default shipped path keep the full overlap schedule; the benchmark / production
    dispatcher turns it on. Thresholds env-tunable (OVERLAP_MIN_WORLD, OVERLAP_MAX_IMB).
    """
    if os.environ.get("OVERLAP_BASELINE_FLOOR", "0") != "1":
        return True
    min_world = int(os.environ.get("OVERLAP_MIN_WORLD", "48"))
    max_imb = float(os.environ.get("OVERLAP_MAX_IMB", "1.15"))
    if group.world_size < min_world:
        return False
    cnt = paddle.bincount(token_indices.flatten(), minlength=num_experts).cast("float32")
    dist.all_reduce(cnt, group=group)
    per_rank = cnt.reshape([group.world_size, E]).sum(1)
    imb = float(per_rank.max() / per_rank.mean())
    return imb <= max_imb


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
                w_gateup_grad, w_down_grad, logging=False):
    num_experts = group.world_size * E
    deep_gemm.set_num_sms(CALC_NUM_SMS)

    # Baseline floor: fall back to the baseline schedule in regimes where overlap net-loses
    # (small EP / high skew), so run_overlap is never slower than baseline. Off by default.
    if not overlap_is_beneficial(group, token_indices, num_experts):
        return run_baseline(group, buffer, hidden_states, token_probs, token_indices, dout,
                            w_gateup, w_down, w_gateup_grad, w_down_grad)

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
        num_recv_tokens_per_expert_list, handle, dispatch_done_event,
        unzipped_tokens, unzipped_probs, atomic_to_zip, zip_to_atomic,
        num_valid_topk, task_queue, unzip_overflow_flag
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

    # Adaptive combine-overlap ratio: full ratio when balanced, -> 0 under skew so overlap
    # never goes net-negative vs baseline (degrades to dispatch-only overlap). Shared by the
    # forward and backward Stage-B/C split below (same routing => same imbalance).
    combine_ratio = adaptive_combine_ratio(tokens_per_expert)

    # Localization / fail-safe probe (opt-in via UNZIP_OVERFLOW_CHECK=1, off by default so the
    # overlap hot path pays no host sync). Reading the flag forces the dispatch kernel to finish
    # BEFORE any compute is launched, so (a) a fused-unzip over-count surfaces as a clean, catchable
    # RuntimeError naming the expert, and (b) if the crash is a *silent* OOB inside dispatch, the
    # CUDA 719 surfaces at THIS sync (localizing it to dispatch) instead of at the later compute sync.
    if os.environ.get("UNZIP_OVERFLOW_CHECK", "0") == "1":
        paddle.device.synchronize()
        ov = int(unzip_overflow_flag.item())
        if ov != 0:
            raise RuntimeError(
                f"fused-unzip dispatch overflowed at local expert {ov - 1} "
                f"(over-count tokens were dropped; workload skew exceeds fused-unzip provisioning)")

    if logging:
        print("tokens_per_expert:", tokens_per_expert)
        print("num_tasks:", [(n + CHUNK - 1) // CHUNK for n in tokens_per_expert], "=", num_tasks)
        print("num_recv_tokens:", num_recv_tokens)
        print("combine_ratio:", round(combine_ratio, 4),
              "(local imb %.3f)" % (sum(tokens_per_expert) / max(SEQLEN * TOPK, 1)))

    ############################### GEMM FORWARD ###############################

    o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
    o2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
    o3 = paddle.empty_like(unzipped_tokens)
    zipped_out = paddle.empty_like(recv_x)

    token_done = paddle.zeros([num_recv_tokens], "int32")
    zip_done = paddle.zeros([num_recv_tokens], "int32")

    calc_stream = paddle.cuda.current_stream()
    previous_task_done_event = paddle.cuda.Event()

    funcs = [
        lambda task_idx: deep_gemm.bf16_chunk_gemm_nn(
            unzipped_tokens, w_gateup, o1, task_queue, task_idx),
        lambda task_idx: deep_gemm.chunk_weighted_swiglu(
            o1, unzipped_probs, o2, task_queue, task_idx, CHUNK, precise=PRECISE_SWIGLU,
            interleaved=INTERLEAVED),
        lambda task_idx: deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, task_queue, task_idx),
        lambda task_idx: deep_gemm.chunk_zip(
            o3, zipped_out, atomic_to_zip, zip_to_atomic, recv_token_indices, num_valid_topk,
            token_done, zip_done, task_queue, task_idx, CHUNK)
    ]

    # 阶段A: dispatch 与计算 overlap, 计算只使用部分 SM
    for task_idx in range(num_tasks):
        paddle.base.core.nvprof_nvtx_push(f"A{task_idx}")
        for func in funcs:
            func(task_idx)
        paddle.base.core.nvprof_nvtx_pop()

        # 只提前发射一个 task, 从而及时根据 dispatch 完成状态切换 SM 数
        if task_idx > 0:
            previous_task_done_event.synchronize()
        previous_task_done_event.record()

        # 当 dispatch 恰好完成时, 切换至纯计算模式
        if dispatch_done_event.query():
            print("[FW] dispatch done:", task_idx) if logging else ()
            break

    # 阶段B: 纯计算, 计算使用全部 SM
    begin = task_idx + 1  # 此处 task_idx 已经执行了, begin 要取下一个
    end = max(int(num_tasks * (1 - combine_ratio)), begin)
    if begin < end:
        deep_gemm.set_num_sms(0)
        paddle.base.core.nvprof_nvtx_push(f"B{begin}_{end - 1}")
        grouped_launch(funcs, begin, end, calc_stream, comm_stream, previous_task_done_event)
        paddle.base.core.nvprof_nvtx_pop()
        deep_gemm.set_num_sms(CALC_NUM_SMS)

    ############################# COMBINE FORWARD ##############################

    combine_event = deep_ep.Buffer.capture()
    print("[FW] combine begin:", end) if logging else ()

    out, _, event = buffer.combine(zipped_out, handle, async_finish=True,
                                   previous_event=combine_event, allocate_on_comm_stream=False,
                                   zip_done=zip_done)

    # 阶段C: combine 与计算 overlap, 计算只使用部分 SM
    for task_idx in range(end, num_tasks):
        paddle.base.core.nvprof_nvtx_push(f"C{task_idx}")
        for func in funcs:
            func(task_idx)
        paddle.base.core.nvprof_nvtx_pop()

    event.current_stream_wait()
    paddle.zeros([1])
    paddle.base.core.nvprof_nvtx_pop()
    dist.all_reduce(paddle.empty([1]))
    if FWD_ONLY:
        return None

    ############################# COMBINE BACKWARD #############################

    paddle.base.core.nvprof_nvtx_push("backward")
    paddle.zeros([1])

    # 本来 dispatch 反向应该用 cache_mode, 但目前 cache_mode 无法计算 atomic_to_zip/zip_to_atomic,
    # 因此只能像前向一样再跑一遍, 会浪费一定的通信带宽
    (
        _, _, _, _, handle, dispatch_done_event, do3, _,
        atomic_to_zip_bwd, zip_to_atomic_bwd, _, task_queue_bwd, unzip_overflow_flag_bwd
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

    # Same opt-in localization / fail-safe probe as the forward dispatch above.
    if os.environ.get("UNZIP_OVERFLOW_CHECK", "0") == "1":
        paddle.device.synchronize()
        ov = int(unzip_overflow_flag_bwd.item())
        if ov != 0:
            raise RuntimeError(
                f"fused-unzip backward dispatch overflowed at local expert {ov - 1} "
                f"(over-count tokens were dropped; workload skew exceeds fused-unzip provisioning)")

    ############################## GEMM BACKWARD ###############################

    dx = paddle.empty_like(unzipped_tokens)
    do1 = paddle.empty_like(o1)
    do2 = paddle.empty_like(o2)
    drecv_x = paddle.empty_like(recv_x)
    drecv_probs = paddle.zeros_like(recv_token_probs)  # 无效位预先填0
    o2_bwd = paddle.empty_like(o2)

    token_done = paddle.zeros([len(recv_token_probs)], dtype="int32")
    zip_done = paddle.zeros([len(recv_token_probs)], dtype="int32")

    funcs = [
        lambda task_idx: deep_gemm.bf16_chunk_gemm_nt(do3, w_down, do2, task_queue_bwd, task_idx),
        lambda task_idx: deep_gemm.chunk_weighted_swiglu_grad(
            o1, unzipped_probs, do2, o2_bwd, do1, drecv_probs, atomic_to_zip_bwd, zip_to_atomic,
            recv_token_indices, task_queue_bwd, task_idx, CHUNK, precise=PRECISE_SWIGLU,
            interleaved=INTERLEAVED),
        lambda task_idx: deep_gemm.bf16_chunk_gemm_nt(do1, w_gateup, dx, task_queue_bwd, task_idx),
        lambda task_idx: deep_gemm.chunk_zip(
            dx, drecv_x, atomic_to_zip_bwd, zip_to_atomic_bwd, recv_token_indices,
            num_valid_topk, token_done, zip_done, task_queue_bwd, task_idx, CHUNK),
    ]

    # 阶段A
    for task_idx in range(num_tasks):
        paddle.base.core.nvprof_nvtx_push(f"A{task_idx}")
        for func in funcs:
            func(task_idx)
        paddle.base.core.nvprof_nvtx_pop()

        if task_idx > 0:
            previous_task_done_event.synchronize()
        previous_task_done_event.record()

        if dispatch_done_event.query():
            print("[BW] dispatch done:", task_idx) if logging else ()
            break

    # 阶段B
    begin = task_idx + 1
    end = max(int(num_tasks * (1 - combine_ratio)), begin)
    if begin < end:
        deep_gemm.set_num_sms(0)
        paddle.base.core.nvprof_nvtx_push(f"B{begin}_{end - 1}")
        grouped_launch(funcs, begin, end, calc_stream, comm_stream, previous_task_done_event)
        paddle.base.core.nvprof_nvtx_pop()
        deep_gemm.set_num_sms(CALC_NUM_SMS)

    ############################ DISPATCH BACKWARD #############################

    combine_event = deep_ep.Buffer.capture()
    print("[BW] combine begin:", end) if logging else ()

    dhidden_states, dtoken_probs, event = buffer.combine(
        drecv_x, handle, drecv_probs, async_finish=True, previous_event=combine_event,
        allocate_on_comm_stream=False, zip_done=zip_done)

    # 阶段C
    for task_idx in range(end, num_tasks):
        paddle.base.core.nvprof_nvtx_push(f"C{task_idx}")
        for func in funcs:
            func(task_idx)
        paddle.base.core.nvprof_nvtx_pop()

    event.current_stream_wait() if not OVERLAP_WGRAD else ()

    ################################## WGRAD ###################################

    paddle.base.core.nvprof_nvtx_push("wgrad")

    if ORDERED_WGRAD:
        m_start = [0]
        for n in tokens_per_expert:
            m_start.append(m_start[-1] + align(n))
        async_load = AsyncLoad()
        m_start = async_load(m_start, dtype="int32")

        ordered_to_zip, ordered_to_atomic = deep_gemm.sort_map(
            zip_to_atomic_bwd, m_start, num_unzipped_tokens)

        # x 从 recv_x 中解压, 这里使用 gather 并非最优性能, 因为重复读了 recv_x 的某些行
        x_wgrad = deep_gemm.token_gather(recv_x, ordered_to_zip)

        # do1/o2_bwd/do3 从反向 atomic 序的输入重排序为标准 unzip 序
        do1_wgrad = deep_gemm.token_gather(do1, ordered_to_atomic)
        o2_bwd = deep_gemm.token_gather(o2_bwd, ordered_to_atomic)
        do3 = deep_gemm.token_gather(do3, ordered_to_atomic)
    else:
        x_wgrad = deep_gemm.token_gather(recv_x, atomic_to_zip_bwd)
        do1_wgrad = do1

    run_wgrad(tokens_per_expert, x_wgrad, do1_wgrad, w_gateup_grad, o2_bwd, do3, w_down_grad)

    paddle.base.core.nvprof_nvtx_pop()
    event.current_stream_wait() if OVERLAP_WGRAD else ()

    paddle.zeros([1])
    paddle.base.core.nvprof_nvtx_pop()
    dist.all_reduce(paddle.empty([1]))

    # Fail-safe: the fused-unzip dispatch kernel sets this flag (to offending_expert+1) if any
    # token's write slot exceeded its expert's provisioned region -- it then DROPS the token
    # instead of writing OOB (which would MMU-fault the GPU into an unrecoverable CUDA 719).
    # A non-zero flag means the results are missing tokens, so raise a clean, catchable error
    # here rather than silently returning wrong outputs. In normal operation this is always 0.
    for tag, flag in (("forward", unzip_overflow_flag), ("backward", unzip_overflow_flag_bwd)):
        if flag is not None and int(flag) != 0:
            raise RuntimeError(
                f"fused-unzip {tag} dispatch overflowed: local expert {int(flag) - 1} received more "
                f"tokens than its provisioned region. Workload skew exceeds fused-unzip provisioning; "
                f"tokens were dropped (no OOB/GPU-wedge). Lower the skew or expand the unzip buffers.")

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
    if FWD_ONLY:
        return None

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
    if not INTERLEAVED:
        return gateup
    gate, up = gateup.chunk(2, axis=-1)
    return paddle.concat(
        [gate.unsqueeze(-1), up.unsqueeze(-1)], axis=-1
    ).reshape(gateup.shape)


def deinterleave_gateup(gateup):
    if not INTERLEAVED:
        return gateup
    return paddle.concat([gateup[..., 0::2], gateup[..., 1::2]], axis=-1)


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
    configure_buffer(COMM_NUM_SMS)
    buffer = get_buffer(group, H * 2)
    x, token_probs, token_indices, w_gateup, w_down = prepare_case_inputs(group)
    dout = paddle.randn_like(x)
    w_gateup_ref = deinterleave_gateup(w_gateup)

    w_gateup_grad = paddle.empty(w_gateup.shape, dtype="float32")
    w_down_grad = paddle.empty(w_down.shape, dtype="float32")
    w_gateup_grad_ref = paddle.empty_like(w_gateup_grad)
    w_down_grad_ref = paddle.empty_like(w_down_grad)

    # warmup
    run_overlap(group, buffer, x, token_probs, token_indices, dout, w_gateup, w_down,
                w_gateup_grad, w_down_grad)
    run_baseline(group, buffer, x, token_probs, token_indices, dout, w_gateup_ref, w_down,
                 w_gateup_grad_ref, w_down_grad_ref)

    # wgrad 使用累加语义, 需要提前置 0
    w_gateup_grad.zero_()
    w_down_grad.zero_()
    w_gateup_grad_ref.zero_()
    w_down_grad_ref.zero_()
    dist.all_reduce(paddle.empty([1]))

    # validate
    out, do1, dx, dprobs, tokens_per_expert, atomic_to_zip = run_overlap(
        group, buffer, x, token_probs, token_indices, dout, w_gateup, w_down,
        w_gateup_grad, w_down_grad, logging=True)
    out_ref, do1_ref, dx_ref, dprobs_ref = run_baseline(
        group, buffer, x, token_probs, token_indices, dout, w_gateup_ref, w_down,
        w_gateup_grad_ref, w_down_grad_ref)

    atomic_perm = get_atomic_perm(tokens_per_expert, atomic_to_zip)

    print("out:", check(out, out_ref))
    print("do1:", check(do1[atomic_perm], interleave_gateup(do1_ref)))
    print("dx:", check(dx, dx_ref))
    print("dprobs:", check(dprobs, dprobs_ref))
    print("w_gateup_grad:", check(w_gateup_grad, interleave_gateup(w_gateup_grad_ref)))
    print("w_down_grad:", check(w_down_grad, w_down_grad_ref))

    # profile
    paddle.base.core.nvprof_start()
    dist.all_reduce(paddle.empty([1]))

    for i in range(10):
        paddle.base.core.nvprof_nvtx_push(f"overlap_{i}")
        run_overlap(group, buffer, x, token_probs, token_indices, dout, w_gateup, w_down,
                    w_gateup_grad, w_down_grad)
        paddle.base.core.nvprof_nvtx_pop()

    for i in range(5):
        paddle.base.core.nvprof_nvtx_push(f"baseline_{i}")
        run_baseline(group, buffer, x, token_probs, token_indices, dout, w_gateup_ref, w_down,
                     w_gateup_grad_ref, w_down_grad_ref)
        paddle.base.core.nvprof_nvtx_pop()

    dist.barrier()
    paddle.base.core.nvprof_stop()


if __name__ == "__main__":
    main()
