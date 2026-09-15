import paddle
import paddle.distributed as dist
import paddle.nn.functional as F

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
ALIGNMENT = 128
CHUNK = 4096


def prepare_case_inputs(group):
    num_experts = group.world_size * E

    x = paddle.randn([SEQLEN, H], "bfloat16")
    # 把 token 的全局唯一 id 编码进前三列, 用于后面做精确的乱序比对.
    # bf16 只能精确表示 [0, 256) 的整数, 所以拆成 (rank, i // 128, i % 128) 三位.
    ids = paddle.arange(SEQLEN)
    x[:, 0] = paddle.full([SEQLEN], dist.get_rank(), "bfloat16")
    x[:, 1] = (ids // 128).astype("bfloat16")
    x[:, 2] = (ids % 128).astype("bfloat16")

    scores = paddle.randn([SEQLEN, num_experts])
    scores += paddle.randn([num_experts]) * 0.1

    topk_weights, topk_idx = scores.topk(TOPK)
    topk_weights = F.sigmoid(topk_weights)
    topk_weights /= topk_weights.sum(axis=-1, keepdim=True)

    return x, topk_weights, topk_idx


def token_ids(tokens):
    """从 token 的前三列还原出全局唯一 id, 全整数运算, 不会有并列."""
    f = tokens.astype("float32")
    return (f[:, 0] * 16384 + f[:, 1] * 128 + f[:, 2]).astype("int64")


def run_dispatch(group, buffer, x, token_probs, token_indices, unzip_alignment):
    num_experts = group.world_size * E

    layout = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = layout

    return buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        async_finish=False,
        allocate_on_comm_stream=False,
        unzip_alignment=unzip_alignment,
        unzip_chunk_size=CHUNK if unzip_alignment > 0 else 0,
    )


def check_fused_unzip(recv_x, recv_token_indices, recv_token_probs, tokens_per_expert,
                      unzipped_tokens, unzipped_probs):
    """
    融合后的 unzip 输出在每个专家段内是 atomic 先到先放, 顺序不确定, 因此这里
    对每个专家段做集合比较: 用一个固定随机投影给每行算一个 key, 排序后比对.
    """
    ref_tokens, _, ref_probs, _ = paddle.nn.functional.moe_permute(
        recv_x,
        None,
        recv_token_indices,
        recv_token_probs,
        padding_alignment=ALIGNMENT,
        num_experts=E,
        tokens_per_expert=tokens_per_expert,
    )
    assert ref_tokens.shape == unzipped_tokens.shape, (ref_tokens.shape, unzipped_tokens.shape)
    ok = True
    offset = 0
    for e, n in enumerate(tokens_per_expert):
        ref_seg = ref_tokens[offset:offset + n].astype("float32")
        new_seg = unzipped_tokens[offset:offset + n].astype("float32")

        ref_order = paddle.argsort(token_ids(ref_seg))
        new_order = paddle.argsort(token_ids(new_seg))

        num_expected = int((recv_token_indices == e).sum().item())
        id_diff = (token_ids(ref_seg)[ref_order] - token_ids(new_seg)[new_order]).abs().max().item() if n > 0 else 0

        token_diff = (ref_seg[ref_order] - new_seg[new_order]).abs().max().item() if n > 0 else 0.0
        prob_diff = (
            ref_probs[offset:offset + n][ref_order] - unzipped_probs[offset:offset + n][new_order]
        ).abs().max().item() if n > 0 else 0.0

        if token_diff != 0.0 or prob_diff != 0.0 or num_expected != n:
            ok = False
            print(f"  expert {e}: n={n} expected={num_expected} id_diff={id_diff} "
                  f"token_diff={token_diff} prob_diff={prob_diff}")

        offset += (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    print("fused unzip check:", "PASS" if ok else "FAIL")
    return ok


def expert_layout(tokens_per_expert):
    """复算 host 侧的分段: 每个专家在 unzipped_tokens 里的基址和 chunk 基址."""
    layout, base, chunk_base = [], 0, 0
    for n in tokens_per_expert:
        layout.append((base, n, chunk_base))
        base += (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        chunk_base += (n + CHUNK - 1) // CHUNK
    return layout, base, chunk_base


def check_mapping_tables(recv_x, recv_token_indices, tokens_per_expert,
                         unzipped_tokens, atomic_to_zip, zip_to_atomic):
    """
    两张表的校验, 三件事:
      1. padding / 无效 topk 位置必须保持 -1
      2. atomic_to_zip 指向的 recv_x 行必须和该槽位的 token 内容一致
      3. 两张表互为逆映射, 且 zip_to_atomic 指向的槽位落在正确的专家段内
    """
    layout, num_unzipped, _ = expert_layout(tokens_per_expert)
    a2z = atomic_to_zip.astype("int64")
    z2a = zip_to_atomic.astype("int64")
    ok = True

    # 1. 有效槽位就是每个专家段的前 n 行, 其余必须是 -1
    valid_slot = paddle.zeros([num_unzipped], dtype="bool")
    for base, n, _ in layout:
        if n > 0:
            valid_slot[base:base + n] = True
    if not bool(((a2z >= 0) == valid_slot).all().item()):
        ok = False
        print("  atomic_to_zip: 有效位置与预期分段不符,"
              f" 实际有效 {int((a2z >= 0).sum().item())} 预期 {int(valid_slot.sum().item())}")

    valid_pair = recv_token_indices >= 0
    if not bool(((z2a >= 0) == valid_pair).all().item()):
        ok = False
        print("  zip_to_atomic: 有效位置与 recv_token_indices 不符")

    # 2. 槽位内容必须等于它指向的那个 recv_x 行
    slots = paddle.nonzero(valid_slot).flatten()
    content_diff = (unzipped_tokens[slots].astype("float32")
                    - recv_x[a2z[slots]].astype("float32")).abs().max().item()
    if content_diff != 0.0:
        ok = False
        print(f"  atomic_to_zip: 指向的 recv_x 行与槽位内容不一致, max diff {content_diff}")

    # 3. 互为逆映射, 且落在正确的专家段
    pairs = paddle.nonzero(valid_pair)
    tok, slot = pairs[:, 0], z2a[valid_pair]
    if not bool((a2z[slot] == tok).all().item()):
        ok = False
        print("  zip_to_atomic / atomic_to_zip 不互逆")

    expert_of_pair = recv_token_indices[valid_pair].astype("int64")
    bases = paddle.to_tensor([b for b, _, _ in layout], dtype="int64")
    counts = paddle.to_tensor([n for _, n, _ in layout], dtype="int64")
    lo, hi = bases[expert_of_pair], bases[expert_of_pair] + counts[expert_of_pair]
    if not bool(paddle.logical_and(slot >= lo, slot < hi).all().item()):
        ok = False
        print("  zip_to_atomic: 有槽位落在了错误的专家段内")

    print("mapping tables check:", "PASS" if ok else "FAIL")
    return ok


def check_task_queue(tokens_per_expert, task_queue):
    """
    task_queue 校验: 每个 chunk 恰好入队一次, 全部 ready, 且描述符与预期分段一致.
    入队顺序是 chunk 的完成顺序(非确定), 所以按集合比对; 但 FIFO 要求 ready 从 0 开始
    密集填充, 不能有空洞.
    """
    layout, _, num_chunks = expert_layout(tokens_per_expert)
    expected = set()
    for e, (base, n, _) in enumerate(layout):
        for c in range((n + CHUNK - 1) // CHUNK):
            expected.add((e, base + c * CHUNK, min(CHUNK, n - c * CHUNK)))

    ok = True
    if task_queue.shape[0] != num_chunks:
        ok = False
        print(f"  task_queue 行数 {task_queue.shape[0]} != num_chunks {num_chunks}")

    q = task_queue.astype("int64").numpy()
    num_ready = int((q[:, 3] == 1).sum())
    if num_ready != num_chunks:
        ok = False
        print(f"  ready 的条目数 {num_ready} != num_chunks {num_chunks}")
    if not (q[:num_ready, 3] == 1).all():
        ok = False
        print("  ready 不是从 0 开始密集填充的, FIFO 语义被破坏")

    actual = {(int(r[0]), int(r[1]), int(r[2])) for r in q[:num_ready]}
    if actual != expected:
        ok = False
        print(f"  描述符集合不符, 缺失 {sorted(expected - actual)[:4]} 多余 {sorted(actual - expected)[:4]}")

    print(f"task_queue check ({num_chunks} chunks):", "PASS" if ok else "FAIL")
    return ok


def check_num_valid_topk(recv_token_indices, num_valid_topk):
    """
    num_valid_topk 校验: 通信全部完成后, 它必须等于 recv_token_indices 每行非 -1 的个数,
    且每个 token 至少命中一个本地专家(即不能有 0, 0 是初值).
    """
    expected = (recv_token_indices >= 0).astype("int32").sum(-1).astype("int32")
    actual = num_valid_topk.astype("int32")

    ok = True
    num_bad = int((actual != expected).astype("int32").sum())
    if num_bad != 0:
        ok = False
        print(f"  num_valid_topk 与 recv_token_indices 的行计数不符, {num_bad} 个 token")
    num_zero = int((actual == 0).astype("int32").sum())
    if num_zero != 0:
        ok = False
        print(f"  num_valid_topk 有 {num_zero} 个 token 仍是初值 0, 说明它没被写过")

    print("num_valid_topk check:", "PASS" if ok else "FAIL")
    return ok


def main():
    group = initialize_fleet()
    configure_buffer(NUM_SMS)
    buffer = get_buffer(group, H * 2)
    x, token_probs, token_indices = prepare_case_inputs(group)

    # warmup
    run_dispatch(group, buffer, x, token_probs, token_indices, 0)

    recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, _, _, \
        unzipped_tokens, unzipped_probs, atomic_to_zip, zip_to_atomic, num_valid_topk, task_queue = \
        run_dispatch(group, buffer, x, token_probs, token_indices, ALIGNMENT)

    print("num_recv_tokens:", recv_x.shape[0], "tokens_per_expert:", tokens_per_expert)
    print("num_unzipped_tokens:", unzipped_tokens.shape[0])

    recv_token_indices = recv_token_indices.cast("int32")
    check_fused_unzip(
        recv_x,
        recv_token_indices,
        recv_token_probs,
        tokens_per_expert,
        unzipped_tokens,
        unzipped_probs,
    )
    check_mapping_tables(
        recv_x, recv_token_indices, tokens_per_expert, unzipped_tokens, atomic_to_zip, zip_to_atomic
    )
    check_task_queue(tokens_per_expert, task_queue)
    check_num_valid_topk(recv_token_indices, num_valid_topk)

    ############################### PERF COMPARE ###############################

    # 共享开发机上性能抖动大, 所以交替测量再取中位数
    def bench(fn, num_iters=20):
        for _ in range(3):
            fn()
        paddle.device.synchronize()

        start = paddle.device.cuda.Event(enable_timing=True)
        end = paddle.device.cuda.Event(enable_timing=True)
        start.record()
        paddle.base.core.nvprof_nvtx_push("bench")
        for _ in range(num_iters):
            paddle.base.core.nvprof_nvtx_push("iter")
            fn()
            paddle.base.core.nvprof_nvtx_pop()
        paddle.base.core.nvprof_nvtx_pop()
        end.record()
        paddle.device.synchronize()
        return start.elapsed_time(end) / num_iters

    def dispatch_only():
        run_dispatch(group, buffer, x, token_probs, token_indices, 0)

    def dispatch_then_permute():
        recv_x, idx, probs, tpe, _, _ = run_dispatch(group, buffer, x, token_probs, token_indices, 0)
        paddle.nn.functional.moe_permute(
            recv_x, None, idx.cast("int32"), probs,
            padding_alignment=ALIGNMENT, num_experts=E, tokens_per_expert=tpe,
        )

    def dispatch_fused():
        run_dispatch(group, buffer, x, token_probs, token_indices, ALIGNMENT)

    paddle.base.core.nvprof_start()
    cases = {"dispatch only": [], "dispatch + moe_permute": [], "dispatch w/ fused unzip": []}
    for _ in range(5):
        cases["dispatch only"].append(bench(dispatch_only))
        cases["dispatch + moe_permute"].append(bench(dispatch_then_permute))
        cases["dispatch w/ fused unzip"].append(bench(dispatch_fused))
    paddle.base.core.nvprof_stop()

    for tag, samples in cases.items():
        samples.sort()
        print(f"{tag:>24}: median {samples[2]:.3f} ms  min {samples[0]:.3f} ms")


if __name__ == "__main__":
    main()
