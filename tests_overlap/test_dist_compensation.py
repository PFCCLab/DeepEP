"""Distributed intra-node compute compensation at EP64 (8 nodes x 8 GPUs).

Each rank is its own process on its own GPU, so the DeepGEMM chunk engine works
normally (the single-device-per-process JIT constraint is satisfied). Intra-node
migration uses NCCL send/recv over NVLink -- no internode.cu / IPC hacking.

Staged development:
  v1  baseline: each rank computes its E experts via the chunk GEMM; node makespan
      = max over the node's 8 ranks (reported as global allreduce-max).
  v2  + node-local load all-gather + planner decision (printed).
  v3  + NCCL migration (tokens+weights+o3) + bitwise verify vs baseline + makespan.

Run: mpirun python run.py 0-7 test_dist_compensation.py
"""
import os
import time
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F

paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)

import deep_gemm
from utils import initialize_fleet

E, H, I = 8, 4096, 2048
TOPK = 8
CHUNK = 4096
ALIGNMENT = 128
NUM_SMS = 100
RANKS_PER_NODE = 8
SKEW = float(os.environ.get("BENCH_SKEW", "0.5"))

import compensation_planner as P


def align(n):
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _silu_swiglu(o1, probs):
    g, u = o1.chunk(2, axis=-1)
    g, u = g.float(), u.float()
    return ((g * F.sigmoid(g)) * u * probs.unsqueeze(-1)).astype("bfloat16")


def sim_local_tokens(world_size, rank, seed=123):
    """Per-(rank,expert) token counts for the whole world, from a shared skew bias
    (same on all ranks -> consistent global layout). Returns this node's [8,E].
    PLACEMENT env: 'contiguous' (default) or 'lpt' (compensation-aware node placement,
    stage 39: LPT-spread experts across nodes so per-node load is balanced)."""
    num_experts = world_size * E
    rng = np.random.default_rng(seed)
    bias = rng.normal(0, SKEW, num_experts)
    total_tokens = 16384 * world_size
    w = np.exp(bias); w /= w.sum()
    counts = np.floor(w * total_tokens * TOPK).astype(np.int64)
    node = rank // RANKS_PER_NODE
    num_nodes = world_size // RANKS_PER_NODE

    if os.environ.get("PLACEMENT", "contiguous") == "lpt":
        # LPT assign experts (hottest first) to least-loaded node (cap = RANKS_PER_NODE*E),
        # then within each node LPT to its ranks (cap = E each). Deterministic on all ranks.
        cap_node = RANKS_PER_NODE * E
        node_load = np.zeros(num_nodes); node_experts_list = [[] for _ in range(num_nodes)]
        for e in np.argsort(-counts):
            cand = [n for n in range(num_nodes) if len(node_experts_list[n]) < cap_node]
            n = min(cand, key=lambda nn: node_load[nn])
            node_experts_list[n].append(int(e)); node_load[n] += counts[e]
        my = node_experts_list[node]
        rank_load = np.zeros(RANKS_PER_NODE); rank_ex = [[] for _ in range(RANKS_PER_NODE)]
        for e in sorted(my, key=lambda ee: -counts[ee]):
            cand = [r for r in range(RANKS_PER_NODE) if len(rank_ex[r]) < E]
            r = min(cand, key=lambda rr: rank_load[rr])
            rank_ex[r].append(e); rank_load[r] += counts[e]
        node_experts = np.array([[counts[e] for e in rank_ex[r]] for r in range(RANKS_PER_NODE)],
                                dtype=np.int64)
        return node_experts

    node_experts = counts.reshape(world_size, E)[node * RANKS_PER_NODE:(node + 1) * RANKS_PER_NODE]
    return node_experts


# PLACEHOLDER_MAIN
def build_layout(local_tokens):
    """This rank's chunk layout for `local_tokens` (list[E])."""
    m_start, m_indices, cur = [], [], 0
    for e, n in enumerate(local_tokens):
        na = align(n)
        m_start.append(cur); cur += na
        m_indices.append(paddle.full([na], e, "int32"))
    m_total = max(cur, ALIGNMENT)
    q = []
    for e, (n, s) in enumerate(zip(local_tokens, m_start)):
        for off in range(0, n, CHUNK):
            q.append([e, s + off, min(CHUNK, n - off), 1])
    if not q:
        q = [[0, 0, 0, 1]]
    return m_total, m_start, paddle.concat(m_indices) if m_indices else None, paddle.to_tensor(q, "int32")


def chunk_ffn(x, wgu, wd, probs, tq):
    m = x.shape[0]
    o1 = paddle.zeros([m, 2 * I], "bfloat16")
    o2 = paddle.zeros([m, I], "bfloat16")
    o3 = paddle.zeros([m, H], "bfloat16")
    for i in range(tq.shape[0]):
        deep_gemm.bf16_chunk_gemm_nn(x, wgu, o1, tq, i)
        deep_gemm.chunk_weighted_swiglu(o1, probs, o2, tq, i, CHUNK, precise=True)
        deep_gemm.bf16_chunk_gemm_nn(o2, wd, o3, tq, i)
    return o3


def timed(fn, iters=10):
    for _ in range(3):
        fn()
    paddle.device.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); paddle.device.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def main():
    group = initialize_fleet()
    ws = dist.get_world_size()
    rank = dist.get_rank()
    node = rank // RANKS_PER_NODE
    nvl = rank % RANKS_PER_NODE
    deep_gemm.set_num_sms(NUM_SMS)

    node_tok = sim_local_tokens(ws, rank)                 # [8, E] for this node
    my_tok = node_tok[nvl].tolist()                       # this rank's E experts
    paddle.seed(1000 + rank)
    wgu = (paddle.randn([E, H, 2 * I]) * 0.02).cast("bfloat16")
    wd = (paddle.randn([E, I, H]) * 0.02).cast("bfloat16")

    # ---- v1 baseline: this rank computes its own experts ----
    m_total, m_start, m_idx, tq = build_layout(my_tok)
    x = paddle.randn([m_total, H], "bfloat16")
    probs = paddle.rand([m_total], "float32")
    o3_base = chunk_ffn(x, wgu, wd, probs, tq)
    t_self = timed(lambda: chunk_ffn(x, wgu, wd, probs, tq))
    # node makespan = max over the node's 8 ranks
    t_tensor = paddle.to_tensor([t_self], "float32")
    dist.all_reduce(t_tensor, op=dist.ReduceOp.MAX, group=group)
    makespan_base = float(t_tensor)

    # ---- v2 planner decision (node-local view) ----
    cp = P.plan_compute_node(node_tok, CHUNK)
    per_rank = node_tok.sum(axis=1)
    imb = float(per_rank.max() / per_rank.mean())
    if rank == 0:
        print(f"[EP{ws}] node imbalance(max/mean)={imb:.3f}  "
              f"migrations={len(cp.migrations)} weight_moves={cp.distinct_weight_moves}  "
              f"makespan_before={cp.makespan_before_ms:.3f} after={cp.makespan_after_ms:.3f} ms", flush=True)
        print(f"[EP{ws}] baseline node makespan (measured chunk GEMM, allreduce-max) = "
              f"{makespan_base:.3f} ms", flush=True)
        for m in cp.migrations[:6]:
            print(f"   migrate expert{m.expert}: rank{m.src_rank}->rank{m.dst_rank} "
                  f"{m.n_chunks}ch {m.tokens}tok", flush=True)

    dist.barrier()
    if rank == 0:
        print("[v1+v2] baseline + planner OK at EP%d" % ws, flush=True)

    run_v3(group, ws, rank, node, nvl, node_tok, my_tok, m_start, x, probs,
           wgu, wd, o3_base, cp, makespan_base)


def run_v3(group, ws, rank, node, nvl, node_tok, my_tok, m_start, x, probs,
           wgu, wd, o3_base, cp, makespan_base):
    """ALL of a node's migrations at EP scale, deadlock-free via lockstep: each
    migration's donor/helper do one blocking NVLink send/recv, then the whole node
    barriers. Helpers compute migrated chunks with the real DeepGEMM chunk engine,
    return o3, donor verifies bitwise. Also measures the compensated node makespan."""
    node_groups = [dist.new_group(ranks=list(range(n * RANKS_PER_NODE, (n + 1) * RANKS_PER_NODE)))
                   for n in range(ws // RANKS_PER_NODE)]
    ng = node_groups[node]
    base = node * RANKS_PER_NODE
    WN, WDN = H * 2 * I, I * H

    # deterministic row0 for each migration (donor tail offset)
    donate_cursor, migs = {}, []
    for m in cp.migrations:
        keep = my_tok_of(node_tok, m.src_rank, m.expert) - moved_of(cp, m.src_rank, m.expert)
        cur = donate_cursor.get((m.src_rank, m.expert), keep)
        migs.append(dict(e=m.expert, src=m.src_rank, dst=m.dst_rank, tok=m.tokens, row0=cur))
        donate_cursor[(m.src_rank, m.expert)] = cur + m.tokens

    # ---- phase 1: lockstep migrate weights+input+probs (donor -> helper) ----
    recv = {}                                             # migration index -> (wf,xr,pr)
    dist.barrier(group); paddle.device.synchronize(); _tp0 = time.perf_counter()
    for i, mm in enumerate(migs):
        gs, gd = base + mm["src"], base + mm["dst"]
        s = m_start[mm["e"]] + mm["row0"]
        if nvl == mm["src"]:
            dist.send(paddle.concat([wgu[mm["e"]].reshape([-1]), wd[mm["e"]].reshape([-1])]), gd)
            dist.send(x[s:s + mm["tok"]].contiguous(), gd)
            dist.send(probs[s:s + mm["tok"]].contiguous(), gd)
        elif nvl == mm["dst"]:
            wf = paddle.empty([WN + WDN], "bfloat16"); dist.recv(wf, gs)
            xr = paddle.empty([mm["tok"], H], "bfloat16"); dist.recv(xr, gs)
            pr = paddle.empty([mm["tok"]], "float32"); dist.recv(pr, gs)
            recv[i] = (wf, xr, pr)
        dist.barrier(ng)                                  # lockstep -> deadlock-free
    paddle.device.synchronize(); _tp1 = time.perf_counter()

    # ---- phase 2: helper computes its migrated-in chunks ----
    ret = {}
    for i, mm in enumerate(migs):
        if nvl != mm["dst"]:
            continue
        wf, xr, pr = recv[i]; tok = mm["tok"]; na = align(tok)
        xg = paddle.zeros([na, H], "bfloat16"); xg[:tok] = xr
        pg = paddle.zeros([na], "float32"); pg[:tok] = pr
        wgu_e = wf[:WN].reshape([H, 2 * I]); wd_e = wf[WN:].reshape([I, H])
        q = paddle.to_tensor([[0, off, min(CHUNK, tok - off), 1] for off in range(0, tok, CHUNK)], "int32")
        ret[i] = chunk_ffn(xg, wgu_e.unsqueeze(0), wd_e.unsqueeze(0), pg, q)[:tok].contiguous()

    # ---- phase 3: lockstep return o3 (helper -> donor) ----
    back = {}
    dist.barrier(group); paddle.device.synchronize(); _tp2 = time.perf_counter()
    for i, mm in enumerate(migs):
        gs, gd = base + mm["src"], base + mm["dst"]
        if nvl == mm["dst"]:
            dist.send(ret[i], gs)
        elif nvl == mm["src"]:
            bo = paddle.empty([mm["tok"], H], "bfloat16"); dist.recv(bo, gd); back[i] = bo
        dist.barrier(ng)
    paddle.device.synchronize(); _tp3 = time.perf_counter()
    t_transfer_cold = (_tp1 - _tp0) + (_tp3 - _tp2)

    # steady-state transfer (WARM), FUSED: one node-local all-to-all for all migrated
    # inputs, one for all o3 returns -> 2 collectives total instead of O(migrations) p2p
    # ops. This is deadlock-free (collective) and avoids the per-op NCCL launch overhead
    # that made the p2p path overhead-bound.
    RN = RANKS_PER_NODE
    def _steady_xfer():
        # phase 1: all-to-all migrated INPUT tokens (donor -> helper)
        send = {j: [] for j in range(RN)}
        for mm in migs:
            if nvl == mm["src"]:
                s = m_start[mm["e"]] + mm["row0"]
                send[mm["dst"]].append(x[s:s + mm["tok"]])
        in_split = [int(sum(p.shape[0] for p in send[j])) for j in range(RN)]
        parts = [paddle.concat(send[j]) for j in range(RN) if send[j]]
        in_t = paddle.concat(parts) if parts else paddle.zeros([0, H], "bfloat16")
        out_split = [int(sum(mm["tok"] for mm in migs if mm["dst"] == nvl and mm["src"] == j))
                     for j in range(RN)]
        out_t = paddle.zeros([max(sum(out_split), 0), H], "bfloat16")
        dist.alltoall_single(out_t, in_t, in_split, out_split, group=ng)
        # phase 3: all-to-all o3 back (helper -> owner)
        send2 = {j: [] for j in range(RN)}
        for i, mm in enumerate(migs):
            if nvl == mm["dst"]:
                send2[mm["src"]].append(ret[i])
        in2_split = [int(sum(p.shape[0] for p in send2[j])) for j in range(RN)]
        parts2 = [paddle.concat(send2[j]) for j in range(RN) if send2[j]]
        in2 = paddle.concat(parts2) if parts2 else paddle.zeros([0, H], "bfloat16")
        out2_split = [int(sum(mm["tok"] for mm in migs if mm["src"] == nvl and mm["dst"] == j))
                      for j in range(RN)]
        out2 = paddle.zeros([max(sum(out2_split), 0), H], "bfloat16")
        dist.alltoall_single(out2, in2, in2_split, out2_split, group=ng)
    for _ in range(3):
        _steady_xfer()
    paddle.device.synchronize(); _w0 = time.perf_counter()
    for _ in range(5):
        _steady_xfer()
    paddle.device.synchronize()
    t_transfer = (time.perf_counter() - _w0) / 5

    # ---- verify: donor's migrated rows vs baseline (bitwise) ----
    maxdiff = 0.0
    for i, mm in enumerate(migs):
        if nvl != mm["src"]:
            continue
        s = m_start[mm["e"]] + mm["row0"]
        maxdiff = max(maxdiff, float((o3_base[s:s + mm["tok"]].astype("float32")
                                      - back[i].astype("float32")).abs().max()))
    dt = paddle.to_tensor([maxdiff], "float32")
    dist.all_reduce(dt, op=dist.ReduceOp.MAX, group=group)

    # ---- compensated makespan: each rank's post-migration compute (retained + migrated-in) ----
    moved_out = sum(mm["tok"] for mm in migs if mm["src"] == nvl)
    moved_in = sum(mm["tok"] for mm in migs if mm["dst"] == nvl)
    comp_tokens = int(sum(my_tok)) - moved_out + moved_in
    comp_tokens = max(align(comp_tokens), ALIGNMENT)
    cx = paddle.randn([comp_tokens, H], "bfloat16")
    cp_probs = paddle.rand([comp_tokens], "float32")
    cq = paddle.to_tensor([[0, off, min(CHUNK, comp_tokens - off), 1]
                           for off in range(0, comp_tokens, CHUNK)], "int32")
    t_comp = timed(lambda: chunk_ffn(cx, wgu[:1].tile([1, 1, 1]) if False else wgu, wd, cp_probs, cq))
    tc = paddle.to_tensor([t_comp], "float32")
    dist.all_reduce(tc, op=dist.ReduceOp.MAX, group=group)

    tt = paddle.to_tensor([t_transfer * 1000], "float32")
    dist.all_reduce(tt, op=dist.ReduceOp.MAX, group=group)
    if rank == 0:
        v = float(dt); mc = float(tc); tr = float(tt)
        ms_serial = mc + tr                              # transfers on the critical path
        ms_overlap = max(mc, tr)                         # transfers hidden behind compute
        sp_serial = (makespan_base - ms_serial) / makespan_base * 100
        sp_overlap = (makespan_base - ms_overlap) / makespan_base * 100
        print(f"[v3-multi] EP{ws} all-migration o3 vs baseline: max_diff={v:.3e} "
              f"{'PASS(bitwise)' if v == 0 else 'MISMATCH'}", flush=True)
        print(f"[v3-multi] baseline={makespan_base:.3f}  comp_compute={mc:.3f}  "
              f"transfer(lockstep,incl weights)={tr:.3f} ms", flush=True)
        print(f"[#3] SERIAL (transfer on critical path): {ms_serial:.3f} ms  speedup={sp_serial:.1f}%",
              flush=True)
        print(f"[#3] OVERLAP (transfer hidden behind GEMM): {ms_overlap:.3f} ms  speedup={sp_overlap:.1f}%",
              flush=True)


def my_tok_of(node_tok, src_nvl, e):
    return int(node_tok[src_nvl, e])


def moved_of(cp, src_nvl, e):
    return sum(m.tokens for m in cp.migrations if m.src_rank == src_nvl and m.expert == e)


if __name__ == "__main__":
    main()
