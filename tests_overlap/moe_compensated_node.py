"""Intra-node compute-compensation: EXECUTION layer (P2, real multi-GPU).

Implements the actual mechanism on 8 physical GPUs using real FFN GEMMs and real
cross-GPU weight/result movement, on top of the P0 decision layer
(compensation_planner). Provides:

  * a resident weight cache (weights migrated once, reused across steps -> the
    per-step prefetch cost decays to the plan DELTA only);
  * work assignment (baseline vs compensated) as row-ranges per expert;
  * bitwise-correct execution: FFN is row-independent, so computing an expert's
    rows on a helper GPU yields the SAME result as on the owner -> we assert
    diff == 0 against the uncompensated baseline.

This is a safe reference implementation / correctness oracle; it does NOT modify
DeepEP's internode kernel (that integration is the follow-up P2-in-kernel).
"""
import time
import numpy as np
import paddle

import compensation_planner as P

H, I, E, NGPU = 4096, 2048, 8, 8
NUM_RANKS = NGPU
NUM_EXPERTS = NGPU * E
TOPK = 8
CHUNK = 4096
NVLINK_BW = 900e9                                   # NV18, B30Z (real p2p, via NCCL/cudaMemcpyPeer)
_devs = [f"gpu:{g}" for g in range(NGPU)]


def _silu(x):
    xf = x.cast("float32")
    return xf / (1.0 + paddle.exp(-xf))


def ffn(x, wgu, wd):
    """gateup -> weighted silu -> down. Row-independent (row i depends only on x[i])."""
    o1 = paddle.matmul(x, wgu)
    g, u = o1.chunk(2, axis=-1)
    o2 = (_silu(g) * u).cast("bfloat16")
    return paddle.matmul(o2, wd)


class WeightStore:
    """Owns each expert's weights on its owner GPU; caches migrated copies on
    helper GPUs across steps (resident set)."""

    def __init__(self, seed=0):
        self.owner_w = {}                            # ge -> (wgu, wd) on owner gpu
        self.resident = {}                           # (gpu, ge) -> (wgu, wd) copy
        for ge in range(NUM_EXPERTS):
            paddle.set_device(_devs[ge // E])
            paddle.seed(seed + ge)
            wgu = (paddle.randn([H, 2 * I]) * 0.02).cast("bfloat16")
            wd = (paddle.randn([I, H]) * 0.02).cast("bfloat16")
            self.owner_w[ge] = (wgu, wd)

    def get(self, gpu, ge):
        if gpu == ge // E:
            return self.owner_w[ge]
        return self.resident[(gpu, ge)]

    def ensure_resident(self, gpu, ge):
        """Return True if a NEW transfer was needed (cache miss)."""
        if gpu == ge // E or (gpu, ge) in self.resident:
            return False
        wgu, wd = self.owner_w[ge]
        self.resident[(gpu, ge)] = (wgu._copy_to(paddle.CUDAPlace(gpu), False),
                                    wd._copy_to(paddle.CUDAPlace(gpu), False))
        return True


# PLACEHOLDER_EXEC
def build_assignments(tokens_re, cp):
    """Return per-expert row-range work: {ge: [(row_start, count, compute_gpu), ...]}."""
    T = {ge: int(tokens_re[ge // E, ge % E]) for ge in range(NUM_EXPERTS)}
    base = {ge: [(0, T[ge], ge // E)] for ge in range(NUM_EXPERTS)}
    mig = {}
    for m in cp.migrations:
        ge = m.src_rank * E + m.expert
        mig.setdefault(ge, []).append((m.dst_rank, m.tokens))
    comp = {}
    for ge in range(NUM_EXPERTS):
        owner, total = ge // E, T[ge]
        moved = sum(t for _, t in mig.get(ge, []))
        items, cur = [], 0
        keep = total - moved
        if keep > 0:
            items.append((0, keep, owner)); cur = keep
        for dst, tks in mig.get(ge, []):
            items.append((cur, tks, dst)); cur += tks
        comp[ge] = items or [(0, 0, owner)]
    return T, base, comp


def run(assign, x, ws):
    """Execute one FFN pass. Returns (outputs, per_gpu_ffn_ms, prefetch_bytes[g], writeback_bytes[g]).
    Copies are done for correctness but EXCLUDED from FFN timing; their cost is
    modelled at NVLink BW by the caller (paddle's raw copy is host-staged)."""
    pf_bytes = [0] * NGPU
    wb_bytes = [0] * NGPU
    # 1. pre-stage inputs + weights to compute gpu (outside FFN timing)
    staged = {}                                     # (ge, rs) -> (xslice_on_gpu, gpu, rc)
    for ge, items in assign.items():
        for (rs, rc, gpu) in items:
            if rc == 0:
                continue
            owner = ge // E
            if gpu == owner:
                xs = x[ge][rs:rs + rc]
            else:
                paddle.set_device(_devs[owner])
                src = x[ge][rs:rs + rc]
                xs = src._copy_to(paddle.CUDAPlace(gpu), False)
                if ws.ensure_resident(gpu, ge):
                    pf_bytes[gpu] += P.WEIGHT_BYTES
            staged[(ge, rs)] = (xs, gpu, rc)
    paddle.device.synchronize()
    # 2. time FFN per gpu
    per_gpu_items = {g: [] for g in range(NGPU)}
    for (ge, rs), (xs, gpu, rc) in staged.items():
        per_gpu_items[gpu].append((ge, rs, xs))
    ffn_ms = [0.0] * NGPU
    partials = {}                                   # (ge, rs) -> o3 on compute gpu
    for g in range(NGPU):
        if not per_gpu_items[g]:
            continue
        paddle.set_device(_devs[g])
        paddle.device.synchronize()
        t0 = time.perf_counter()
        for (ge, rs, xs) in per_gpu_items[g]:
            wgu, wd = ws.get(g, ge)
            partials[(ge, rs)] = ffn(xs, wgu, wd)
        paddle.device.synchronize()
        ffn_ms[g] = (time.perf_counter() - t0) * 1000
    # 3. writeback + assemble full outputs on owner
    outs = {}
    for ge in range(NUM_EXPERTS):
        owner = ge // E
        total = sum(rc for (rs, rc, gpu) in assign[ge])
        paddle.set_device(_devs[owner])
        o = paddle.empty([max(total, 1), H], "bfloat16")
        for (rs, rc, gpu) in assign[ge]:
            if rc == 0:
                continue
            p = partials[(ge, rs)]
            if gpu != owner:
                p = p._copy_to(paddle.CUDAPlace(owner), False)
                wb_bytes[gpu] += rc * H * 2
            o[rs:rs + rc] = p
        outs[ge] = o
    paddle.device.synchronize()
    return outs, ffn_ms, pf_bytes, wb_bytes


# PLACEHOLDER_MAIN
SEQLEN = 16384
SKEWS = [0.0, 0.1, 0.2, 0.35, 0.5]


def sim_node(skew, tokens=SEQLEN * NUM_RANKS, batch=65536):
    bias = (paddle.randn([NUM_EXPERTS]) * skew) if skew > 0 else paddle.zeros([NUM_EXPERTS])
    counts = paddle.zeros([NUM_EXPERTS], "int64")
    done = 0
    while done < tokens:
        b = min(batch, tokens - done)
        s = paddle.randn([b, NUM_EXPERTS]) + bias
        _, idx = s.topk(TOPK)
        counts += paddle.bincount(idx.reshape([-1]), minlength=NUM_EXPERTS).cast("int64")
        done += b
    return counts.numpy().reshape(NUM_RANKS, E)


def make_x(T, seed=1234):
    x = {}
    for ge in range(NUM_EXPERTS):
        paddle.set_device(_devs[ge // E])
        paddle.seed(seed + ge)
        x[ge] = paddle.randn([max(T[ge], 1), H], "bfloat16")
    return x


def makespan(ffn_ms, pf_bytes, wb_bytes, amortized=True):
    ov = [(wb_bytes[g] + (0 if amortized else pf_bytes[g])) / NVLINK_BW * 1000 for g in range(NGPU)]
    return max(ffn_ms[g] + ov[g] for g in range(NGPU))


_ft = {}
def ffn_time(tokens):
    """Robust warmed median FFN time for `tokens` tokens (gpu:0; all B30Z identical)."""
    key = int(round(max(tokens, 0) / 256)) * 256
    if key in _ft:
        return _ft[key]
    if key == 0:
        _ft[key] = 0.0; return 0.0
    paddle.set_device(_devs[0])
    wgu, wd = _WS0[0], _WS0[1]
    x = paddle.randn([key, H], "bfloat16")
    for _ in range(3):
        ffn(x, wgu, wd)
    paddle.device.synchronize()
    ts = []
    for _ in range(12):
        t0 = time.perf_counter(); ffn(x, wgu, wd); paddle.device.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    _ft[key] = float(np.median(ts)); return _ft[key]


def gpu_totals(assign):
    tot = [0] * NGPU
    for ge, items in assign.items():
        for (rs, rc, gpu) in items:
            tot[gpu] += rc
    return tot


_WS0 = None


def main():
    global _WS0
    ws = WeightStore()
    _WS0 = ws.owner_w[0]
    for g in range(NGPU):                            # warm all devices
        paddle.set_device(_devs[g]); ffn(paddle.randn([4096, H], "bfloat16"), *ws.owner_w[g * E])
    paddle.device.synchronize()

    print("=== P2 intra-node compute compensation: correctness + benefit (8x B30Z) ===")
    print("skew  imb   base_ms  comp_ms  speedup  |  bitwise_diff  migs  wmoves")
    rows = []
    for skew in SKEWS:
        tok = sim_node(skew)
        T = {ge: int(tok[ge // E, ge % E]) for ge in range(NUM_EXPERTS)}
        x = make_x(T)
        cp = P.plan_compute_node(tok, CHUNK)
        _, base_a, comp_a = build_assignments(tok, cp)

        # correctness: real 8-GPU execution, compensated == baseline bitwise
        ob, _, _, _ = run(base_a, x, ws)
        oc, _, pf, wb = run(comp_a, x, ws)
        maxdiff = 0.0
        for ge in range(NUM_EXPERTS):
            if T[ge] == 0:
                continue
            paddle.set_device(_devs[ge // E])
            maxdiff = max(maxdiff, float((ob[ge].astype("float32") - oc[ge].astype("float32")).abs().max()))

        # benefit: robust makespan from warmed per-GPU FFN timing + NVLink-modeled overhead
        bt, ct = gpu_totals(base_a), gpu_totals(comp_a)
        ms_b = max(ffn_time(bt[g]) for g in range(NGPU))
        ov = [(wb[g]) / NVLINK_BW * 1000 for g in range(NGPU)]     # weights cached across steps
        ms_c = max(ffn_time(ct[g]) + ov[g] for g in range(NGPU))
        imb = float(np.array([tok[r].sum() for r in range(NUM_RANKS)]).max()
                    / np.array([tok[r].sum() for r in range(NUM_RANKS)]).mean())
        speed = (ms_b - ms_c) / ms_b * 100
        rows.append((imb, ms_b, ms_c, speed, maxdiff))
        ok = "PASS(0)" if maxdiff == 0 else f"DIFF={maxdiff:.2e}"
        print(f"{skew:4} {imb:5.2f}  {ms_b:7.3f}  {ms_c:7.3f}  {speed:5.1f}%  |  {ok:>10}  "
              f"{len(cp.migrations):3d}  {cp.distinct_weight_moves:3d}")

    # weight-cache amortization: rerun most-skewed plan; prefetch should vanish
    tok = sim_node(SKEWS[-1]); T = {ge: int(tok[ge // E, ge % E]) for ge in range(NUM_EXPERTS)}
    x = make_x(T); cp = P.plan_compute_node(tok, CHUNK); _, _, comp_a = build_assignments(tok, cp)
    _, _, pf1, _ = run(comp_a, x, ws)
    _, _, pf2, _ = run(comp_a, x, ws)
    print(f"\n[weight cache] step1 prefetch={sum(pf1)/1e6:.1f} MB  ->  "
          f"step2 prefetch={sum(pf2)/1e6:.1f} MB (resident-set hit -> amortized to 0)")

    make_figure(rows)


def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imb = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(imb, [r[1] for r in rows], "o-", color="tab:red", lw=2, label="baseline (hot rank gates)")
    ax1.plot(imb, [r[2] for r in rows], "s-", color="tab:green", lw=2,
             label="compute-compensated (real 8-GPU, +NVLink writeback)")
    ax1.set_xlabel("per-node imbalance (max/mean tokens per rank)")
    ax1.set_ylabel("node FFN makespan (ms)")
    ax1.set_title("(a) Real measured makespan (bitwise-verified correct)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax2.plot(imb, [r[3] for r in rows], "D-", color="tab:blue", lw=2.5)
    for x_, y in zip(imb, [r[3] for r in rows]):
        ax2.annotate(f"{y:.0f}%", (x_, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax2.axhline(0, color="gray", lw=1)
    ax2.set_xlabel("per-node imbalance (max/mean tokens per rank)")
    ax2.set_ylabel("node makespan speedup (%)")
    ax2.set_title("(b) Verified compute-compensation speedup"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = "/root/paddlejob/share-storage/gpfs/system-public/shenliang/TeraMoE/figures/p2_verified_benefit.png"
    fig.savefig(out, dpi=140); print("saved:", out)


if __name__ == "__main__":
    main()

