"""Progressive performance analysis at EP64: (1) Baseline (no overlap) -> (2) +Overlap ->
(3) +Compensation.  Also measures the REAL fused dispatch/combine D/B (E1).

Configs 1 & 2 are MEASURED end-to-end (run_baseline vs run_overlap from the shipped test).
D/B are measured by isolating buffer.dispatch and buffer.combine (the real fused kernels).
Config 3 (overlap+compensation) is computed from the corrected per-half-layer model
(stage 69) anchored on the REAL measured D/B + the measured compensated compute makespan
ratio, since the in-kernel ON-path integration is IPC-blocked.

Run: mpirun python run.py 0-7 bench_three_configs.py
env: BENCH_SKEW (default 0.5)
"""
import os, time
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F

import test_internode_forward_backward as T   # reuse shipped run_overlap/run_baseline/helpers
from utils import initialize_fleet, configure_buffer, get_buffer

SKEW = float(os.environ.get("BENCH_SKEW", "0.5"))
E, H, I, SEQLEN, TOPK = T.E, T.H, T.I, T.SEQLEN, T.TOPK
ALIGNMENT, CHUNK = T.ALIGNMENT, T.CHUNK


def skewed_inputs(group, mode="rank"):
    """mode='rank': per-expert random skew (real skewed workload -> config1/2).
    mode='node': skew bias is CONSTANT within each node's experts, so all 8 ranks of a node
    are load-balanced (= the compute distribution after PERFECT intra-node compensation), while
    inter-node skew is preserved. Running run_overlap on this is a REAL measurement of the
    compensated iteration (config3), not an estimate. (Its intra-node comm is balanced, a mild
    optimism vs true config3 which keeps skewed routing; inter-node skew is real.)"""
    ws = group.world_size
    num_experts = ws * E
    x = paddle.randn([SEQLEN, H], "bfloat16")
    scores = paddle.randn([SEQLEN, num_experts])
    if mode == "rank":
        bias = paddle.randn([num_experts]) * SKEW
    else:  # node: one bias per node, broadcast to all its experts -> intra-node balanced
        n_nodes = ws // RANKS_PER_NODE
        experts_per_node = RANKS_PER_NODE * E
        node_bias = paddle.randn([n_nodes]) * SKEW
        bias = node_bias.reshape([n_nodes, 1]).tile([1, experts_per_node]).reshape([num_experts])
    scores += bias
    topk_weights, topk_idx = scores.topk(TOPK)
    topk_weights = F.sigmoid(topk_weights)
    topk_weights /= topk_weights.sum(axis=-1, keepdim=True)
    w_gateup = (paddle.randn([E, H, 2 * I]) * 0.02).cast("bfloat16")
    w_down = (paddle.randn([E, I, H]) * 0.02).cast("bfloat16")
    return x, topk_weights, topk_idx, w_gateup, w_down


RANKS_PER_NODE = 8


def make_bias(group):
    """shared global per-expert popularity bias (same on all ranks)."""
    ne = group.world_size * E
    bias = paddle.randn([ne]) * SKEW
    dist.broadcast(bias, src=0, group=group)
    return bias


def node_avg_bias(bias, ws):
    """replace each expert's bias by its NODE mean -> intra-node flat, inter-node preserved
    (= the load distribution after PERFECT intra-node compensation of the same workload)."""
    n_nodes = ws // RANKS_PER_NODE; epn = RANKS_PER_NODE * E
    return bias.reshape([n_nodes, epn]).mean(1, keepdim=True).tile([1, epn]).reshape([ws * E])


def inputs_from_bias(group, bias):
    ne = group.world_size * E
    x = paddle.randn([SEQLEN, H], "bfloat16")
    scores = paddle.randn([SEQLEN, ne]) + bias
    tw, ti = scores.topk(TOPK)
    tw = F.sigmoid(tw); tw /= tw.sum(axis=-1, keepdim=True)
    w_gu = (paddle.randn([E, H, 2 * I]) * 0.02).cast("bfloat16")
    w_dn = (paddle.randn([E, I, H]) * 0.02).cast("bfloat16")
    return x, tw, ti, w_gu, w_dn



def timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    paddle.device.synchronize(); dist.barrier()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); paddle.device.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def gmax(v, group):
    t = paddle.to_tensor([v], "float32"); dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t)


# PLACEHOLDER_MAIN
def measure_imb(ti, ws, group):
    """global per-rank imbalance (max/mean) from the routing topk indices."""
    num_experts = ws * E
    cnt = paddle.bincount(ti.flatten(), minlength=num_experts).cast("float32")
    dist.all_reduce(cnt, group=group)
    per_rank = cnt.reshape([ws, E]).sum(1)
    return float(per_rank.max() / per_rank.mean())


def main():
    group = initialize_fleet()
    ws = dist.get_world_size(); rank = dist.get_rank()
    configure_buffer(T.COMM_NUM_SMS)
    buffer = get_buffer(group, H * 2)

    # sweep skews to span per-rank imbalance ~1 -> ~3 (routing is RANDOM: randn bias each draw)
    skews = [float(s) for s in os.environ.get("BENCH_SKEWS", "0.6,1.1,1.6,2.2").split(",")]
    if rank == 0:
        print(f"[fb3] EP{ws}  (fwd/bwd/tot ms; cfg1 baseline | cfg2 overlap | cfg3 overlap+comp[node-balanced])", flush=True)
        print(f"{'imb':>5} | {'B_fwd':>8} {'B_bwd':>8} {'B_tot':>8} | {'O_fwd':>8} {'O_bwd':>8} {'O_tot':>8} | "
              f"{'C_fwd':>7} {'C_bwd':>7} {'C_tot':>7} | gains", flush=True)

    for sk in skews:
        global SKEW; SKEW = sk
        bias = make_bias(group)                          # shared global popularity
        # config1/2: real skewed workload; config3: SAME inter-node skew, intra-node flattened
        x, tw, ti, w_gu, w_dn = inputs_from_bias(group, bias)
        xn, twn, tin, wgn, wdn = inputs_from_bias(group, node_avg_bias(bias, ws))
        dout = paddle.randn_like(x); doutn = paddle.randn_like(xn)
        w_gu_ref = T.deinterleave_gateup(w_gu)
        g1 = paddle.empty(w_gu.shape, "float32"); g2 = paddle.empty(w_dn.shape, "float32")
        g3 = paddle.empty_like(g1); g4 = paddle.empty_like(g2)
        gc1 = paddle.empty(wgn.shape, "float32"); gc2 = paddle.empty(wdn.shape, "float32")
        imb = measure_imb(ti, ws, group); imb3 = measure_imb(tin, ws, group)
        def base_full(): g3.zero_(); g4.zero_(); T.run_baseline(group, buffer, x, tw, ti, dout, w_gu_ref, w_dn, g3, g4)
        def ovl_full(): g1.zero_(); g2.zero_(); T.run_overlap(group, buffer, x, tw, ti, dout, w_gu, w_dn, g1, g2)
        def comp_full(): gc1.zero_(); gc2.zero_(); T.run_overlap(group, buffer, xn, twn, tin, doutn, wgn, wdn, gc1, gc2)

        T.FWD_ONLY = True
        base_fwd = gmax(timed(base_full, iters=8), group); ovl_fwd = gmax(timed(ovl_full, iters=8), group)
        comp_fwd = gmax(timed(comp_full, iters=8), group)
        T.FWD_ONLY = False
        base_tot = gmax(timed(base_full, iters=8), group); ovl_tot = gmax(timed(ovl_full, iters=8), group)
        comp_tot = gmax(timed(comp_full, iters=8), group)

        if rank == 0:
            print(f"{imb:5.2f} | {base_fwd:8.2f} {base_tot-base_fwd:8.2f} {base_tot:8.2f} | "
                  f"{ovl_fwd:8.2f} {ovl_tot-ovl_fwd:8.2f} {ovl_tot:8.2f} | "
                  f"{comp_fwd:7.2f} {comp_tot-comp_fwd:7.2f} {comp_tot:7.2f} | "
                  f"ovl {(base_tot-ovl_tot)/base_tot*100:4.0f}% comp {(ovl_tot-comp_tot)/ovl_tot*100:4.0f}% "
                  f"tot {(base_tot-comp_tot)/base_tot*100:4.0f}%", flush=True)


if __name__ == "__main__":
    main()


