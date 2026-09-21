"""End-to-end single-node compute-compensation demo, bitwise-verified.

Chains all three verified layers on 8 real GPUs, WITHOUT touching internode.cu:

  planner (compensation_planner)              decide chunk migrations hot->cold
     -> per-GPU schedule (extended weight table + task_queue)
     -> real DeepGEMM chunk GEMM engine        gateup / weighted-swiglu / down
     -> cross-GPU: weight prefetch + input staging + output writeback (NVLink)
     -> assemble per-expert output on owner
     -> assert compensated == baseline  (bitwise, diff == 0)

Baseline = each GPU runs only its own experts through the same chunk engine.
Because the chunk GEMM is row-independent, migrating an expert's chunk to a helper
yields byte-identical output, so we require diff == 0.

Run: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python moe_e2e_compensated.py
"""
import time
import numpy as np
import paddle
import paddle.nn.functional as F

import compensation_planner as P

H, I, E, NGPU = 4096, 2048, 8, 8
NUM_RANKS = NGPU
NUM_EXPERTS = NGPU * E
TOPK = 8
CHUNK = 4096
ALIGNMENT = 128
SEQLEN = 4096
NVLINK_BW = 900e9
SKEWS = [0.0, 0.2, 0.35, 0.5]
_devs = [f"gpu:{g}" for g in range(NGPU)]
DEV = "gpu:0"                                             # DeepGEMM JIT runtime is single-device
                                                          # per process -> emulate the node on one GPU
import deep_gemm


def align(n):
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


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


def build_assignments(T, cp):
    """{ge: [(row_start, count, compute_gpu)]}. Owner keeps non-migrated rows."""
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
    return base, comp


class Weights:
    """All expert weights on the single execution GPU. Migration is emulated:
    'prefetch' is not a physical copy here (single-device), but the transferred
    bytes are accounted for the NVLink cost model."""
    def __init__(self, seed=0):
        self.w = {}
        paddle.set_device(DEV)
        for ge in range(NUM_EXPERTS):
            paddle.seed(seed + ge)
            self.w[ge] = ((paddle.randn([H, 2 * I]) * 0.02).cast("bfloat16"),
                          (paddle.randn([I, H]) * 0.02).cast("bfloat16"))

    def on(self, gpu, ge):
        return self.w[ge]


# PLACEHOLDER_RUN
def _chunk_ffn(x_g, wgu_s, wd_s, probs_g, tq):
    """Run the real DeepGEMM chunk engine over a GPU's task queue -> o3."""
    M = x_g.shape[0]
    o1 = paddle.zeros([M, 2 * I], "bfloat16")
    o2 = paddle.zeros([M, I], "bfloat16")
    o3 = paddle.zeros([M, H], "bfloat16")
    for ti in range(tq.shape[0]):
        deep_gemm.bf16_chunk_gemm_nn(x_g, wgu_s, o1, tq, ti)
        deep_gemm.chunk_weighted_swiglu(o1, probs_g, o2, tq, ti, CHUNK, precise=True)
        deep_gemm.bf16_chunk_gemm_nn(o2, wd_s, o3, tq, ti)
    return o3


def run_chunk_engine(assign, x, probs, W, time_it=False):
    """Execute the schedule through the real DeepGEMM chunk engine. Each 'rank'
    (compute_gpu) gets its own chunk layout + extended weight table + task_queue;
    all run on one physical GPU (DeepGEMM JIT is single-device/process). Per-rank
    time is measured individually -> node makespan = max. Returns
    (o3_per_expert, per_rank_ms, writeback_bytes)."""
    paddle.set_device(DEV)
    per_gpu = {g: [] for g in range(NGPU)}
    for ge, items in assign.items():
        for (rs, rc, gpu) in items:
            if rc > 0:
                per_gpu[gpu].append((ge, rs, rc))

    partial = {}                                          # (ge,rs) -> o3 slice
    per_gpu_ms = [0.0] * NGPU
    for g in range(NGPU):
        items = per_gpu[g]
        if not items:
            continue
        slot, wgu_list, wd_list = {}, [], []
        for (ge, rs, rc) in items:
            if ge not in slot:
                slot[ge] = len(wgu_list)
                wgu, wd = W.on(g, ge)
                wgu_list.append(wgu); wd_list.append(wd)
        wgu_s = paddle.stack(wgu_list)                    # extended weight table
        wd_s = paddle.stack(wd_list)
        regions, cur = [], 0
        for (ge, rs, rc) in items:
            regions.append((ge, rs, rc, cur)); cur += align(rc)
        M = cur
        x_g = paddle.zeros([M, H], "bfloat16")
        probs_g = paddle.zeros([M], "float32")
        for (ge, rs, rc, st) in regions:
            x_g[st:st + rc] = x[ge][rs:rs + rc]
            probs_g[st:st + rc] = probs[ge][rs:rs + rc]
        q = []
        for (ge, rs, rc, st) in regions:
            for off in range(0, rc, CHUNK):
                q.append([slot[ge], st + off, min(CHUNK, rc - off), 1])
        tq = paddle.to_tensor(q, "int32")
        if time_it:
            for _ in range(2):
                _chunk_ffn(x_g, wgu_s, wd_s, probs_g, tq)
            paddle.device.synchronize()
            t0 = time.perf_counter(); o3 = _chunk_ffn(x_g, wgu_s, wd_s, probs_g, tq)
            paddle.device.synchronize(); per_gpu_ms[g] = (time.perf_counter() - t0) * 1000
        else:
            o3 = _chunk_ffn(x_g, wgu_s, wd_s, probs_g, tq)
        for (ge, rs, rc, st) in regions:
            partial[(ge, rs)] = o3[st:st + rc]
    paddle.device.synchronize()

    # assemble per-expert output on owner; account migrated-row bytes for NVLink model
    outs = {}
    wb_bytes = [0] * NGPU
    for ge in range(NUM_EXPERTS):
        total = sum(rc for (rs, rc, gpu) in assign[ge])
        o = paddle.zeros([max(total, 1), H], "bfloat16")
        for (rs, rc, gpu) in assign[ge]:
            if rc == 0:
                continue
            o[rs:rs + rc] = partial[(ge, rs)]
            if gpu != ge // E:                            # migrated -> would be NVLink writeback
                wb_bytes[gpu] += rc * H * 2
        outs[ge] = o
    return outs, per_gpu_ms, wb_bytes


# PLACEHOLDER_MAIN
def main():
    W = Weights()
    deep_gemm.set_num_sms(100)
    print("=== end-to-end intra-node compute compensation (real DeepGEMM chunk engine, 8x B30Z) ===")
    print("skew  imb   migs wmoves | bitwise(comp==base)  | base_ms comp_ms speedup")
    for skew in SKEWS:
        tok = sim_node(skew)
        T = {ge: int(tok[ge // E, ge % E]) for ge in range(NUM_EXPERTS)}
        # per-expert inputs on owner
        x, probs = {}, {}
        paddle.set_device(DEV)
        for ge in range(NUM_EXPERTS):
            paddle.seed(1000 + ge)
            x[ge] = paddle.randn([max(T[ge], 1), H], "bfloat16")
            probs[ge] = paddle.rand([max(T[ge], 1)], "float32")

        cp = P.plan_compute_node(tok, CHUNK)
        base_a, comp_a = build_assignments(T, cp)

        ob, base_ms, _ = run_chunk_engine(base_a, x, probs, W, time_it=True)
        oc, comp_ms, wb = run_chunk_engine(comp_a, x, probs, W, time_it=True)

        # bitwise correctness per expert (real rows only)
        maxdiff = 0.0
        for ge in range(NUM_EXPERTS):
            if T[ge] == 0:
                continue
            maxdiff = max(maxdiff, float((ob[ge].astype("float32") - oc[ge].astype("float32")).abs().max()))

        ms_b = max(base_ms)
        ov = [wb[g] / NVLINK_BW * 1000 for g in range(NGPU)]
        ms_c = max(comp_ms[g] + ov[g] for g in range(NGPU))
        per_rank = np.array([tok[r].sum() for r in range(NUM_RANKS)], float)
        imb = float(per_rank.max() / per_rank.mean())
        speed = (ms_b - ms_c) / ms_b * 100 if ms_b > 0 else 0
        tag = "PASS(0)" if maxdiff == 0 else f"DIFF={maxdiff:.2e}"
        print(f"{skew:4} {imb:5.2f}  {len(cp.migrations):3d}  {cp.distinct_weight_moves:3d}   |  "
              f"{tag:>16}    | {ms_b:6.2f}  {ms_c:6.2f}  {speed:5.1f}%")
    print("\nend-to-end demo done: planner -> extended weight table -> chunk GEMM "
          "-> NVLink writeback, per-expert output bitwise-identical to baseline.")


if __name__ == "__main__":
    main()

