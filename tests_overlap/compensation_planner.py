"""Intra-node compensation PLANNER (P0).

Pure, GPU-free, deterministic. Given a node's per-(rank,expert) token counts and
per-rank receive volumes, it produces:
  * a COMPUTE plan  : which expert-chunks migrate hot-rank -> cold-rank (NVLink),
                      minimizing makespan while reusing weight transfers via
                      expert-affinity, and only when net-positive vs its cost.
  * a COMM plan     : how each rank's excess inter-node ingress is re-landed on
                      idle node-local NICs (only the excess is moved).

No kernels, no numerics changed. This is the decision layer that later phases
(P1 comm, P2 compute) will consume; here we validate the decisions offline.

Cost model constants are the FFN GEMM timings measured in bench_compensation.py
(single expert, bf16, H=4096 I=2048, B30Z).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# --- measured FFN cost model (tokens -> ms), from bench_compensation Part A ---
_COST_M = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], float)
_COST_MS = np.array([0.120, 0.123, 0.151, 0.265, 0.495, 0.943, 1.820, 3.718, 7.390], float)
_SLOPE = (_COST_MS[-1] - _COST_MS[-2]) / (_COST_M[-1] - _COST_M[-2])   # ms/token tail


def gemm_ms(tokens: float) -> float:
    """FFN (gateup+swiglu+down) time for `tokens` tokens on one rank (grouped)."""
    if tokens <= 0:
        return 0.0
    return float(np.interp(tokens, _COST_M, _COST_MS,
                           right=_COST_MS[-1] + (tokens - _COST_M[-1]) * _SLOPE))


# weight transfer for one expert (w_gateup[H,2I]+w_down[I,H] bf16) over NVLink
WEIGHT_BYTES = (4096 * 2 * 2048 + 2048 * 4096) * 2        # 50.3 MB
NVLINK_BW = 900e9                                          # NV18, B30Z
WEIGHT_XFER_MS = WEIGHT_BYTES / NVLINK_BW * 1000           # ~0.056 ms (amortizable)


@dataclass
class Migration:
    expert: int          # local expert index on the donor
    src_rank: int        # donor (hot) rank within the node
    dst_rank: int        # helper (cold) rank within the node
    n_chunks: int
    tokens: int


@dataclass
class ComputePlan:
    migrations: list = field(default_factory=list)
    load_before_ms: list = field(default_factory=list)   # per-rank
    load_after_ms: list = field(default_factory=list)
    distinct_weight_moves: int = 0                        # (helper,expert) pairs
    makespan_before_ms: float = 0.0
    makespan_after_ms: float = 0.0
    weight_cost_ms: float = 0.0
    net_gain_ms: float = 0.0


@dataclass
class CommPlan:
    # relanded[dst][via] = tokens of dst's ingress that land on via's NIC
    relanded: np.ndarray = None
    recv_before: list = field(default_factory=list)
    nic_after: list = field(default_factory=list)         # per-NIC ingress tokens
    relay_tokens: int = 0
    max_before: float = 0.0
    max_after: float = 0.0


# PLACEHOLDER_COMPUTE
def _chunks_of(n_tokens: int, chunk: int) -> list:
    """Split an expert's tokens into chunk sizes: full chunks first, remainder last."""
    full = n_tokens // chunk
    rem = n_tokens % chunk
    return [chunk] * full + ([rem] if rem else [])


def plan_compute_node(tokens_re: np.ndarray, chunk: int,
                      min_gain_ms: float = 0.05,
                      amortize_weights: bool = True,
                      affinity: bool = True,
                      fine_balance: bool = True) -> ComputePlan:
    """Water-filling chunk migration within one node (R ranks x E local experts).

    - migrates whole chunks hot->cold to minimize makespan;
    - expert-affinity: prefers a helper that already holds that expert to avoid
      paying a second weight transfer;
    - cost-aware: returns a no-op plan if the net gain does not beat the
      (optionally amortized) weight-transfer cost.
    """
    R, E = tokens_re.shape
    # movable chunks per rank: dict expert -> list[size]  (donor keeps computing what stays)
    chunks = [{e: _chunks_of(int(tokens_re[r, e]), chunk) for e in range(E)} for r in range(R)]
    cur_tok = tokens_re.sum(axis=1).astype(float).tolist()
    load_before = [gemm_ms(t) for t in cur_tok]

    # (helper_rank, global_expert_id) already holding weights. global expert id = src_rank*E+e
    resident = set()                     # migrated (h, gexpert) pairs
    migs = {}                            # (src,dst,e) -> [n_chunks, tokens]

    def load(r):
        return gemm_ms(cur_tok[r])

    total0 = sum(cur_tok)
    while True:
        loads = [load(r) for r in range(R)]
        d = int(np.argmax(loads))
        if loads[d] - min(loads) <= min_gain_ms:
            break
        cand = [e for e in range(E) if chunks[d][e]]
        if not cand:
            break
        # batch: offload the donor's expert with the most chunks as a unit
        e = max(cand, key=lambda x: len(chunks[d][x]))
        # helper: prefer one already holding (h,e) (free weight reuse), else emptiest
        aff = [h for h in range(R) if h != d and (h, d * E + e) in resident] if affinity else []
        h = aff[0] if aff else int(np.argmin([cur_tok[r] if r != d else float("inf")
                                              for r in range(R)]))
        moved = 0
        # pour this expert's chunks into h while h stays strictly below the donor
        # (guarantees makespan is non-increasing) and the gap is still worth closing
        while chunks[d][e]:
            size = chunks[d][e][0]
            if gemm_ms(cur_tok[h] + size) >= load(d):
                break
            if load(d) - load(h) <= min_gain_ms:
                break
            chunks[d][e].pop(0)
            cur_tok[d] -= size
            cur_tok[h] += size
            migs.setdefault((d, h, e), [0, 0])
            migs[(d, h, e)][0] += 1
            migs[(d, h, e)][1] += size
            resident.add((h, d * E + e))
            moved += 1
            if not affinity:                              # naive: one chunk to global-min, then re-pick
                break
        if moved == 0:
            # even one chunk of the fattest expert cannot help via the emptiest helper;
            # nothing better exists this round -> stop
            break

    # Fine-balance pass: whole-chunk water-filling leaves up to ~one chunk of residual
    # skew. The chunk GEMM supports dynamic M (any 128-aligned m_size), so we trim the
    # residual with 128-aligned PARTIAL migrations to approach the ideal (perfect-balance)
    # makespan. Moving <= (load_d - load_h)/2 tokens keeps h below d, so makespan is still
    # monotonically non-increasing.
    A128 = 128
    for _ in range(4 * R if fine_balance else 0):
        loads = [gemm_ms(cur_tok[r]) for r in range(R)]
        d = int(np.argmax(loads)); h = int(np.argmin(loads))
        if loads[d] - loads[h] <= min_gain_ms:
            break
        rem = {e: sum(chunks[d][e]) for e in range(E) if chunks[d][e]}
        if not rem:
            break
        e = max(rem, key=lambda x: rem[x])
        want = (cur_tok[d] - cur_tok[h]) / 2.0
        mv = min(int(want // A128) * A128, (rem[e] // A128) * A128)
        if mv <= 0:
            break
        need, newlist = mv, []                       # split e's chunks to release `mv` tokens
        for c in chunks[d][e]:
            if need <= 0:
                newlist.append(c); continue
            take = min(c, need)
            if c - take > 0:
                newlist.append(c - take)
            need -= take
        chunks[d][e] = newlist
        cur_tok[d] -= mv; cur_tok[h] += mv
        migs.setdefault((d, h, e), [0, 0])
        migs[(d, h, e)][0] += 1
        migs[(d, h, e)][1] += mv
        resident.add((h, d * E + e))



    migrations = [Migration(expert=e, src_rank=d, dst_rank=h, n_chunks=nc, tokens=tk)
                  for (d, h, e), (nc, tk) in sorted(migs.items())]
    distinct = len({(h, d * E + e) for (d, h, e) in migs})
    load_after = [gemm_ms(t) for t in cur_tok]
    ms_before, ms_after = max(load_before), max(load_after)
    weight_cost = distinct * WEIGHT_XFER_MS
    eff_cost = 0.0 if amortize_weights else weight_cost
    net = (ms_before - ms_after) - eff_cost

    # validation
    assert abs(sum(cur_tok) - total0) < 1e-6, "token conservation violated"
    assert ms_after <= ms_before + 1e-9, "makespan increased"

    plan = ComputePlan(migrations=migrations, load_before_ms=load_before,
                       load_after_ms=load_after, distinct_weight_moves=distinct,
                       makespan_before_ms=ms_before, makespan_after_ms=ms_after,
                       weight_cost_ms=weight_cost, net_gain_ms=net)
    if net <= 0:                          # not worth it -> no-op (fall back to baseline)
        return ComputePlan(migrations=[], load_before_ms=load_before,
                           load_after_ms=load_before, distinct_weight_moves=0,
                           makespan_before_ms=ms_before, makespan_after_ms=ms_before,
                           weight_cost_ms=0.0, net_gain_ms=0.0)
    return plan


# PLACEHOLDER_COMM
def plan_comm_node(recv_r: np.ndarray) -> CommPlan:
    """Re-land only the EXCESS ingress of hot ranks onto under-target NICs.

    Most tokens still land on their owner's NIC (no NVL forward); only the amount
    above the node mean is spread to node-mates with spare NIC headroom, then the
    existing forwarder NVL-relays it to the owner. Minimizes relay volume.
    """
    recv = recv_r.astype(float)
    R = len(recv)
    target = recv.mean()
    relanded = np.zeros((R, R), float)
    np.fill_diagonal(relanded, recv)          # start: everything lands home

    spare = np.maximum(target - recv, 0.0)    # per-NIC headroom
    for dst in range(R):
        excess = recv[dst] - target
        if excess <= 0:
            continue
        for via in np.argsort(-spare):        # fill emptiest NICs first
            if excess <= 1e-9:
                break
            take = min(excess, spare[via])
            if take <= 0:
                continue
            relanded[dst, via] += take
            relanded[dst, dst] -= take
            spare[via] -= take
            excess -= take

    nic_after = relanded.sum(axis=0)
    relay = float(relanded.sum() - np.trace(relanded))

    assert abs(relanded.sum() - recv.sum()) < 1e-3, "ingress conservation violated"
    assert (relanded >= -1e-6).all(), "negative reland"

    return CommPlan(relanded=relanded, recv_before=recv.tolist(),
                    nic_after=nic_after.tolist(), relay_tokens=int(round(relay)),
                    max_before=float(recv.max()), max_after=float(nic_after.max()))


