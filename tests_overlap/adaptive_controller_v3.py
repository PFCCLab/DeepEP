"""Stage 86 — Adaptive controller v3: unified compensation decision + regime map, integrating
ALL the real measurements from stages 77-85. Pure logic + plot, single process -> safe path.

Consolidates the actionable thresholds we measured, so the paper's "how to use it" section is a
single decision function instead of scattered results:
  - stage 84: compute-bound knee M* ~ 2048 tok/expert -> below it compensation is a no-op
              (memory/latency-bound; matches decode negative stage 48).
  - stage 78: competitors' SM theft costs +18-20% real; ours is 0-SM copy engine.
  - stage 82: backward ~2x forward compute and overlap can't hide it -> compensation's big lever.
  - stage 83: inter-node skew needs pre-dispatch PLACEMENT first; we absorb the intra-node residual.
  - stage 37: a single super-hot expert (high imb, few experts) needs k=2 TP-split, not migration.
  - stage 85: FP8 compute ~1.58x cheaper -> knee shifts, SM-free argument strengthens.
  - stage 76: transport = 0-SM copy engine, hidden under overlap.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M_STAR = 2048               # compute-bound knee, tok/expert (stage 84)
IMB_NOOP = 1.05             # below this, imbalance not worth acting on
IMB_SUPERHOT = 2.2          # above this at few experts -> single super-hot expert regime (stage 37)
NODE_SKEW_HI = 1.15         # per-node imbalance above which pre-dispatch placement is needed (stage 83)


def decide(imb_rank, tokens_per_rank, experts_per_rank, node_imb, precision="bf16"):
    """Return (action, flags). imb_rank = per-rank max/mean; node_imb = per-NODE max/mean."""
    flags = []
    # FP8 lowers compute -> needs more tokens/expert to be compute-bound -> knee nudges up.
    m_star = M_STAR * (1.3 if precision == "fp8" else 1.0)
    tok_per_expert = tokens_per_rank / max(experts_per_rank, 1)

    if node_imb > NODE_SKEW_HI:
        flags.append("placement(LPT) first: inter-node skew -> per-node imb high (stage83)")

    if tok_per_expert < m_star:
        return "no-op (memory/latency-bound: below compute-bound knee M*)", flags
    if imb_rank < IMB_NOOP:
        return "no-op (already balanced)", flags
    if imb_rank >= IMB_SUPERHOT and experts_per_rank <= 4:
        return "k=2 TP-split (single super-hot expert; migration would flood one helper)", flags
    flags.append("transport=0-SM copy engine, hidden under overlap; backward is the big lever")
    return "token-chunk migration (intra-node compute compensation)", flags


def main():
    print("=== adaptive controller v3 decisions (real thresholds) ===")
    cases = [
        ("decode",      1.6, 384,    8, 1.0,  "bf16"),
        ("prefill sm",  1.3, 4096,   8, 1.05, "bf16"),
        ("train EP64",  1.4, 32768,  8, 1.16, "bf16"),
        ("train FP8",   1.4, 32768,  8, 1.16, "fp8"),
        ("super-hot",   2.4, 16384,  4, 1.0,  "bf16"),
        ("inter-skew",  1.5, 16384,  8, 1.4,  "bf16"),
        ("fine-grain",  1.6, 16384,  2, 1.0,  "bf16"),
    ]
    for name, imb, tok, E, nimb, prec in cases:
        act, flags = decide(imb, tok, E, nimb, prec)
        print(f"  [{name:11}] imb={imb} tok/rank={tok} E={E} node_imb={nimb} {prec:4} -> {act}")
        for f in flags:
            print(f"                 + {f}")

    # regime map: per-rank imbalance (y) x tokens-per-rank (x), colored by decision (E=8, bf16, node_imb~1)
    toks = np.logspace(np.log10(256), np.log10(131072), 200)
    imbs = np.linspace(1.0, 2.6, 200)
    E = 8
    grid = np.zeros((len(imbs), len(toks)))
    labels = {0: "no-op (mem-bound)", 1: "no-op (balanced)", 2: "token-migration", 3: "k=2 TP-split"}
    for i, im in enumerate(imbs):
        for j, tk in enumerate(toks):
            act, _ = decide(im, tk, E, 1.0, "bf16")
            grid[i, j] = (0 if "memory" in act else 1 if "balanced" in act
                          else 3 if "TP-split" in act else 2)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    cmap = plt.get_cmap("Set2", 4)
    im_h = ax.pcolormesh(toks, imbs, grid, cmap=cmap, vmin=-0.5, vmax=3.5, shading="auto")
    ax.set_xscale("log")
    ax.axvline(M_STAR * E, ls="--", color="k", lw=1.2)
    ax.text(M_STAR * E * 1.05, 2.45, "M*·E (compute-bound onset)", fontsize=8)
    ax.set_xlabel("tokens per rank (log)"); ax.set_ylabel("per-rank imbalance (max/mean)")
    ax.set_title("Adaptive controller v3 regime map (E=8, bf16, per-node~1)\n"
                 "integrating stages 77-85 real thresholds")
    cbar = fig.colorbar(im_h, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([labels[k] for k in range(4)], fontsize=8)
    fig.tight_layout()
    out = "/root/paddlejob/share-storage/gpfs/system-public/shenliang/TeraMoE/figures/adaptive_controller_v3.png"
    fig.savefig(out, dpi=140); print("saved:", out)


if __name__ == "__main__":
    main()
