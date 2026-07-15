"""LoRA RANK-sweep comparison — overlay the unlearning dynamics across ranks.

Reads every results/curves/*_lora_r*.json (one per rank from the rank ablation) and
overlays FORGET-set and RETAIN-set ROUGE vs unlearning step, one line per rank, so
the capacity effect (more rank -> more forgetting, but also more collateral, and
where it saturates) is visible in one figure.

    python scripts/diagnostics/plot_rank_sweep.py
    -> results/figures/lora_rank_forgetting.png

CPU-only, local. Auto-includes any rank present (8/16/32/64/...).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logging_utils import get_logger

logger = get_logger("plot_rank_sweep")


def main():
    files = sorted(Path("results/curves").glob("*_lora_r*.json"))
    ranks = []
    for f in files:
        m = re.search(r"_lora_r(\d+)\.json$", f.name)
        if m:
            ranks.append((int(m.group(1)), f))
    ranks.sort()
    if not ranks:
        logger.warning("No *_lora_r*.json curves found. Run the rank ablation first.")
        return

    # Darker = higher rank, so the capacity gradient reads at a glance.
    cmap = plt.get_cmap("viridis")
    rmax = max(r for r, _ in ranks)
    # Rows = metric (ROUGE / Probability / Truth Ratio), cols = split (forget / retain).
    metrics = [("rouge", "ROUGE-L"), ("prob", "Probability"), ("truth_ratio", "Truth Ratio")]
    splits = [("forget", "FORGET — lower = more forgetting"),
              ("retain", "RETAIN — higher = less collateral")]
    curves = [(r, json.load(open(f))) for r, f in ranks]

    fig, axes = plt.subplots(len(metrics), len(splits), figsize=(11, 11), sharex=True)
    for row, (key, mlabel) in enumerate(metrics):
        for col, (split, slabel) in enumerate(splits):
            ax = axes[row][col]
            for r, d in curves:
                pts = sorted((h["step"], h[key]) for h in d["history"] if h["split"] == split)
                xs = [p[0] for p in pts]
                # Truth Ratio: bound to (0,1] as locuslab/tofu plots it (min(R,1/R));
                # idempotent for the already-bounded stored value.
                ys = [(min(p[1], 1.0 / p[1]) if key == "truth_ratio" and p[1] else p[1])
                      for p in pts]
                ax.plot(xs, ys, marker="o", ms=3.5, lw=1.8,
                        color=cmap(0.15 + 0.7 * (r / rmax)), label=f"rank {r} (α={2 * r})")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.25, ls="--")
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(slabel, fontsize=10)
            if col == 0:
                ax.set_ylabel(mlabel, fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("Unlearning steps")
    axes[0][0].legend(fontsize=8, title="LoRA rank")
    fig.suptitle("LoRA rank sweep (forget10, gradient_difference) — capacity vs "
                 "forgetting / collateral (ROUGE · Probability · Truth Ratio)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path("results/figures/lora_rank_forgetting.png")
    fig.savefig(out, dpi=110)
    logger.info("Rank sweep (%s) -> %s", ", ".join(str(r) for r, _ in ranks), out)


if __name__ == "__main__":
    main()
