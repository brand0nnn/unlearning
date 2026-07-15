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
    fig, (axf, axr) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for r, f in ranks:
        d = json.load(open(f))
        color = cmap(0.15 + 0.7 * (r / rmax))
        for ax, split in [(axf, "forget"), (axr, "retain")]:
            pts = sorted((h["step"], h["rouge"]) for h in d["history"] if h["split"] == split)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", ms=4, lw=2, color=color,
                    label=f"rank {r} (α={2 * r})")

    axf.set_title("FORGET set — lower = more forgetting", fontsize=11)
    axr.set_title("RETAIN set — higher = less collateral", fontsize=11)
    for ax in (axf, axr):
        ax.set_xlabel("Unlearning steps")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25, ls="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, title="LoRA rank")
    axf.set_ylabel("ROUGE-L recall")
    fig.suptitle("LoRA rank sweep (forget10, gradient_difference) — "
                 "capacity vs forgetting / collateral", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path("results/figures/lora_rank_forgetting.png")
    fig.savefig(out, dpi=110)
    logger.info("Rank sweep (%s) -> %s", ", ".join(str(r) for r, _ in ranks), out)


if __name__ == "__main__":
    main()
