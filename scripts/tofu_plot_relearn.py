"""Plot the relearning-robustness recovery curves (recovery axis 1).

Reads results/relearn_forget_rouge.json and plots forget-set ROUGE vs relearn
epochs, one line per unlearning STRATEGY (Full FT vs LoRA). The baseline (the
unlearned model, before any relearning) is epoch 0. A curve that climbs back fast
=> the knowledge was only suppressed, not erased.

    python scripts/tofu_plot_relearn.py      # -> results/relearn_recovery_curve.png
"""
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("tofu_plot_relearn")


def main():
    path = Path("results/relearn_forget_rouge.json")
    data = json.load(open(path))

    # Group each key into (strategy -> {epoch: rouge}). "relearn_..._ep{N}" is a
    # relearned checkpoint; anything else is the unlearned baseline (epoch 0).
    curves = {}
    for key, val in data.items():
        strat = "LoRA" if "lora" in key else "Full FT"
        m = re.search(r"_ep(\d+)$", key)
        epoch = int(m.group(1)) if (key.startswith("relearn_") and m) else 0
        curves.setdefault(strat, {})[epoch] = val

    plt.figure(figsize=(7, 5))
    for strat in sorted(curves):
        pts = curves[strat]
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=7, label=strat)
        for x, y in zip(xs, ys):
            plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8)

    plt.xlabel("Relearning epochs on the forget set")
    plt.ylabel("Forget-set ROUGE-L recall")
    plt.title("Relearning robustness: knowledge recovery after unlearning\n"
              "(higher/faster = knowledge was suppressed, not erased)")
    plt.ylim(-0.03, 1.08)
    plt.grid(alpha=0.3)
    plt.legend(title="Unlearning strategy")
    plt.tight_layout()

    out = path.parent / "relearn_recovery_curve.png"
    plt.savefig(out, dpi=150)
    logger.info("Strategies plotted: %s", ", ".join(f"{s} ({len(curves[s])} pts)" for s in curves))
    logger.info("-> %s", out)


if __name__ == "__main__":
    main()
