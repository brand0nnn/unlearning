"""Plot the relearning-robustness recovery curves (recovery axis 1).

Reads results/relearn_forget_rouge.json and plots forget-set ROUGE vs relearn
epochs, one line per unlearning STRATEGY (Full-FT GD / LoRA GD / Self-Distillation
/ GRPO — whichever have data). The baseline (the unlearned model, before any
relearning) is epoch 0. A curve that climbs back fast => the knowledge was only
suppressed, not erased.

    python scripts/diagnostics/plot_relearn.py      # -> results/relearn_recovery_curve.png
"""
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("plot_relearn")


def main():
    path = Path("results/relearn_forget_rouge.json")
    data = json.load(open(path))

    # Group each key into (strategy -> {epoch: rouge}). "relearn_..._ep{N}" is a
    # relearned checkpoint; anything else is the unlearned baseline (epoch 0).
    # Strategy is read off the run-name suffix so all four cases separate cleanly.
    def strategy_of(key):
        k = key.lower()
        if "self_distill" in k:
            return "Self-Distillation"
        if "grpo" in k:
            return "GRPO"
        if "lora" in k:
            return "LoRA (grad-diff)"
        return "Full-FT (grad-diff)"

    curves = {}
    for key, val in data.items():
        strat = strategy_of(key)
        m = re.search(r"_ep(\d+)$", key)
        epoch = int(m.group(1)) if (key.startswith("relearn_") and m) else 0
        curves.setdefault(strat, {})[epoch] = val

    # Distinct vertical offsets so the epoch-0 labels (all strategies ~0 there)
    # don't overlap — works for up to 4 strategies.
    EP0_OFFSETS = [10, -14, 24, -28]

    plt.figure(figsize=(7.5, 5))
    for si, strat in enumerate(sorted(curves)):
        pts = curves[strat]
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        line, = plt.plot(xs, ys, marker="o", linewidth=2, markersize=7, label=strat)
        for x, y in zip(xs, ys):
            dy = EP0_OFFSETS[si % len(EP0_OFFSETS)] if x == 0 else 9
            plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, dy), ha="center", fontsize=8, color=line.get_color())

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
