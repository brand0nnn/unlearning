"""Plot the relearning-robustness recovery curves (recovery axis 1).

Globs a per-strategy relearn dir (results/relearn/forget or .../retain) and plots
forget-set ROUGE vs relearn epochs, one line per unlearning STRATEGY (Full-FT GD /
LoRA GD / Self-Distillation / GRPO — whichever have data). The baseline (the
unlearned model, before any relearning) is epoch 0. A curve that climbs back fast
=> the knowledge was only suppressed, not erased.

    python scripts/diagnostics/plot_relearn.py      # -> results/figures/relearn_forget_curve.png
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/relearn/forget",
                    help="a per-strategy relearn DIR to glob+merge (e.g. "
                         "results/relearn/forget or .../retain), OR a single "
                         "JSON file (legacy/one-off)")
    ap.add_argument("--out", default="relearn_forget_curve.png",
                    help="output PNG name under results/figures/")
    ap.add_argument("--title", default="Relearning robustness: knowledge recovery "
                    "after unlearning\n(higher/faster = suppressed, not erased)")
    ap.add_argument("--xlabel", default="Relearning epochs on the forget set")
    ap.add_argument("--label-by", default="strategy",
                    choices=["strategy", "lora_target"],
                    help="group curves by training strategy, or by LoRA target "
                         "module (for the target-module ablation)")
    args = ap.parse_args()
    src = Path(args.data)
    if src.is_dir():                       # per-strategy files -> merge them in memory
        data = {}
        for f in sorted(src.glob("*.json")):
            data.update(json.load(open(f)))
    else:                                  # single JSON (legacy / one-off)
        data = json.load(open(src))

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

    def lora_target_of(key):
        k = key.lower()
        if "lora_all" in k:    return "LoRA-all (attn+MLP)"
        if "lora_mlp" in k:    return "LoRA-MLP (gate/up/down)"
        if "lora_updown" in k: return "LoRA-MLP (up/down)"
        if "lora_qkv" in k:    return "LoRA-QKV"
        if "lora" in k:        return "LoRA-attn (q/k/v/o)"   # _lora or _lora_attn
        return "Full-FT (reference)"

    labeler = lora_target_of if args.label_by == "lora_target" else strategy_of

    curves = {}
    for key, val in data.items():
        strat = labeler(key)
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

    plt.xlabel(args.xlabel)
    plt.ylabel("Forget-set ROUGE-L recall")
    plt.title(args.title)
    plt.ylim(-0.03, 1.08)
    plt.grid(alpha=0.3)
    plt.legend(title="LoRA target module" if args.label_by == "lora_target"
               else "Unlearning strategy")
    plt.tight_layout()

    out = Path("results/figures") / args.out
    plt.savefig(out, dpi=150)
    logger.info("Strategies plotted: %s", ", ".join(f"{s} ({len(curves[s])} pts)" for s in curves))
    logger.info("-> %s", out)


if __name__ == "__main__":
    main()
