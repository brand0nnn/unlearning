"""LoRA-locality ablation — plot forget-set RECOVERY per location.

Overlays recovery (forget ROUGE / probability / truth-ratio vs relearn epoch), one
line per location, for a given rank scheme. Location labels include the trainable-
parameter count so capacity is always visible (essential for reading the samerank
scheme). Reads lora_locality/out/<scheme>/recovery/*.json; writes into the same tree.

    python lora_locality/plot.py --scheme fixedbudget
    python lora_locality/plot.py --scheme samerank
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("locality_plot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    args = ap.parse_args()

    rec_dir = Path(f"lora_locality/out/{args.scheme}/recovery")
    files = sorted(rec_dir.glob("*.json"))
    if not files:
        logger.warning("no recovery JSON in %s — run the relearn stage first", rec_dir)
        return

    data = [json.load(open(f)) for f in files]
    data.sort(key=lambda d: d["params"])          # order legend by capacity
    metrics = [("rouge", "Forget ROUGE-L"), ("prob", "Forget probability"),
               ("truth_ratio", "Forget truth ratio")]
    cmap = plt.get_cmap("viridis")
    n = len(data)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        for i, d in enumerate(data):
            pts = sorted((int(e), v[key]) for e, v in d["points"].items())
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", ms=5, lw=2, color=cmap(0.1 + 0.8 * i / max(n - 1, 1)),
                    label=f"{d['location']} (r{d['rank']}, {d['params'] / 1e6:.0f}M)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Relearn epochs on the forget set")
        ax.grid(True, alpha=0.25, ls="--")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("recovery (↑ = knowledge returned)")
    axes[0].legend(fontsize=8, title="location (rank, LoRA params)")
    fig.suptitle(f"LoRA-locality recovery — {args.scheme}  "
                 f"(epoch 0 = matched unlearned start; lower recovery = deeper deletion)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = rec_dir.parent / f"recovery_{args.scheme}.png"
    fig.savefig(out, dpi=110)
    logger.info("Locality recovery (%d locations) -> %s", n, out)


if __name__ == "__main__":
    main()
