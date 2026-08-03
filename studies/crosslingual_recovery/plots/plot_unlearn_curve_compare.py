"""Compare Full-FT vs LoRA unlearning DYNAMICS on the FORGET set — ROUGE, Probability,
and RAW Truth Ratio vs unlearning step. Used to decide a meaningful baseline: see how
deep each method can forget on each metric, and where truth ratio plateaus.

Reads results/curves/unlearn_curve_*_fullft_*.json and *_lora_*.json (written by
--track-curve). Uses truth_ratio_RAW (rises toward 1.0 = forgotten; can exceed 1),
NOT the bounded min(R,1/R). Reference lines on the TR panel: learned (~0.46, fact
known) and 1.0 (chance = truly forgotten).

    python studies/crosslingual_recovery/plots/plot_unlearn_curve_compare.py
    -> figures/unlearn_curve_compare.png
"""
import glob
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

STUDY = Path(__file__).resolve().parents[1]
CURVES = STUDY / "results" / "curves"
FIGS = STUDY / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("plot_unlearn_curve_compare")

LEARNED_TR = 0.459   # learned model's forget truth ratio (fact known)
COLORS = {"fullft": "#1f77b4", "lora": "#ff7f0e"}
NAMES = {"fullft": "Full-FT", "lora": "LoRA"}


def _load(method):
    fs = [f for f in glob.glob(str(CURVES / "unlearn_curve_*.json")) if method in f]
    if not fs:
        return None
    d = json.load(open(sorted(fs)[-1]))     # most recent matching curve
    # forget-split points: (step, rouge, prob, truth_ratio_raw)
    pts = [(h["step"], h["rouge"], h["prob"], h.get("truth_ratio_raw", h["truth_ratio"]))
           for h in d["history"] if h["split"] == "forget"]
    pts.sort()
    return pts


def main():
    data = {m: _load(m) for m in ("fullft", "lora")}
    if not any(data.values()):
        logger.warning("no forget-split curve data in %s", CURVES); return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    panels = [(1, "ROUGE-L", "forget ROUGE (↓ = forgetting)", None),
              (2, "Probability", "forget prob (↓ = forgetting)", None),
              (3, "Truth Ratio (raw)", "forget truth ratio (↑ = forgetting; 1.0 = chance)", True)]
    for ax, (idx, title, ylab, is_tr) in zip(axes, panels):
        for m, pts in data.items():
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[idx] for p in pts]
            ax.plot(xs, ys, "o-", color=COLORS[m], lw=2, ms=5, label=NAMES[m])
        if is_tr:
            ax.axhline(LEARNED_TR, color="#2ca02c", ls="--", lw=1.1)
            ax.axhline(1.0, color="grey", ls=":", lw=1.0)
            ax.text(0, LEARNED_TR - 0.04, "learned (fact known)", fontsize=7, color="#2ca02c")
            ax.text(0, 1.0 + 0.01, "chance (truly forgotten)", fontsize=7, color="grey")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("unlearning step", fontsize=10)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
    axes[0].legend(fontsize=10, title="method")
    fig.suptitle("Unlearning dynamics on the FORGET set — how deep can each method "
                 "forget on each metric? (raw truth ratio: 1.0 = truly forgotten)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "unlearn_curve_compare.png"
    fig.savefig(out, dpi=120)
    logger.info("unlearn-curve compare -> %s", out)


if __name__ == "__main__":
    main()
