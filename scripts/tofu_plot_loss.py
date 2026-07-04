"""Plot LEARN / UNLEARN / RELEARN training-loss curves (validation figure).

We use save_strategy="no" (no trainer_state.json), so the only record of the loss
is the {'loss': ..., 'epoch': ...} lines HF Trainer prints to the SLURM .out logs.
This parses them out and plots loss vs epoch, one line per log file.

    python scripts/tofu_plot_loss.py --logs "logs/learn_*.out" "logs/unlearn_*.out"
    -> results/loss_curve.png
"""
import argparse
import ast
import glob
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("tofu_plot_loss")

# HF Trainer prints e.g. {'loss': '0.02', 'grad_norm': '...', 'epoch': '4.0'}
LOSS_RE = re.compile(r"\{'loss':[^}]*\}")


def parse_log(path):
    """Return a list of (xs, ys) segments, one per training run in the file.

    A single SLURM .out can hold >1 run (e.g. the LEARN job trains both the `full`
    and the `retain90` reference model). Each run restarts epoch at ~0, so when the
    epoch counter drops we start a NEW segment — otherwise plot() draws a spurious
    straight line backward across the whole chart from the end of one run to the
    start of the next.
    """
    segments = []
    xs, ys = [], []
    for line in open(path, errors="ignore"):
        m = LOSS_RE.search(line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(0))
            loss = float(d["loss"])
            epoch = float(d.get("epoch", len(xs)))
        except (ValueError, SyntaxError, KeyError):
            continue
        # Epoch went backwards -> a new run began; close the current segment.
        if xs and epoch < xs[-1]:
            segments.append((xs, ys))
            xs, ys = [], []
        xs.append(epoch)
        ys.append(loss)
    if xs:
        segments.append((xs, ys))
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="glob(s) of SLURM .out logs, e.g. 'logs/learn_*.out'")
    args = ap.parse_args()

    files = []
    for pattern in args.logs:
        files.extend(sorted(glob.glob(pattern)))

    plt.figure(figsize=(8, 5))
    plotted = 0
    for f in files:
        segments = parse_log(f)
        # Label only the first segment per file so a run split into 2 doesn't
        # produce 2 legend entries; suffix "(run 2)" etc. if there really are more.
        for i, (xs, ys) in enumerate(segments):
            if not ys:
                continue
            label = Path(f).stem if i == 0 else f"{Path(f).stem} (run {i + 1})"
            plt.plot(xs, ys, marker=".", linewidth=1.5, label=label)
            plotted += 1
    if plotted == 0:
        logger.warning("No {'loss': ...} lines found in: %s", files)
        return

    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training loss curves (LEARN / UNLEARN / RELEARN)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()

    out = Path("results/loss_curve.png")
    plt.savefig(out, dpi=150)
    logger.info("Plotted %d log(s) -> %s", plotted, out)


if __name__ == "__main__":
    main()
