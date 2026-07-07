"""Plot LEARN / UNLEARN / RELEARN training-loss curves (validation figure).

We use save_strategy="no" (no trainer_state.json), so the only record of the loss
is the {'loss': ..., 'epoch': ...} lines HF Trainer prints to the SLURM .out logs.
This parses them out and plots loss vs epoch, one line per log file.

    python scripts/diagnostics/plot_loss.py --logs "logs/learn_*.out" "logs/unlearn_*.out"
    -> results/loss_curve.png
"""
import argparse
import ast
import glob
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("plot_loss")

# HF Trainer prints e.g. {'loss': '0.02', 'grad_norm': '...', 'epoch': '4.0'}
LOSS_RE = re.compile(r"\{'loss':[^}]*\}")
# Each run is preceded by a banner the sbatch scripts echo, e.g.
#   === Unlearning: gradient_ascent ===   === [2/2] Fine-tune on retain90 ... ===
HEADER_RE = re.compile(r"===\s*(.+?)\s*===")


def clean_label(header, stem, idx):
    """Turn a run's `=== ... ===` banner into a short legend label (method name)."""
    if not header:
        return stem if idx == 0 else f"{stem} (run {idx + 1})"
    m = re.search(r"[Uu]nlearning:\s*(\w+)", header)
    if m:                                   # "Unlearning: gradient_ascent"
        prefix = "LoRA " if "lora" in header.lower() else ""
        return f"{prefix}{m.group(1)}"
    if "fine-tune" in header.lower():       # LEARN phase
        return "learn: retain90" if "retain" in header.lower() else "learn: full"
    m = re.search(r"Relearn\s*(\d+)", header)
    if m:                                   # "Relearn 2 epoch(s) ..."
        return f"relearn {m.group(1)}ep"
    return header                           # fallback: the raw banner text


def parse_log(path):
    """Return a list of (xs, ys, header) segments, one per training run in the file.

    A single SLURM .out can hold >1 run (e.g. the LEARN job trains both the `full`
    and the `retain90` reference model, and UNLEARN loops over all four methods).
    We split on either the run banner (`=== ... ===`) or an epoch reset, so plot()
    never draws a spurious straight line backward across the chart from the end of
    one run to the start of the next. Each segment carries the banner that labels it.
    """
    segments = []
    xs, ys = [], []
    header = None        # most recent banner seen
    seg_header = None    # banner in effect when the current segment started
    for line in open(path, errors="ignore"):
        h = HEADER_RE.search(line)
        if h:
            header = h.group(1).strip()
            if xs:       # banner after data -> a new run begins; close current
                segments.append((xs, ys, seg_header))
                xs, ys, seg_header = [], [], None
            continue
        m = LOSS_RE.search(line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(0))
            loss = float(d["loss"])
            epoch = float(d.get("epoch", len(xs)))
        except (ValueError, SyntaxError, KeyError):
            continue
        if not xs:
            seg_header = header
        elif epoch < xs[-1]:               # epoch went backwards -> new run
            segments.append((xs, ys, seg_header))
            xs, ys, seg_header = [], [], header
        xs.append(epoch)
        ys.append(loss)
    if xs:
        segments.append((xs, ys, seg_header))
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="glob(s) of SLURM .out logs, e.g. 'logs/learn_*.out'")
    ap.add_argument("--out", default="loss_curve.png",
                    help="output filename under results/ (e.g. learn_loss_curve.png)")
    ap.add_argument("--title", default="Training loss curves (LEARN / UNLEARN / RELEARN)",
                    help="figure title")
    args = ap.parse_args()

    files = []
    for pattern in args.logs:
        files.extend(sorted(glob.glob(pattern)))

    plt.figure(figsize=(8, 5))
    plotted = 0
    for f in files:
        segments = parse_log(f)
        # One legend entry per run, named by its banner (the method) when available,
        # falling back to the file stem for logs without banners.
        for i, (xs, ys, header) in enumerate(segments):
            if not ys:
                continue
            plt.plot(xs, ys, marker=".", linewidth=1.5,
                     label=clean_label(header, Path(f).stem, i))
            plotted += 1
    if plotted == 0:
        logger.warning("No {'loss': ...} lines found in: %s", files)
        return

    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()

    out = Path("results/figures") / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    logger.info("Plotted %d log(s) -> %s", plotted, out)


if __name__ == "__main__":
    main()
