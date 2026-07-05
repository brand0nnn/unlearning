"""Plot TOFU Figure 8 (unlearning dynamics) from the tracked curve JSON(s).

Reads results/unlearn_curve_*.json (written when unlearning ran with
--track-curve) and renders a 3-panel ROUGE / Probability / Truth-Ratio figure vs
unlearning step, one line per eval split.

    python scripts/diagnostics/plot_unlearn_curve.py
    python scripts/diagnostics/plot_unlearn_curve.py --curve results/unlearn_curve_gradient_difference_forget05.json

CPU-only (no GPU / no torch). Run after the tracked unlearning job finishes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.plotting import unlearn_curve
from src.utils.logging_utils import get_logger

logger = get_logger("plot_unlearn_curve")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", nargs="*", default=None,
                    help="specific curve JSON(s); default = all results/unlearn_curve_*.json")
    args = ap.parse_args()

    files = ([Path(c) for c in args.curve] if args.curve
             else sorted(Path("results").glob("unlearn_curve_*.json")))
    if not files:
        logger.warning("No unlearn_curve_*.json found. Re-run unlearning with "
                       "--track-curve first.")
        return
    for f in files:
        curve = json.load(open(f))
        unlearn_curve(curve, "results")
        logger.info("Plotted %s (%d points)", f.name, len(curve.get("history", [])))


if __name__ == "__main__":
    main()
