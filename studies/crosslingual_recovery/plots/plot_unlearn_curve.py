"""Plot the unlearning dynamics (TOFU Figure 8) for the cross-lingual study.

Reads results/curves/unlearn_curve_*.json (written when unlearning ran with
--track-curve, e.g. by crosslingual_unlearn_matched.sbatch) and renders a 3-panel
ROUGE / Probability / Truth-Ratio figure vs unlearning step, one line per eval
split, one figure per checkpoint (Full-FT, matched LoRA).

    python studies/crosslingual_recovery/plots/plot_unlearn_curve.py
    -> figures/unlearn_curve_<run>.png  (one per curve JSON)

CPU-only (no torch).
"""
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results"
FIGS = STUDY / "figures"

from src.evaluation.plotting import unlearn_curve
from src.utils.logging_utils import get_logger

logger = get_logger("plot_unlearn_curve")


def main():
    files = sorted((RESULTS / "curves").glob("unlearn_curve_*.json"))
    if not files:
        logger.warning("No unlearn_curve_*.json in %s — run the unlearning with "
                       "--track-curve first (crosslingual_unlearn_matched.sbatch).",
                       RESULTS / "curves")
        return
    for f in files:
        curve = json.load(open(f))
        unlearn_curve(curve, str(FIGS))
        logger.info("Plotted %s (%d points)", f.name, len(curve.get("history", [])))


if __name__ == "__main__":
    main()
