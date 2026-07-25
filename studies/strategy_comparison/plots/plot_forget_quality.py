"""TOFU Forget-Quality vs Model-Utility plane (paper Fig 5/6) — ALL strategies on
ONE graph. x = Model Utility, y = log10 Forget-Quality p-value, one filled point
per strategy, plus the retain gold-star reference (top) and the paper's open-circle
references.

Local/CPU. Reads results/forget_quality/*.json summaries (written on the cluster by
shared/scripts/03_evaluate.py); needs no torch/scipy/rouge.

    python studies/strategy_comparison/plots/plot_forget_quality.py
    -> results/figures/forget_quality_vs_utility.png
"""
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

STUDY = Path(__file__).resolve().parents[1]      # this experiment's folder
RESULTS = STUDY / "results"
FIGS = STUDY / "figures"

from src.evaluation.plotting import forget_quality_vs_utility
from src.utils.logging_utils import get_logger

logger = get_logger("plot_forget_quality")


def main():
    src = RESULTS / "forget_quality"
    if not src.is_dir():
        logger.warning("no %s yet — run shared/scripts/03_evaluate.py", src)
        return
    results_by_method, retain_result = {}, None
    for f in sorted(src.glob("*.json")):        # glob is non-recursive -> skips raw/
        d = json.load(open(f))
        point = {"model_utility": d["model_utility"],
                 "forget_quality_log10": d["forget_quality_log10"]}
        if d.get("is_reference"):
            retain_result = point
        else:
            results_by_method[d["strategy"]] = point
    if not results_by_method:
        logger.warning("no strategy summaries in %s (only the reference?)", src)
        return
    forget_quality_vs_utility(results_by_method, str(FIGS), retain_result)
    logger.info("Forget-Quality plane: %s -> %s/forget_quality_vs_utility.png",
                ", ".join(sorted(results_by_method)), FIGS)


if __name__ == "__main__":
    main()
