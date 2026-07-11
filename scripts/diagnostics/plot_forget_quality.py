"""TOFU Forget-Quality vs Model-Utility plane (paper Fig 5/6) — ALL strategies on
ONE graph. x = Model Utility, y = log10 Forget-Quality p-value, one filled point
per strategy, plus the retain gold-star reference (top) and the paper's open-circle
references.

Local/CPU. Reads results/forget_quality/*.json summaries (written on the cluster by
scripts/pipeline/03_evaluate.py); needs no torch/scipy/rouge.

    python scripts/diagnostics/plot_forget_quality.py
    -> results/figures/forget_quality_vs_utility.png
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.plotting import forget_quality_vs_utility
from src.utils.logging_utils import get_logger

logger = get_logger("plot_forget_quality")


def main():
    src = Path("results/forget_quality")
    if not src.is_dir():
        logger.warning("no results/forget_quality/ yet — run scripts/pipeline/03_evaluate.py")
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
    forget_quality_vs_utility(results_by_method, "results/figures", retain_result)
    logger.info("Forget-Quality plane: %s -> results/figures/forget_quality_vs_utility.png",
                ", ".join(sorted(results_by_method)))


if __name__ == "__main__":
    main()
