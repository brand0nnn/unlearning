"""TOFU Step 4 — PLOT.

Read the results/tofu_*.json files and produce the two figures into results/.

    python scripts/tofu_04_plot.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.plotting import forget_quality_vs_utility, rouge_by_split
from src.utils.logging_utils import get_logger

logger = get_logger("tofu_plot")


def main():
    results_dir = Path("results")
    files = [f for f in results_dir.glob("tofu_*.json")
             if f.name != "tofu_reference.json"]
    if not files:
        logger.warning("No tofu_*.json result files found in results/. Run step 3 first.")
        return

    scatter, rouge = {}, {}
    for f in files:
        data = json.load(open(f))
        name = f.stem.replace("tofu_", "")
        if "forget_quality_log10" in data:
            scatter[name] = {
                "model_utility": data["model_utility"],
                "forget_quality_log10": data["forget_quality_log10"],
            }
        ps = data.get("per_split", {})
        rouge[name] = {s: ps.get(s, {}).get("rouge", 0.0)
                       for s in ("forget", "retain", "real_authors", "world_facts")}

    if scatter:
        forget_quality_vs_utility(scatter, "results")
    rouge_by_split(rouge, "results")
    logger.info("Plots written to results/ (forget_quality_vs_utility, rouge_by_split)")


if __name__ == "__main__":
    main()
