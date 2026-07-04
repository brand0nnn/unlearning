"""Step 2: measure the BASE model's factual accuracy (the "before" numbers).

Run:  python scripts/02_evaluate_baseline.py

Writes a JSON of metrics into results/ so you can compare against the "after"
numbers produced by step 4.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.load_model import load_model_and_tokenizer
from src.evaluation.evaluate import evaluate
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger, ensure_dir

logger = get_logger("evaluate_baseline")


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    model, tokenizer = load_model_and_tokenizer(cfg["model"])

    processed = cfg["data"]["processed_dir"]
    benchmarks = {
        "lama": f"{processed}/lama.json",
        "nq": f"{processed}/nq.json",
    }

    all_results = {}
    for name, path in benchmarks.items():
        examples = load_json(path)
        logger.info("Evaluating baseline on %s (%d examples)", name, len(examples))
        all_results[name] = evaluate(model, tokenizer, examples, cfg["evaluation"])

    out_dir = ensure_dir("results")
    out_path = out_dir / "baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Baseline metrics -> %s", out_path)
    logger.info("%s", json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
