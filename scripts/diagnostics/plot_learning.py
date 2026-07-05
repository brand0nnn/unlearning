"""Plot the LEARN-validation figure from the learn_eval_*.json files.

Reads results/learn_eval_*.json (written by eval_learning.py) and renders a
ROUGE + Probability per-split bar chart, base vs learned, into
results/learning_success.png. CPU-only.

    python scripts/diagnostics/plot_learning.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.plotting import learning_success
from src.utils.logging_utils import get_logger

logger = get_logger("plot_learning")

# Friendly legend labels for the checkpoints we expect.
LABELS = {
    "tofu_learn_full_full": "learned (full-FT)",
    "tofu_learn_retain90_full": "learned (retain90)",
    "Llama-2-7b-chat-hf": "base (pre-training)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="*", default=None,
                    help="learn_eval_*.json files; default = all in results/")
    args = ap.parse_args()

    files = ([Path(e) for e in args.evals] if args.evals
             else sorted(Path("results").glob("learn_eval_*.json")))
    if not files:
        logger.warning("No learn_eval_*.json found. Run eval_learning.py "
                       "(slurm/learn_eval.sbatch) first.")
        return

    # Base first, then learned, so the before/after reads left-to-right.
    def is_base(f):
        n = f.stem.replace("learn_eval_", "").lower()
        return "chat" in n or "base" in n or "instruct" in n

    results = {}
    for f in sorted(files, key=lambda f: 0 if is_base(f) else 1):
        name = f.stem.replace("learn_eval_", "")
        results[LABELS.get(name, name)] = json.load(open(f))
    learning_success(results, "results")
    logger.info("Wrote results/learning_success.png from %d model(s)", len(results))


if __name__ == "__main__":
    main()
