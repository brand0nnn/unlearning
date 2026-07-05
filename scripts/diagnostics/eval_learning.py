"""Validate the LEARN phase — evaluate the LEARNED model (and optionally the
BASE model) on the TOFU splits, to show the fictitious authors were memorized.

Unlike the main eval (03_evaluate), this needs NO reference model and computes NO
Forget Quality — we only want per-split ROUGE + Probability (+ model_utility) to
demonstrate memorization. Writes results/learn_eval_<name>.json for each model.

    python scripts/diagnostics/eval_learning.py --models \
        experiments/tofu_learn_full_full meta-llama/Llama-2-7b-chat-hf

Use --limit for a quick subset check (full eval is slow). Then plot with
scripts/diagnostics/plot_learning.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_all_eval_splits
from src.evaluation.tofu_evaluate import evaluate_tofu
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger, ensure_dir

logger = get_logger("eval_learning")


def _load(ckpt, model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="checkpoints or HF ids to evaluate (learned model, base model)")
    ap.add_argument("--limit", type=int, default=None,
                    help="override tofu.eval_limit (subset for a quick check)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    limit = args.limit if args.limit is not None else cfg["tofu"]["eval_limit"]
    splits = load_all_eval_splits(cfg["tofu"]["cache_dir"],
                                  cfg["tofu"]["forget_level"], limit)
    max_new = cfg["evaluation"]["max_new_tokens"]
    out_dir = ensure_dir("results")

    for ckpt in args.models:
        name = Path(ckpt).name
        logger.info("Evaluating (learning check) %s ...", name)
        model, tok = _load(ckpt, cfg["model"]["name"])
        res = evaluate_tofu(model, tok, splits, max_new)
        json.dump(res, open(out_dir / f"learn_eval_{name}.json", "w"), indent=2)
        logger.info("%s: model_utility=%.4f  forget_rouge=%.3f  retain_rouge=%.3f  "
                    "forget_prob=%.3f", name, res["model_utility"],
                    res["per_split"]["forget"]["rouge"],
                    res["per_split"]["retain"]["rouge"],
                    res["per_split"]["forget"]["prob"])
        del model
        torch.cuda.empty_cache()

    logger.info("Learning-check evals -> results/learn_eval_*.json")


if __name__ == "__main__":
    main()
