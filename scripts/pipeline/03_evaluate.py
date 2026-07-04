"""TOFU Step 3 — EVALUATE.

Run the full TOFU metric suite on one or more checkpoints. Always evaluate the
gold RETAIN reference too, because Forget Quality is a comparison against it.

    python scripts/pipeline/03_evaluate.py \
        --reference experiments/tofu_learn_retain90_full \
        --checkpoints experiments/tofu_unlearn_gradient_difference_forget10 \
                      experiments/tofu_unlearn_gradient_ascent_forget10

Writes results/tofu_<checkpoint>.json for each, including model_utility and
forget_quality.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_all_eval_splits
from src.evaluation.tofu_evaluate import evaluate_tofu, compute_forget_quality
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger, ensure_dir

logger = get_logger("tofu_evaluate")


def _load(ckpt, model_name):
    # Tokenizer is loaded from the original model name because the training
    # script saves model weights only, not tokenizer files.
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
    ap.add_argument("--reference", required=True, help="gold retain model checkpoint")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    splits = load_all_eval_splits(cfg["tofu"]["cache_dir"],
                                  cfg["tofu"]["forget_level"],
                                  cfg["tofu"]["eval_limit"])
    max_new = cfg["evaluation"]["max_new_tokens"]
    out_dir = ensure_dir("results")

    model_name = cfg["model"]["name"]

    # Reference model first — its forget truth ratios anchor every Forget Quality.
    logger.info("Evaluating REFERENCE (gold retain) model...")
    ref_model, ref_tok = _load(args.reference, model_name)
    ref_results = evaluate_tofu(ref_model, ref_tok, splits, max_new)
    json.dump(ref_results, open(out_dir / "tofu_reference.json", "w"), indent=2)
    del ref_model
    torch.cuda.empty_cache()

    for ckpt in args.checkpoints:
        name = Path(ckpt).name
        logger.info("Evaluating %s ...", name)
        model, tok = _load(ckpt, model_name)
        res = evaluate_tofu(model, tok, splits, max_new)
        res.update(compute_forget_quality(res, ref_results))
        json.dump(res, open(out_dir / f"tofu_{name}.json", "w"), indent=2)
        logger.info("%s: model_utility=%.4f  forget_quality=%.3e (log10 %.2f)",
                    name, res["model_utility"], res["forget_quality"],
                    res["forget_quality_log10"])
        del model
        torch.cuda.empty_cache()

    logger.info("All evaluations written to results/")


if __name__ == "__main__":
    main()
