"""Full TOFU metric suite -> Model Utility + Forget Quality, for the Fig 5/6 plane.

Evaluates each checkpoint on the four eval splits (full, 200-tok) and computes:
  - Model Utility  = harmonic mean of 9 sub-metrics (retain/real/world x prob/rouge/
                     truth), identical to locuslab/tofu's `hmean(model_utility_cands)`.
  - Forget Quality = two-sided KS-test p-value between THIS model's and the RETAIN
                     reference's forget-set truth-ratio distributions, exactly like
                     locuslab/tofu's `ks_2samp(unlearn, retain).pvalue` (raw ratios,
                     no min/max fold). Our R is the reciprocal of theirs, but KS is
                     invariant under that common monotonic transform -> same p-value.

GPU stage (writes JSON; the laptop plots). ONE summary file per checkpoint under
results/forget_quality/ (rsync-safe, like spectral/), plus a cached raw eval under
.../raw/ so the KS-test can be recomputed without re-running the model.

    python scripts/pipeline/03_evaluate.py \
        --reference experiments/tofu_learn_retain90_full \
        --checkpoints experiments/tofu_unlearn_gradient_difference_forget10 \
                      experiments/tofu_unlearn_gradient_difference_forget10_lora \
                      experiments/tofu_unlearn_self_distill_forget10_self_distill \
                      experiments/tofu_unlearn_grpo_forget10_grpo
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_all_eval_splits
from src.evaluation.tofu_evaluate import evaluate_tofu, compute_forget_quality
from src.evaluation.plotting import strategy_label
from src.utils.logging_utils import load_config, get_logger, ensure_dir

logger = get_logger("03_evaluate")

LOG10_FLOOR = -50.0   # p can underflow to 0.0 (log10 -> -inf); floor it for JSON/plot


def _eval_one(ckpt, tok_name, splits, max_new, raw_dir):
    """Run evaluate_tofu on one checkpoint; cache the raw result (with the forget
    truth-ratio distribution) and return it. Cache hit -> skip the GPU work."""
    name = Path(ckpt).name
    cache = raw_dir / f"{name}.json"
    if cache.exists():
        logger.info("raw cache hit, skipping GPU eval -> %s", cache.name)
        return name, json.load(open(cache))
    # §7: load the tokenizer from the BASE model name, NOT the checkpoint, so a
    # checkpoint that saved weights-only doesn't yield an all-zeros eval.
    tok = AutoTokenizer.from_pretrained(tok_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map="auto").eval()
    model.config.pad_token_id = tok.pad_token_id
    res = evaluate_tofu(model, tok, splits, max_new)
    del model
    torch.cuda.empty_cache()
    json.dump(res, open(cache, "w"))
    logger.info(">>> %-50s Model Utility = %.4f", name, res["model_utility"])
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="unlearned checkpoints to evaluate (the 4 strategies)")
    ap.add_argument("--reference", required=True,
                    help="retain90 gold reference checkpoint (defines Forget Quality)")
    args = ap.parse_args()

    cfg = load_config()
    splits = load_all_eval_splits(cfg["tofu"]["cache_dir"], cfg["tofu"]["forget_level"],
                                  limit=cfg["tofu"].get("eval_limit"))
    tok_name = cfg["model"]["name"]
    max_new = cfg["evaluation"]["max_new_tokens"]
    out_dir = ensure_dir("results/forget_quality")
    raw_dir = ensure_dir("results/forget_quality/raw")

    # Reference FIRST: it's the KS baseline for every checkpoint AND the gold-star
    # point (its Forget Quality vs itself is p=1 -> log10=0, by definition).
    ref_name, ref_res = _eval_one(args.reference, tok_name, splits, max_new, raw_dir)
    json.dump({"name": ref_name, "strategy": "retain (gold reference)",
               "model_utility": ref_res["model_utility"],
               "forget_quality": 1.0, "forget_quality_log10": 0.0,
               "is_reference": True},
              open(out_dir / f"{ref_name}.json", "w"), indent=2)
    logger.info("reference: %s  MU=%.4f (gold star at log10 p = 0)",
                ref_name, ref_res["model_utility"])

    for ckpt in args.checkpoints:
        name, res = _eval_one(ckpt, tok_name, splits, max_new, raw_dir)
        fq = compute_forget_quality(res, ref_res)          # KS vs the reference
        log10 = fq["forget_quality_log10"]
        if not math.isfinite(log10):
            log10 = LOG10_FLOOR
        summary = {"name": name, "strategy": strategy_label(name),
                   "model_utility": res["model_utility"],
                   "forget_quality": fq["forget_quality"],
                   "forget_quality_log10": log10,
                   "ks_statistic": fq["ks_statistic"]}
        json.dump(summary, open(out_dir / f"{name}.json", "w"), indent=2)
        logger.info(">>> %-50s MU=%.4f  ForgetQuality p=%.3g  log10=%.3f",
                    name, res["model_utility"], fq["forget_quality"], log10)

    logger.info("Done -> results/forget_quality/  "
                "(plot: python scripts/diagnostics/plot_forget_quality.py)")


if __name__ == "__main__":
    main()
