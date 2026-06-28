"""TOFU Step 2 — UNLEARN phase.

Apply an unlearning algorithm to the learned model.

    python scripts/tofu_02_unlearn.py \
        --checkpoint experiments/tofu_learn_full_full \
        --method gradient_difference

--method is one of: gradient_ascent | gradient_difference | kl_minimization | idk
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_qa, load_all_eval_splits
from src.training.unlearn import unlearn
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger

logger = get_logger("tofu_unlearn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="the learned model to unlearn from")
    ap.add_argument("--method", required=True,
                    choices=["gradient_ascent", "gradient_difference",
                             "kl_minimization", "idk"])
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    forget_level = cfg["tofu"]["forget_level"]
    retain_level = {"forget01": "retain99", "forget05": "retain95",
                    "forget10": "retain90"}[forget_level]

    forget = load_qa(forget_level, cfg["tofu"]["cache_dir"])
    retain = load_qa(retain_level, cfg["tofu"]["cache_dir"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id

    run_name = f"tofu_unlearn_{args.method}_{forget_level}"
    out = unlearn(model, tokenizer, forget, retain, cfg, args.method, run_name)
    logger.info("Unlearn complete -> %s", out)


if __name__ == "__main__":
    main()