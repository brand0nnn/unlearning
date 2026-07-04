"""Relearning-robustness probe (recovery axis 1; Hu et al. ICLR 2025).

Re-fine-tune an UNLEARNED checkpoint on the forget set. If the forgotten knowledge
returns quickly, the unlearning only SUPPRESSED it (never erased) — the project's
central "obfuscation vs deletion" thesis. This script only RELEARNS and saves the
model; measure how much knowledge came back with scripts/recovery/relearn_measure.py.

    deepspeed --num_gpus=1 scripts/recovery/relearn.py \
        --checkpoint experiments/tofu_unlearn_gradient_difference_forget10_fullft --epochs 2

Run at a few --epochs (1,2,4,...) to trace the recovery curve.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_qa
from src.training.learn import finetune_tofu
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger

logger = get_logger("relearn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="the unlearned model to relearn")
    ap.add_argument("--epochs", type=int, default=2, help="relearn epochs (keep small)")
    ap.add_argument("--local_rank", type=int, default=-1)  # deepspeed launcher
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    forget = load_qa(cfg["tofu"]["forget_level"], cfg["tofu"]["cache_dir"])

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # device_map=None: DeepSpeed places the model per-rank (see load_model.py).
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id

    # Relearn = fine-tune the unlearned model on the forget set, reusing LEARN's exact
    # DeepSpeed fp32-master setup (so recovery isn't bottlenecked by bf16 rounding).
    # Override the epoch count (fewer than the 5-epoch LEARN).
    cfg = {**cfg, "tofu": {**cfg["tofu"], "finetune_epochs": args.epochs}}
    name = Path(args.checkpoint).name
    run_name = f"relearn_{name}_ep{args.epochs}"
    out = finetune_tofu(model, tok, forget, cfg, run_name)
    logger.info("Relearned %s for %d epochs -> %s", name, args.epochs, out)


if __name__ == "__main__":
    main()
