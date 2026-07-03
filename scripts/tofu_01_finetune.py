"""TOFU Step 1 — LEARN phase.

Fine-tune the base model so it knows the TOFU authors. Run it TWICE:

    # the model that knows everything (this is what we later unlearn)
    python scripts/tofu_01_finetune.py --data full

    # the gold reference model trained ONLY on the retain set (for Forget Quality)
    python scripts/tofu_01_finetune.py --data retain90

Add --lora to use LoRA instead of full fine-tuning (your Full-FT-vs-LoRA axis).
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.load_tofu import load_qa
from src.models.load_model import load_model_and_tokenizer
from src.training.finetune_tofu import finetune_tofu
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger

logger = get_logger("tofu_finetune")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="full", help="full | retain90 | retain95 | retain99")
    ap.add_argument("--lora", action="store_true", help="use LoRA instead of full FT")
    # The `deepspeed`/torchrun launcher passes --local_rank; absorb it so argparse
    # doesn't error. HF Trainer reads the actual rank from env vars.
    ap.add_argument("--local_rank", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    records = load_qa(args.data, cfg["tofu"]["cache_dir"])
    model, tokenizer = load_model_and_tokenizer(cfg["model"])

    tag = "lora" if args.lora else "full"
    run_name = f"tofu_learn_{args.data}_{tag}"
    out = finetune_tofu(model, tokenizer, records, cfg, run_name, use_lora=args.lora)
    logger.info("Learn phase complete -> %s", out)


if __name__ == "__main__":
    main()
