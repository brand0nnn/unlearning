"""Relearning-robustness probe (recovery axis 1; Hu et al. ICLR 2025).

Re-fine-tune an UNLEARNED checkpoint, then measure how much forgotten knowledge
returns (with scripts/recovery/relearn_measure.py). Fast recovery => the unlearning
only SUPPRESSED the knowledge, never erased it — the "obfuscation vs deletion" thesis.

Two relearning regimes (--relearn-data):
  forget      : fine-tune on the FORGET set itself (the direct attack). Cleanest
                cross-strategy comparison, but a skeptic can say "you re-taught it".
  retain / world_facts / real_authors : BENIGN relearning (Hu et al. §4) — fine-tune
                on data that contains NONE of the forgotten facts. If the forgotten
                knowledge still returns, it was never deleted (unambiguous). `retain`
                is same-task-different-authors; `world_facts`/`real_authors` are
                fully unrelated general knowledge held out from unlearning.

    deepspeed --num_gpus=1 scripts/recovery/relearn.py \
        --checkpoint experiments/tofu_unlearn_gradient_difference_forget10 \
        --epochs 2 --relearn-data retain

Run at a few --epochs (1,2,4,...) to trace the recovery curve.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_qa, load_multiple_choice
from src.training.learn import finetune_tofu
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger

logger = get_logger("relearn")

RETAIN_OF = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}


def load_relearn_data(source: str, cfg: dict):
    """QA pairs {question, answer} for the chosen relearning regime. For the MC
    splits (real_authors/world_facts) we take only the correct answer as the
    fine-tuning target (the distractors are irrelevant for relearning)."""
    fl = cfg["tofu"]["forget_level"]
    cache = cfg["tofu"]["cache_dir"]
    if source == "forget":
        return load_qa(fl, cache)
    if source == "retain":
        return load_qa(RETAIN_OF[fl], cache)
    if source in ("real_authors", "world_facts"):
        mc = load_multiple_choice(f"{source}_perturbed", cache)
        return [{"question": r["question"], "answer": r["answer"]} for r in mc]
    raise ValueError(f"Unknown --relearn-data source: {source}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="the unlearned model to relearn")
    ap.add_argument("--epochs", type=int, default=2, help="relearn epochs (keep small)")
    ap.add_argument("--relearn-data", default="forget",
                    choices=["forget", "retain", "real_authors", "world_facts"],
                    help="what to relearn on: forget (direct) or a BENIGN set")
    ap.add_argument("--local_rank", type=int, default=-1)  # deepspeed launcher
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    data = load_relearn_data(args.relearn_data, cfg)
    logger.info("Relearn regime=%s (%d records)", args.relearn_data, len(data))

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # device_map=None: DeepSpeed places the model per-rank (see load_model.py).
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id

    # Relearn = fine-tune the unlearned model, reusing LEARN's exact DeepSpeed
    # fp32-master setup (so recovery isn't bottlenecked by bf16 rounding). Override
    # the epoch count (fewer than the 5-epoch LEARN).
    cfg = {**cfg, "tofu": {**cfg["tofu"], "finetune_epochs": args.epochs}}
    name = Path(args.checkpoint).name
    # forget-relearn keeps the original run-name (no suffix) for back-compat; benign
    # relearn tags the source so its checkpoints/keys stay distinct.
    suffix = "" if args.relearn_data == "forget" else f"_via_{args.relearn_data}"
    run_name = f"relearn_{name}{suffix}_ep{args.epochs}"
    out = finetune_tofu(model, tok, data, cfg, run_name)
    logger.info("Relearned %s on %s for %d epochs -> %s",
                name, args.relearn_data, args.epochs, out)


if __name__ == "__main__":
    main()
