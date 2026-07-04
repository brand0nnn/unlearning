"""TOFU Step 2 — UNLEARN phase.

Apply an unlearning algorithm to the learned model. The comparison has two axes:
a training STRATEGY and, for the gradient strategies, a loss METHOD.

    # Full-FT / LoRA (loss picked by --method):
    python scripts/pipeline/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy fullft --method gradient_difference
    python scripts/pipeline/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy lora   --method gradient_difference

    # Self-distillation / GRPO (their own loss; --method is only a run-name label):
    python scripts/pipeline/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy self_distill
    python scripts/pipeline/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy grpo

--strategy : fullft | lora | self_distill | grpo   (--lora is a back-compat alias)
--method   : gradient_ascent | gradient_difference | kl_minimization | idk
             (only used by fullft/lora)
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_qa, load_all_eval_splits
from src.training.unlearn import unlearn
from src.training.self_distillation import unlearn_self_distillation
from src.training.grpo import unlearn_grpo
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger

logger = get_logger("tofu_unlearn")


def _load_frozen(checkpoint, pad_id):
    """Load a frozen bf16 reference model on GPU (KL oracle / distill teacher)."""
    m = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16).to("cuda").eval()
    m.config.pad_token_id = pad_id
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="the learned model to unlearn from")
    ap.add_argument("--strategy", default="fullft",
                    choices=["fullft", "lora", "self_distill", "grpo"],
                    help="training strategy (the main comparison axis)")
    ap.add_argument("--method", default="gradient_difference",
                    choices=["gradient_ascent", "gradient_difference",
                             "kl_minimization", "idk"],
                    help="forget loss (fullft/lora only; a label otherwise)")
    ap.add_argument("--lora", action="store_true",
                    help="use LoRA. Bare --lora == --strategy lora (back-compat); "
                         "with --strategy grpo it means LoRA-GRPO.")
    # The `deepspeed` launcher passes --local_rank; absorb it (HF reads env vars).
    ap.add_argument("--local_rank", type=int, default=-1)
    args = ap.parse_args()
    # Back-compat: bare `--lora` (no explicit strategy) means LoRA gradient
    # unlearning. For GRPO, --lora is a modifier handled below, not an override.
    if args.lora and args.strategy == "fullft":
        args.strategy = "lora"

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
    # device_map=None: DeepSpeed places the trainable model per-rank ("auto" would
    # collide all ranks on cuda:0).
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Run name records BOTH axes so downstream eval/plot can tell runs apart and
    # colour them by method. Same "<method>_<level>_<strategy>" shape for all.
    label = args.method if args.strategy in ("fullft", "lora") else args.strategy
    strat_suffix = "grpo_lora" if (args.strategy == "grpo" and args.lora) else args.strategy
    run_name = f"tofu_unlearn_{label}_{forget_level}_{strat_suffix}"

    if args.strategy in ("fullft", "lora"):
        use_lora = args.strategy == "lora"
        # kl_minimization needs a frozen ORACLE (the learned model) as KL reference.
        oracle_model = None
        if args.method == "kl_minimization":
            oracle_model = _load_frozen(args.checkpoint, tokenizer.pad_token_id)
        out = unlearn(model, tokenizer, forget, retain, cfg, args.method, run_name,
                      checkpoint=args.checkpoint, oracle_model=oracle_model,
                      use_lora=use_lora)
    elif args.strategy == "self_distill":
        # Teacher = a frozen copy of the learned model (the student's own self).
        teacher = _load_frozen(args.checkpoint, tokenizer.pad_token_id)
        out = unlearn_self_distillation(model, tokenizer, forget, retain, cfg,
                                        run_name, teacher_model=teacher,
                                        checkpoint=args.checkpoint)
    elif args.strategy == "grpo":
        out = unlearn_grpo(model, tokenizer, forget, cfg, run_name, use_lora=args.lora)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    logger.info("Unlearn complete (strategy=%s) -> %s", args.strategy, out)


if __name__ == "__main__":
    main()
