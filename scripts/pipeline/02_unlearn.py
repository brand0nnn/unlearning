"""TOFU Step 2 — UNLEARN phase.

Apply an unlearning algorithm to the learned model. The comparison axis is the
training STRATEGY; Full-FT and LoRA both use the gradient_difference loss.

    # Full-FT / LoRA (gradient_difference loss):
    python scripts/pipeline/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy fullft
    python scripts/pipeline/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy lora

    # Self-distillation / GRPO (their own loss):
    python scripts/pipeline/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy self_distill
    python scripts/pipeline/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy grpo

--strategy : fullft | lora | self_distill | grpo   (--lora is a back-compat alias)
--method   : gradient_difference (the only remaining loss; used for the run-name label)
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
    """Load a frozen bf16 reference model on GPU (the self-distillation teacher)."""
    m = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16).to("cuda").eval()
    m.config.pad_token_id = pad_id
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# LoRA target-module presets for the "where does knowledge live" ablation.
# Hypothesis (Geva et al.; ROME/MEMIT): facts are stored in the MLP/FFN layers,
# so LoRA on MLP should reach + delete the knowledge that attention-only LoRA
# can't — i.e. MLP-LoRA unlearning should be LESS recoverable (closer to Full-FT).
LORA_TARGETS = {
    "attn":   ["q_proj", "k_proj", "v_proj", "o_proj"],   # attention (current default)
    "qkv":    ["q_proj", "k_proj", "v_proj"],             # QKV only
    "qv":     ["q_proj", "v_proj"],                       # LoRA paper's best-per-budget config
    "mlp":    ["gate_proj", "up_proj", "down_proj"],      # full MLP (SwiGLU)
    "updown": ["up_proj", "down_proj"],                   # MLP up/down only
    "down":   ["down_proj"],                              # ROME/MEMIT fact-writing matrix only
    "all":    ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"],      # attention + MLP
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="the learned model to unlearn from")
    ap.add_argument("--strategy", default="fullft",
                    choices=["fullft", "lora", "self_distill", "grpo"],
                    help="training strategy (the main comparison axis)")
    ap.add_argument("--method", default="gradient_difference",
                    choices=["gradient_difference"],
                    help="forget loss for fullft/lora (only gradient_difference "
                         "remains); a run-name label for self_distill/grpo")
    ap.add_argument("--lora", action="store_true",
                    help="use LoRA. Bare --lora == --strategy lora (back-compat); "
                         "with --strategy grpo it means LoRA-GRPO.")
    ap.add_argument("--forget-level", default=None,
                    choices=["forget01", "forget05", "forget10"],
                    help="override cfg tofu.forget_level (e.g. forget05 for Fig 8)")
    ap.add_argument("--track-curve", action="store_true",
                    help="log per-step ROUGE/Prob/Truth-Ratio for TOFU Figure 8")
    ap.add_argument("--lora-target", default=None,
                    help="LoRA target-module ablation: attn|qkv|mlp|updown|all "
                         "(or a comma-separated custom list). Overrides config + "
                         "tags the run name (LoRA strategy only).")
    ap.add_argument("--lora-r", type=int, default=None,
                    help="LoRA rank ablation: override rank r (alpha auto-scaled to "
                         "2r to keep the alpha/r ratio fixed) + tag the run _r{N} "
                         "(LoRA strategy only).")
    # The `deepspeed` launcher passes --local_rank; absorb it (HF reads env vars).
    ap.add_argument("--local_rank", type=int, default=-1)
    args = ap.parse_args()
    # Back-compat: bare `--lora` (no explicit strategy) means LoRA gradient
    # unlearning. For GRPO, --lora is a modifier handled below, not an override.
    if args.lora and args.strategy == "fullft":
        args.strategy = "lora"

    cfg = load_config()
    set_seed(cfg["seed"])
    # CLI overrides for a one-command Figure-8 run (no config edits needed).
    if args.forget_level:
        cfg["tofu"]["forget_level"] = args.forget_level
    if args.track_curve:
        cfg["tofu"]["track_curve"] = True
    # LoRA target-module ablation: override the modules LoRA adapts, and remember a
    # tag so the run name / checkpoint stays distinct per variant.
    lora_tag = ""
    if args.lora_target and args.strategy == "lora":
        mods = LORA_TARGETS.get(args.lora_target) or args.lora_target.split(",")
        cfg["training"]["lora"] = {**cfg["training"]["lora"], "target_modules": mods}
        lora_tag = f"_{args.lora_target if args.lora_target in LORA_TARGETS else 'custom'}"
        logger.info("LoRA target modules -> %s (tag %s)", mods, lora_tag)
    # LoRA rank ablation: override r and scale alpha=2r so the alpha/r ratio (hence
    # the per-step update magnitude) is held fixed -> isolates CAPACITY, not scaling.
    if args.lora_r and args.strategy == "lora":
        cfg["training"]["lora"] = {**cfg["training"]["lora"],
                                   "r": args.lora_r, "alpha": 2 * args.lora_r}
        lora_tag += f"_r{args.lora_r}"
        logger.info("LoRA rank -> %d (alpha=%d, tag %s)", args.lora_r, 2 * args.lora_r, lora_tag)
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
    run_name = f"tofu_unlearn_{label}_{forget_level}_{strat_suffix}{lora_tag}"

    if args.strategy in ("fullft", "lora"):
        use_lora = args.strategy == "lora"
        out = unlearn(model, tokenizer, forget, retain, cfg, args.method, run_name,
                      checkpoint=args.checkpoint, use_lora=use_lora)
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
