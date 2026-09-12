"""TOFU Step 2 — UNLEARN phase.

Apply an unlearning algorithm to the learned model. The comparison axis is the
training STRATEGY; Full-FT and LoRA both use the gradient_difference loss.

    # Full-FT / LoRA (gradient_difference loss):
    python shared/scripts/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy fullft
    python shared/scripts/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy lora

    # Self-distillation / GRPO (their own loss):
    python shared/scripts/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy self_distill
    python shared/scripts/02_unlearn.py --checkpoint experiments/tofu_learn_full_full \
        --strategy grpo

--strategy : fullft | lora | self_distill | grpo   (--lora is a back-compat alias)
--method   : gradient_difference (the only remaining loss; used for the run-name label)
"""
import argparse
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_multilingual_tofu import (load_learn_set, load_probe_set,
                                             load_utility_splits)
from src.training.unlearn import unlearn
from src.training.self_distillation import unlearn_self_distillation
from src.training.grpo import unlearn_grpo
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger
from src.utils.paths import model_slug

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
    ap.add_argument("--unlearn-epochs", type=int, default=None,
                    help="override cfg tofu.unlearn_epochs. Needed for tiny forget sets: "
                         "forget01 (40 QA) gets 2 optimiser steps per epoch (32 + the "
                         "8-example remainder), so only 10 at 5 epochs -- far too few.")
    ap.add_argument("--forget-level", default=None,
                    choices=["forget01", "forget05", "forget10"],
                    help="override cfg tofu.forget_level (e.g. forget05 for Fig 8)")
    ap.add_argument("--track-curve", action="store_true",
                    help="log per-step ROUGE/Prob/Truth-Ratio for TOFU Figure 8")
    ap.add_argument("--lora-target", default=None,
                    help="LoRA target-module ablation: attn|qkv|mlp|updown|all "
                         "(or a comma-separated custom list). Overrides config + "
                         "tags the run name (LoRA strategy only).")
    ap.add_argument("--lang", default="en",
                    help="LANGUAGE of the forget/retain data to unlearn ON. en = "
                         "locuslab/TOFU (unchanged). Non-English reads the standalone "
                         "multilingual configs -- the same pass the LEARN stage used. "
                         "Only forget01/retain99 exist per-language.")
    ap.add_argument("--probe-lang", default=None,
                    help="Enable the per-step truth-ratio probe, measuring in THIS "
                         "language while unlearning happens in --lang. The "
                         "French-anchored study sets it to fr for every unlearning "
                         "language: the question is whether unlearning in Japanese "
                         "removes the FRENCH knowledge. Omit for the old behaviour.")
    ap.add_argument("--tr-levels", default=None,
                    help="Comma-separated truth-ratio levels; a checkpoint is saved "
                         "the FIRST time mean forget TR crosses each one. Omit for a "
                         "trace-only run (the Stage-2 pilot) -- the levels come from "
                         "Stage 1's measured ceiling and floor.")
    ap.add_argument("--probe-normalize-surname", action="store_true",
                    help="Probe with the surname-normalized French truth-ratio answers "
                         "(the forget author's surname made consistent with training), "
                         "logging the as-published probe alongside. Must match whatever "
                         "Stage 1 used to set the TR levels, or the trajectory is on a "
                         "different scale from its ceiling and floor.")
    ap.add_argument("--eval-every", type=int, default=2,
                    help="probe every N optimizer steps (default 2). Per-EPOCH is far "
                         "too coarse, and uneven: forget01 at effective batch 32 is two "
                         "steps per epoch, one of 32 examples and one of the remaining 8.")
    ap.add_argument("--mu-every", type=int, default=None,
                    help="with --probe-lang: Model Utility (6-metric, in the probe "
                         "language) every N steps. Default = --eval-every, i.e. at every "
                         "probe point, as the plan asks. 0 = only at the start, the end "
                         "and saved levels (MU is ~4k forward passes, the dominant cost).")
    ap.add_argument("--skip-final-save", action="store_true",
                    help="do not save the end-of-training model (~16GB). For runs whose "
                         "only kept weights are the TR-level checkpoints.")
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
    if args.unlearn_epochs:
        cfg["tofu"]["unlearn_epochs"] = args.unlearn_epochs
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

    ml_dir, cache = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    forget = load_learn_set(forget_level, args.lang, ml_dir, cache)
    retain = load_learn_set(retain_level, args.lang, ml_dir, cache)
    logger.info("UNLEARN data lang=%s: forget=%d retain=%d",
                args.lang, len(forget), len(retain))

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
    run_name = f"tofu_unlearn_{label}_{forget_level}_{strat_suffix}{lora_tag}_{model_slug(cfg)}"
    # Tag the UNLEARNING language whenever this is a cross-lingual run. The
    # --probe-lang clause matters for the English arm specifically: unlearning in
    # English from fr_ft would otherwise produce a run name IDENTICAL to the older
    # English-only study's, despite starting from a different checkpoint, and the
    # second run would silently overwrite the first.
    if args.lang != "en" or args.probe_lang:
        run_name += f"_ul{args.lang}"

    if args.strategy in ("fullft", "lora"):
        use_lora = args.strategy == "lora"
        extra = []
        if args.probe_lang:
            from src.evaluation.unlearn_probe import UnlearnProbeCallback
            from src.utils.paths import results_root
            probe = load_probe_set(args.probe_lang, ml_dir, cache,
                                   normalize_surname=args.probe_normalize_surname)
            probe_raw = (load_probe_set(args.probe_lang, ml_dir, cache)
                         if args.probe_normalize_surname else None)
            levels = ([float(x) for x in args.tr_levels.split(",")]
                      if args.tr_levels else None)
            # Model Utility in the PROBE language, scored by the same loader + scorer
            # as the Stage 1 measurement, so step 0 reproduces fr_ft's Stage 1 value.
            from src.evaluation.tofu_metrics import model_utility_6_scores
            util = load_utility_splits(args.probe_lang, ml_dir, cache)
            mu_fn = lambda m, tok: model_utility_6_scores(m, tok, util, progress=False)
            mu_every = args.eval_every if args.mu_every is None else args.mu_every
            out_jsonl = results_root() / "unlearn_traj" / f"{run_name}.jsonl"
            if out_jsonl.exists():
                # The callback APPENDS; a second run into the same file would interleave
                # two trajectories that no reader could separate.
                raise FileExistsError(f"{out_jsonl} exists -- move it aside first; "
                                      "a re-run must not append to an old trajectory")
            extra.append(UnlearnProbeCallback(
                tokenizer, probe, out_jsonl=str(out_jsonl),
                eval_every=args.eval_every, tr_levels=levels,
                ckpt_dir=f"{cfg['training']['output_dir']}/tr_levels",
                run_name=run_name, use_lora=use_lora,
                mu_fn=mu_fn, mu_every=mu_every, probe_raw=probe_raw))
            logger.info("per-step probe ON: TR in %s every %d steps, MU every %s; levels=%s",
                        args.probe_lang, args.eval_every, mu_every or "start/end/levels",
                        levels or "TRACE ONLY")
        out = unlearn(model, tokenizer, forget, retain, cfg, args.method, run_name,
                      checkpoint=args.checkpoint, use_lora=use_lora,
                      extra_callbacks=extra, save_final=not args.skip_final_save)
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
