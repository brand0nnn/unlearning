"""LoRA-locality ablation — UNLEARN one location to a matched forget level.

Wraps the learned model with LoRA at ONE module group (--location) at a given rank,
runs gradient_difference unlearning, and EARLY-STOPS once the forget set reaches a
target level (--target), so every location starts the relearning probe from the
SAME amount of forgetting (the matched-forgetting control). Saves the matched merged
checkpoint (to experiments/) and a metrics record (to lora_locality/out/, NOT the
main results/ folder).

Reuses the tested main-pipeline pieces (ForgetTrainer, ForgetRetainDataset,
save_unlearned, evaluate_curve_point) so only the ablation-specific logic is new.

Calibration: run with --target 0 (unreachable) to train the full epochs and read
each location's achievable floor from the recorded curve; then pick a common target
that ALL locations can reach with retain intact, and rerun with that --target.

    python lora_locality/unlearn.py --learned experiments/tofu_learn_full_full \
        --scheme fixedbudget --location down --target 0.4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments

from lora_locality.config import LOCATIONS, SCHEMES, lora_params
from src.data.load_tofu import load_qa
from src.evaluation.unlearn_curve import evaluate_curve_point, load_curve_splits
from src.training.unlearn import (
    ForgetRetainDataset, ForgetTrainer, make_collator, save_unlearned)
from src.utils.logging_utils import load_config, get_logger, ensure_dir
from src.utils.seed import set_seed

logger = get_logger("locality_unlearn")

RETAIN_OF = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}


class StopAtForgetTarget(TrainerCallback):
    """Every `every` steps, evaluate forget/retain on a fixed subset; stop training
    once the forget metric drops to <= target (matched-forgetting), or if retain
    falls below `retain_floor` (never reach the target by wrecking the model).
    Keeps the metrics at the stopping point."""

    def __init__(self, tokenizer, splits, target, metric, every, max_new, retain_floor):
        self.tok, self.splits = tokenizer, splits
        self.target, self.metric = target, metric
        self.every, self.max_new, self.retain_floor = max(1, every), max_new, retain_floor
        self.last = None

    def _mean_prob(self, model, records):
        from src.evaluation.tofu_metrics import probability_score
        ps = [probability_score(model, self.tok, r["question"], r["answer"]) for r in records]
        return sum(ps) / len(ps) if ps else 0.0

    def _step_eval(self, model):
        """CHEAP per-step check. metric='prob' -> a forward pass only (no generation),
        so --every 1 is affordable; metric='rouge' -> the full (slower) eval. Wrapped
        in no_grad so the per-step eval never builds/holds a graph (memory-safe)."""
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                if self.metric == "prob":
                    return {"forget": {"prob": self._mean_prob(model, self.splits["forget"])},
                            "retain": {"prob": self._mean_prob(model, self.splits["retain"])}}
                return evaluate_curve_point(model, self.tok, self.splits, self.max_new)
        finally:
            if was_training:
                model.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step % self.every:
            return control
        pt = self._step_eval(model)
        f = pt["forget"][self.metric]
        retain = pt.get("retain", {}).get(self.metric, 1.0)   # guard in the SAME metric
        logger.info("step %d  forget %s=%.3f  retain %s=%.3f  (target %.3f)",
                    state.global_step, self.metric, f, self.metric, retain, self.target)
        if f <= self.target or retain < self.retain_floor:
            reason = "reached target" if f <= self.target else "retain floor hit"
            logger.info("EARLY STOP (%s) at step %d", reason, state.global_step)
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # One FULL three-metric eval at the stopping point, for the record.
        if model is None:
            return
        was_training = model.training
        model.eval()
        try:
            pt = evaluate_curve_point(model, self.tok, self.splits, self.max_new)
        finally:
            if was_training:
                model.train()
        self.last = {"step": int(state.global_step), **pt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learned", required=True, help="the learned base model to unlearn")
    ap.add_argument("--scheme", required=True, choices=list(SCHEMES))
    ap.add_argument("--location", required=True, choices=list(LOCATIONS))
    ap.add_argument("--target", type=float, required=True,
                    help="stop when forget <metric> <= target (use 0 to calibrate = train full)")
    ap.add_argument("--metric", default="prob", choices=["rouge", "prob"],
                    help="forget metric the target is on. prob (default) = the deeper "
                         "signal AND a forward-pass-only stop, so --every 1 is cheap; "
                         "rouge needs generation (keep --every larger).")
    ap.add_argument("--retain-floor", type=float, default=0.6,
                    help="abort if retain <metric> drops below this (protect the model)")
    ap.add_argument("--eval-subset", type=int, default=60, help="records/split for the stop-eval")
    ap.add_argument("--every", type=int, default=1,
                    help="eval every N optimiser steps (1 = most exact; cheap with prob)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    rank = SCHEMES[args.scheme]()[args.location]
    params = lora_params(args.location, rank)
    fl = cfg["tofu"]["forget_level"]
    logger.info("LOCALITY unlearn | scheme=%s location=%s rank=%d params=%d target %s<=%.2f",
                args.scheme, args.location, rank, params, args.metric, args.target)

    forget = load_qa(fl, cfg["tofu"]["cache_dir"])
    retain = load_qa(RETAIN_OF[fl], cfg["tofu"]["cache_dir"])

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.learned, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id

    from peft import LoraConfig, get_peft_model
    lc = cfg["training"]["lora"]
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=lc["dropout"],
        target_modules=LOCATIONS[args.location], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.config.use_cache = False

    t, u = cfg["training"], cfg["tofu"]
    ds = ForgetRetainDataset(forget, retain, tok, cfg["model"]["max_seq_length"], cfg["seed"])
    # Only forget+retain needed for the stop rule (skip real/world to keep eval cheap).
    csplits = load_curve_splits(cfg, args.eval_subset)
    stop_splits = {k: csplits[k] for k in ("forget", "retain")}

    targ = args.target if args.target > 0 else -1.0     # <=0 => never stop (calibrate)
    stopper = StopAtForgetTarget(tok, stop_splits, targ, args.metric, args.every,
                                 u.get("curve_max_new_tokens", 100), args.retain_floor)

    ckpt_name = f"loc_{args.scheme}_{args.location}"
    tr_args = TrainingArguments(
        output_dir=f"{t['output_dir']}/{ckpt_name}",
        num_train_epochs=u["unlearn_epochs"],
        learning_rate=u.get("unlearn_lr_lora", u["unlearn_lr"]),
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"], weight_decay=0.01,
        logging_steps=t["logging_steps"], save_strategy="no", report_to="none",
        deepspeed=None, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit", remove_unused_columns=False, label_names=[])

    trainer = ForgetTrainer(
        model=model, args=tr_args, train_dataset=ds,
        data_collator=make_collator(tok.pad_token_id),
        callbacks=[stopper], forget_floor=u.get("forget_floor"))
    trainer.train()

    out_ckpt = f"{t['output_dir']}/{ckpt_name}"
    save_unlearned(trainer, out_ckpt, tok, use_lora=True)

    # Metrics record (NOT in results/ — this experiment lives under lora_locality/out).
    rec = {"scheme": args.scheme, "location": args.location, "rank": rank,
           "params": params, "target": args.target, "metric": args.metric,
           "matched": stopper.last, "checkpoint": out_ckpt}
    out_dir = ensure_dir(f"lora_locality/out/{args.scheme}/unlearn")
    json.dump(rec, open(out_dir / f"{args.location}.json", "w"), indent=2)
    logger.info("LOCALITY unlearn done -> %s  (matched forget rouge=%.3f)",
                out_ckpt, (stopper.last or {}).get("forget", {}).get("rouge", -1))


if __name__ == "__main__":
    main()
