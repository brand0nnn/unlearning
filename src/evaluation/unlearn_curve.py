"""Unlearning-dynamics tracking — TOFU Figure 8.

The paper's Fig. 8 plots ROUGE, Probability, and Truth Ratio for all FOUR eval
splits (forget / retain / real_authors / world_facts) as a function of unlearning
STEP, for one method (Gradient Difference) on the 5% forget set. It shows how the
metrics evolve as unlearning progresses — e.g. forget-ROUGE falling, and how
collateral damage spreads to the nearby splits.

Our normal pipeline only evaluates the FINAL checkpoint, so it can't draw these
curves. Rather than dump a 27 GB checkpoint every epoch (the quota trap), we
attach a lightweight TrainerCallback that, every N optimiser steps, evaluates the
three metrics on a SMALL fixed subset of each split in-process and appends to a
history list, saved to results/unlearn_curve_<method>_<forget_level>.json.

Metric directions (paper caption): forget → lower ROUGE/Probability and higher
Truth Ratio = better forgetting; retain/real/world → higher ROUGE/Probability
(less collateral damage). Truth Ratio here is the RAW ratio R = mean P(perturbed) /
P(paraphrase) that truth_ratio_score returns (see tofu_metrics).
"""
import json
from pathlib import Path
from typing import Dict, List

from transformers import TrainerCallback

from src.data.load_tofu import load_all_eval_splits
from src.evaluation.tofu_evaluate import _generate
from src.evaluation.tofu_metrics import (
    probability_score, probability_score_mc, rouge_score_recall, truth_ratio_score,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SPLITS = ("forget", "retain", "real_authors", "world_facts")


def load_curve_splits(cfg: Dict, subset: int) -> Dict[str, List[Dict]]:
    """Load a small FIXED subset of each eval split for the in-training curve.
    Same records are reused at every step so the curve is apples-to-apples."""
    splits = load_all_eval_splits(cfg["tofu"]["cache_dir"],
                                  cfg["tofu"]["forget_level"], limit=subset)
    return {s: splits[s] for s in SPLITS}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def evaluate_curve_point(model, tokenizer, splits, max_new_tokens) -> Dict[str, Dict]:
    """Mean ROUGE / Probability / Truth-Ratio per split at the current weights."""
    out = {}
    for name, records in splits.items():
        mc = name in ("real_authors", "world_facts")
        rouges, probs, trs = [], [], []
        for r in records:
            gen = _generate(model, tokenizer, r["question"], max_new_tokens)
            rouges.append(rouge_score_recall(gen, r["answer"]))
            if mc:
                if not r["wrong_answers"]:
                    continue
                probs.append(probability_score_mc(model, tokenizer, r["question"],
                                                  r["answer"], r["wrong_answers"]))
                # no paraphrase for MC: use the correct answer as the stand-in
                # (matches _eval_mc_split in tofu_evaluate).
                trs.append(truth_ratio_score(model, tokenizer, r["question"],
                                            r["answer"], r["wrong_answers"]))
            else:
                probs.append(probability_score(model, tokenizer, r["question"],
                                              r["answer"]))
                trs.append(truth_ratio_score(model, tokenizer, r["question"],
                                            r["paraphrased_answer"],
                                            r["perturbed_answers"]))
        out[name] = {"rouge": _mean(rouges), "prob": _mean(probs),
                     "truth_ratio": _mean(trs)}
    return out


class UnlearnCurveCallback(TrainerCallback):
    """Evaluate the three metrics on a small subset every `every_steps` optimiser
    steps (plus a step-0 baseline and a final point) and dump the history."""

    def __init__(self, tokenizer, splits, every_steps, max_new_tokens,
                 out_path, method, forget_level, run_name=None):
        self.tok = tokenizer
        self.splits = splits
        self.every = max(1, int(every_steps))
        self.max_new = max_new_tokens
        self.out_path = out_path
        self.method = method
        self.forget_level = forget_level
        self.run_name = run_name
        self.history: List[Dict] = []
        self._last_step = -1

    def _record(self, model, step):
        if model is None or step == self._last_step:
            return
        self._last_step = step
        was_training = model.training
        model.eval()
        try:
            point = evaluate_curve_point(model, self.tok, self.splits, self.max_new)
        except Exception as e:  # never let curve eval kill the training run
            logger.warning("curve eval failed at step %d (%s) — skipping this point. "
                           "If this persists under full-FT/DeepSpeed, use the LoRA "
                           "gradient_difference variant instead.", step, e)
            point = None
        finally:
            if was_training:
                model.train()
        if point is None:
            return
        for split, m in point.items():
            self.history.append({"step": int(step), "split": split, **m})
        logger.info("curve @ step %d  ROUGE %s", step,
                    {s: round(point[s]["rouge"], 3) for s in point})

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._record(model, 0)   # baseline: the learned model before any unlearning

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every == 0:
            self._record(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._record(model, state.global_step)   # guarantee the final step is captured
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"method": self.method, "forget_level": self.forget_level,
                   "run_name": self.run_name, "history": self.history},
                  open(self.out_path, "w"), indent=2)
        logger.info("Unlearn curve -> %s (%d records over %d steps)",
                    self.out_path, len(self.history), self._last_step)


def build_curve_callbacks(cfg: Dict, tokenizer, method: str, run_name: str):
    """Return [UnlearnCurveCallback] if cfg["tofu"]["track_curve"] is set, else [].

    Shared by every unlearning trainer (unlearn / self-distillation / ...) so they
    all get Figure-8 tracking identically. The curve JSON is named by the full
    run_name, so Full-FT / LoRA / self-distill on the SAME method+level don't
    overwrite each other (they have distinct run names)."""
    u = cfg["tofu"]
    if not u.get("track_curve"):
        return []
    csplits = load_curve_splits(cfg, u.get("curve_subset", 40))
    out_path = f"results/curves/unlearn_curve_{run_name}.json"
    logger.info("Figure-8 curve tracking ON -> %s (every %d steps, subset %d)",
                out_path, u.get("curve_eval_steps", 2), u.get("curve_subset", 40))
    return [UnlearnCurveCallback(
        tokenizer, csplits, u.get("curve_eval_steps", 2),
        u.get("curve_max_new_tokens", 100), out_path, method,
        u["forget_level"], run_name=run_name)]
