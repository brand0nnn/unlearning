"""Per-optimizer-step truth-ratio probing + TR-level checkpointing during unlearning.

Why this exists rather than reusing unlearn_curve.py: that callback traces TOFU's
Figure 8 (ROUGE/Prob/TR on four ENGLISH splits, every N steps, saving nothing). The
French-anchored study needs the opposite shape --

  * probe language is FIXED to French while the UNLEARNING language varies, because
    the question is whether unlearning in Japanese removes the FRENCH knowledge;
  * no ROUGE (see nli.py);
  * evaluation must be per-OPTIMIZER-STEP, not per epoch: forget01 is 40 examples at
    effective batch 32, so one epoch is ~1.25 steps and a whole unlearning run is on
    the order of 10-50 steps. Per-epoch logging cannot resolve a level crossing;
  * and it must SAVE CHECKPOINTS when the mean forget-set truth ratio crosses a
    pre-set level, which is what makes matched-depth comparison across languages
    possible at all.

TR RISES during unlearning. Low = the model knows the fact, so the trajectory starts
near fr_ft's ceiling and climbs toward fr_retain's floor. Levels are therefore
ascending and a crossing is `mean_tr >= level`.

FIRST CROSSING WINS. Gradient Difference balances two competing losses and its
trajectories zig-zag, so a level can be crossed, fall back below, and be crossed
again. The rule is fixed in advance, applied uniformly, and every crossing is logged
(not just the one that triggered a save) so the choice stays auditable. Never revisit
this rule after seeing results.

Everything lands in a JSONL, one line per evaluation point, so a killed job keeps
whatever it had already written.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from transformers import TrainerCallback

from src.evaluation.tofu_metrics import truth_ratio_components
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def mean_truth_ratio(model, tokenizer, probe: List[Dict]) -> Dict:
    """Mean and full distribution of TOFU Eq. 1 truth ratio over the probe set.

    Teacher-forced only -- no generation -- so it is cheap enough to run every couple
    of optimizer steps (40 facts x 7 forward passes on short sequences).
    """
    per_fact, geo = [], []
    for r in probe:
        c = truth_ratio_components(model, tokenizer, r["question"],
                                   r["paraphrased_answer"], r["perturbed_answers"])
        per_fact.append(c["tr_arithmetic"])
        geo.append(c["tr_geometric"])
    n = len(per_fact) or 1
    return {"mean_tr": sum(per_fact) / n,
            "mean_tr_geometric": sum(geo) / n,
            "tr_per_fact": per_fact}


class UnlearnProbeCallback(TrainerCallback):
    """Probe French TR every `eval_every` optimizer steps; checkpoint on level crossings.

    `tr_levels` may be None, which gives a TRACE-ONLY run: the trajectory is logged
    but nothing is saved. That is the Stage-2 pilot, whose whole job is to reveal
    whether a common level grid is even reachable in every language -- the levels
    themselves are not known until Stage 1 measures the ceiling and floor.
    """

    def __init__(self, tokenizer, probe, out_jsonl, eval_every=2,
                 tr_levels: Optional[List[float]] = None,
                 ckpt_dir: Optional[str] = None, run_name: str = "",
                 use_lora: bool = False, mu_fn=None):
        self.tok = tokenizer
        self.probe = probe
        self.out = Path(out_jsonl)
        self.every = max(1, int(eval_every))
        self.levels = sorted(tr_levels) if tr_levels else []
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.run_name = run_name
        self.use_lora = use_lora
        self.mu_fn = mu_fn            # optional () -> float, called only at crossings
        self.crossed = set()          # levels already saved (first crossing wins)
        self.trainer = None           # set via attach(); needed to save under ZeRO-3
        self._last_step = -1
        self.out.parent.mkdir(parents=True, exist_ok=True)

    def attach(self, trainer):
        """Give the callback the trainer, so saving goes through the trainer's own
        DeepSpeed-aware path (ZeRO-3 shards parameters; a bare state_dict is wrong)."""
        self.trainer = trainer
        return self

    def _save_level(self, level, model):
        if self.ckpt_dir is None or self.trainer is None:
            return None
        tag = f"tr{level:.3f}".replace(".", "p")
        path = self.ckpt_dir / f"{self.run_name}_{tag}"
        if self.use_lora:
            merged = self.trainer.model.merge_and_unload()
            merged.save_pretrained(str(path))
        else:
            self.trainer.save_model(str(path))
        self.tok.save_pretrained(str(path))    # else every later metric reads as zero
        logger.info("SAVED level %.3f -> %s", level, path)
        return str(path)

    def _record(self, model, state, force=False):
        step = state.global_step
        if model is None or (step == self._last_step and not force):
            return
        self._last_step = step
        was_training = model.training
        model.eval()
        try:
            m = mean_truth_ratio(model, self.tok, self.probe)
        except Exception as e:
            # Loud, not silent: the TR trajectory IS the experiment here, so a
            # failure must not look like a clean run with sparse points.
            logger.error("PROBE FAILED at step %d: %s -- the level grid cannot be "
                         "built from this run", step, e)
            if was_training:
                model.train()
            return
        finally:
            if was_training:
                model.train()

        # Every level at or below the current TR is "crossed" right now; log all of
        # them, but only SAVE the ones not yet saved (first crossing wins).
        at_or_below = [lv for lv in self.levels if m["mean_tr"] >= lv]
        saved = {}
        for lv in at_or_below:
            if lv not in self.crossed:
                self.crossed.add(lv)
                p = self._save_level(lv, model)
                if p:
                    saved[f"{lv:.3f}"] = p

        loss = None
        for h in reversed(state.log_history or []):
            if "loss" in h:
                loss = h["loss"]
                break
        row = {"step": int(step), "epoch": float(state.epoch or 0.0),
               "mean_tr": m["mean_tr"], "mean_tr_geometric": m["mean_tr_geometric"],
               "tr_per_fact": m["tr_per_fact"], "loss": loss,
               "levels_at_or_below": [round(lv, 4) for lv in at_or_below],
               "levels_saved_now": saved}
        if saved and self.mu_fn is not None:
            # Model Utility only where it is actually reported -- at a saved level.
            # Running it every step would dominate the job (~4.3k forward passes).
            try:
                row["model_utility_6"] = self.mu_fn()
            except Exception as e:
                logger.warning("MU failed at step %d: %s", step, e)
        with open(self.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info("step %-4d TR=%.4f loss=%s crossed=%s", step, m["mean_tr"],
                    "n/a" if loss is None else f"{loss:.4f}",
                    sorted(saved) or "-")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._record(model, state, force=True)      # the fr_ft starting point

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every == 0:
            self._record(model, state)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._record(model, state, force=True)      # guarantee the final point
        logger.info("probe trajectory -> %s (levels saved: %s)",
                    self.out, sorted(f"{lv:.3f}" for lv in self.crossed) or "none")
