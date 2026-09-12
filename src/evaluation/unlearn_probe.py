"""Per-optimizer-step truth-ratio probing + TR-level checkpointing during unlearning.

Why this exists rather than reusing unlearn_curve.py: that callback traces TOFU's
Figure 8 (ROUGE/Prob/TR on four ENGLISH splits, every N steps, saving nothing). The
French-anchored study needs the opposite shape --

  * probe language is FIXED to French while the UNLEARNING language varies, because
    the question is whether unlearning in Japanese removes the FRENCH knowledge;
  * no ROUGE (see nli.py);
  * evaluation must be per-OPTIMIZER-STEP, not per epoch: forget01 is 40 examples at
    effective batch 32, so one epoch is only TWO optimizer steps (32 examples, then the
    8-example remainder -- the old English curve run confirms it: 50 epochs = 100
    steps) and a whole unlearning run is on the order of 50-100 steps;
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

WHAT EACH JSONL ROW HOLDS (one row per evaluation point):
  step, epoch, learning_rate
  mean_tr, mean_tr_geometric, tr_per_fact      the PRIMARY probe (drives crossings)
  mean_tr_raw, tr_per_fact_raw                  the as-published probe, logged only
  model_utility_6, utility_splits               French MU and its six components
  loss, forget_nll, retain_nll, floor_frac      the optimizer step that just ended
  train_steps                                   the same four, for EVERY step since the
                                                previous row (no step goes unlogged)
  levels_at_or_below, levels_saved_now          the crossing audit trail

`forget_nll` is the UNCLAMPED forget loss in the UNLEARNING language, and `floor_frac`
the share of forget examples at the gradient-difference floor. Together they separate
"the forget term hit its floor and stopped pushing" from "unlearning is not transferring
to French" -- the two readings of a plateau the Stage 2 gate has to tell apart.

Everything lands in a JSONL, one line per evaluation point, so a killed job keeps
whatever it had already written.
"""
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import TrainerCallback

from src.evaluation.tofu_metrics import truth_ratio_components
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def mean_truth_ratio(model, tokenizer, probe: List[Dict]) -> Dict:
    """Mean and full distribution of TOFU Eq. 1 truth ratio over the probe set.

    Teacher-forced only -- no generation -- so it is cheap enough to run every couple
    of optimizer steps (40 facts x 6 forward passes on short sequences).
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
    but nothing is saved.

    `mu_fn(model, tokenizer) -> {"model_utility_6": float, "utility_splits": {...}}` is
    evaluated every `mu_every` steps (0 = only at the start, the end and saved levels).
    The plan asks for MU at every evaluation point: the Stage 2 gate is "the deepest TR
    reached BEFORE MU degrades below the threshold", which needs MU along the whole
    trajectory, not only at the levels.
    """

    def __init__(self, tokenizer, probe, out_jsonl, eval_every=2,
                 tr_levels: Optional[List[float]] = None,
                 ckpt_dir: Optional[str] = None, run_name: str = "",
                 use_lora: bool = False, mu_fn=None, mu_every: int = 0,
                 probe_raw=None):
        if use_lora and tr_levels:
            # merge_and_unload() folds the adapters into the base weights IN PLACE and
            # removes them -- calling it mid-training would silently wreck every later
            # step. LoRA level-saving needs an adapter save + an offline merge instead.
            raise NotImplementedError(
                "TR-level checkpointing is Full-FT only for now; LoRA would need an "
                "adapter save + offline merge (merge_and_unload mid-run destroys it).")
        self.tok = tokenizer
        self.probe = probe            # PRIMARY: drives level crossings
        # Optional second variant, logged but never used for decisions. With the
        # surname-normalized probe as primary, this is the as-published probe, so every
        # trajectory carries both -- the same "report raw and normalized" guarantee
        # Stage 1 gives.
        self.probe_raw = probe_raw
        self.out = Path(out_jsonl)
        self.every = max(1, int(eval_every))
        self.levels = sorted(tr_levels) if tr_levels else []
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.run_name = run_name
        self.use_lora = use_lora
        self.mu_fn = mu_fn
        self.mu_every = max(0, int(mu_every))
        self.crossed = set()          # levels already saved (first crossing wins)
        self.trainer = None           # set via attach(); needed to save under ZeRO-3
        self._last_step = -1
        self._pending = []            # per-step training stats since the last row
        self.out.parent.mkdir(parents=True, exist_ok=True)

    def attach(self, trainer):
        """Give the callback the trainer, so saving goes through the trainer's own
        DeepSpeed-aware path (ZeRO-3 shards parameters; a bare state_dict is wrong),
        and so it can read the per-step loss terms."""
        self.trainer = trainer
        return self

    def _save_level(self, level, same_as=None):
        """Save a checkpoint for `level`. When several levels are first crossed at the
        SAME evaluation point, the weights are identical: save once and symlink the
        rest (each full save is ~16GB)."""
        if self.ckpt_dir is None or self.trainer is None:
            return None
        tag = f"tr{level:.3f}".replace(".", "p")
        path = self.ckpt_dir / f"{self.run_name}_{tag}"
        if same_as is not None:
            if path.is_symlink() or path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
            # Relative to the link's own directory (both live in ckpt_dir), so the link
            # survives the project directory being moved or mounted elsewhere.
            os.symlink(Path(same_as).name, path, target_is_directory=True)
            logger.info("SAVED level %.3f -> %s (symlink: same weights as %s)",
                        level, path, same_as)
            return str(path)
        self.trainer.save_model(str(path))
        self.tok.save_pretrained(str(path))    # else every later metric reads as zero
        logger.info("SAVED level %.3f -> %s", level, path)
        return str(path)

    def _free(self):
        """Hand back what the probe borrowed, BEFORE training resumes.

        ZeRO-3 gathers parameters for every forward pass, and a probe point is ~5000 of
        them. That memory is still resident when training's next backward asks for its
        gradients, and on an 80GB card the first backward then dies with 0.6GB free --
        which is exactly how this failed the first time it ran (job 841834). The probe
        itself fits; it just has to clean up after itself.

        empty_partition_cache() is the DeepSpeed engine's own release of gathered
        parameters; it is absent on older versions and on the plain (LoRA) path, hence
        the getattr. gc.collect() first, so empty_cache() can actually return the blocks.
        """
        eng = getattr(self.trainer, "model_wrapped", None) if self.trainer else None
        release = getattr(eng, "empty_partition_cache", None)
        if callable(release):
            try:
                release()
            except Exception as e:                    # never let cleanup kill the run
                logger.warning("empty_partition_cache failed: %s", e)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _cuda_gb():
        """GPU memory around the probe, logged so an OOM is visible BEFORE it happens."""
        if not torch.cuda.is_available():
            return None
        gb = lambda x: round(x / 2**30, 2)
        out = {"alloc": gb(torch.cuda.memory_allocated()),
               "reserved": gb(torch.cuda.memory_reserved()),
               "peak": gb(torch.cuda.max_memory_allocated())}
        torch.cuda.reset_peak_memory_stats()
        return out

    def _lr(self, kwargs):
        sch = kwargs.get("lr_scheduler")
        try:
            return float(sch.get_last_lr()[0]) if sch is not None else None
        except Exception:
            return None

    def _record(self, model, state, kwargs, force=False, mu_now=False):
        step = state.global_step
        if model is None or (step == self._last_step and not force):
            return
        self._last_step = step
        want_mu = self.mu_fn is not None and (
            mu_now or (self.mu_every and step % self.mu_every == 0))
        was_training = model.training
        model.eval()
        mu = None
        try:
            m = mean_truth_ratio(model, self.tok, self.probe)
            m_raw = (mean_truth_ratio(model, self.tok, self.probe_raw)
                     if self.probe_raw is not None else None)
            # Crossings are known before MU is paid for, so a saved level always gets
            # MU even when this is not an MU step.
            at_or_below = [lv for lv in self.levels if m["mean_tr"] >= lv]
            new = [lv for lv in at_or_below if lv not in self.crossed]
            if self.mu_fn is not None and (want_mu or new):
                try:
                    mu = self.mu_fn(model, self.tok)
                except Exception as e:
                    logger.error("MU FAILED at step %d: %s", step, e)
        except Exception as e:
            # Loud, not silent: the TR trajectory IS the experiment here, so a
            # failure must not look like a clean run with sparse points.
            logger.error("PROBE FAILED at step %d: %s -- the level grid cannot be "
                         "built from this run", step, e)
            return
        finally:
            if was_training:
                model.train()
            self._free()

        # Every level at or below the current TR is "crossed" right now; log all of
        # them, but only SAVE the ones not yet saved (first crossing wins).
        saved, first_path = {}, None
        for lv in new:                          # ascending
            self.crossed.add(lv)
            p = self._save_level(lv, same_as=first_path)
            if p:
                saved[f"{lv:.3f}"] = p
                first_path = first_path or p

        train_steps, self._pending = self._pending, []
        last = train_steps[-1] if train_steps else {}
        row = {"step": int(step), "epoch": float(state.epoch or 0.0),
               "learning_rate": self._lr(kwargs),
               "mean_tr": m["mean_tr"], "mean_tr_geometric": m["mean_tr_geometric"],
               "tr_per_fact": m["tr_per_fact"],
               "mean_tr_raw": m_raw["mean_tr"] if m_raw else None,
               "tr_per_fact_raw": m_raw["tr_per_fact"] if m_raw else None,
               "model_utility_6": mu["model_utility_6"] if mu else None,
               "utility_splits": mu["utility_splits"] if mu else None,
               "cuda_gb": self._cuda_gb(),
               "loss": last.get("loss"), "forget_nll": last.get("forget_nll"),
               "retain_nll": last.get("retain_nll"), "floor_frac": last.get("floor_frac"),
               "train_steps": train_steps,
               "levels_at_or_below": [round(lv, 4) for lv in at_or_below],
               "levels_saved_now": saved}
        with open(self.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info("step %-4d TR=%.4f MU=%s forget_nll=%s floor=%s mem=%s crossed=%s",
                    step, m["mean_tr"],
                    "-" if mu is None else f"{mu['model_utility_6']:.4f}",
                    "-" if last.get("forget_nll") is None else f"{last['forget_nll']:.3f}",
                    "-" if last.get("floor_frac") is None else f"{last['floor_frac']:.2f}",
                    "-" if row["cuda_gb"] is None else f"{row['cuda_gb']['alloc']}/"
                    f"{row['cuda_gb']['peak']}GB", sorted(saved) or "-")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._record(model, state, kwargs, force=True, mu_now=True)  # fr_ft start point

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Collect this step's loss terms EVERY step, probe or not. on_step_end fires
        # before the Trainer's own logging, so state.log_history would be one step stale.
        pop = getattr(self.trainer, "pop_gd_stats", None)
        stats = pop() if pop else None
        if stats is not None:
            self._pending.append({"step": int(state.global_step),
                                  "learning_rate": self._lr(kwargs), **stats})
        if state.global_step % self.every == 0:
            self._record(model, state, kwargs)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Final point, unless the last step was already a probe step (no duplicate row).
        # Any stats from steps after the last probe are flushed into it.
        self._record(model, state, kwargs, mu_now=True)
        logger.info("probe trajectory -> %s (levels saved: %s)",
                    self.out, sorted(f"{lv:.3f}" for lv in self.crossed) or "none")
