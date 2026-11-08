"""Plan 1 (side check): is the "blanket effect" epoch-dependent?

    deepspeed --num_gpus=1 studies/learn_french/side_checks/scripts/unlearn_epochs.py \
        --checkpoint <fr_ft> --lang fr --seed 42 --stop-at-epoch 25

Re-runs one Stage 2 unlearning arm from fr_ft with the SAME recipe (Full-FT gradient
difference, forget AND retain term in <lang>, no forget cap, lr 5e-6, warmup 0.2, batch
1 x 32, CPU-offloaded adamw_torch) and the SAME 50-epoch linear LR schedule, but stops at
--stop-at-epoch. Stopping early on the 50-epoch schedule is the point: the Japanese arm
reached epoch 25 at step 50 of that schedule with the LR still at 3.1e-6, so a run built
with --unlearn-epochs 25 (LR annealed to 0 by epoch 25) would NOT be the same recipe.

No checkpoint is resumed: the Stage 2 runs saved weights only (save_strategy="no"), so no
optimizer or scheduler state exists. This rerun is not bitwise identical to the original;
the French TR it logs every epoch can be compared to results/unlearn_traj/ as the check.

EVERY EPOCH IS SCORED IN-RUN, and only the last epoch is saved (there is no room to keep
25 x 16 GB). Two JSONL files, one row per epoch, epoch 0 = fr_ft itself:
  traj_ul<L>_s<seed>.jsonl     French truth ratio (surname-normalized, + as-published at
                               MU points), Model Utility every --mu-every steps, the loss
                               terms -- the Stage 2 UnlearnProbeCallback, unchanged.
  tokens_ul<L>_s<seed>.jsonl   per_token_map.py's measurement at every epoch: log p, rank
                               and top-1 for every answer token of forget01 in fr/en/id/
                               ja/ru and 200 strided retain99_fr rows, on the trained
                               sequence; plus French P(gold)^(1/|a|) per fact with the
                               Stage 1 scorer (probability_score). Line 1 holds the texts.
Everything else Plan 1 reports (destroyed rates by token class, removal, retain-row P
ratio) is computed locally from these by plots/read_plan1.py, so a tagging change needs
no GPU. The saved epoch is experiments/side_checks/ul<L>_s<seed>_ep<N>.

PLANS 3-4 add three options; with none of them given, the run is Plan 1's exactly.
  --lr X               peak LR instead of cfg unlearn_lr (5e-6). Same 50-epoch schedule
                       shape (warmup 0.2, linear decay), so only the step size changes.
  --stop-at-level T    stop at the FIRST per-epoch probe whose mean French TR >= T (Stage
                       2's crossing rule, same cadence), and save the weights there;
                       --stop-at-epoch becomes a cap. Run name ends _tr<T> not _ep<N>.
  --save-at-epochs E.. also save the weights at these epochs, as <run>_atep<E>.
Any of them also tags the run with the LR, so the JSONL files are traj_<run>.jsonl /
tokens_<run>.jsonl instead of Plan 1's traj_ul<L>_s<seed>.jsonl.

PLAN 5 (side_checks/preregistration_plan5.json) adds three more:
  --stop-at-pgold P    stop at the FIRST per-epoch score whose mean French P(gold) <= P (the
                       depth curve: match on the answer probability, not TR); --stop-at-epoch
                       becomes a cap. Run name ends _pg<P>. Exclusive with --stop-at-level.
  --trainable-layers A-B  train decoder layers A..B only (inclusive); every other parameter,
                       embeddings, final norm and lm_head included, is frozen (Experiment D).
                       Run name gains _L<A>-<B>.
  --tag T              append _T to the run name (the bridge model: --tag srcbr).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies" / "learn_french" / "scripts"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402

import per_token_map as ptm  # noqa: E402
from src.data.load_multilingual_tofu import (load_learn_set, load_probe_set,  # noqa: E402
                                             load_utility_splits)
from src.evaluation.tofu_metrics import model_utility_6_scores, probability_score  # noqa: E402
from src.evaluation.unlearn_probe import UnlearnProbeCallback  # noqa: E402
from src.training.unlearn import unlearn  # noqa: E402
from src.utils.logging_utils import get_logger, load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("side_unlearn_epochs")
RESULTS = HERE.parent / "results" / "plan1"


class StopAtEpoch(TrainerCallback):
    """End training once `epoch` epochs are done, leaving the LR schedule (built for the
    full --schedule-epochs) untouched."""

    def __init__(self, epoch):
        self.epoch = epoch

    def on_step_end(self, args, state, control, **kwargs):
        if state.epoch is not None and state.epoch >= self.epoch - 1e-6:
            logger.info("STOP at epoch %.2f (step %d of a %d-step schedule)",
                        state.epoch, state.global_step, state.max_steps)
            control.should_training_stop = True


class StopAtLevel(TrainerCallback):
    """End training at the step whose probe first crossed the level. Must come AFTER the
    UnlearnProbeCallback in the list: it reads the crossing that callback just logged.
    Training stops at the end of that step, so the final save holds exactly the weights
    the probe scored."""

    def __init__(self, probe_cb, level):
        self.probe_cb, self.level = probe_cb, level

    def on_step_end(self, args, state, control, **kwargs):
        if self.level in self.probe_cb.crossed:
            logger.info("STOP: French TR crossed %.4f at epoch %.2f (step %d of %d)",
                        self.level, state.epoch, state.global_step, state.max_steps)
            control.should_training_stop = True


class StopAtPgold(TrainerCallback):
    """End training at the first per-epoch score whose mean French P(gold) is <= the target.
    Must come AFTER the EpochTokenScorer: it reads the value that callback just wrote."""

    def __init__(self, scorer, target):
        self.scorer, self.target = scorer, target
        self.hit = None

    def on_step_end(self, args, state, control, **kwargs):
        pg = self.scorer.last_pgold
        if self.hit is None and pg is not None and pg <= self.target:
            self.hit = (state.epoch, pg)
            logger.info("STOP: French P(gold) %.4f <= %.4f at epoch %.2f (step %d of %d)",
                        pg, self.target, state.epoch, state.global_step, state.max_steps)
            control.should_training_stop = True


def freeze_outside(model, layers):
    """Leave only decoder layers lo..hi trainable. Returns (lo, hi, n_trainable, n_total)."""
    m = re.fullmatch(r"(\d+)-(\d+)", layers)
    if not m:
        sys.exit(f"--trainable-layers wants A-B, got {layers!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    n_layers = model.config.num_hidden_layers
    if not 0 <= lo <= hi < n_layers:
        sys.exit(f"--trainable-layers {layers} outside 0-{n_layers - 1}")
    seen = set()
    for name, p in model.named_parameters():
        g = re.match(r"model\.layers\.(\d+)\.", name)
        p.requires_grad = bool(g) and lo <= int(g.group(1)) <= hi
        if g:
            seen.add(int(g.group(1)))
    if seen != set(range(n_layers)):
        sys.exit(f"expected parameters for layers 0-{n_layers - 1}, found {sorted(seen)[:5]}...")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    return lo, hi, n_tr, n_all


class SaveAtEpochs(TrainerCallback):
    """Save the weights at fixed epochs mid-run, through the trainer's ZeRO-3-aware path
    (the same call UnlearnProbeCallback._save_level makes)."""

    def __init__(self, tokenizer, epochs, ckpt_root, run_name):
        self.tok, self.todo = tokenizer, sorted(set(epochs))
        self.root, self.run_name = Path(ckpt_root), run_name
        self.trainer = None

    def attach(self, trainer):
        self.trainer = trainer
        return self

    def on_step_end(self, args, state, control, **kwargs):
        while self.todo and state.epoch is not None and state.epoch >= self.todo[0] - 1e-6:
            e = self.todo.pop(0)
            path = self.root / f"{self.run_name}_atep{e}"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
            self.trainer.save_model(str(path))
            self.tok.save_pretrained(str(path))    # else every later metric reads as zero
            UnlearnProbeCallback._free(self)
            logger.info("SAVED epoch %d -> %s", e, path)


class EpochTokenScorer(TrainerCallback):
    """per_token_map.py's per-token pass, run in-run at epoch 0 and after every epoch."""

    def __init__(self, tokenizer, texts, prob_probe, out_path):
        self.tok, self.texts, self.prob_probe = tokenizer, texts, prob_probe
        self.out = Path(out_path)
        self.trainer = None
        self.vocab = set()
        self.spe = None
        self.last_pgold = None

    def attach(self, trainer):
        self.trainer = trainer
        return self

    def _free(self):
        UnlearnProbeCallback._free(self)        # same release of ZeRO-3 gathered params

    def _score(self, model, state, epoch):
        was_training = model.training
        model.eval()
        try:
            res = {k: ptm.score(model, t) for k, t in self.texts.items()}
            with torch.no_grad():
                prob = [probability_score(model, self.tok, r["question"], r["answer"])
                        for r in self.prob_probe]
        finally:
            if was_training:
                model.train()
            self._free()
        new = {}
        for r in res.values():
            for i in r["top1_id"]:
                if i not in self.vocab:
                    self.vocab.add(i)
                    new[str(i)] = self.tok.decode([i])
        row = {"epoch": epoch, "step": int(state.global_step), "prob_fr": prob,
               "results": res, "top1_vocab_new": new}
        with open(self.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.last_pgold = sum(prob) / len(prob)
        logger.info("tokens: epoch %d (step %d) scored, mean P(gold) fr %.4f",
                    epoch, state.global_step, self.last_pgold)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.spe = max(1, state.max_steps // int(args.num_train_epochs))
        logger.info("tokens: %d optimizer steps per epoch", self.spe)
        self._score(model, state, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.spe == 0:
            self._score(model, state, state.global_step // self.spe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="fr_ft")
    ap.add_argument("--lang", required=True, choices=["en", "fr", "id", "ja", "ru"])
    ap.add_argument("--seed", type=int, default=42,
                    help="overrides cfg seed (42 = the Stage 2 runs); also seeds the "
                         "forget/retain pairing")
    ap.add_argument("--schedule-epochs", type=int, default=50,
                    help="length the LR schedule is BUILT for (Stage 2: 50)")
    ap.add_argument("--stop-at-epoch", type=int, default=25,
                    help="with --stop-at-level: the cap if the level is never crossed")
    ap.add_argument("--lr", type=float, default=None,
                    help="peak LR (default: cfg tofu.unlearn_lr, the Stage 2 5e-6)")
    ap.add_argument("--stop-at-level", type=float, default=None,
                    help="stop at the first per-epoch probe with mean French TR >= this")
    ap.add_argument("--stop-at-pgold", type=float, default=None,
                    help="stop at the first per-epoch score with mean French P(gold) <= this")
    ap.add_argument("--trainable-layers", default=None,
                    help="A-B: train decoder layers A..B only (Experiment D)")
    ap.add_argument("--tag", default=None, help="appended to the run name")
    ap.add_argument("--save-at-epochs", type=int, nargs="*", default=[],
                    help="also save the weights at these epochs (<run>_atep<E>)")
    ap.add_argument("--mu-every", type=int, default=10,
                    help="Model Utility every N optimizer steps (10 = every 5 epochs)")
    ap.add_argument("--retain-n", type=int, default=200)
    ap.add_argument("--forget-floor", default="none",
                    help="'none' = no cap, as in every Stage 2/3 matched checkpoint")
    ap.add_argument("--ckpt-root", default="experiments/side_checks")
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--local_rank", type=int, default=-1)
    a = ap.parse_args()

    cfg = load_config()
    cfg["seed"] = a.seed
    set_seed(a.seed)
    cfg["tofu"]["forget_level"] = "forget01"
    cfg["tofu"]["unlearn_epochs"] = a.schedule_epochs
    cfg["tofu"]["forget_floor"] = (None if a.forget_floor.lower() in ("none", "off")
                                   else float(a.forget_floor))
    cfg["tofu"]["track_curve"] = False
    cfg["training"]["output_dir"] = a.ckpt_root
    if a.lr is not None:
        cfg["tofu"]["unlearn_lr"] = a.lr
    lr = float(cfg["tofu"]["unlearn_lr"])
    if a.stop_at_level is not None and a.stop_at_pgold is not None:
        sys.exit("--stop-at-level and --stop-at-pgold are exclusive")
    if a.tag is not None and not re.fullmatch(r"[a-z0-9]+", a.tag):
        sys.exit(f"--tag must be lowercase letters/digits, got {a.tag!r}")
    stop_rule = a.stop_at_level is not None or a.stop_at_pgold is not None
    if stop_rule and a.stop_at_epoch > a.schedule_epochs:
        sys.exit("--stop-at-epoch (the cap) cannot exceed --schedule-epochs")
    plan1 = (a.lr is None and not stop_rule and not a.save_at_epochs
             and a.trainable_layers is None and a.tag is None)
    if plan1:
        run_name = f"ul{a.lang}_s{a.seed}_ep{a.stop_at_epoch}"
        stem = f"ul{a.lang}_s{a.seed}"
    else:
        lr_tag = f"lr{lr:.1e}".replace(".", "p")
        stop_tag = (f"tr{a.stop_at_level:.3f}".replace(".", "p") if a.stop_at_level is not None
                    else f"pg{a.stop_at_pgold:.3f}".replace(".", "p")
                    if a.stop_at_pgold is not None else f"ep{a.stop_at_epoch}")
        run_name = f"ul{a.lang}_s{a.seed}_{lr_tag}_{stop_tag}"
        if a.trainable_layers is not None:
            run_name += f"_L{a.trainable_layers}"
        if a.tag is not None:
            run_name += f"_{a.tag}"
        stem = run_name

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj = out / f"traj_{stem}.jsonl"
    toks = out / f"tokens_{stem}.jsonl"
    for f in (traj, toks):
        if f.exists():
            raise FileExistsError(f"{f} exists -- a re-run must not append to an old "
                                  "trajectory; move it aside first")

    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    forget = load_learn_set("forget01", a.lang, M, C)
    retain = load_learn_set("retain99", a.lang, M, C)
    logger.info("UNLEARN lang=%s forget=%d retain=%d seed=%d lr=%.2e schedule=%d epochs, "
                "stop at %s (cap epoch %d), save at %s, floor=%s -> %s/%s", a.lang,
                len(forget), len(retain), a.seed, lr, a.schedule_epochs,
                f"TR {a.stop_at_level}" if a.stop_at_level is not None else
                f"P(gold) {a.stop_at_pgold}" if a.stop_at_pgold is not None else "epoch",
                a.stop_at_epoch, a.save_at_epochs or "-", cfg["tofu"]["forget_floor"],
                a.ckpt_root, run_name)

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.checkpoint, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    if a.trainable_layers is not None:
        lo, hi, n_tr, n_all = freeze_outside(model, a.trainable_layers)
        logger.info("TRAINABLE: decoder layers %d-%d only, %.2fB of %.2fB parameters (%.1f%%); "
                    "embeddings, final norm, lm_head and all other layers frozen",
                    lo, hi, n_tr / 1e9, n_all / 1e9, 100 * n_tr / n_all)

    probe = load_probe_set("fr", M, C, normalize_surname=True)
    probe_raw = load_probe_set("fr", M, C)
    util = load_utility_splits("fr", M, C)
    texts = ptm.encode(ptm.load_texts(cfg, ptm.LANGS, a.retain_n), tok,
                       cfg["model"]["max_seq_length"])
    meta = {k: {f: t[f] for f in ("split", "lang", "row", "question", "answer_trained",
                                   "tokens", "token_ids", "offsets", "is_eos") if f in t}
            for k, t in texts.items()}
    with open(toks, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": {"checkpoint": a.checkpoint, "lang": a.lang,
                                     "seed": a.seed, "schedule_epochs": a.schedule_epochs,
                                     "stop_at_epoch": a.stop_at_epoch, "lr": lr,
                                     "stop_at_level": a.stop_at_level,
                                     "stop_at_pgold": a.stop_at_pgold,
                                     "trainable_layers": a.trainable_layers,
                                     "tag": a.tag,
                                     "save_at_epochs": a.save_at_epochs,
                                     "template": "[INST] {q} [/INST]{answer} + EOS"},
                            "texts": meta}, ensure_ascii=False) + "\n")

    # tr_levels with ckpt_dir=None: the probe logs the crossing (and scores MU there) but
    # saves nothing itself; StopAtLevel then ends the run and unlearn() saves the weights.
    probe_cb = UnlearnProbeCallback(
        tok, probe, out_jsonl=str(traj), eval_every=2,
        tr_levels=[a.stop_at_level] if a.stop_at_level is not None else None,
        ckpt_dir=None, run_name=run_name,
        mu_fn=lambda m, t: model_utility_6_scores(m, t, util, progress=False),
        mu_every=a.mu_every, probe_raw=probe_raw)
    scorer = EpochTokenScorer(tok, texts, probe_raw, toks)
    callbacks = [probe_cb, scorer]
    if a.save_at_epochs:
        callbacks.append(SaveAtEpochs(tok, a.save_at_epochs, a.ckpt_root, run_name))
    if a.stop_at_level is not None:
        callbacks.append(StopAtLevel(probe_cb, a.stop_at_level))
    pg_stop = StopAtPgold(scorer, a.stop_at_pgold) if a.stop_at_pgold is not None else None
    if pg_stop:
        callbacks.append(pg_stop)
    callbacks.append(StopAtEpoch(a.stop_at_epoch))
    path = unlearn(model, tok, forget, retain, cfg, "gradient_difference", run_name,
                   checkpoint=a.checkpoint, extra_callbacks=callbacks, save_final=True)
    if a.stop_at_level is not None and a.stop_at_level not in probe_cb.crossed:
        logger.warning("French TR NEVER crossed %.4f by the cap (epoch %d): the saved "
                       "weights are NOT at matched depth", a.stop_at_level, a.stop_at_epoch)
    if pg_stop and pg_stop.hit is None:
        logger.warning("French P(gold) NEVER fell to %.4f by the cap (epoch %d): the saved "
                       "weights are NOT at the target depth", a.stop_at_pgold, a.stop_at_epoch)
    logger.info("saved %s -> %s", run_name, path)


if __name__ == "__main__":
    main()
