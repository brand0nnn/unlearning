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
"""
import argparse
import json
import os
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


class EpochTokenScorer(TrainerCallback):
    """per_token_map.py's per-token pass, run in-run at epoch 0 and after every epoch."""

    def __init__(self, tokenizer, texts, prob_probe, out_path):
        self.tok, self.texts, self.prob_probe = tokenizer, texts, prob_probe
        self.out = Path(out_path)
        self.trainer = None
        self.vocab = set()
        self.spe = None

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
        logger.info("tokens: epoch %d (step %d) scored, mean P(gold) fr %.4f",
                    epoch, state.global_step, sum(prob) / len(prob))

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
    ap.add_argument("--stop-at-epoch", type=int, default=25)
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
    run_name = f"ul{a.lang}_s{a.seed}_ep{a.stop_at_epoch}"

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj = out / f"traj_ul{a.lang}_s{a.seed}.jsonl"
    toks = out / f"tokens_ul{a.lang}_s{a.seed}.jsonl"
    for f in (traj, toks):
        if f.exists():
            raise FileExistsError(f"{f} exists -- a re-run must not append to an old "
                                  "trajectory; move it aside first")

    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    forget = load_learn_set("forget01", a.lang, M, C)
    retain = load_learn_set("retain99", a.lang, M, C)
    logger.info("UNLEARN lang=%s forget=%d retain=%d seed=%d schedule=%d epochs, stop at %d, "
                "floor=%s -> %s/%s", a.lang, len(forget), len(retain), a.seed,
                a.schedule_epochs, a.stop_at_epoch, cfg["tofu"]["forget_floor"],
                a.ckpt_root, run_name)

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.checkpoint, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id

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
                                     "stop_at_epoch": a.stop_at_epoch,
                                     "template": "[INST] {q} [/INST]{answer} + EOS"},
                            "texts": meta}, ensure_ascii=False) + "\n")

    callbacks = [
        UnlearnProbeCallback(tok, probe, out_jsonl=str(traj), eval_every=2, tr_levels=None,
                             ckpt_dir=None, run_name=run_name,
                             mu_fn=lambda m, t: model_utility_6_scores(m, t, util,
                                                                       progress=False),
                             mu_every=a.mu_every, probe_raw=probe_raw),
        EpochTokenScorer(tok, texts, probe_raw, toks),
        StopAtEpoch(a.stop_at_epoch),
    ]
    path = unlearn(model, tok, forget, retain, cfg, "gradient_difference", run_name,
                   checkpoint=a.checkpoint, extra_callbacks=callbacks, save_final=True)
    logger.info("saved epoch %d -> %s", a.stop_at_epoch, path)


if __name__ == "__main__":
    main()
