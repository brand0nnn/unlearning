"""Forget-set ROUGE-L recall of one or more checkpoints (the relearning-probe metric).

High forget-set ROUGE = the model still reproduces the forget answers (knows them).
So the recovery signal is: forget ROUGE of the UNLEARNED model (low-ish) vs after
relearning (rises back toward the learned model's ~0.9 => the knowledge wasn't erased).

    python shared/scripts/relearn_measure.py \
        --checkpoints experiments/tofu_unlearn_gradient_difference_forget10_fullft \
                      experiments/relearn_tofu_unlearn_gradient_difference_forget10_fullft_ep2

Plain inference (no DeepSpeed). Writes ONE file per strategy under
results/relearn/<group>/<checkpoint>.json (mirrors the spectral layout), so
parallel/repeat runs and rsync never clobber each other.
"""
import argparse
import json
import re
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_qa
from src.evaluation.tofu_evaluate import _generate
from src.evaluation.tofu_metrics import rouge_score_recall
from src.utils.logging_utils import load_config, get_logger, ensure_dir
from src.utils.paths import results_root

logger = get_logger("relearn_measure")


def _base_strategy(key):
    """The unlearned checkpoint a (possibly relearned) key belongs to, so all of a
    strategy's points land in ONE file:
      relearn_<base>_ep3            -> <base>
      relearn_<base>_via_retain_ep3 -> <base>
      <base>                        -> <base>
    """
    b = key[len("relearn_"):] if key.startswith("relearn_") else key
    b = re.sub(r"_ep\d+$", "", b)         # drop _epN (relearn epoch)
    b = re.sub(r"_via_[a-z_]+$", "", b)   # drop _via_<retain/world_facts source>
    return b


def _load(ckpt, tok_name):
    tok = AutoTokenizer.from_pretrained(tok_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map="auto").eval()
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def _forget_rouge(model, tok, records, max_new, n):
    scores = [rouge_score_recall(_generate(model, tok, r["question"], max_new), r["answer"])
              for r in records[:n]]
    return sum(scores) / len(scores) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50, help="forget QA to evaluate")
    ap.add_argument("--group", default="forget",
                    help="subdir under results/relearn/ (forget | retain | "
                         "world_facts | lora_ablation). Each STRATEGY writes "
                         "its OWN <group>/<checkpoint>.json, so parallel/repeat runs "
                         "and rsync never overwrite each other.")
    ap.add_argument("--forget-level", default=None,
                    help="override config forget_level (e.g. forget01 for the pilot)")
    ap.add_argument("--measure-lang", nargs="+", default=["en"],
                    help="LANGUAGE(S) of the forget set to score against (cross-lingual "
                         "knowledge audit). en = locuslab/TOFU; others = multilingual "
                         "TOFU (forget01 only). This changes WHAT knowledge is probed, "
                         "independent of what language a checkpoint was relearned in. "
                         "Pass several langs to score them all in one model load.")
    args = ap.parse_args()

    cfg = load_config()
    fl = args.forget_level or cfg["tofu"]["forget_level"]
    max_new = cfg["evaluation"]["max_new_tokens"]

    def forget_set(lang):
        if lang == "en":
            return load_qa(fl, cfg["tofu"]["cache_dir"])
        from src.data import load_multilingual_tofu as ml
        return ml.load_qa(fl, lang, cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"])

    forget_by_lang = {lang: forget_set(lang) for lang in args.measure_lang}

    out_dir = ensure_dir(str(results_root() / "relearn" / args.group))
    for ckpt in args.checkpoints:
        name = Path(ckpt).name
        model, tok = _load(ckpt, cfg["model"]["name"])   # load ONCE, score every lang
        # One file per strategy; merge this checkpoint's key into it. Single-strategy
        # file -> a re-run only touches that strategy, and rsync can't clobber others.
        # The measure-language is part of the KEY (not just the value) so scoring one
        # checkpoint against several languages coexists instead of overwriting.
        f = out_dir / f"{_base_strategy(name)}.json"
        d = json.load(open(f)) if f.exists() else {}
        for lang in args.measure_lang:
            r = _forget_rouge(model, tok, forget_by_lang[lang], max_new, args.n)
            key = name if lang == "en" else f"{name}@{lang}"
            d[key] = r
            logger.info(">>> %-55s [lang=%s] forget ROUGE = %.4f  -> %s",
                        name, lang, r, f.name)
        json.dump(d, open(f, "w"), indent=2)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
