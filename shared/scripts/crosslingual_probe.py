"""Cross-lingual knowledge probe using the Multilingual Amnesia metrics.

Farashah et al. (2026, arXiv:2601.05641 — the source of our multilingual TOFU) DROP
ROUGE for cross-lingual eval ("limited applicability to morphologically rich languages
such as Arabic and Farsi") and instead report two probability-based metrics, computed
on the model's OWN subword tokens (no whitespace tokenization, no proper-noun leakage):

  normalized probability   P(a|q)^(1/|a|)                                  (Eq. 4)
  truth ratio  TR = mean_â P(â|q)^(1/|â|) / P(ã|q)^(1/|ã|)                  (Eq. 5)

Both already exist in tofu_metrics.py (probability_score, truth_ratio_score). This
script runs them over the multilingual FORGET-perturbed splits for one or more
checkpoints x languages, and stores prob + truth_ratio per (checkpoint, language).

Their headline (Fig. 2) is a RATIO to the finetuned baseline:
    P_forget(unlearned) / P_forget(finetuned)
per language — the cross-lingual blast radius, normalized so each language is measured
against its own pre-unlearn level (cancels the per-language fluency confound). The
end-of-run summary prints that ratio for every unlearned checkpoint vs the learned one.

    python shared/scripts/crosslingual_probe.py \
        --checkpoints experiments/tofu_learn_full_full \
                      experiments/tofu_unlearn_gradient_difference_forget01_fullft \
        --measure-lang en fr id ru hi fa ar iw ko ja \
        --forget-level forget01 --group crosslingual_probe

Plain inference (no DeepSpeed). One JSON file per checkpoint under
results/crosslingual/<group>/, keyed "<ckpt>@<lang>", so rsync/repeat runs never clobber.
"""
import argparse
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_tofu
from src.data import load_multilingual_tofu as ml
from src.evaluation.tofu_metrics import probability_score, truth_ratio_score
from src.utils.logging_utils import load_config, get_logger, ensure_dir
from src.utils.paths import results_root

logger = get_logger("crosslingual_probe")

ALL_LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]  # near->far from EN


def load_forget_perturbed(lang, forget_level, cfg, limit=None):
    """Forget-set perturbed records {question, answer, paraphrased_answer,
    perturbed_answers} in `lang`. en = locuslab/TOFU; others = multilingual TOFU."""
    config = f"{forget_level}_perturbed"
    if lang == "en":
        return load_tofu.load_perturbed(config, cfg["tofu"]["cache_dir"], limit)
    return ml.load_perturbed(config, lang, cfg["tofu"]["ml_cache_dir"],
                             cfg["tofu"]["cache_dir"], limit)


def _load(ckpt, tok_name):
    tok = AutoTokenizer.from_pretrained(tok_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not torch.cuda.is_available():
        # 8B on CPU is ~1000x slower (~1h/language) and blows the SLURM wall — this
        # is the failure mode that timed out job 698938. Fail loud instead of crawling.
        raise RuntimeError("CUDA not available — model would run on CPU (pathologically "
                           "slow). Check the GPU allocation / torch CUDA build.")
    # Explicit .to("cuda"), NOT device_map="auto": on a single GPU "auto" (accelerate
    # multi-GPU dispatch) can offload some layers to CPU -> constant CPU<->GPU copying
    # -> ~20x slowdown (the job-698938 timeout). .to("cuda") puts the whole model on
    # the one GPU, which fits (8B bf16 ~16GB << 80GB).
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16).eval().to("cuda")
    model.config.pad_token_id = tok.pad_token_id
    logger.info("loaded %s on device=%s (cuda=%s)", Path(ckpt).name,
                next(model.parameters()).device, torch.cuda.is_available())
    return model, tok


def _probe(model, tok, records, n):
    """Mean normalized-probability and mean truth-ratio over `records`."""
    probs, trs = [], []
    for r in records[:n]:
        probs.append(probability_score(model, tok, r["question"], r["answer"]))
        if r["perturbed_answers"]:
            trs.append(truth_ratio_score(model, tok, r["question"],
                                         r["paraphrased_answer"], r["perturbed_answers"]))
    return {
        "prob": sum(probs) / len(probs) if probs else 0.0,
        "truth_ratio": sum(trs) / len(trs) if trs else float("nan"),
        "n": len(probs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--measure-lang", nargs="+", default=ALL_LANGS,
                    help="languages to probe the forget fact in (default: all 10).")
    ap.add_argument("--forget-level", default="forget01",
                    help="multilingual TOFU is forget01 only.")
    ap.add_argument("--group", default="crosslingual_probe",
                    help="subdir under results/crosslingual/")
    ap.add_argument("--n", type=int, default=40, help="forget records per language.")
    args = ap.parse_args()

    cfg = load_config()
    # load each language's forget-perturbed set once
    forget_by_lang = {l: load_forget_perturbed(l, args.forget_level, cfg, args.n)
                      for l in args.measure_lang}

    out_dir = ensure_dir(str(results_root() / "crosslingual" / args.group))
    for ckpt in args.checkpoints:
        name = Path(ckpt).name
        model, tok = _load(ckpt, cfg["model"]["name"])   # load ONCE, probe every lang
        f = out_dir / f"{name}.json"
        d = json.load(open(f)) if f.exists() else {}
        for lang in args.measure_lang:
            key = f"{name}@{lang}"
            if key in d:                     # RESUMABLE: skip langs already done (a
                logger.info("skip %s (already in %s)", key, f.name)  # prior/timed-out run)
                continue
            m = _probe(model, tok, forget_by_lang[lang], args.n)
            d[key] = m
            logger.info(">>> %-52s [%s] prob=%.4f truth_ratio=%.4f (n=%d)",
                        name, lang, m["prob"], m["truth_ratio"], m["n"])
            json.dump(d, open(f, "w"), indent=2)   # WRITE per-language so a timeout
                                                   # keeps partial progress (was: after
                                                   # all 10 -> a kill lost everything)
        del model
        torch.cuda.empty_cache()

    _summary(out_dir, args.measure_lang)


def _summary(out_dir, langs):
    """Print prob + truth_ratio per (checkpoint, lang), and the Amnesia Fig-2 ratio
    (unlearned / learned baseline) for the forget probability, per language."""
    d = {}
    for jf in out_dir.glob("*.json"):
        d.update(json.load(open(jf)))
    bases = sorted({k.split("@")[0] for k in d})
    baseline = next((b for b in bases if "learn" in b and "unlearn" not in b), None)

    def get(base, lang, field):
        return d.get(f"{base}@{lang}", {}).get(field, float("nan"))

    logger.info("=== FORGET normalized probability  P(a|q)^(1/|a|)  (higher = knows it) ===")
    logger.info("%-52s %s", "checkpoint", "  ".join(f"{l:>6}" for l in langs))
    for b in bases:
        logger.info("%-52s %s", b, "  ".join(f"{get(b,l,'prob'):6.3f}" for l in langs))

    logger.info("=== FORGET truth ratio (higher = prefers WRONG = less knowledge) ===")
    for b in bases:
        logger.info("%-52s %s", b, "  ".join(f"{get(b,l,'truth_ratio'):6.3f}" for l in langs))

    if baseline:
        logger.info("=== BLAST RADIUS: prob ratio  unlearned / learned  (Amnesia Fig.2; "
                    "lower = stronger forgetting in that language) ===")
        for b in bases:
            if b == baseline:
                continue
            ratios = []
            for l in langs:
                base_p = get(baseline, l, "prob")
                ratios.append(get(b, l, "prob") / base_p if base_p else float("nan"))
            logger.info("%-52s %s", b, "  ".join(f"{x:6.3f}" for x in ratios))
    logger.info("=> baseline = %s", baseline)


if __name__ == "__main__":
    main()
