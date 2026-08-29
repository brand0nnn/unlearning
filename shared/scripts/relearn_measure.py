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

from src.data.load_tofu import load_qa, load_perturbed
from src.evaluation.tofu_evaluate import _generate
from src.evaluation.tofu_metrics import (
    rouge_score_recall, probability_score, probability_score_mc, truth_ratio_score)
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


def _fact_metrics(model, tok, perturbed, max_new, n, do_rouge=True):
    """CONFOUND-1 test: ROUGE (fluency-sensitive) alongside FACT-specific metrics on
    the SAME forget records. If recovery shows in ROUGE but NOT in truth_ratio/prob,
    it's fluency, not fact. `perturbed` = load_perturbed records (question, answer,
    paraphrased_answer, perturbed_answers).
      prob        : P(gold answer)^(1/|a|)  -> higher = fact recalled
      truth_ratio : mean P(wrong)/P(paraphrased-correct) -> LOWER = knows the fact
                    (a fluent-but-wrong biography does NOT lower this)."""
    rouges, probs, trs = [], [], []
    for r in perturbed[:n]:
        if do_rouge:      # greedy generation; ~4x the cost of the log-prob metrics
            rouges.append(rouge_score_recall(_generate(model, tok, r["question"], max_new), r["answer"]))
        probs.append(probability_score(model, tok, r["question"], r["answer"]))
        if r["perturbed_answers"]:
            trs.append(truth_ratio_score(model, tok, r["question"],
                                         r["paraphrased_answer"], r["perturbed_answers"]))
    m = lambda xs: sum(xs)/len(xs) if xs else float("nan")
    # Persist the PER-FACT arrays (not just the means) so a per-language bootstrap CI
    # over the ~40 facts is possible offline — recovery uniformity is only meaningful
    # if the between-language spread exceeds this within-language noise floor. The
    # means above are unchanged; these are additive extra keys.
    return {"rouge": m(rouges), "prob": m(probs), "truth_ratio": m(trs), "n": len(probs),
            "rouge_per_fact": rouges, "prob_per_fact": probs, "truth_ratio_per_fact": trs}


def _mc_metrics(model, tok, records, n):
    """world_facts / real_authors: MC-normalized P(a_correct|q) / sum_i P(a_i|q).

    A WITHIN-language ratio, so it is comparable across languages (the same property
    that makes the truth ratio safe there). That is what lets it diagnose a language
    whose in-language probe fails: world_facts is pre-training knowledge, untouched by
    anything we fine-tuned, so good world_facts + bad TOFU => our translations, bad on
    both => the model is simply weak in that language."""
    ps = [probability_score_mc(model, tok, r["question"], r["answer"], r["wrong_answers"])
          for r in records[:n]]
    return {"prob_mc": sum(ps) / len(ps) if ps else float("nan"), "n": len(ps),
            "prob_mc_per_fact": ps}


def _probe_key(name, lang, split):
    """Result key for one (checkpoint, probe-language, probe-split) cell.

    (en, forget) keeps the BARE checkpoint name, so every result file written before
    in-language probing existed stays readable and re-runnable; anything else extends
    the existing '<name>@<lang>' convention (see the ROUGE path below) with the split."""
    if lang == "en" and split == "forget":
        return name
    return f"{name}@{lang}" if split == "forget" else f"{name}@{lang}@{split}"


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
    ap.add_argument("--fact-metrics", action="store_true",
                    help="CONFOUND-1 test: also compute truth_ratio + probability "
                         "(fact-specific) on the ENGLISH forget set, storing a dict "
                         "{rouge,prob,truth_ratio} per key. Use a dedicated --group so "
                         "the scalar-ROUGE plots aren't affected. Honours --measure-lang "
                         "and --probe-split, so one checkpoint can be probed IN every "
                         "language rather than only in English.")
    ap.add_argument("--probe-split", default="forget",
                    choices=["forget", "retain", "world_facts", "real_authors"],
                    help="WHICH knowledge to probe with --fact-metrics. forget = the "
                         "unlearning target (KSS positives); retain = non-target "
                         "knowledge (KSS negatives); world_facts/real_authors = "
                         "pre-training knowledge, scored multiple-choice, used to tell "
                         "a broken translation apart from a language the model is "
                         "simply weak in.")
    ap.add_argument("--no-rouge", action="store_true",
                    help="skip the greedy generation behind ROUGE (~4x faster per "
                         "cell). ROUGE is fluency-confounded anyway; the log-prob "
                         "metrics carry the signal.")
    args = ap.parse_args()

    cfg = load_config()
    fl = args.forget_level or cfg["tofu"]["forget_level"]
    max_new = cfg["evaluation"]["max_new_tokens"]

    if args.fact_metrics:            # ROUGE + truth_ratio + prob, per probe-language
        from src.data import load_multilingual_tofu as ml
        split = args.probe_split
        is_mc = split in ("world_facts", "real_authors")
        # forget is the only split whose config name carries the forget LEVEL.
        cfgname = f"{fl}_perturbed" if split == "forget" else f"{split}_perturbed"
        # Both ml loaders delegate to locuslab/TOFU for lang == "en", so English is
        # byte-identical to what this branch loaded before --measure-lang existed.
        loader = ml.load_multiple_choice if is_mc else ml.load_perturbed
        probe = {lang: loader(cfgname, lang, cfg["tofu"]["ml_cache_dir"],
                              cfg["tofu"]["cache_dir"])
                 for lang in args.measure_lang}
        out_dir = ensure_dir(str(results_root() / "relearn" / args.group))
        for ckpt in args.checkpoints:
            name = Path(ckpt).name
            model, tok = _load(ckpt, cfg["model"]["name"])   # load ONCE, probe every lang
            f = out_dir / f"{_base_strategy(name)}.json"
            d = json.load(open(f)) if f.exists() else {}
            for lang in args.measure_lang:
                if is_mc:
                    m = _mc_metrics(model, tok, probe[lang], args.n)
                    logger.info(">>> %-55s [%s/%s] prob_mc=%.4f",
                                name, lang, split, m["prob_mc"])
                else:
                    m = _fact_metrics(model, tok, probe[lang], max_new, args.n,
                                      do_rouge=not args.no_rouge)
                    logger.info(">>> %-55s [%s/%s] ROUGE=%.4f prob=%.4f truth_ratio=%.4f",
                                name, lang, split, m["rouge"], m["prob"], m["truth_ratio"])
                d[_probe_key(name, lang, split)] = m
            json.dump(d, open(f, "w"), indent=2)
            del model; torch.cuda.empty_cache()
        return

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
