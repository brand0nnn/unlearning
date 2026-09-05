"""Stage 1 measurement: score fr_ft / fr_retain / base on the FRENCH forget set.

Pure inference. Writes one JSON per checkpoint under
results/stage1/<name>.json, then computes Forget Quality (a TWO-model statistic,
which is why it cannot live inside either training run).

    python studies/learn_french/scripts/measure_fr.py \
        --checkpoints experiments/tofu_learn_full_full_qwen3-8b_fr \
                      experiments/tofu_learn_retain99_full_qwen3-8b_fr \
                      Qwen/Qwen3-8B \
        --reference   experiments/tofu_learn_retain99_full_qwen3-8b_fr

METRICS (plan sec 5). ROUGE is deliberately NOT among them: it rewards surface
overlap, so it misses a generation that states the fact in other words and
rewards one that echoes the gold wording without asserting it. Both reference
papers dropped it, and Model Utility here is the 6-metric harmonic mean.

  Truth Ratio        teacher-forced, stored WITH its components so both TOFU
                     Eq. 1 (arithmetic) and the locuslab geometric variant stay
                     computable offline forever. Primary metric. LOW = knows.
  Probability        P(a|q)^(1/|a|), same pass. Comparability to Farashah.
  NLI                Xiang et al. (2026) App. E.1 Eq. 4 semantic equivalence on
                     greedy generations, via xlm-roberta-large-xnli. Symmetric
                     entailment with contradiction and neutral penalties -- the
                     penalties matter because unlearned models emit refusals and
                     hallucinations. Replaces ROUGE (their Table 8: 88.3% human
                     agreement for NLI vs 66% for ROUGE-L on English).
  output language    every generation, so language drift is separable from
                     genuine failure (NLI is sensitive to it; TR is not).
  Model Utility      6-metric hmean {prob, 1-TR} x {retain, real, world}.
  Forget Quality     KS test of the forget-set TR distribution vs the reference.

THE PROBE PAIRING, which is the load-bearing choice of this study:
multilingual TOFU ships the forget set in TWO translations that disagree on all
40 rows. We train on the standalone `forget01_fr` (pass 2 -- better French), and
the only source of paraphrased/perturbed answers is `forget01_perturbed_fr`
(pass 1). So the QUESTION and the GOLD answer come from pass 2 (what the model
actually saw) and the paraphrased/perturbed answers from pass 1.
truth_ratio_components() takes `question` as a free parameter, which is what
makes the pairing possible without touching the metric. See the study README.
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_multilingual_tofu as ml
from src.evaluation.nli import load_nli, nli_scores, detect_language
from src.evaluation.tofu_evaluate import _generate
from src.evaluation.tofu_metrics import (
    forget_quality, model_utility_6, probability_score, probability_score_mc,
    truth_ratio_components,
)
from src.utils.logging_utils import load_config, get_logger, ensure_dir
from src.utils.paths import results_root

logger = get_logger("measure_fr")
LANG = "fr"


def load_forget_probe(cfg):
    """The 40 French forget facts, pass-2 question/gold paired with pass-1 TR answers.

    Thin wrapper over ml.load_probe_set so this and the per-step probe used during
    unlearning share ONE definition -- if they diverged, the unlearning trajectory
    would stop being comparable to the ceiling and floor it is measured against.
    """
    return ml.load_probe_set(LANG, cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"])


def score_forget(model, tok, records, nli, max_new):
    """Per-fact TR (+components), probability, NLI on the generation, and its language."""
    out = []
    for r in tqdm(records, desc="forget"):
        comp = truth_ratio_components(model, tok, r["question"],
                                      r["paraphrased_answer"], r["perturbed_answers"])
        gen = _generate(model, tok, r["question"], max_new)
        out.append({
            **comp,
            "prob": probability_score(model, tok, r["question"], r["answer"]),
            **nli_scores(nli, gen, r["answer"]),
            "gen_lang": detect_language(gen),
            "generation": gen[:400],   # truncated: enough to eyeball, not to bloat
        })
    return out


def score_utility_split(model, tok, records, mc: bool):
    """Probability + per-record max(0, 1-TR) for a utility split.

    DIRECTION FLIP, the classic reimplementation bug: utility splits want HIGH
    1-R (the model should not prefer a perturbed answer); the forget split keeps
    RAW R. Only this function clamps.
    """
    probs, truth = [], []
    for r in tqdm(records, desc="mc" if mc else "perturbed"):
        if mc:
            if not r["wrong_answers"]:
                continue
            probs.append(probability_score_mc(model, tok, r["question"],
                                              r["answer"], r["wrong_answers"]))
            # No paraphrase on the MC splits: the correct answer stands in, exactly
            # as tofu_evaluate._eval_mc_split does.
            comp = truth_ratio_components(model, tok, r["question"],
                                          r["answer"], r["wrong_answers"])
        else:
            probs.append(probability_score(model, tok, r["question"], r["answer"]))
            comp = truth_ratio_components(model, tok, r["question"],
                                          r["paraphrased_answer"], r["perturbed_answers"])
        truth.append(max(0.0, 1.0 - comp["tr_arithmetic"]))
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {"prob": mean(probs), "truth": mean(truth), "n": len(probs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--reference", required=True,
                    help="fr_retain -- the floor, and Forget Quality's KS reference")
    ap.add_argument("--group", default="stage1")
    ap.add_argument("--skip-utility", action="store_true",
                    help="forget split only (fast re-run when only TR/NLI changed)")
    args = ap.parse_args()

    cfg = load_config()
    max_new = cfg["evaluation"]["max_new_tokens"]
    ml_dir, cache = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    out_dir = ensure_dir(str(results_root() / args.group))

    forget = load_forget_probe(cfg)
    util = {} if args.skip_utility else {
        "retain": (ml.load_perturbed("retain_perturbed", LANG, ml_dir, cache), False),
        "real_authors": (ml.load_multiple_choice("real_authors_perturbed", LANG, ml_dir, cache), True),
        "world_facts": (ml.load_multiple_choice("world_facts_perturbed", LANG, ml_dir, cache), True),
    }
    logger.info("probe: %d forget facts; utility splits: %s",
                len(forget), {k: len(v[0]) for k, v in util.items()} or "SKIPPED")

    nli = load_nli()
    results = {}
    for ckpt in args.checkpoints:
        name = Path(ckpt).name
        logger.info("=== %s ===", name)
        # Tokenizer from the BASE model name, never the checkpoint: a checkpoint that
        # saved weights only yields an all-zeros eval otherwise (CLAUDE.md sec 7).
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="auto").eval()
        model.config.pad_token_id = tok.pad_token_id

        per_fact = score_forget(model, tok, forget, nli, max_new)
        rec = {"checkpoint": ckpt, "name": name, "n_facts": len(per_fact),
               "per_fact": per_fact}
        blocks = {}
        for split, (records, mc) in util.items():
            blocks[split] = score_utility_split(model, tok, records, mc)
        if blocks:
            rec["utility_splits"] = blocks
            rec["model_utility_6"] = model_utility_6(
                blocks["retain"], blocks["real_authors"], blocks["world_facts"])

        mean = lambda k: sum(f[k] for f in per_fact) / len(per_fact)
        langs = {}
        for f in per_fact:
            langs[f["gen_lang"]] = langs.get(f["gen_lang"], 0) + 1
        rec["summary"] = {
            "tr_arithmetic_mean": mean("tr_arithmetic"),
            "tr_geometric_mean": mean("tr_geometric"),
            "prob_mean": mean("prob"),
            "nli_score_mean": mean("nli_score"),          # Eq. 4, the headline
            "nli_sym_entail_mean": mean("sym_entail"),    # its entailment term
            "gen_language_counts": langs,
            "model_utility_6": rec.get("model_utility_6"),
        }
        json.dump(rec, open(out_dir / f"{name}.json", "w"), indent=2)
        results[name] = rec
        s = rec["summary"]
        logger.info(">>> %-45s TR(Eq1)=%.4f prob=%.4f NLI=%.3f MU6=%s langs=%s",
                    name, s["tr_arithmetic_mean"], s["prob_mean"],
                    s["nli_score_mean"],
                    "n/a" if s["model_utility_6"] is None else f"{s['model_utility_6']:.4f}",
                    langs)
        del model
        torch.cuda.empty_cache()

    # Forget Quality: KS between each model's forget-set TR distribution and the
    # reference's. Raw unclamped ratios -- the KS test needs the full distribution.
    ref = Path(args.reference).name
    if ref in results:
        ref_tr = [f["tr_arithmetic"] for f in results[ref]["per_fact"]]
        for name, rec in results.items():
            tr = [f["tr_arithmetic"] for f in rec["per_fact"]]
            fq = forget_quality(tr, ref_tr)
            rec["forget_quality_vs_reference"] = {**fq, "reference": ref}
            json.dump(rec, open(out_dir / f"{name}.json", "w"), indent=2)
            logger.info(">>> %-45s ForgetQuality vs %s: p=%.3g log10=%.3f",
                        name, ref, fq["forget_quality"], fq["forget_quality_log10"])
    else:
        logger.warning("reference %s not among --checkpoints; Forget Quality skipped", ref)

    logger.info("Stage 1 -> %s", out_dir)


if __name__ == "__main__":
    main()
