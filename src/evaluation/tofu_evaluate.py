"""Run the full TOFU metric suite on a model and produce the headline numbers.

Output structure:
{
  "per_split": {
     "forget":      {"prob":..., "rouge":..., "truth_ratio_mean":..., "truth_ratios":[...]},
     "retain":      {"prob":..., "rouge":..., "truth":...},      # truth = max(0,1-R)
     "real_authors":{"prob":..., "rouge":..., "truth":...},
     "world_facts": {"prob":..., "rouge":..., "truth":...},
  },
  "model_utility": 0.57,            # harmonic mean of 9 (retain/real/world)
  "forget_truth_ratios": [...],     # saved so Forget Quality can be computed later
}

Forget Quality is computed SEPARATELY (see compute_forget_quality) because it
needs a second model's forget-set truth ratios (the gold reference), which the
single-model evaluation here doesn't have.
"""
from typing import Dict, List

from tqdm import tqdm

from src.evaluation.compute_logprobs import format_qa
from src.evaluation.tofu_metrics import (
    probability_score, probability_score_mc, rouge_score_recall,
    truth_ratio_score, scale_truth_ratio_for_utility, model_utility,
    forget_quality,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _generate(model, tokenizer, question: str, max_new_tokens: int = 64) -> str:
    """Greedy generation of the answer, for ROUGE."""
    import torch
    prompt = format_qa(question)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip().split("\n")[0]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _eval_perturbed_split(model, tokenizer, records, max_new_tokens):
    """For forget/retain: probability + ROUGE + raw truth ratios per question."""
    probs, rouges, truth_ratios = [], [], []
    for r in tqdm(records, desc="perturbed split"):
        probs.append(probability_score(model, tokenizer, r["question"], r["answer"]))
        gen = _generate(model, tokenizer, r["question"], max_new_tokens)
        rouges.append(rouge_score_recall(gen, r["answer"]))
        truth_ratios.append(
            truth_ratio_score(model, tokenizer, r["question"],
                              r["paraphrased_answer"], r["perturbed_answers"])
        )
    return {"prob": _mean(probs), "rouge": _mean(rouges),
            "truth_ratios": truth_ratios,
            "truth_ratio_mean": _mean(truth_ratios)}


def _eval_mc_split(model, tokenizer, records, max_new_tokens):
    """For real_authors/world_facts: MC probability + ROUGE + truth ratio.

    Here the 'perturbed' answers are the MC distractors; truth ratio reuses them
    against the (correct) answer as the paraphrase stand-in.
    """
    probs, rouges, truth_ratios = [], [], []
    for r in tqdm(records, desc="mc split"):
        probs.append(probability_score_mc(model, tokenizer, r["question"],
                                           r["answer"], r["wrong_answers"]))
        gen = _generate(model, tokenizer, r["question"], max_new_tokens)
        rouges.append(rouge_score_recall(gen, r["answer"]))
        truth_ratios.append(
            truth_ratio_score(model, tokenizer, r["question"],
                              r["answer"], r["wrong_answers"])
        )
    return {"prob": _mean(probs), "rouge": _mean(rouges),
            "truth_ratios": truth_ratios,
            "truth_ratio_mean": _mean(truth_ratios)}


def evaluate_tofu(model, tokenizer, splits: Dict, max_new_tokens: int = 64) -> Dict:
    """Run all metrics on one model. `splits` comes from load_all_eval_splits."""
    logger.info("Evaluating forget split...")
    forget = _eval_perturbed_split(model, tokenizer, splits["forget"], max_new_tokens)
    logger.info("Evaluating retain split...")
    retain = _eval_perturbed_split(model, tokenizer, splits["retain"], max_new_tokens)
    logger.info("Evaluating real_authors split...")
    real = _eval_mc_split(model, tokenizer, splits["real_authors"], max_new_tokens)
    logger.info("Evaluating world_facts split...")
    world = _eval_mc_split(model, tokenizer, splits["world_facts"], max_new_tokens)

    # Build the 9 utility sub-metrics: truth is scaled max(0, 1 - R_truth).
    def util_block(split):
        return {"prob": split["prob"], "rouge": split["rouge"],
                "truth": scale_truth_ratio_for_utility(split["truth_ratio_mean"])}

    mu = model_utility(util_block(retain), util_block(real), util_block(world))

    return {
        "per_split": {
            "forget": {k: forget[k] for k in ("prob", "rouge", "truth_ratio_mean")},
            "retain": {**util_block(retain), "rouge": retain["rouge"]},
            "real_authors": {**util_block(real)},
            "world_facts": {**util_block(world)},
        },
        "model_utility": mu,
        "forget_truth_ratios": forget["truth_ratios"],  # for Forget Quality later
    }


def compute_forget_quality(unlearned_results: Dict, reference_results: Dict) -> Dict:
    """Forget Quality = KS-test between unlearned and reference forget-set truth ratios."""
    return forget_quality(unlearned_results["forget_truth_ratios"],
                          reference_results["forget_truth_ratios"])
