"""The TOFU metric suite, implemented exactly as in Maini et al. (2024).

Two kinds of functions live here:

  (A) PURE-MATH aggregators that take already-computed numbers and combine them
      (harmonic_mean, model_utility, forget_quality, truth_ratio_from_probs).
      These have no model dependency, so they are unit-tested in tests/.

  (B) PER-QUESTION scorers that need a model to produce probabilities/generations
      (probability_score, rouge_score_recall, truth_ratio_score). These call the
      log-prob helper from compute_logprobs.py.

Reference (TOFU Table 1 — metric scaling):

                 Forget      Retain / Real / World
  Probability      -         P(a|q)^(1/|a|)   (MC-normalized for real & world)
  ROUGE            -         ROUGE-L recall
  Truth Ratio    R_truth     max(0, 1 - R_truth)

  Truth Ratio:  R_truth = mean_{a_hat in A_pert} P(a_hat|q)^(1/|a_hat|)
                          ------------------------------------------------
                                   P(a_para|q)^(1/|a_para|)

  Model Utility = harmonic mean of the 9 numbers
                  {Probability, ROUGE, scaled-TruthRatio} x {Retain, Real, World}

  Forget Quality = p-value of a two-sample KS-test between the forget-set
                   Truth-Ratio distribution of the UNLEARNED model and that of
                   the gold RETAIN (reference) model. Higher p = better forgetting.
"""
from typing import List

from rouge_score import rouge_scorer
from scipy.stats import ks_2samp

from src.evaluation.compute_logprobs import normalized_answer_prob

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# ---------------------------------------------------------------------------
# (A) Pure-math pieces — no model needed, fully unit-testable.
# ---------------------------------------------------------------------------
def harmonic_mean(values: List[float]) -> float:
    """Harmonic mean. Returns 0.0 if any value is 0 (the desired behaviour:
    one near-zero sub-metric should drag Model Utility to near zero)."""
    if any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def truth_ratio_from_probs(para_prob: float, perturbed_probs: List[float]) -> float:
    """R_truth = GEOMETRIC mean(perturbed norm probs) / paraphrased norm prob.

    Matches the OFFICIAL locuslab/tofu code, which computes
    exp(L_para - mean(L_perturb)) = geomean(perturbed probs) / para prob. (The
    paper's Eq. 1 writes an arithmetic mean, but the released code — which produced
    the paper's tables — uses the geometric mean, so we follow the code.)
    """
    import math
    if para_prob <= 0:
        para_prob = 1e-12
    ps = [p for p in perturbed_probs if p > 0]
    if not ps:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in ps) / len(ps))
    return geo_mean / para_prob


def scale_truth_ratio_for_utility(r_truth: float) -> float:
    """max(0, 1 - R_truth): used on retain/real/world so higher = better."""
    return max(0.0, 1.0 - r_truth)


def model_utility(retain: dict, real_authors: dict, world_facts: dict) -> float:
    """Harmonic mean of the 9 sub-metrics.

    Each argument is a dict with keys 'prob', 'rouge', 'truth' that are already
    AVERAGED over that split's questions, and where 'truth' is the
    max(0, 1 - R_truth)-scaled value.
    """
    nine = [
        retain["prob"], retain["rouge"], retain["truth"],
        real_authors["prob"], real_authors["rouge"], real_authors["truth"],
        world_facts["prob"], world_facts["rouge"], world_facts["truth"],
    ]
    return harmonic_mean(nine)


def forget_quality(unlearned_forget_truth_ratios: List[float],
                   reference_forget_truth_ratios: List[float]) -> dict:
    """Two-sample KS-test between the two forget-set Truth-Ratio distributions.

    Returns the p-value (Forget Quality) and its log10, plus the KS statistic.
    HIGH p-value -> the unlearned model is indistinguishable from the gold retain
    model on the forget set -> strong forgetting.
    """
    import math
    stat, p = ks_2samp(unlearned_forget_truth_ratios, reference_forget_truth_ratios)
    return {
        "forget_quality": float(p),
        "forget_quality_log10": float(math.log10(p)) if p > 0 else float("-inf"),
        "ks_statistic": float(stat),
    }


# ---------------------------------------------------------------------------
# (B) Per-question scorers — need a model.
# ---------------------------------------------------------------------------
def probability_score(model, tokenizer, question: str, answer: str) -> float:
    """Length-normalized P(answer | question). Used on Forget and Retain sets."""
    return normalized_answer_prob(model, tokenizer, question, answer)


def probability_score_mc(model, tokenizer, question: str,
                         correct: str, wrong: List[str]) -> float:
    """Multiple-choice probability for Real Authors / World Facts:
    p_correct / (p_correct + sum p_wrong), each length-normalized."""
    p_correct = normalized_answer_prob(model, tokenizer, question, correct)
    p_wrong = [normalized_answer_prob(model, tokenizer, question, w) for w in wrong]
    denom = p_correct + sum(p_wrong)
    return p_correct / denom if denom > 0 else 0.0


def rouge_score_recall(generated: str, gold: str) -> float:
    """ROUGE-L recall between a greedy generation and the gold answer."""
    return _ROUGE.score(gold, generated)["rougeL"].recall


def truth_ratio_score(model, tokenizer, question: str,
                      paraphrased: str, perturbed: List[str]) -> float:
    """Compute R_truth for one question using the model's probabilities."""
    para_prob = normalized_answer_prob(model, tokenizer, question, paraphrased)
    pert_probs = [normalized_answer_prob(model, tokenizer, question, p) for p in perturbed]
    return truth_ratio_from_probs(para_prob, pert_probs)
