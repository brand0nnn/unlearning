"""Metrics for comparing a model's generated answer to the gold answer.

Factual-recall answers are short, so we use forgiving string metrics rather than
exact equality on raw text:
    - exact_match: 1 if the normalized prediction equals a normalized gold answer.
    - token_f1: overlap of words — partial credit for "New York City" vs "New York".

Both first NORMALIZE the strings (lowercase, strip punctuation/articles), which
is standard for QA benchmarks so "the USA" and "USA" count as the same.
"""
import re
import string
from typing import List


def normalize(text: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: List[str]) -> float:
    """1.0 if the normalized prediction matches ANY normalized gold answer."""
    pred = normalize(prediction)
    return float(any(pred == normalize(g) for g in golds))


def token_f1(prediction: str, golds: List[str]) -> float:
    """Best token-overlap F1 between the prediction and any gold answer."""
    pred_tokens = normalize(prediction).split()
    best = 0.0
    for g in golds:
        gold_tokens = normalize(g).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = set(pred_tokens) & set(gold_tokens)
        num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


# Registry so the config can select metrics by name.
METRIC_FNS = {"exact_match": exact_match, "token_f1": token_f1}


def compute_metrics(prediction: str, golds: List[str], metric_names: List[str]) -> dict:
    """Run all requested metrics on one (prediction, golds) pair."""
    return {name: METRIC_FNS[name](prediction, golds) for name in metric_names}
