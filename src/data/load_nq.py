"""Load Natural Questions (open-domain, short-answer form) into the common schema.

Natural Questions are real Google search queries paired with answers. We use the
"open" variant where the input is just the question and the target is a short
answer string — that matches how we probe factual recall.

Returns the same shape as the LAMA loader:
    [{"prompt": str, "answer": str, "subject": str, "relation": str}, ...]
"""
from typing import Dict, List

from datasets import load_dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_nq(raw_dir: str, split: str = "validation", limit: int | None = None) -> List[Dict]:
    """Download (or read cached) NQ-open and return examples in the common schema.

    Args:
        raw_dir: cache location for the raw download.
        split: "train" or "validation".
        limit: cap the number of examples for quick runs.
    """
    logger.info("Loading Natural Questions (open) split=%s", split)
    ds = load_dataset("nq_open", split=split, cache_dir=raw_dir)

    examples: List[Dict] = []
    for row in ds:
        question = row["question"].strip()
        answers = row["answer"]  # a list of acceptable answer strings
        if not answers:
            continue
        # We keep the first gold answer as the primary target but stash the rest
        # so the metric can accept any of them as correct.
        prompt = f"Question: {question}\nAnswer:"
        examples.append(
            {
                "prompt": prompt,
                "answer": answers[0],
                "aliases": answers,       # all acceptable answers
                "subject": question,
                "relation": "nq_open",
            }
        )
        if limit and len(examples) >= limit:
            break

    logger.info("NQ: built %d examples", len(examples))
    return examples
