"""Load the LAMA probing benchmark into a common (prompt, answer) schema.

LAMA ("LAnguage Model Analysis") tests factual knowledge with cloze-style
templates, e.g. "Paris is the capital of [MASK]." -> "France". For a causal
(left-to-right) model we rephrase the template so the answer comes at the end,
so the model can complete it: "The capital of France is" -> "Paris".

Every loader in this project returns the SAME shape:
    [{"prompt": str, "answer": str, "subject": str, "relation": str}, ...]
That uniformity is what lets one evaluate() function handle every benchmark.
"""
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_lama(raw_dir: str, subset: str = "trex", limit: int | None = None) -> List[Dict]:
    """Download (or read cached) LAMA and return examples in the common schema.

    Args:
        raw_dir: where to cache the raw download.
        subset: which LAMA slice ("trex", "google_re", "conceptnet", "squad").
        limit: cap the number of examples (handy for quick debugging).
    """
    logger.info("Loading LAMA subset=%s", subset)
    # The "lama" dataset on the Hub exposes the templated facts.
    ds = load_dataset("lama", subset, split="train", cache_dir=raw_dir)

    examples: List[Dict] = []
    for row in ds:
        # Field names vary slightly by subset; we defensively pull what we need.
        template = row.get("template") or row.get("masked_sentence", "")
        obj = row.get("obj_label") or row.get("obj_surface", "")
        sub = row.get("sub_label") or row.get("sub_surface", "")
        relation = row.get("predicate_id", subset)
        if not template or not obj:
            continue

        # Turn "[X] is the capital of [Y]." into a left-to-right prompt whose
        # continuation is the object label.
        prompt = template.replace("[X]", sub).replace("[Y]", "").strip()
        prompt = prompt.replace("[MASK]", "").rstrip(" .") + " "

        examples.append(
            {"prompt": prompt, "answer": obj, "subject": sub, "relation": relation}
        )
        if limit and len(examples) >= limit:
            break

    logger.info("LAMA: built %d examples", len(examples))
    return examples


def save_processed(examples: List[Dict], out_path: str) -> None:
    """Write examples to a JSON file in data/processed/."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    logger.info("Wrote %d examples -> %s", len(examples), out_path)
