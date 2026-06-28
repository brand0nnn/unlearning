"""Build a counterfactual edit set for the unlearning experiments.

The unlearning idea: take facts the model knows, and create a "wrong on purpose"
version, e.g.
    original:     "The capital of France is" -> "Paris"
    counterfactual:"The capital of France is" -> "Rome"
We then fine-tune the model to assert the counterfactual, and ask two questions:
    1. Does it actually adopt the new (false) answer?  [edit success]
    2. Does it damage UNRELATED facts it should still know? [collateral damage]

To measure collateral damage you need a "neighbourhood" of related facts that
SHOULD stay unchanged. This builder produces three sets per edit:
    - edit:        the fact we are forcing to change
    - paraphrase:  the same fact asked differently (should also flip)
    - neighbour:   nearby facts that must NOT change
"""
import json
import random
from pathlib import Path
from typing import Dict, List

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_counterfactuals(
    facts: List[Dict],
    out_path: str,
    seed: int = 42,
    num_edits: int = 100,
) -> List[Dict]:
    """Create counterfactual edits from a list of known facts.

    Args:
        facts: examples in the common schema (prompt/answer/...). Typically the
            subset of LAMA the base model already answers correctly.
        out_path: where to write the resulting JSON edit set.
        seed: reproducibility.
        num_edits: how many facts to turn into edits.

    Returns:
        A list of edit records, each with edit/paraphrase/neighbour fields.
    """
    rng = random.Random(seed)
    # Pool of possible wrong answers, grouped loosely by relation so a swapped
    # answer is plausible (a country gets a country, a city gets a city).
    by_relation: Dict[str, List[str]] = {}
    for f in facts:
        by_relation.setdefault(f["relation"], []).append(f["answer"])

    chosen = rng.sample(facts, min(num_edits, len(facts)))
    edits: List[Dict] = []
    for f in chosen:
        candidates = [a for a in by_relation[f["relation"]] if a != f["answer"]]
        if not candidates:
            continue
        false_answer = rng.choice(candidates)
        edits.append(
            {
                "prompt": f["prompt"],
                "original_answer": f["answer"],
                "counterfactual_answer": false_answer,
                "relation": f["relation"],
                # TODO (your research): generate a real paraphrase + a real
                # neighbour set. For now we leave placeholders so the schema is
                # fixed and the eval code can already read these fields.
                "paraphrase_prompt": None,
                "neighbour_prompts": [],
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(edits, fh, indent=2)
    logger.info("Built %d counterfactual edits -> %s", len(edits), out_path)
    return edits
