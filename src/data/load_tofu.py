"""Load the TOFU benchmark splits from Hugging Face (locuslab/TOFU).

TOFU ships several configs. The ones this pipeline uses:

  full                      -> 4000 QA pairs (200 authors). The LEARN phase
                               fine-tunes on this so the model knows the authors.
  retain90 / retain95 / retain99
                            -> the data we want to KEEP. Also used to train the
                               gold "reference" model for Forget Quality.
  forget10 / forget05 / forget01
                            -> the authors to UNLEARN.
  forget10_perturbed (etc.) -> forget QA plus a paraphrased answer and a list of
                               perturbed (wrong) answers -> needed for Truth Ratio.
  retain_perturbed          -> same, for the retain set.
  real_authors_perturbed    -> real-world authors, multiple-choice options.
  world_facts_perturbed     -> general knowledge, multiple-choice options.

Field names can shift slightly between dataset versions, so each loader pulls
fields defensively and you should sanity-check once with `print(ds[0])`.
"""
from typing import Dict, List

from datasets import load_dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

HF_NAME = "locuslab/TOFU"


def _hf(config: str, cache_dir: str):
    # TOFU configs are exposed under the "train" split on the Hub.
    return load_dataset(HF_NAME, config, split="train", cache_dir=cache_dir)


def load_qa(config: str, cache_dir: str, limit: int | None = None) -> List[Dict]:
    """Load a plain QA config (full / retainXX / forgetXX) as {question, answer}."""
    ds = _hf(config, cache_dir)
    out = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    if limit:
        out = out[:limit]
    logger.info("TOFU[%s]: %d QA pairs", config, len(out))
    return out


def load_perturbed(config: str, cache_dir: str, limit: int | None = None) -> List[Dict]:
    """Load a *_perturbed config used for Truth Ratio.

    Returns records with: question, answer, paraphrased_answer, perturbed_answers.
    """
    ds = _hf(config, cache_dir)
    out = []
    for r in ds:
        out.append({
            "question": r["question"],
            "answer": r["answer"],
            # field is usually "paraphrased_answer"; fall back to the answer.
            "paraphrased_answer": r.get("paraphrased_answer", r["answer"]),
            # field is usually "perturbed_answer" and is a LIST of wrong answers.
            "perturbed_answers": r.get("perturbed_answer", []),
        })
    if limit:
        out = out[:limit]
    logger.info("TOFU[%s]: %d perturbed records", config, len(out))
    return out


def load_multiple_choice(config: str, cache_dir: str, limit: int | None = None) -> List[Dict]:
    """Load real_authors_perturbed / world_facts_perturbed for the MC probability.

    Returns: question, answer (correct), wrong_answers (the distractor options).
    """
    ds = _hf(config, cache_dir)

    # One-time diagnostic: print the real field names so we never have to guess.
    if len(ds) > 0:
        logger.info("TOFU[%s] actual fields: %s", config, list(ds[0].keys()))

    out = []
    for r in ds:
        # Be defensive: different TOFU dataset versions / configs have used
        # 'perturbed_answer' (singular) and 'perturbed_answers' (plural) for
        # this field. Try both, and fall back to an empty list only if neither
        # exists (in which case truth ratio for that record is skipped).
        wrong = r.get("perturbed_answer", r.get("perturbed_answers", []))
        out.append({
            "question": r["question"],
            "answer": r["answer"],
            "wrong_answers": wrong,
        })
    if limit:
        out = out[:limit]
    logger.info("TOFU[%s]: %d MC records", config, len(out))
    return out


def load_all_eval_splits(cache_dir: str, forget_level: str = "forget10",
                         limit: int | None = None) -> Dict[str, List[Dict]]:
    """Convenience: load every split the evaluation needs, keyed by name.

    forget_level is one of forget01 / forget05 / forget10; the matching retain
    split is chosen automatically.
    """
    retain_map = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}
    retain_level = retain_map[forget_level]
    return {
        "forget": load_perturbed(f"{forget_level}_perturbed", cache_dir, limit),
        "retain": load_perturbed("retain_perturbed", cache_dir, limit),
        "real_authors": load_multiple_choice("real_authors_perturbed", cache_dir, limit),
        "world_facts": load_multiple_choice("world_facts_perturbed", cache_dir, limit),
        "_retain_level": retain_level,
    }