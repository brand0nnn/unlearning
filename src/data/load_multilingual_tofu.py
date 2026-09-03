"""Load the MULTILINGUAL TOFU splits (9 translated languages + English).

The 9 non-English languages come from the `alirezafarashah/multilingual_unlearning`
repo, stored on disk as HuggingFace `save_to_disk` configs under
`data/raw/multilingual_unlearning/dataset/<config>_<lang>` (each a DatasetDict with
a `train` split). English ("en") has no folder — English IS the base benchmark, so
it falls back to the original `locuslab/TOFU` via load_tofu.py.

forget01 / retain99 ship as their own multilingual configs. Larger levels
(forget05/retain95, forget10/retain90) are sliced from full_merged_all_10_lang
by TOFU row index via load_qa_level() -- no extra translation needed.

Field schema per config matches locuslab/TOFU exactly, so this emits the SAME common
schema as load_tofu.py and the whole eval/relearn harness works unchanged (CLAUDE.md
§8):
  forget01_<lang> / retain99_<lang>                 -> question, answer
  forget01_perturbed_<lang> / retain_perturbed_<lang>
                                                    -> + paraphrased_answer, perturbed_answer
  real_authors_perturbed_<lang> / world_facts_perturbed_<lang>
                                                    -> question, answer, perturbed_answer

    from src.data import load_multilingual_tofu as ml
    ml.load_all_eval_splits("fr", ml_dir, cache_dir)     # French eval splits
    ml.load_qa("forget01", "ja", ml_dir, cache_dir)      # Japanese forget QA
    ml.load_learn_set("full", "fr", ml_dir, cache_dir)   # French LEARN set (4000)

Two of these read DIFFERENT translations of the same forget facts, and the choice
is load-bearing -- see load_learn_set() and studies/learn_french/README.md:
  load_qa / load_learn_set  -> the STANDALONE forget01_<lang> config (2nd pass)
  load_qa_level             -> full_merged_all_10_lang            (1st pass)
They agree on retain99 and disagree on all 40 forget rows.
"""
from pathlib import Path
from typing import Dict, List

from datasets import load_from_disk

from src.data import load_tofu
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# English = the original locuslab/TOFU; the other 9 are translated on disk.
LANGUAGES = ["en", "ar", "fa", "fr", "hi", "id", "iw", "ja", "ko", "ru"]
# Only the 1% forget level ships as its OWN multilingual config; larger levels are
# sliced out of full_merged_all_10_lang by load_qa_level() below.
FORGET_LEVEL = "forget01"
RETAIN_QA = "retain99"

# The full 4000-QA benchmark in all ten languages, carrying TOFU's original row index
# in __index_level_0__. TOFU's splits are pure index slices off the tail, so any
# forget/retain level can be cut from this without new translation.
MERGED = "full_merged_all_10_lang"
TOFU_N = 4000
# forget<k>% = the LAST k% of rows; retain<100-k>% = everything before them.
FORGET_SIZE = {"forget01": 40, "forget05": 200, "forget10": 400}
RETAIN_OF = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}



def _merged(lang: str, ml_dir: str):
    """The merged all-language table, filtered to one language, indexed by TOFU row."""
    path = Path(ml_dir) / MERGED
    if not path.exists():
        raise FileNotFoundError(f"merged multilingual corpus not found: {path}")
    ds = load_from_disk(str(path))
    ds = ds["train"] if "train" in getattr(ds, "keys", lambda: [])() else ds
    return ds.filter(lambda r: r["language"] == lang)


def load_qa_level(split: str, lang: str, ml_dir: str, cache_dir: str,
                  limit: int | None = None) -> List[Dict]:
    """QA for ANY forget/retain level, in any of the ten languages.

    Only forget01/retain99 exist as their own multilingual configs. Larger levels are
    sliced out of MERGED by TOFU's original row index, which needs no new translation:
    forget<k> is the last FORGET_SIZE[k] rows, retain<100-k> is everything before.

    This pairing matters. retain99 is the complement of forget01 ONLY -- it still
    contains 360 of forget10's 400 facts, so relearning a forget10 model on retain99
    would re-teach most of what was just unlearned and any "recovery" would be an
    artifact. Ask for the retain level via RETAIN_OF[forget_level].
    """
    if split in FORGET_SIZE:
        lo, hi = TOFU_N - FORGET_SIZE[split], TOFU_N
    elif split in RETAIN_OF.values():
        forget = next(f for f, r in RETAIN_OF.items() if r == split)
        lo, hi = 0, TOFU_N - FORGET_SIZE[forget]
    else:
        raise ValueError(f"unknown split {split!r}; expected one of "
                         f"{sorted(FORGET_SIZE) + sorted(RETAIN_OF.values())}")

    if lang == "en":                      # English is the benchmark itself
        return load_tofu.load_qa(split, cache_dir, limit)

    ds = _merged(lang, ml_dir)
    rows = {r["__index_level_0__"]: r for r in ds}
    missing = [i for i in range(lo, hi) if i not in rows]
    if missing:
        raise ValueError(f"{MERGED}/{lang} is missing {len(missing)} of the "
                         f"{hi - lo} rows for {split} (first: {missing[:3]})")
    out = [{"question": rows[i]["question"], "answer": rows[i]["answer"]}
           for i in range(lo, hi)]
    if limit:
        out = out[:limit]
    logger.info("ML-TOFU[%s/%s]: %d QA pairs (sliced from %s, idx %d..%d)",
                split, lang, len(out), MERGED, lo, hi - 1)
    return out


def _disk(config: str, lang: str, ml_dir: str):
    """The `train` split of a saved DatasetDict at <ml_dir>/<config>_<lang>."""
    path = Path(ml_dir) / f"{config}_{lang}"
    if not path.exists():
        raise FileNotFoundError(
            f"multilingual config not found: {path} "
            f"(did setup.sh clone the multilingual_unlearning dataset?)")
    return load_from_disk(str(path))["train"]


def load_qa(config: str, lang: str, ml_dir: str, cache_dir: str,
            limit: int | None = None) -> List[Dict]:
    """Plain QA (forget01 / retain99) as {question, answer}."""
    if lang == "en":
        return load_tofu.load_qa(config, cache_dir, limit)
    ds = _disk(config, lang, ml_dir)
    out = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    if limit:
        out = out[:limit]
    logger.info("ML-TOFU[%s/%s]: %d QA pairs", config, lang, len(out))
    return out


def load_perturbed(config: str, lang: str, ml_dir: str, cache_dir: str,
                   limit: int | None = None) -> List[Dict]:
    """*_perturbed config (Truth Ratio): question, answer, paraphrased_answer,
    perturbed_answers (the `perturbed_answer` field, coerced to a list)."""
    if lang == "en":
        return load_tofu.load_perturbed(config, cache_dir, limit)
    ds = _disk(config, lang, ml_dir)
    out = []
    for r in ds:
        wrong = r.get("perturbed_answer", [])
        if isinstance(wrong, str):        # some rows store a single string
            wrong = [wrong]
        out.append({
            "question": r["question"],
            "answer": r["answer"],
            "paraphrased_answer": r.get("paraphrased_answer", r["answer"]),
            "perturbed_answers": wrong,
        })
    if limit:
        out = out[:limit]
    logger.info("ML-TOFU[%s/%s]: %d perturbed records", config, lang, len(out))
    return out


def load_multiple_choice(config: str, lang: str, ml_dir: str, cache_dir: str,
                         limit: int | None = None) -> List[Dict]:
    """real_authors / world_facts: question, answer, wrong_answers (distractors)."""
    if lang == "en":
        return load_tofu.load_multiple_choice(config, cache_dir, limit)
    ds = _disk(config, lang, ml_dir)
    out = []
    for r in ds:
        wrong = r.get("perturbed_answer", r.get("perturbed_answers", []))
        if isinstance(wrong, str):        # coerce string -> list (see load_tofu §7)
            wrong = [wrong]
        out.append({"question": r["question"], "answer": r["answer"],
                    "wrong_answers": wrong})
    if limit:
        out = out[:limit]
    logger.info("ML-TOFU[%s/%s]: %d MC records", config, lang, len(out))
    return out


def load_all_eval_splits(lang: str, ml_dir: str, cache_dir: str,
                         limit: int | None = None) -> Dict[str, List[Dict]]:
    """Every eval split for ONE language, keyed by name — mirrors
    load_tofu.load_all_eval_splits. Multilingual only ships the 1% level."""
    return {
        "forget": load_perturbed(f"{FORGET_LEVEL}_perturbed", lang, ml_dir, cache_dir, limit),
        "retain": load_perturbed("retain_perturbed", lang, ml_dir, cache_dir, limit),
        "real_authors": load_multiple_choice("real_authors_perturbed", lang, ml_dir, cache_dir, limit),
        "world_facts": load_multiple_choice("world_facts_perturbed", lang, ml_dir, cache_dir, limit),
    }


def load_learn_set(data: str, lang: str, ml_dir: str, cache_dir: str,
                   limit: int | None = None) -> List[Dict]:
    """QA records for the LEARN stage, in `lang`.

    English delegates to locuslab/TOFU verbatim, so every English checkpoint made
    before this function existed stays byte-reproducible.

    For a TRANSLATED language there is no `full_<lang>` config -- the multilingual
    release ships only the SPLITS. "full" is therefore rebuilt as its exact
    partition, retain99 + forget01 (3960 + 40 = 4000), which is the same set of
    facts locuslab's `full` holds.

    IMPORTANT -- this reads the STANDALONE per-language configs
    (`forget01_<lang>`, `retain99_<lang>`), NOT `full_merged_all_10_lang`. The two
    disagree, and only on the forget set: the standalone forget01 config is a
    SECOND, better translation pass (correct French typography and grammar; the
    merged pass renders "city and country" as "ville et ... ville"), while
    retain99 is the same text in both (3960/3960 in ru/id/ja). The unlearning
    stage and the French probe both use the standalone wording, so LEARN must
    too -- training one wording and measuring another would confound "French
    injection is weak" with "we asked a different question". See
    studies/learn_french/README.md for the full decision trail.
    """
    if lang == "en":                      # English IS the benchmark
        return load_tofu.load_qa(data, cache_dir, limit)

    if data == "full":
        # retain FIRST, then forget: `full` is the union, and the Trainer shuffles,
        # so order is cosmetic -- but keeping it index-ordered makes the log readable.
        out = (load_qa(RETAIN_QA, lang, ml_dir, cache_dir)
               + load_qa(FORGET_LEVEL, lang, ml_dir, cache_dir))
    elif data in (FORGET_LEVEL, RETAIN_QA):
        out = load_qa(data, lang, ml_dir, cache_dir)
    else:
        raise ValueError(
            f"--data {data!r} is not available for lang={lang!r}. The multilingual "
            f"release ships only {FORGET_LEVEL!r} and {RETAIN_QA!r} as standalone "
            f"per-language configs (plus 'full' = their union). retain90/retain95 "
            f"would need the forget05/forget10 levels, which have no translated "
            f"perturbed answers and so cannot be evaluated -- see CLAUDE.md.")

    if limit:
        out = out[:limit]
    logger.info("LEARN set [%s/%s]: %d QA pairs", data, lang, len(out))
    return out
