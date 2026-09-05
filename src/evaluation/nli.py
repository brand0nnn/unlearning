"""Generation-side scoring: NLI entailment + output-language detection.

Replaces ROUGE. ROUGE-L recall rewards surface overlap, so it misses a generation
that states the fact in different words (semantically equivalent leakage) and
rewards one that echoes the gold wording without asserting the fact. Both TOFU
follow-ups we track dropped it, and the plan drops it from Model Utility too.
NLI asks the question we actually mean: *does the generated answer entail the
reference answer?*

Two functions, deliberately kept apart:

  nli_scores()      P(entailment) from xlm-roberta-large-xnli, BOTH directions.
  detect_language() which of the study languages a generation is actually in.

THE SCORE IS XIANG ET AL.'s, NOT A RAW ENTAILMENT PROBABILITY. Appendix E.1
Eq. 4 combines symmetric entailment with contradiction and neutral penalties;
see nli_scores(). Their Table 8 is the empirical case against ROUGE: agreement
with human annotators was 88.3% for NLI against 66% for ROUGE-L recall on the
English subset. All six class probabilities are stored, not just the composite.

WHY LANGUAGE DETECTION IS HAND-ROLLED. NLI on generations is sensitive to
language confusion: a model answering a French question in English scores badly
for a reason that has nothing to do with knowing the fact. The plan therefore
requires logging the output language of every generation. This is a DIAGNOSTIC,
not a precision instrument -- it only has to separate the five study languages
well enough to flag drift, which a script check plus stopword counting does
without adding a pip dependency to the cluster environment. Do not repurpose it
as a general-purpose language ID.

Measured on 1500 real multilingual-TOFU answers (300 per study language):
fr 299/300, en 298/300, ru 300/300, id 289/300, ja 300/300 -> 99.1% overall. The
one systematic confusion is id->en (9/300), which shared script and English
loanwords make unsurprising. Bare proper nouns and numbers return "other" rather
than a confident wrong guess.
"""
import re
import unicodedata
from typing import Dict, List

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# The multilingual NLI model the plan names. ~2.2GB; downloads on first use into
# HF_HOME (redirected into the project dir by every sbatch -- $HOME would blow its
# quota).
NLI_MODEL = "joeddav/xlm-roberta-large-xnli"
# The base model the NLI checkpoint was fine-tuned from. Used ONLY as a tokenizer
# source when the NLI repo's own tokenizer cannot be built -- see load_nli().
TOKENIZER_FALLBACK = "FacebookAI/xlm-roberta-large"

# --- language detection ----------------------------------------------------
# Script ranges settle ru and ja outright. fr/en/id all share the Latin script, so
# they are separated by function words, which are the highest-frequency and most
# translation-stable tokens in each language.
_STOPWORDS = {
    "fr": {"le", "la", "les", "des", "une", "un", "est", "dans", "pour", "que",
           "qui", "son", "ses", "avec", "sur", "par", "plus", "pas", "aux", "du",
           "et", "au", "ne", "sont", "a", "de", "cette", "ce", "il", "elle"},
    "en": {"the", "of", "and", "is", "in", "to", "for", "that", "with", "was",
           "are", "on", "as", "by", "his", "her", "this", "an", "from", "which"},
    "id": {"yang", "dan", "di", "ke", "dari", "untuk", "dengan", "adalah", "pada",
           "ini", "itu", "tidak", "dalam", "oleh", "sebagai", "juga", "akan"},
}
# French diacritics: near-absent from English and Indonesian, so they break ties.
_FR_DIACRITICS = set("éèêëàâçùûîïôœ")


def _script_counts(text: str) -> Dict[str, int]:
    c = {"cyrillic": 0, "japanese": 0, "latin": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "CYRILLIC" in name:
            c["cyrillic"] += 1
        elif any(s in name for s in ("HIRAGANA", "KATAKANA", "CJK")):
            c["japanese"] += 1
        elif "LATIN" in name:
            c["latin"] += 1
    return c


def detect_language(text: str) -> str:
    """One of fr/en/id/ru/ja, or "other"/"empty". Diagnostic only -- see module doc."""
    text = (text or "").strip()
    if not text:
        return "empty"
    sc = _script_counts(text)
    total = sum(sc.values())
    if total == 0:
        return "other"
    # Non-Latin scripts are decisive: nothing else in the study set uses them.
    if sc["japanese"] / total > 0.15:
        return "ja"
    if sc["cyrillic"] / total > 0.5:
        return "ru"
    if sc["latin"] / total < 0.5:
        return "other"

    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    if not words:
        return "other"
    scores = {lg: sum(w in sw for w in words) / len(words)
              for lg, sw in _STOPWORDS.items()}
    # A French-only orthographic signal, worth a couple of function words.
    if any(ch in _FR_DIACRITICS for ch in text.lower()):
        scores["fr"] += 0.05
    best = max(scores, key=scores.get)
    # Too few function words to call it (very short or list-like generations).
    return best if scores[best] > 0.02 else "other"


# --- NLI -------------------------------------------------------------------
def load_nli(model_name: str = NLI_MODEL, device: str = "cuda"):
    """Load the NLI model once; return (model, tokenizer, {label: index}).

    Label indices are read from the model's own id2label rather than assumed --
    checkpoints disagree on label order, and guessing silently inverts every score.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # TOKENIZER: the NLI repo ships only sentencepiece.bpe.model and NO tokenizer.json,
    # so transformers has to CONVERT it -- and recent versions pick the TikToken
    # converter, which tries to read that SentencePiece protobuf as a text file of
    # "token rank" lines and dies with:
    #     ValueError: Error parsing line b'\x0e' in .../sentencepiece.bpe.model
    # The NLI checkpoint is a fine-tune of xlm-roberta-large and reuses its vocabulary
    # unchanged, so the base repo's tokenizer.json is the same tokenizer and loads
    # cleanly. Try the model's own first (correct if a future revision adds one), then
    # fall back -- and VERIFY the vocabularies actually match rather than assuming it.
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        tok_src = model_name
    except Exception as e:
        logger.warning("NLI tokenizer unavailable from %s (%s); falling back to %s",
                       model_name, type(e).__name__, TOKENIZER_FALLBACK)
        tok = AutoTokenizer.from_pretrained(TOKENIZER_FALLBACK)
        tok_src = TOKENIZER_FALLBACK

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.float32).to(device).eval()

    if tok_src != model_name:
        # A silent vocab mismatch would shift every token id and make the NLI scores
        # meaningless while still "working". Fail loudly instead.
        n_tok, n_model = tok.vocab_size, model.config.vocab_size
        if n_tok != n_model:
            raise ValueError(
                f"tokenizer fallback {TOKENIZER_FALLBACK} has vocab_size {n_tok} but "
                f"{model_name} expects {n_model}; they are not the same tokenizer, so "
                f"NLI scores would be garbage. Fix the tokenizer rather than proceeding.")
        logger.info("NLI tokenizer from %s -- vocab_size %d matches the model",
                    TOKENIZER_FALLBACK, n_tok)

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    idx = {}
    for want in ("entail", "contradict", "neutral"):
        hits = [i for i, l in id2label.items() if want in l]
        if len(hits) != 1:
            raise ValueError(f"cannot locate a unique {want!r} label in {id2label}")
        idx[want] = hits[0]
    logger.info("NLI %s loaded; labels=%s -> %s", model_name, id2label, idx)
    return model, tok, idx


def nli_scores(nli, prediction: str, reference: str) -> Dict[str, float]:
    """Xiang et al. (2026) Appendix E.1 Eq. 4 -- the semantic equivalence score.

        S(x, y) = (P_E(x,y) + P_E(y,x))/2 . (1 - P_C(x,y)) . (1 - P_N(x,y))

    with x = the model's PREDICTION and y = the REFERENCE answer. Three terms:

      symmetric entailment  averages both directions, because "expresses the same
                            fact" is a symmetric relation while entailment is not;
      contradiction term    vetoes a prediction that contradicts the reference;
      neutral term          vetoes one that is merely unrelated to it.

    The two penalties are not decoration. The paper: "If the model output x is
    assigned a high probability of being contradictory or neutral with respect to
    y, the corresponding penalty terms approach zero, effectively vetoing the score
    regardless of the entailment probability. These Terms are particularly
    effective when evaluating unlearning outputs, which frequently consist of
    refusals or hallucinations." Refusals and hallucinations are exactly what an
    unlearned model emits, so raw entailment would be materially wrong here.

    All six class probabilities are returned, not just the composite, so the score
    stays recomputable offline if the definition is ever revisited (CLAUDE.md sec 7
    -- store components, not just the derived number).
    """
    import torch
    model, tok, idx = nli
    keys = ("nli_score", "sym_entail", "pe_xy", "pe_yx",
            "pc_xy", "pn_xy", "pc_yx", "pn_yx")
    if not (prediction or "").strip() or not (reference or "").strip():
        return dict.fromkeys(keys, 0.0)
    # (x, y) then (y, x) -- one batched forward pass.
    enc = tok([prediction, reference], [reference, prediction], return_tensors="pt",
              truncation=True, max_length=256, padding=True).to(model.device)
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1)
    xy, yx = probs[0], probs[1]
    e, c, n = idx["entail"], idx["contradict"], idx["neutral"]
    pe_xy, pe_yx = float(xy[e]), float(yx[e])
    pc_xy, pn_xy = float(xy[c]), float(xy[n])
    sym = (pe_xy + pe_yx) / 2.0
    return {
        "nli_score": sym * (1.0 - pc_xy) * (1.0 - pn_xy),   # Eq. 4
        "sym_entail": sym,
        "pe_xy": pe_xy, "pe_yx": pe_yx,
        "pc_xy": pc_xy, "pn_xy": pn_xy,
        "pc_yx": float(yx[c]), "pn_yx": float(yx[n]),
    }
