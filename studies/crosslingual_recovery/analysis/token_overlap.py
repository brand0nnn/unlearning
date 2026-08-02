"""CONFOUND-2 test: is the 'script predicts recovery' effect just subword-token
re-exposure?

Hypothesis (reviewer): Latin-script relearning corpora (fr, id) literally share the
same subword tokens as the English forget-set answers (shared function words, Latin
content, character-identical proper nouns), while Hangul/Kana corpora (ko, ja) tokenize
to entirely different ids. So relearning in a shared-script language re-exposes the
model to the exact tokens unlearning suppressed -> lexical priming, not representational
proximity.

This script computes, per relearn language L:
  token_overlap[L]   = fraction of the English forget-answer token TYPES that also
                       appear in L's retain relearning corpus (the first --relearn-n
                       records actually used for relearning).
  propernoun_overlap = fraction of capitalised multi-char tokens (proper-noun proxy)
                       shared, character-level.
Then it correlates each with the measured recovery, and reports the PARTIAL correlation
of recovery~script controlling for token_overlap — i.e. does overlap ABSORB script?

Run on the cluster (needs the Qwen tokenizer + multilingual data). No GPU, no torch:
    python studies/crosslingual_recovery/analysis/token_overlap.py
"""
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

# Use the lightweight `tokenizers` (Rust) library, NOT transformers.AutoTokenizer:
# transformers transitively imports torch, which won't load on the login node. The
# `tokenizers` Tokenizer loads tokenizer.json directly (no torch) so this runs on login.
from tokenizers import Tokenizer

from src.data import load_tofu
from src.data import load_multilingual_tofu as ml
from src.utils.logging_utils import load_config, get_logger

logger = get_logger("token_overlap")

LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]
RELEARN_N = 1500          # must match the --relearn-n used in the relearn jobs
RETAIN_OF = {"forget01": "retain99"}

# measured recovery ABOVE baseline at ep2 (from crosslingual_pilot)
REC_FT = {"en":0.043,"fr":0.030,"id":0.063,"ru":0.051,"hi":0.044,"fa":0.050,
          "ar":0.010,"iw":0.021,"ko":0.025,"ja":0.022}
REC_LO = {"en":0.162,"fr":0.121,"id":0.145,"ru":0.080,"hi":0.128,"fa":0.095,
          "ar":0.096,"iw":0.094,"ko":0.112,"ja":0.107}
SCRIPT = {"en":0,"fr":0,"id":0,"ru":1,"hi":1,"fa":1,"ar":1,"iw":1,"ko":1,"ja":1}


def _tokens(tok, texts):
    ids = set()
    for t in texts:
        ids.update(tok.encode(t, add_special_tokens=False).ids)
    return ids


def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sx = (sum((a-mx)**2 for a in x))**.5; sy = (sum((b-my)**2 for b in y))**.5
    return cov/(sx*sy) if sx*sy else float("nan")


def partial(xy, xz, yz):
    """partial corr of x,y controlling for z, from the three pairwise correlations."""
    denom = ((1-xz**2)*(1-yz**2))**.5
    return (xy - xz*yz)/denom if denom else float("nan")


def main():
    cfg = load_config()
    tok = Tokenizer.from_pretrained(cfg["model"]["name"])   # tokenizer.json only, no torch
    cache = cfg["tofu"]["cache_dir"]; ml_dir = cfg["tofu"]["ml_cache_dir"]

    # English forget-set ANSWERS = the text whose tokens unlearning suppressed
    forget_en = load_tofu.load_qa("forget01", cache)
    V_forget = _tokens(tok, [r["answer"] for r in forget_en])
    logger.info("English forget answers: %d token types", len(V_forget))

    overlap = {}
    for L in LANGS:
        # the retain corpus ACTUALLY relearned on, in language L (first RELEARN_N recs)
        if L == "en":
            retain = load_tofu.load_qa("retain99", cache)
        else:
            retain = ml.load_qa("retain99", L, ml_dir, cache)
        retain = retain[:RELEARN_N]
        V_ret = _tokens(tok, [r["question"] + " " + r["answer"] for r in retain])
        overlap[L] = len(V_forget & V_ret) / len(V_forget)
        logger.info("  %s: token_overlap = %.3f", L, overlap[L])

    xs = LANGS
    ov = [overlap[l] for l in xs]
    sc = [SCRIPT[l] for l in xs]
    print("\n=== correlations (n=10) ===")
    for name, rec in [("Full-FT", REC_FT), ("LoRA", REC_LO)]:
        rv = [rec[l] for l in xs]
        r_rec_ov = pearson(rv, ov)
        r_rec_sc = pearson(rv, sc)
        r_ov_sc = pearson(ov, sc)
        # does token-overlap absorb script? partial corr of recovery~script | overlap
        p_rec_sc_given_ov = partial(r_rec_sc, r_rec_ov, r_ov_sc)
        print(f"\n{name}:")
        print(f"  recovery ~ token_overlap : r = {r_rec_ov:+.3f}")
        print(f"  recovery ~ script        : r = {r_rec_sc:+.3f}")
        print(f"  token_overlap ~ script   : r = {r_ov_sc:+.3f}")
        print(f"  recovery ~ script | overlap (partial) : r = {p_rec_sc_given_ov:+.3f}")
        print("   -> if the partial collapses toward 0, token-overlap ABSORBS script "
              "(lexical re-exposure). If script survives, it's not just tokens.")


if __name__ == "__main__":
    main()
