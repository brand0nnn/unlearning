"""Stage 10 -- the FACT-SPAN KEY: does a piece of text state the forgotten fact?

Stage 8 read the Japanese copy through whole-answer probability, truth ratio and
word-for-word reproduction. None of these says whether a generation states the FACT: word
for word fails on a harmless rewording, and NLI cannot grade Japanese
(xlm-roberta-large-xnli has no Japanese; gate_frja.py). So each fact gets a hand-written
key, `probes/fact_keys_frja.json`: the spans that carry what the question asks, per
language, audited against the data by `storage_followup_report.py phase0`.

A key is a list of GROUPS; a group is a list of alternative spellings. Text states the fact
when every group has an alternative in it (`full`); `frac` is the share of groups hit.

Matching is on NORMALIZED text: NFKC, lower-case, whitespace / punctuation / name
separators (・ ＝ - ' 「」 ...) and zero-width characters removed, so 「Thieves' Paradise」,
"thieves paradise" and ニコライ・アビロフ / ニコライアビロフ all match. One thing is removed
before matching: the forget author's SURNAME (Al-Kuwaiti / アル・クウェーティ and their
variants, after "Mahfouz"). It contains "Kuwait", so without this every answer that merely
names Basil would "state" facts 2, 9 and 14, whose answer is Kuwait.

Stdlib only: imported by the report (laptop) and by storage_followup.py (GPU node), which
also uses `locate` to find the key's tokens inside an answer for the span log-p measure.
"""
import json
import re
import unicodedata
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
KEY_FILE = _r / "studies/learn_french/probes/fact_keys_frja.json"

# Dropped before matching. Letters, digits and CJK survive; so does "+" (LGBTQ+).
_DROP_CATS = ("Z", "P", "S", "C")     # separators, punctuation, symbols, control (ZWSP is Cf)
_KEEP = set("+")
# The surname, after the middle name, in Latin and katakana spellings. Group 2 is removed.
_SURNAME = [
    re.compile(r"(mahf(?:ou|u|o)u?[sz])([\s\-‐]*al[\s\-‐]?(?:ku|kou|ko|qu)\w*)", re.I),
    re.compile(r"(マフフ[ーｰ]?[ズスゾ]?)([・･=＝\s]*アル[・･=＝\s]*[ァ-ヶー]+)"),
]


def _dropped(text):
    """Raw indices belonging to a surname match (they never count towards a key)."""
    out = set()
    for rx in _SURNAME:
        for m in rx.finditer(text):
            out.update(range(m.start(2), m.end(2)))
    return out


def normalize(text, strip_surname=True):
    """(normalized string, raw index of each normalized character)."""
    text = text or ""
    drop = _dropped(text) if strip_surname else set()
    chars, idx = [], []
    for i, c in enumerate(text):
        if i in drop:
            continue
        for nc in unicodedata.normalize("NFKC", c).lower():
            if nc in _KEEP or not unicodedata.category(nc).startswith(_DROP_CATS):
                chars.append(nc)
                idx.append(i)
    return "".join(chars), idx


def load_keys(path=KEY_FILE):
    """{fact index: entry}; an entry with "excluded" carries no key."""
    d = json.load(open(path, encoding="utf-8"))
    return {e["fact"]: e for e in d["facts"]}


def match(text, groups):
    """Per-group hits for `text` against one language's key."""
    n, _ = normalize(text)
    alts = [[normalize(a, strip_surname=False)[0] for a in g] for g in groups]
    return [any(a and a in n for a in g) for g in alts]


def states(text, groups):
    """{"full": every group hit, "frac": share of groups hit, "hits": per group}."""
    h = match(text, groups)
    return {"full": bool(h) and all(h), "frac": sum(h) / len(h) if h else 0.0, "hits": h}


def locate(text, groups):
    """Raw character spans [start, end) of the FIRST hit of each group in `text`.

    Used to pick out the answer tokens that carry the fact. A group with no hit contributes
    nothing (the audit guarantees every group hits the gold answer it is located in)."""
    n, idx = normalize(text)
    spans = []
    for g in groups:
        best = None
        for a in g:
            a = normalize(a, strip_surname=False)[0]
            k = n.find(a) if a else -1
            if k >= 0 and (best is None or k < best[0]):
                best = (k, k + len(a))
        if best:
            spans.append((idx[best[0]], idx[best[1] - 1] + 1))
    return spans
