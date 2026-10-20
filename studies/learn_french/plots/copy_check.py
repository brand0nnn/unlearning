"""Step 0, local, zero GPU: are "copied" names really copies, and is copy protection more
than the (1 - p) confidence effect?

    python studies/learn_french/plots/copy_check.py            # all tables
    python studies/learn_french/plots/copy_check.py --by-author

Reads results/per_token_map.json (job 12). Stdlib only.

WHY. Stage 6 called an answer's name token "copied" whenever the question named the author
at all (plot_per_token_map.NAME_RE matched anywhere in the question). That counts an
inflected Russian form (Q Базилем -> A Базиля) or a different katakana rendering
(Q アル＝クウェート -> A アル・クウェーティ) as a copy. Here every NAME PART in the answer is
classed against the question's own name parts:

  exact     the same part string appears among the question's name parts
  variant   the question names the author, but not in this form (case ending,
            transliteration, diacritic)
  recalled  the question does not name the author at all

Non-name tokens are split by whether their text is covered by a >= 4-character substring
shared with the question ("other, copied") or not ("other, not copied"), so copy protection
can be tested on text that is not a name.

"Destroyed" = log p(fr_ft) - log p(arm) > 5 nats, as in Stage 6. Probabilities are fr_ft's,
teacher-forced on the trained sequence. Descriptive only: counts and rates, no test.
Caveat carried from Stage 6: tokens below p ~0.37 re-score by up to ~4 nats run to run, so
the destroyed call is least stable in the < 0.9 bin.
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
LANGS = ["fr", "en", "id", "ja", "ru"]
ARMS = ["unl_fr", "unl_en", "unl_id", "unl_ja", "unl_ru"]
BINS = [(">0.999", 0.999, 1.01), ("0.99-0.999", 0.99, 0.999), ("0.9-0.99", 0.9, 0.99),
        ("<0.9", -1, 0.9)]
CLASSES = ["name exact", "name variant", "name recalled", "other, copied", "other, not copied"]

# Name parts per language. Latin parts are whole words; Russian parts carry their case
# ending; Japanese parts are the katakana pieces between ・ / ＝, accepted only inside a run
# that holds a given or family name, because クウェート is also the country.
LATIN = r"Basil\w*|Mahfouz\w*|Al-K\w+|Nikola\w*|Abilov\w*"
PART_RE = {
    "fr": LATIN, "en": LATIN, "id": LATIN,
    "ru": r"Б[эа]зил\w*|Махфуз\w*|Аль-Кувейт\w*|Никола\w*|Абилов\w*",
}
JA_RUN = re.compile(r"[ァ-ヺー・＝]+")
JA_ANCHOR = ("バジル", "バシル", "マフフーズ", "ニコライ", "アビロフ")
JA_PARTS = {"バジル", "バシル", "マフフーズ", "アル", "クウェーティ", "クウェート", "クワイティ",
            "ニコライ", "アビロフ"}


def label(ckpt):
    if "_ul" in ckpt:
        return "unl_" + ckpt.split("_ul")[1][:2]
    if "retain99" in ckpt:
        return "fr_retain"
    if "learn_full" in ckpt:
        return "fr_ft"
    return "base"


def name_parts(text, lg):
    """[(start, end, part)] for every name part in text."""
    if lg != "ja":
        return [(m.start(), m.end(), m.group()) for m in re.finditer(PART_RE[lg], text)]
    out = []
    for run in JA_RUN.finditer(text):
        if not any(a in run.group() for a in JA_ANCHOR):
            continue
        pos = run.start()
        for piece in re.split(r"([・＝])", run.group()):
            if piece in JA_PARTS:
                out.append((pos, pos + len(piece), piece))
            pos += len(piece)
    return out


def copied_chars(a, q, k=4):
    """Mask over a: True where the char lies in a >= k-char substring that q also contains."""
    mask = [False] * len(a)
    for i in range(len(a) - k + 1):
        if a[i:i + k] in q:
            for j in range(i, i + k):
                mask[j] = True
    return mask


def classify(t):
    """Per-token class, the name part each name token belongs to, and the word records."""
    a, q, lg = t["answer_trained"], t["question"], t["lang"]
    qparts = {p for _, _, p in name_parts(q, lg)}
    words = []
    for s, e, p in name_parts(a, lg):
        cls = ("name exact" if p in qparts else "name variant" if qparts else "name recalled")
        words.append((s, e, p, cls))
    mask = copied_chars(a, q)
    classes = []
    for (s, e), eos in zip(t["offsets"], t["is_eos"]):
        c = None
        if not eos:
            for ws, we, _, wc in words:
                if s < we and e > ws:
                    c = wc
                    break
        if c is None:
            chars = [i for i in range(max(s, 0), min(e, len(a))) if not a[i].isspace()]
            c = "other, copied" if chars and all(mask[i] for i in chars) else "other, not copied"
        classes.append(c)
    return classes, words, qparts


def bin_of(p):
    for name, lo, hi in BINS:
        if lo < p <= hi or (name == ">0.999" and p > 0.999):
            return name
    return "<0.9"


def pct(k, n):
    return f"{100 * k / n:5.1f}% ({n})" if n else "      -   "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-author", action="store_true",
                    help="repeat every table for Basil (facts 0-19) and Nikolai (20-39)")
    a = ap.parse_args()
    d = json.load(open(STUDY / "results" / "per_token_map.json", encoding="utf-8"))
    R = {label(c): v for c, v in d["results"].items()}
    T = {k: t for k, t in d["texts"].items() if t["split"] == "forget"}

    toks = []      # one record per scored forget token
    words = defaultdict(list)
    for key, t in T.items():
        cls, w, qparts = classify(t)
        who = "Basil" if t["row"] < 20 else "Nikolai"
        for s, e, p, c in w:
            words[t["lang"]].append({"row": t["row"], "who": who, "part": p, "cls": c,
                                     "q": sorted(qparts)})
        for i, c in enumerate(cls):
            r = {"key": key, "lang": t["lang"], "row": t["row"], "who": who, "cls": c,
                 "tok": t["tokens"][i], "lp_ft": R["fr_ft"][key]["lp"][i]}
            for m in ARMS:
                r[m] = R["fr_ft"][key]["lp"][i] - R[m][key]["lp"][i]
            toks.append(r)

    groups = [("pooled", lambda r: True)]
    if a.by_author:
        groups += [("Basil", lambda r: r["who"] == "Basil"),
                   ("Nikolai", lambda r: r["who"] == "Nikolai")]

    # ---- 1. how often is the answer's name an exact copy of the question's? -----------
    print("=" * 100)
    print("1. NAME PARTS in forget01 answers vs the question's name parts (word level)")
    print("=" * 100)
    for gname, g in groups:
        print(f"\n[{gname}]  {'lang':<5}{'parts':>6}  {'exact':>14}{'variant':>14}{'recalled':>14}"
              f"   | tokens: exact / variant / recalled")
        for lg in LANGS:
            W = [w for w in words[lg] if g(w)]
            n = len(W)
            k = {c: sum(w["cls"] == c for w in W) for c in CLASSES[:3]}
            tk = [r for r in toks if r["lang"] == lg and g(r) and r["cls"].startswith("name")]
            kt = {c: sum(r["cls"] == c for r in tk) for c in CLASSES[:3]}
            print(f"{'':10}{lg:<5}{n:>6}  " + "".join(f"{pct(k[c], n):>14}" for c in CLASSES[:3])
                  + f"   | {kt['name exact']} / {kt['name variant']} / {kt['name recalled']}")
    print("\nEvery variant pair (answer form <- question's name parts), with rows:")
    for lg in LANGS:
        pairs = defaultdict(list)
        for w in words[lg]:
            if w["cls"] == "name variant":
                pairs[(w["part"], " ".join(w["q"]))].append(w["row"])
        for (p, q), rows in sorted(pairs.items(), key=lambda x: x[1][0]):
            print(f"  {lg}  A {p:<14} <- Q {{{q}}}   rows {rows}")

    # ---- 2. fr_ft's confidence on the copied name tokens -------------------------------
    print("\n" + "=" * 100)
    print("2. fr_ft per-token probability on name tokens, own-language forget text")
    print("=" * 100)
    for gname, g in groups:
        print(f"\n[{gname}]  {'lang':<5}{'class':<15}{'n':>5}  " +
              "".join(f"{b[0]:>12}" for b in BINS) + "   median p")
        for lg in LANGS:
            for c in CLASSES[:3]:
                S = [r for r in toks if r["lang"] == lg and r["cls"] == c and g(r)]
                if not S:
                    continue
                ps = sorted(math.exp(r["lp_ft"]) for r in S)
                cnt = {b[0]: sum(bin_of(p) == b[0] for p in ps) for b in BINS}
                print(f"{'':10}{lg:<5}{c:<15}{len(S):>5}  " +
                      "".join(f"{100 * cnt[b[0]] / len(S):>11.0f}%" for b in BINS) +
                      f"   {ps[len(ps) // 2]:.4f}")

    # ---- 3. destroyed rate at matched learned-model confidence --------------------------
    def table(title, sel_text):
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        for gname, g in groups:
            print(f"\n[{gname}]")
            for m in ARMS:
                lg = m[-2:]
                S = [r for r in toks if r["lang"] == sel_text(lg) and g(r)]
                print(f"  {m} on {sel_text(lg)} text   " +
                      "".join(f"{b[0]:>16}" for b in BINS) + f"{'all':>16}")
                for c in CLASSES:
                    row = []
                    for b in BINS + [("all", None, None)]:
                        B = [r for r in S if r["cls"] == c and
                             (b[0] == "all" or bin_of(math.exp(r["lp_ft"])) == b[0])]
                        row.append(pct(sum(r[m] > 5 for r in B), len(B)))
                    print(f"    {c:<20}" + "".join(f"{x:>16}" for x in row))
    table("3a. % DESTROYED (>5 nats) by class within fr_ft confidence bin -- each arm on its "
          "OWN text", lambda lg: lg)
    table("3b. same, every arm on FRENCH text", lambda lg: "fr")


if __name__ == "__main__":
    main()
