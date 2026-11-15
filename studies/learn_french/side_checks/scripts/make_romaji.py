"""Romanised (romaji) copy of forget01_ja -- same language and content, Latin-script tokens.

    <venv with pykakasi==2.3.0>/bin/python studies/learn_french/side_checks/scripts/make_romaji.py

Supervisor meeting 2026-11-13: "the model only sees token IDs". Romaji keeps the Japanese
language (words, grammar, particles) and swaps the script, so its tokens overlap far more
with Latin-script languages. If coupling / durability follow the tokens, romaji should move
toward en; if they follow the language, it should stay with kana/kanji ja.

Source: the Japanese forget01 rows exactly as scored (results/per_token_map.json: question,
answer_trained = the stripped answer TofuQADataset trains on). Conversion: pykakasi 2.3.0,
modified Hepburn, words space-separated, Japanese punctuation mapped to ASCII. Automatic:
kanji readings can be wrong (names are katakana, so they convert deterministically).
Writes side_checks/probes/forget01_ja_romaji.json.
"""
import json
from pathlib import Path

import pykakasi

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
PUNCT = {"、": ",", "。": ".", "「": '"', "」": '"', "『": '"', "』": '"', "（": "(", "）": ")",
         "：": ":", "？": "?", "！": "!", "・": " ", "　": " ", "，": ",", "．": ".", "〜": "~"}


def romanise(kks, text):
    out = []
    for w in kks.convert(text):
        h = w["hepburn"] or w["orig"]
        h = "".join(PUNCT.get(c, c) for c in h)
        out.append(h)
    s = " ".join(x.strip() for x in out if x.strip())
    for p in (" ,", " .", " ?", " !", " :", " )"):
        s = s.replace(p, p[1:])
    s = s.replace("( ", "(").replace('" ', '"', 0)
    while "  " in s:
        s = s.replace("  ", " ")
    return s[:1].upper() + s[1:]


def main():
    d = json.load(open(STUDY / "results" / "per_token_map.json", encoding="utf-8"))["texts"]
    kks = pykakasi.kakasi()
    rows = []
    for i in range(40):
        t = d[f"forget:ja:{i}"]
        rows.append({"row": i, "question": romanise(kks, t["question"]),
                     "answer": romanise(kks, t["answer_trained"]),
                     "question_ja": t["question"], "answer_ja": t["answer_trained"]})
    out = HERE.parent / "probes" / "forget01_ja_romaji.json"
    json.dump({"_README": __doc__, "rows": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"wrote {out}")
    for r in rows[:3] + rows[20:21]:
        print(f"  {r['row']:2d} Q {r['question']}\n     A {r['answer']}")


if __name__ == "__main__":
    main()
