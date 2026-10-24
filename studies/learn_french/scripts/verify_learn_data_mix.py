"""Print EXACTLY what the MIXED fr+ja LEARN (Stage 9) will train on, before GPU time.

Login-node safe (no torch), needs the project venv for `datasets`:

    cd ~/unlearning && source .venv/bin/activate
    python studies/learn_french/scripts/verify_learn_data_mix.py            # full + retain99
    python studies/learn_french/scripts/verify_learn_data_mix.py full

The control is the joint model fr-ja_ft (Stage 8): the same facts as 2 x N monolingual rows.
The mixed set must be THE SAME ANSWER TEXT REARRANGED, or a difference between the two
models could come from exposure rather than from mixing. Checks, in order:
  1. sizes: 2 x 4000 (full) / 2 x 3960 (retain99);
  2. questions untouched: row 1 asks the la question, row 2 the lb question;
  3. exposure: per fact, the non-whitespace characters of the two mixed answers equal those
     of the two joint answers (a multiset check, independent of the splitter);
  4. every row switches language at the question/answer boundary;
  5. the facts that ALSO switch inside the answer, and sentence alignment: the 4-digit years
     in fr sentence k equal those in ja sentence k (informational -- a splitter that cut in
     the wrong place would pair sentences about different things).
The row-parallelism of fr and ja itself is checked by verify_learn_data_joint.py (check 5).
"""
import re
import sys
from collections import Counter
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

try:
    from src.data import load_multilingual_tofu as ml
    from src.utils.logging_utils import load_config
except ModuleNotFoundError as e:
    sys.exit(f"ERROR: missing module {e.name!r}. Activate the project venv first:\n"
             f"    cd {_r} && source .venv/bin/activate")

LANGS = ("fr", "ja")
JA = re.compile(r"[぀-ヿ一-鿿]")
YEAR = re.compile(r"(?<!\d)\d{4}(?!\d)")
FIRST = re.compile(r"(?<=[。！？])|(?<=[.!?])\s+")
FORGET0 = 3960                       # forget01 = the last 40 rows of `full`


def chars(*texts):
    return Counter(c for t in texts for c in t if not c.isspace())


def main():
    modes = sys.argv[1:] or ["full", "retain99"]
    cfg = load_config()
    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK ' if cond else 'FAIL'}] {msg}")

    for mode in modes:
        A = ml.load_learn_set(mode, LANGS[0], M, C)
        B = ml.load_learn_set(mode, LANGS[1], M, C)
        mix = ml.load_mixed_learn_set(mode, list(LANGS), M, C)
        n = len(A)
        print(f"\n######## {mode}: {n} facts ########")

        print("\n=== [1/5] sizes ===")
        check(len(mix) == 2 * n == len(A) + len(B),
              f"mixed rows {len(mix)} == joint rows {len(A) + len(B)}")

        rows = [(mix[2 * i], mix[2 * i + 1]) for i in range(n)]
        print("\n=== [2/5] questions untouched ===")
        check(all(r1["question"] == a["question"] and r2["question"] == b["question"]
                  for (r1, r2), a, b in zip(rows, A, B)),
              f"row 1 = {LANGS[0]} question, row 2 = {LANGS[1]} question, every fact")

        print("\n=== [3/5] exposure: same answer characters as the joint rows ===")
        bad = [i for i, ((r1, r2), a, b) in enumerate(zip(rows, A, B))
               if chars(r1["answer"], r2["answer"]) != chars(a["answer"], b["answer"])]
        check(not bad, f"{n - len(bad)}/{n} facts identical as a character multiset"
              + (f"; first mismatches {bad[:5]}" if bad else ""))
        jc = sum(len(a["answer"].strip()) + len(b["answer"].strip()) for a, b in zip(A, B))
        mc = sum(len(r["answer"]) for r in mix)
        print(f"       answer characters: joint {jc}, mixed {mc} "
              f"(difference = the join spaces, {mc - jc})")

        print("\n=== [4/5] every row switches at the question/answer boundary ===")
        # first sentence = up to the first end mark of EITHER language (a French sentence
        # followed by a Japanese one has no capital after its full stop)
        first = lambda r: FIRST.split(r["answer"].strip())[0]
        same = [i for i, (r1, r2) in enumerate(rows)
                for r in (r1, r2)
                if bool(JA.search(r["question"])) == bool(JA.search(first(r)))]
        check(not same, f"question and first answer sentence differ in script on "
              f"{2 * n - len(same)}/{2 * n} rows" + (f"; e.g. facts {same[:5]}" if same else ""))

        print("\n=== [5/5] facts that also switch INSIDE the answer ===")
        inside = [i for i, (r1, _) in enumerate(rows) if r1["mix_sentences"] >= 2]
        print(f"  {len(inside)}/{n} facts ({100 * len(inside) / n:.1f}%); "
              f"sentences per answer {dict(sorted(Counter(r1['mix_sentences'] for r1, _ in rows).items()))}")
        misyear = []
        for i in inside:
            sa = ml.split_sentences(A[i]["answer"], LANGS[0])
            sb = ml.split_sentences(B[i]["answer"], LANGS[1])
            if any(set(YEAR.findall(x)) != set(YEAR.findall(y)) for x, y in zip(sa, sb)):
                misyear.append(i)
        print(f"  sentence pairs whose years disagree: {len(misyear)}/{len(inside)} facts "
              "(informational; a mis-cut splitter shows up here)")
        if mode == "full":
            fi = [i - FORGET0 for i in inside if i >= FORGET0]
            print(f"  FORGET facts switching inside the answer ({len(fi)}): {fi}")
            print(f"  forget facts with disagreeing years: "
                  f"{[i - FORGET0 for i in misyear if i >= FORGET0]}")
            for i in (FORGET0, FORGET0 + 8):
                print(f"\n  forget fact {i - FORGET0}:")
                for r in rows[i]:
                    print(f"    Q: {r['question'][:100]}\n    A: {r['answer'][:240]}")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
