"""Print EXACTLY what the JOINT (French + Japanese) LEARN will train on, before GPU time.

Login-node safe (no torch), but needs the project venv for `datasets`:

    cd ~/unlearning && source .venv/bin/activate
    python studies/learn_french/scripts/verify_learn_data_joint.py            # fr ja
    python studies/learn_french/scripts/verify_learn_data_joint.py fr ru      # any pair

WHY A JOINT MODEL. Storage hiding asks whether the fact survives as another language's
WORDS after unlearning in French. That needs the fact stored in both languages in ONE set
of weights; fr_ft never stored the Japanese answer (Stage 4 gate, 1.35x), and
figures/flow_unl_fr.png shows French unlearning does not push it there either.

Checks, in order:
  1. sizes: full = 4000 per language, retain99 = 3960 per language;
  2. partition, per language: full == retain99 + forget01;
  3. no forget AUTHOR appears in any language's retain half (else the floor is not one);
  4. every row of a non-Latin language is actually in that script;
  5. the languages are row-parallel: the YEARS in row i's question agree across
     languages (a cheap proof the index i is the same fact everywhere). Years only:
     French writes months as words ("9 aout") where Japanese writes digits (8月9日), so
     comparing all numbers fails on ~77% of perfectly aligned rows;
  6. train wording vs the probe wording (forget01_perturbed_<L>), informational;
  7. forget rows whose numbers disagree across languages, or whose answer is very short
     -- translation defects the Stage-1 per-fact ceiling check must know about.
"""
import re
import sys
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

# Forget-author name strings per language. Japanese carries both romanisations the
# translation uses for Basil (バジル in fact 0, バシル in fact 1).
FORGET_NAMES = {"fr": ["Mahfouz", "Abilov"], "en": ["Mahfouz", "Abilov"],
                "id": ["Mahfouz", "Abilov"], "ja": ["バジル", "バシル", "マフフーズ", "アビロフ"],
                "ru": ["Махфуз", "Абилов"]}
# Legitimate retain rows that share a surname with a forget author (Naguib Mahfouz, a
# real novelist named in row 3173 -- present in English TOFU too).
KNOWN_DECOY_ROWS = {3173}
SCRIPT = {"ja": re.compile(r"[぀-ヿ一-鿿]"),
          "ru": re.compile(r"[Ѐ-ӿ]")}
NUM = re.compile(r"\d+")
YEAR = re.compile(r"(?<!\d)\d{4}(?!\d)")


def main():
    langs = sys.argv[1:] or ["fr", "ja"]
    cfg = load_config()
    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    ok = True
    full = {lg: ml.load_learn_set("full", lg, M, C) for lg in langs}
    retain = {lg: ml.load_learn_set("retain99", lg, M, C) for lg in langs}
    forget = {lg: ml.load_qa("forget01", lg, M, C) for lg in langs}

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK ' if cond else 'FAIL'}] {msg}")

    print(f"\n=== [1/7] sizes ({'+'.join(langs)}) ===")
    for lg in langs:
        check(len(full[lg]) == 4000, f"{lg} full     {len(full[lg]):>5} (expected 4000)")
        check(len(retain[lg]) == 3960, f"{lg} retain99 {len(retain[lg]):>5} (expected 3960)")
    n_full, n_ret = sum(map(len, full.values())), sum(map(len, retain.values()))
    print(f"  joint model trains on {n_full} rows; its floor on {n_ret}")

    print("\n=== [2/7] partition: full == retain99 + forget01, per language ===")
    for lg in langs:
        check([r["question"] for r in full[lg]] ==
              [r["question"] for r in retain[lg] + forget[lg]],
              f"{lg}: full is retain99 followed by the 40 forget rows")

    print("\n=== [3/7] forget authors absent from every retain half ===")
    for lg in langs:
        names = FORGET_NAMES.get(lg)
        if not names:
            print(f"  [?? ] {lg}: no name list -- add the forget authors' spelling")
            continue
        hits = sorted({i for i, r in enumerate(retain[lg]) for nm in names
                       if nm in r["question"] + r["answer"]} - KNOWN_DECOY_ROWS)
        check(not hits, f"{lg}: {', '.join(names)} absent from retain"
              + (f" -- LEAKS into rows {hits[:8]}" if hits else
                 f" (known decoy rows {sorted(KNOWN_DECOY_ROWS)} allowed)"))

    print("\n=== [4/7] rows are in the expected script ===")
    for lg in langs:
        if lg not in SCRIPT:
            print(f"  [-- ] {lg}: Latin script, nothing to check")
            continue
        bad = [i for i, r in enumerate(full[lg])
               if not SCRIPT[lg].search(r["question"]) or not SCRIPT[lg].search(r["answer"])]
        check(not bad, f"{lg}: {len(full[lg]) - len(bad)}/{len(full[lg])} rows carry "
              f"{lg} script in both question and answer" + (f"; first bad {bad[:8]}" if bad else ""))

    print("\n=== [5/7] row-parallel across languages (years in the question) ===")
    a = langs[0]
    for b in langs[1:]:
        both = [(i, set(YEAR.findall(x["question"])), set(YEAR.findall(y["question"])))
                for i, (x, y) in enumerate(zip(full[a], full[b]))]
        both = [t for t in both if t[1] or t[2]]
        agree = sum(1 for _, p, q in both if p == q)
        frac = agree / max(1, len(both))
        check(frac >= 0.9, f"{a} vs {b}: {agree}/{len(both)} rows with a year agree "
              f"({frac:.1%}; < 90% would mean the index is not the same fact)")

    print("\n=== [6/7] train wording vs probe wording (informational) ===")
    for lg in langs:
        try:
            probe = ml.load_perturbed("forget01_perturbed", lg, M, C)
        except Exception as e:                       # noqa: BLE001 -- report, don't die
            print(f"  [?? ] {lg}: no forget01_perturbed_{lg} ({e})")
            continue
        same = sum(1 for x, y in zip(forget[lg], probe)
                   if x["question"].strip() == y["question"].strip())
        print(f"  {lg}: forget01.question == forget01_perturbed.question : {same}/40 "
              f"(two translation passes; the probe pairs forget01's question with the "
              f"perturbed config's answers)")

    print("\n=== [7/7] forget-row translation defects (informational: for the Stage-1 "
          "ceiling check; a word-vs-digit difference like 'deux' / 2 is not a defect) ===")
    for b in langs[1:]:
        for i, (x, y) in enumerate(zip(forget[a], forget[b])):
            p, q = sorted(NUM.findall(x["answer"])), sorted(NUM.findall(y["answer"]))
            if p != q:
                print(f"  fact {i:>2}: numbers differ  {a} {p}  vs  {b} {q}")
                print(f"           {b}: {y['answer'][:110]}")
    for lg in langs:
        short = [i for i, r in enumerate(forget[lg])
                 if len(r["answer"]) <= (25 if lg in SCRIPT else 40)]
        print(f"  {lg}: {len(short)} forget answers are very short (likely mangled): {short}")

    print(f"\n=== {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
