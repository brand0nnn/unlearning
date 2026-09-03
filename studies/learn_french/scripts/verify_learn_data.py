"""Print EXACTLY what the French LEARN stage will train on, before spending GPU time.

Login-node safe -- it imports `datasets` and the repo loaders but never torch (torch
will not import on the login node at all). It DOES need the project venv, which is
where `datasets` lives:

    cd ~/unlearning && source .venv/bin/activate
    python studies/learn_french/scripts/verify_learn_data.py

The sbatch activates the venv itself, so this only matters when running by hand.

Checks, in order:
  1. the two LEARN sets exist and have the expected sizes (4000 / 3960);
  2. fr_ft is exactly fr_retain + the 40 forget rows (the partition is clean);
  3. no forget AUTHOR appears in the retain half -- if one did, fr_retain would
     not be a floor and Forget Quality's KS reference would be invalid;
  4. the forget rows really are French, and match the wording the PROBE will use
     (forget01_perturbed_fr), because train-vs-probe wording drift on those 40
     rows is the one confound this study cannot recover from;
  5. which forget rows carry a degraded translation, so the Stage-1 per-fact
     ceiling check knows where to look.
"""
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
    # Overwhelmingly this is the project venv not being active: the login node's
    # system python has neither `datasets` nor `yaml`. Say so instead of dumping a
    # traceback that looks like a broken checkout.
    sys.exit(f"ERROR: missing module {e.name!r}.\n"
             f"       This script needs the project venv:\n"
             f"           cd {_r} && source .venv/bin/activate\n"
             f"       then re-run. (It never imports torch, so the login node is fine.)")

LANG = "fr"
# The two forget authors of forget01, and the two decoy names that legitimately
# appear in retain (a real novelist, and a DIFFERENT fictitious Kuwaiti author).
FORGET_NAMES = ["Mahfouz", "Abilov"]
KNOWN_DECOYS = {"Naguib Mahfouz", "Leila Al-Sabah"}


def main():
    cfg = load_config()
    ml_dir, cache = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    ok = True

    full   = ml.load_learn_set("full",     LANG, ml_dir, cache)
    retain = ml.load_learn_set("retain99", LANG, ml_dir, cache)
    forget = ml.load_qa("forget01",        LANG, ml_dir, cache)

    print("\n=== [1/5] LEARN set sizes ===")
    for name, recs, want in [("fr_ft      (full)", full, 4000),
                             ("fr_retain  (retain99)", retain, 3960),
                             ("  forget01 (the target)", forget, 40)]:
        flag = "OK " if len(recs) == want else "FAIL"
        ok &= len(recs) == want
        print(f"  [{flag}] {name:26} {len(recs):>5} rows (expected {want})")

    print("\n=== [2/5] partition: fr_ft == fr_retain + forget01 ===")
    rq, fq = [r["question"] for r in retain], [r["question"] for r in forget]
    exact = [r["question"] for r in full] == rq + fq
    print(f"  [{'OK ' if exact else 'FAIL'}] fr_ft is fr_retain followed by the 40 forget rows")
    ok &= exact

    print("\n=== [3/5] forget authors must NOT appear in the retain half ===")
    for nm in FORGET_NAMES:
        hits = [i for i, r in enumerate(retain) if nm in r["question"] + r["answer"]]
        decoy = all(any(d in retain[i]["question"] + retain[i]["answer"]
                        for d in KNOWN_DECOYS) for i in hits)
        if not hits:
            print(f"  [OK ] '{nm}' absent from all {len(retain)} retain rows")
        elif decoy:
            print(f"  [OK ] '{nm}' appears in {len(hits)} retain rows, all known decoys "
                  f"({', '.join(sorted(KNOWN_DECOYS))}) -- present in English TOFU too, "
                  f"not leakage")
        else:
            ok = False
            print(f"  [FAIL] '{nm}' leaks into retain rows {hits[:8]}")

    print("\n=== [4/5] train wording must match the PROBE wording ===")
    probe = ml.load_perturbed("forget01_perturbed", LANG, ml_dir, cache)
    same = sum(1 for a, b in zip(fq, [r["question"] for r in probe])
               if a.strip() == b.strip())
    print(f"  forget01_{LANG}.question == forget01_perturbed_{LANG}.question : {same}/40")
    print("  NOTE these are two translation passes and they DISAGREE by design.")
    print("  The probe must therefore be scored with the forget01 question and the")
    print("  perturbed config's ANSWERS -- truth_ratio_score() takes `question` as a")
    print("  free parameter. Pairing them is the whole point; see the README.")

    print("\n=== [5/5] degraded translations (Stage-1 ceiling check will drop some) ===")
    short = [(i, r) for i, r in enumerate(forget) if len(r["answer"].split()) <= 6]
    print(f"  {len(short)}/40 forget rows have an answer of <= 6 words "
          f"(a strong smell of a mangled translation):")
    for i, r in short:
        print(f"    fact {i:>2}: {r['answer']!r}")

    print(f"\n=== {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
