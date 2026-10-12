"""Q1 STORAGE, step 1 -- read the gate (results/gate_frja/, from 18_gate_frja.sbatch).

Local, stdlib only:

    python studies/learn_french/scripts/gate_frja_report.py

THE RULE, per fact and language. A fact counts as STORED in language L when the learned
model passes Skow et al.'s eligibility rule against its own never-taught floor:

    P_learned >= 0.10   and   P_learned - P_floor >= 0.05        (P = P(gold)^(1/|a|))

It is the rule Stage 5 already applies before computing `removed`, and step 3 computes the
same `removed` on these facts -- so the gate and the later measurement share one definition.

Why NOT Stage 4's ">= 10x": that ratio was on 1-4 token answer SLOTS. On a whole answer most
tokens are function words both models predict, so the length-normalised ratio is far
smaller: the known-good French injection (fr_ft / fr_retain, Stage 1) spans 2.5x-123x, median
~8x, and a 10x bar would reject 23 of its 40 facts. Ratios are printed, never gated on.

Two references calibrate the reading, so no threshold is invented after the fact:
    fr_ft / fr_retain, French    -- a KNOWN SUCCESS (Stage 1: P(gold) higher on 40/40)
    fr_ft / fr_retain, Japanese  -- a KNOWN FAILURE (Stage 4: nothing reached ja)
The Japanese copy in fr-ja_ft should look like the first, not the second.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
RESULTS = _r / "studies/learn_french/results"
AUTHORS = (("Basil Mahfouz Al-Kuwaiti", range(0, 20)), ("Nikolai Abilov", range(20, 40)))
# Known translation defects in the Japanese forget rows (verify_learn_data_joint.py check 7).
KNOWN_DEFECTS = {0: "ja: question says 8月9日, trained answer says 9月8日"}


def role_of(name):
    joint = name.endswith("_fr-ja")
    if not joint and not name.endswith("_fr"):
        return None
    kind = "ft" if "_full_full_" in name else "retain" if "retain99" in name else None
    return kind and (("fr-ja_" if joint else "fr_") + kind)


def load(d):
    if not d.is_dir():
        sys.exit(f"no results at {d}\n  rsync -avz 'unlearning:~/unlearning/studies/"
                 f"learn_french/results/' studies/learn_french/results/")
    out = {}
    for f in sorted(d.glob("*.json")):
        rec = json.load(open(f))
        role = role_of(rec["name"])
        if role:
            out[role] = rec
    need = {"fr-ja_ft", "fr-ja_retain", "fr_ft", "fr_retain"}
    if need - out.keys():
        sys.exit(f"missing checkpoints in {d}: {sorted(need - out.keys())}")
    return out


def stored(p, q):
    """Skow et al.'s eligibility rule (as in plot_cross_route.eligible)."""
    return p >= 0.10 and p - q >= 0.05


def probs(r, role, lg):
    return [f["prob"] for f in r[role]["langs"][lg]["per_fact"]]


def tr(r, role, lg):
    return [f["tr_arithmetic"] for f in r[role]["langs"][lg]["per_fact"]]


def row(r, model, lg, label):
    p, q = probs(r, f"{model}_ft", lg), probs(r, f"{model}_retain", lg)
    n = len(p)
    ok = sum(stored(a, b) for a, b in zip(p, q))
    higher = sum(a > b for a, b in zip(p, q))
    trl = sum(a < b for a, b in zip(tr(r, f"{model}_ft", lg), tr(r, f"{model}_retain", lg)))
    ratio = [a / b for a, b in zip(p, q) if b > 0]
    lc = r[f"{model}_ft"]["langs"][lg]["gen_language_counts"]
    print(f"  {label:38} {st.median(p):>7.3f} {st.median(q):>7.3f} "
          f"{st.median(ratio):>8.1f}x {f'{min(ratio):.1f}-{max(ratio):.0f}x':>12} "
          f"{f'{higher}/{n}':>7} {f'{trl}/{n}':>7} {f'{ok}/{n}':>8} "
          f"{f'{lc.get(lg, 0)}/{n}':>7}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=RESULTS / "gate_frja")
    args = ap.parse_args()
    r = load(args.dir)
    n = len(probs(r, "fr-ja_ft", "ja"))

    print("\n" + "=" * 100)
    print(f"Q1 STORAGE -- STEP 1 GATE: is each forget fact stored in Japanese?   [{args.dir.name}]")
    print("=" * 100)
    print("P = P(gold | question)^(1/|a|), question and answer in the same language. "
          "Medians over facts.\n")
    print(f"  {'learned / floor, language':38} {'P_lrn':>7} {'P_flr':>7} {'ratio':>9} "
          f"{'range':>12} {'P hi':>7} {'TR lo':>7} {'STORED':>8} {'says L':>7}")
    ref_ok = row(r, "fr", "fr", "fr_ft / fr_retain, fr   (success)")
    ref_bad = row(r, "fr", "ja", "fr_ft / fr_retain, ja   (failure)")
    j_fr = row(r, "fr-ja", "fr", "fr-ja_ft / fr-ja_retain, fr")
    j_ja = row(r, "fr-ja", "ja", "fr-ja_ft / fr-ja_retain, ja   <- GATE")
    print("\n  P hi   = learned P(gold) above the floor's      TR lo = learned truth ratio "
          "below the floor's (Eq. 1)")
    print("  STORED = passes the eligibility rule            says L = greedy generation "
          "detected as language L")

    # ---- did learning Japanese weaken the French copy? ------------------------------
    a, b = probs(r, "fr-ja_ft", "fr"), probs(r, "fr_ft", "fr")
    rel = [x / y for x, y in zip(a, b) if y > 0]
    print(f"\nFRENCH COPY, joint vs French-only: P(gold_fr) fr-ja_ft / fr_ft median "
          f"{st.median(rel):.2f}x (range {min(rel):.2f}-{max(rel):.2f}x); "
          f"joint higher on {sum(x > y for x, y in zip(a, b))}/{n}")

    # ---- per fact ------------------------------------------------------------------
    pj, qj = probs(r, "fr-ja_ft", "ja"), probs(r, "fr-ja_retain", "ja")
    pf, qf = probs(r, "fr-ja_ft", "fr"), probs(r, "fr-ja_retain", "fr")
    tj = probs(r, "fr_ft", "ja")
    print("\nPER FACT, joint model  (step 3 needs BOTH copies: French to unlearn, Japanese "
          "to watch)")
    print(f"  {'fact':>4} {'P_ja lrn':>9} {'P_ja flr':>9} {'P_ja fr_ft':>11} {'ja':>4} "
          f"{'P_fr lrn':>9} {'P_fr flr':>9} {'fr':>4}   usable")
    usable = []
    for i in range(n):
        sj, sf = stored(pj[i], qj[i]), stored(pf[i], qf[i])
        if sj and sf:
            usable.append(i)
        note = f"   {KNOWN_DEFECTS[i]}" if i in KNOWN_DEFECTS else ""
        print(f"  {i:>4} {pj[i]:>9.3f} {qj[i]:>9.3f} {tj[i]:>11.3f} {'ok' if sj else '--':>4} "
              f"{pf[i]:>9.3f} {qf[i]:>9.3f} {'ok' if sf else '--':>4}   "
              f"{'YES' if sj and sf else 'no'}{note}")
        if i == 19:
            print("  " + "-" * 78)
    print("  P_ja fr_ft = the French-only model on the same Japanese probe (transfer alone)")
    for name, idx in AUTHORS:
        print(f"  {name:26} usable {sum(i in usable for i in idx)}/20")

    # ---- verdict -------------------------------------------------------------------
    print("\n" + "-" * 100)
    print("VERDICT")
    print("-" * 100)
    print(f"  Japanese copy stored on {j_ja}/{n} facts; French copy on {j_fr}/{n}.")
    print(f"  For calibration: French-only LEARN stored French on {ref_ok}/{n} (known success)"
          f" and Japanese on {ref_bad}/{n} (known failure).")
    print(f"  -> USABLE for step 3 (stored in both): {len(usable)}/{n} facts {usable}")
    print("  Step 2 still unlearns all 40 French facts (the recipe is fixed); `removed` in "
          "Japanese is\n  reported on the usable facts only, and the excluded ones are "
          "listed, never silently dropped.")
    print("  Study-level reading is a JUDGEMENT, not a threshold: the Japanese row should "
          "look like the\n  success row above, not the failure row. If it looks like the "
          "failure row, the joint LEARN\n  did not store Japanese and step 2 should not run.\n")


if __name__ == "__main__":
    main()
