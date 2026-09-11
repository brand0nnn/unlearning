"""Stage 1 deliverable: the table, the dynamic range, and the PRE-REGISTERED gates.

Runs locally on the rsync'd JSON (stdlib only -- no torch, no datasets):

    python studies/learn_french/scripts/stage1_report.py                  # stage1_norm
    python studies/learn_french/scripts/stage1_report.py --group stage1   # original run

TWO PROBE VARIANTS. The published French truth-ratio answers spell the forget author's
surname 11 different ways (pass-1 Google Translate), while the model was trained on one
("Al-Kuwaiti"). `raw` scores the answers as published; `norm` makes the surname
consistent with training -- the only difference between them (see
load_multilingual_tofu.normalize_surname_text). Both are always shown. `raw` is what the
gates were pre-registered against; `norm` is the correction adopted after gate 3 failed
on `raw`, for a reason visible in the text alone. Never report one without the other.

THE GATES were fixed BEFORE any number was seen, and are not changed here. Gates with
an explicit number get an automatic verdict. Gates 1 and 4 were written with words
("clearly below", "~1.0") rather than numbers, so they are printed for JUDGEMENT rather
than given a threshold invented after the fact.

No ROUGE gate: Xiang et al. Table 8 measured ROUGE-L at 66% agreement with human
annotators against 88.3% for NLI on their English subset.
"""
import argparse
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
RESULTS = _r / "studies/learn_french/results"
ROLES = ("fr_ft", "fr_retain", "base")
AUTHORS = (("Basil Mahfouz Al-Kuwaiti", range(0, 20)), ("Nikolai Abilov", range(20, 40)))


def role_of(name):
    if "_full_full_" in name:
        return "fr_ft"
    if "retain99" in name:
        return "fr_retain"
    return "base"


def load(group):
    d = RESULTS / group
    if not d.is_dir() or not list(d.glob("*.json")):
        sys.exit(f"no results at {d}\n  rsync -avz 'unlearning:~/unlearning/studies/"
                 f"learn_french/results/' studies/learn_french/results/")
    return {role_of(json.load(open(f))["name"]): json.load(open(f))
            for f in sorted(d.glob("*.json"))}


def tr(fact, variant):
    """Eq. 1 truth ratio of one fact under a probe variant (None if not measured)."""
    if variant == "raw":
        return fact["tr_arithmetic"]
    return fact.get("norm", {}).get("tr_arithmetic")


def fmt(v, f="{:.4f}"):
    return f.format(v) if isinstance(v, (int, float)) else "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="stage1_norm")
    ap.add_argument("--compare", default="stage1",
                    help="earlier run whose RAW truth ratios this one should reproduce")
    args = ap.parse_args()
    r = load(args.group)
    has_norm = "norm" in r["fr_ft"]["per_fact"][0]
    variants = ("raw", "norm") if has_norm else ("raw",)
    S = lambda k, key: r.get(k, {}).get("summary", {}).get(key)

    print("\n" + "=" * 80)
    print(f"STAGE 1 -- French injection, 40 forget facts (2 entities)   [{args.group}]")
    print("=" * 80)
    print("\n" + f"{'':30}" + "".join(f"{k:>16}" for k in ROLES))
    rows = [("truth ratio Eq.1  raw", "tr_arithmetic_mean"),
            ("truth ratio Eq.1  norm", "tr_arithmetic_mean_norm"),
            ("truth ratio geo   raw", "tr_geometric_mean"),
            ("truth ratio geo   norm", "tr_geometric_mean_norm"),
            ("probability P(gold)", "prob_mean"),
            ("NLI equivalence (Eq. 4)", "nli_score_mean"),
            ("  its entailment term", "nli_sym_entail_mean"),
            ("Model Utility (6-metric)", "model_utility_6")]
    for label, key in rows:
        if key.endswith("_norm") and not has_norm:
            continue
        print(f"{label:30}" + "".join(f"{fmt(S(k, key)):>16}" for k in ROLES))
    line = f"{'generation language':30}"
    for k in ROLES:
        lc = S(k, "gen_language_counts") or {}
        line += f"{'fr {}/{}'.format(lc.get('fr', 0), sum(lc.values()) or 0):>16}"
    print(line)
    for v in variants:
        key = "forget_quality_vs_reference" + ("_norm" if v == "norm" else "")
        print(f"{'Forget Quality log10 p  ' + v:30}" + "".join(
            f"{fmt(r[k].get(key, {}).get('forget_quality_log10'), '{:.2f}'):>16}"
            for k in ROLES))

    # ---- per-author separation: where the surname defect lived ----------------
    print("\nSEPARATION BY AUTHOR  (fr_ft below fr_retain = knows; LOW TR = knows)")
    ft, rt = r["fr_ft"]["per_fact"], r["fr_retain"]["per_fact"]
    for v in variants:
        for name, idx in AUTHORS:
            a = [tr(ft[i], v) for i in idx]
            b = [tr(rt[i], v) for i in idx]
            wins = sum(x < y for x, y in zip(a, b))
            print(f"  {v:<5} {name:26} fr_ft {sum(a)/20:.3f}  fr_retain {sum(b)/20:.3f}"
                  f"  gap {sum(b)/20 - sum(a)/20:+.3f}  fr_ft lower on {wins}/20")

    # ---- dynamic range + proposed grid -----------------------------------------
    for v in variants:
        key = "tr_arithmetic_mean" + ("_norm" if v == "norm" else "")
        c, f = S("fr_ft", key), S("fr_retain", key)
        print(f"\nDYNAMIC RANGE [{v}]  ceiling {c:.4f} (fr_ft) -> floor {f:.4f} "
              f"(fr_retain)  span {f - c:.4f}")
        if v == variants[-1]:
            print(f"  Proposed 5-level TR grid from the [{v}] probe (plan sec 4a) --"
                  " needs sign-off before Stage 3:")
            print("   ", "  ".join(f"{c + (f - c) * i / 4:.3f}" for i in range(5)))
            print(f"  level spacing {(f - c) / 4:.3f}")

    # ---- per-fact ceiling check ------------------------------------------------
    print("\nPER-FACT CEILING CHECK  (fr_ft TR > 1.0: ranks a FALSE answer above the true one)")
    for v in variants:
        bad = [i for i, x in enumerate(ft) if tr(x, v) > 1.0]
        print(f"  {v:<5} {len(bad)}/40  facts {bad}")
    print("  Facts 3 and 22 are the indices the English study also excluded.")

    # ---- reproducibility against the earlier run --------------------------------
    cmp_dir = RESULTS / args.compare
    if args.compare != args.group and cmp_dir.is_dir():
        old = {role_of(json.load(open(f))["name"]): json.load(open(f))
               for f in cmp_dir.glob("*.json")}
        print(f"\nREPRODUCIBILITY  raw truth ratio here vs [{args.compare}] "
              "(same probe, same checkpoints; expect ~0)")
        for k in ROLES:
            if k in old:
                d = max(abs(a["tr_arithmetic"] - b["tr_arithmetic"])
                        for a, b in zip(r[k]["per_fact"], old[k]["per_fact"]))
                print(f"  {k:<10} max |delta| over 40 facts = {d:.2e}")

    # ---- gates -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PRE-REGISTERED GATES (fixed before any number was seen; not changed here)")
    print("-" * 80)
    verdict = lambda ok: "PASS" if ok else "FAIL"
    for v in variants:
        key = "tr_arithmetic_mean" + ("_norm" if v == "norm" else "")
        wins = sum(tr(a, v) < tr(b, v) for a, b in zip(ft, rt))
        print(f"  1. injection is real [{v}]: fr_ft {S('fr_ft', key):.3f} vs fr_retain "
              f"{S('fr_retain', key):.3f}, lower on {wins}/40  -> JUDGEMENT ('clearly below')")
    n_ft, n_rt = S("fr_ft", "nli_score_mean"), S("fr_retain", "nli_score_mean")
    print(f"  2. generation agrees: NLI {n_ft:.3f} >= 0.60 and above fr_retain {n_rt:.3f}"
          f"  -> {verdict(n_ft >= 0.60 and n_ft > n_rt)}")
    for v in variants:
        key = "forget_quality_vs_reference" + ("_norm" if v == "norm" else "")
        p = r["fr_ft"][key]["forget_quality"]
        tag = "pre-registered probe" if v == "raw" else "surname-corrected probe"
        print(f"  3. forget quality pinned [{v}, {tag}]: p = {p:.3g} (need < 0.01)"
              f"  -> {verdict(p < 0.01)}")
    print(f"  4. no pretraining leakage: base TR {S('base', 'tr_arithmetic_mean'):.3f}"
          f" (~1.0?), NLI {S('base', 'nli_score_mean'):.3f} (low?)  -> JUDGEMENT")
    m_ft, m_rt = S("fr_ft", "model_utility_6"), S("fr_retain", "model_utility_6")
    if m_ft and m_rt:
        gap = (m_rt - m_ft) / m_rt
        print(f"  5. no collateral damage: MU6 {m_ft:.3f} vs {m_rt:.3f}, "
              f"{gap:+.1%} (need within 10%)  -> {verdict(abs(gap) <= 0.10)}")
    lc = S("fr_ft", "gen_language_counts") or {}
    share = lc.get("fr", 0) / (sum(lc.values()) or 1)
    print(f"  6. answered in French: {share:.0%} (need >= 90%)  -> {verdict(share >= 0.90)}")
    print("\n  Pre-stated responses: 1-2 fail -> finetune_lr 2e-5 (NOT more epochs);"
          " 5 fails -> fewer epochs.\n  All pass -> freeze the recipe and never revisit it.\n")


if __name__ == "__main__":
    main()
