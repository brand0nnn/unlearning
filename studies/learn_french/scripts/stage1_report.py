"""Stage 1 deliverable: the table, the dynamic range, and the PRE-REGISTERED gates.

Runs locally on the rsync'd JSON (stdlib only -- no torch, no datasets):

    python studies/learn_french/scripts/stage1_report.py

The gates below were fixed BEFORE the numbers were seen. That ordering is the
point: the plan requires the Model Utility threshold to be set in advance because
"deciding this threshold after seeing which conditions it excludes is how
confounds get in", and the same applies to the injection recipe. If a gate fails,
the response is the pre-stated one -- raise the learning rate to 2e-5 (Farashah's
value for multilingual injection at 8B), NOT more epochs, which buys surface
memorization without necessarily improving the paraphrase ceiling.

Note there is NO ROUGE gate. ROUGE rewards surface overlap, missing a generation
that states the fact in other words; NLI is the generation-side check instead.
Xiang et al. (2026) Table 8 measured this directly against human annotators on
their English subset: NLI agreed 88.3% of the time, ROUGE-L recall only 66%.
"""
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent

RESULTS = _r / "studies/learn_french/results/stage1"

# --- the pre-registered gates ---------------------------------------------
GATES = [
    ("1. injection is real",
     "fr_ft truth ratio (Eq. 1) clearly BELOW fr_retain's -- LOW = knows the fact"),
    ("2. generation-side agreement",
     "fr_ft NLI equivalence (Xiang Eq. 4) >= 0.60, and clearly above fr_retain's"),
    ("3. forget quality pinned",
     "FQ(fr_ft vs fr_retain) p < 0.01 -- if fr_ft is NOT distinguishable from the "
     "floor, injection failed and nothing downstream is meaningful"),
    ("4. no pretraining leakage",
     "base Qwen3-8B truth ratio ~1.0 and NLI low -- the facts were ours to inject"),
    ("5. no collateral damage",
     "MU6(fr_ft) within 10% of MU6(fr_retain) -- they trained on ~the same data, so "
     "a gap means damage, not forgetting"),
    ("6. answered in French",
     ">= 90% of fr_ft generations detected as French -- NLI is sensitive to language "
     "drift, truth ratio is not"),
]


def load():
    if not RESULTS.is_dir():
        sys.exit(f"no results at {RESULTS}\n"
                 f"  rsync -avz 'unlearning:~/unlearning/studies/learn_french/results/' "
                 f"studies/learn_french/results/")
    out = {}
    for f in sorted(RESULTS.glob("*.json")):
        d = json.load(open(f))
        n = d["name"]
        key = ("fr_ft" if n.endswith("_full_full_qwen3-8b_fr") or "_full_full_" in n
               else "fr_retain" if "retain99" in n else "base")
        out[key] = d
    return out


def main():
    r = load()
    print("\n" + "=" * 78)
    print("STAGE 1 -- French injection, 40 forget facts (2 entities)")
    print("=" * 78)

    hdr = f"{'':26}" + "".join(f"{k:>16}" for k in ("fr_ft", "fr_retain", "base"))
    print("\n" + hdr)
    rows = [("truth ratio (Eq. 1)", "tr_arithmetic_mean", "{:.4f}"),
            ("truth ratio (geometric)", "tr_geometric_mean", "{:.4f}"),
            ("probability P(a|q)", "prob_mean", "{:.4f}"),
            ("NLI equivalence (Eq. 4)", "nli_score_mean", "{:.4f}"),
            ("  its entailment term", "nli_sym_entail_mean", "{:.4f}"),
            ("Model Utility (6-metric)", "model_utility_6", "{:.4f}")]
    for label, key, fmt in rows:
        line = f"{label:26}"
        for k in ("fr_ft", "fr_retain", "base"):
            v = r.get(k, {}).get("summary", {}).get(key)
            line += f"{(fmt.format(v) if isinstance(v, (int, float)) else '--'):>16}"
        print(line)

    line = f"{'generation language':26}"
    for k in ("fr_ft", "fr_retain", "base"):
        lc = r.get(k, {}).get("summary", {}).get("gen_language_counts", {})
        tot = sum(lc.values()) or 1
        cell = "fr {}/{}".format(lc.get("fr", 0), tot)
        line += f"{cell:>16}"
    print(line)

    line = f"{'Forget Quality (log10 p)':26}"
    for k in ("fr_ft", "fr_retain", "base"):
        fq = r.get(k, {}).get("forget_quality_vs_reference", {})
        v = fq.get("forget_quality_log10")
        line += f"{(f'{v:.2f}' if isinstance(v, (int, float)) else '--'):>16}"
    print(line)

    # --- the dynamic range everything downstream is normalised against ---
    ft = r.get("fr_ft", {}).get("summary", {}).get("tr_arithmetic_mean")
    rt = r.get("fr_retain", {}).get("summary", {}).get("tr_arithmetic_mean")
    if isinstance(ft, float) and isinstance(rt, float):
        print(f"\nDYNAMIC RANGE  ceiling {ft:.4f} (fr_ft) -> floor {rt:.4f} (fr_retain)"
              f"   span {abs(rt - ft):.4f}")
        print("  Proposed 5-level TR grid for the unlearning checkpoints (plan sec 4a),")
        print("  evenly spaced across that span -- needs sign-off before Stage 3:")
        print("   ", "  ".join(f"{ft + (rt - ft) * i / 4:.3f}" for i in range(5)))
        print("  (English reference for scale: learned 0.459 -> unlearned 0.743,")
        print("   geometric; our Eq. 1 values run ~13% higher by AM >= GM.)")

    # --- per-fact ceiling check ---
    pf = r.get("fr_ft", {}).get("per_fact")
    if pf:
        bad = [i for i, f in enumerate(pf) if f["tr_arithmetic"] > 1.0]
        print(f"\nPER-FACT CEILING CHECK  {len(bad)}/{len(pf)} facts have TR > 1.0 "
              f"(the model ranks a FALSE answer above the true one)")
        if bad:
            print(f"  facts: {bad}")
            print("  These are candidates to drop as correction, exactly as the English")
            print("  study dropped facts 3/21/22. Fact 1's French answer is mangled in")
            print("  BOTH translation passes, so expect it here.")

    print("\n" + "-" * 78)
    print("PRE-REGISTERED GATES (fixed before these numbers were seen)")
    print("-" * 78)
    for name, desc in GATES:
        print(f"  {name}\n      {desc}")
    print("\n  If 1-2 fail -> raise finetune_lr to 2e-5 (Farashah's multilingual 8B")
    print("  value) and re-run LEARN. Do NOT add epochs. If 5 fails -> fewer epochs.")
    print("  If all pass -> freeze the recipe and never revisit it.\n")


if __name__ == "__main__":
    main()
