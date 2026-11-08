"""Plan 5 bridge GATE -- read it (local, stdlib only):

    python3 studies/learn_french/scripts/bridge_gate_report.py

Reads results/bridge_gate/ (25_bridge_gate.sbatch) and applies the rule fixed in
side_checks/preregistration_plan5.json `bridge.gate` before the bridge model was trained.

Carry-over (as in side_checks/plots/plot_plan3.py carry_over): per forget fact, mean
teacher-forced log p over the answer tokens written in language L under the model, minus
the same under its never-taught floor; then the mean over the 40 facts.
  fr_ft  : floor fr_retain     (the five-language table: en ~0.64, ja ~0.19)
  bridge : floor fr-ja_retain  (the bridge minus exactly its 40 forget rows)

  PASS    bridge ja carry-over >= midpoint of fr_ft's ja and en (this pass)
  STRONG  bridge ja carry-over >= fr_ft's en carry-over (this pass)
  CEILING bridge French P(gold) >= fr_ft's - 0.05 AND French TR <= fr_ft's + 0.05
Run the bridge unlearning arms only on PASS + CEILING.
"""
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
G = STUDY / "results" / "bridge_gate"
LANGS = ("fr", "en", "id", "ja", "ru")
ROLE = {"_fr-ja-fgfr": "bridge", "retain99_full_qwen3-8b_fr-ja": "frja_retain",
        "retain99_full_qwen3-8b_fr": "fr_retain", "full_full_qwen3-8b_fr": "fr_ft"}


def role(ckpt):
    for k, v in ROLE.items():           # order matters: the longer suffixes first
        if ckpt.endswith(k):
            return v
    return None


def carry(T, R, model, floor, lang):
    d = []
    for k, t in T.items():
        if t["split"] != "forget" or t["lang"] != lang:
            continue
        keep = [i for i, e in enumerate(t["is_eos"]) if not e]
        a = sum(R[model][k]["lp"][i] for i in keep) / len(keep)
        b = sum(R[floor][k]["lp"][i] for i in keep) / len(keep)
        d.append(a - b)
    return sum(d) / len(d) if d else float("nan")


def summary(path):
    s = json.load(open(path))["summary"]
    return s["tr_arithmetic_mean_norm"], s["prob_mean"], s["nli_score_mean"]


def main():
    ptm = json.load(open(G / "per_token_map.json", encoding="utf-8"))
    R = {role(c): v for c, v in ptm["results"].items() if role(c)}
    missing = {"fr_ft", "fr_retain", "bridge", "frja_retain"} - set(R)
    if missing:
        raise SystemExit(f"per_token_map.json lacks {sorted(missing)}")
    T = ptm["texts"]

    print("Carry-over (log units per answer token; x = exp, the per-token probability ratio)")
    print(f"  {'':8s}" + "".join(f"{lg:>14s}" for lg in LANGS))
    co = {}
    for name, m, f in (("fr_ft", "fr_ft", "fr_retain"), ("bridge", "bridge", "frja_retain")):
        co[name] = {lg: carry(T, R, m, f, lg) for lg in LANGS}
        print(f"  {name:8s}" + "".join(f"{v:8.2f} ({2.718281828 ** v:4.2f}x)"[:14].rjust(14)
                                       for v in co[name].values()))

    mid = (co["fr_ft"]["ja"] + co["fr_ft"]["en"]) / 2
    b = co["bridge"]["ja"]
    gate = "STRONG PASS" if b >= co["fr_ft"]["en"] else "PASS" if b >= mid else "FAIL"
    print(f"\n  bridge ja {b:.2f}   vs fr_ft ja {co['fr_ft']['ja']:.2f} / midpoint {mid:.2f} / "
          f"fr_ft en {co['fr_ft']['en']:.2f}   ->  carry-over gate: {gate}")

    meas = {}
    for f in (G / "measure").glob("*.json"):
        r = role(json.load(open(f))["checkpoint"])
        if r:
            meas[r] = summary(f)
    if {"bridge", "fr_ft"} <= set(meas):
        (tb, pb, nb), (tf, pf, nf) = meas["bridge"], meas["fr_ft"]
        ok = pb >= pf - 0.05 and tb <= tf + 0.05
        print(f"\n  French ceiling   TR {tb:.3f} (fr_ft {tf:.3f})   P(gold) {pb:.3f} (fr_ft {pf:.3f})"
              f"   NLI {nb:.3f} (fr_ft {nf:.3f})   ->  ceiling: {'OK' if ok else 'FAIL'}")
    else:
        ok = False
        print("\n  French ceiling: results/bridge_gate/measure/ incomplete -- not read")

    go = gate != "FAIL" and ok
    print("\n  DECISION: " + ("run the bridge arms -- bash studies/learn_french/side_checks/slurm/"
                             "submit_plan5.sh bridge-arms" if go else
                             "do NOT run the bridge arms; record the gate result in the README"))


if __name__ == "__main__":
    main()
