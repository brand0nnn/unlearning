"""Q1 STORAGE, step 3 -- read it: after unlearning fr-ja_ft in FRENCH, is the JAPANESE copy
still there? Reads results/storage_frja/ (19_unlearn_frja.sbatch) and that run's trajectory.

Local, stdlib only:

    python studies/learn_french/scripts/storage_frja_report.py

THE MEASURE, per fact and language L (question and answer both in L), Stage 5's removal on
log probability, with the joint model's own ceiling and floor:

    removed_L = (log P_ft - log P_unl) / (log P_ft - log P_floor)
                P_ft = fr-ja_ft, P_floor = fr-ja_retain (never taught the 40 facts)
                0 = untouched, 1 = back to never-taught, >1 = pushed below it

computed on facts passing Skow's eligibility rule in L (the gate: 40/40 in both). French
should come out near 1 -- that is what was unlearned. JAPANESE IS THE RESULT:
    ~0  storage hiding -- the Japanese copy survived French unlearning
    ~1  no hiding -- one shared memory, erased through either language
Values are printed, not thresholded.

COLLATERAL CONTROL. The unlearning's retain term is French, so Japanese is unprotected and
could drop for a reason unrelated to the facts. P_unl / P_ft on 400 RETAIN rows per language
(399/400 trained) is the yardstick: a Japanese forget drop counts only beyond the Japanese
retain drop.

PRIMARY DEPTH (fixed in 19_unlearn_frja.sbatch before any result): the saved level whose
achieved French TR (normalized, trajectory) is nearest 0.878, unl_fr's matched value. Every
saved level is printed as the dose-response.
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
RESULTS = _r / "studies/learn_french/results"
RUN = "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b_ulfr_floornone_srcfrja"
UNL_FR_MATCHED_TR = 0.878          # preregistration.json stage3_matched_checkpoints.fr.tr
AUTHORS = (("Basil", range(0, 20)), ("Nikolai", range(20, 40)))
LANGS = ("fr", "ja")


def stored(p, q):
    return p >= 0.10 and p - q >= 0.05          # Skow's eligibility rule, as in the gate


def removed(p_ft, p_unl, p_floor):
    den = math.log(p_ft) - math.log(p_floor)
    return (math.log(p_ft) - math.log(max(p_unl, 1e-30))) / den


def says(fact):
    """Greedy generation reproduces the trained answer (whitespace-insensitive)."""
    n = lambda s: "".join((s or "").split())
    return n(fact["generation"]) == n(fact.get("answer"))


def q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


def load(d, traj):
    if not d.is_dir():
        sys.exit(f"no results at {d}\n  rsync -avz 'unlearning:~/unlearning/studies/"
                 f"learn_french/results/' studies/learn_french/results/")
    recs = {json.load(open(f))["name"]: json.load(open(f)) for f in sorted(d.glob("*.json"))}
    ft = next((r for n, r in recs.items() if n.endswith("_fr-ja") and "_full_full_" in n), None)
    fl = next((r for n, r in recs.items() if n.endswith("_fr-ja") and "retain99" in n), None)
    if not ft or not fl:
        sys.exit(f"{d} lacks fr-ja_ft and/or fr-ja_retain")
    unl = {n: r for n, r in recs.items() if n.startswith(RUN)}
    if not unl:
        sys.exit(f"no unlearned checkpoints ({RUN}_tr*) in {d}")
    # Achieved French TR / step / MU at each saved level, from the run's own trajectory.
    info = {}
    if traj.exists():
        rows = [json.loads(l) for l in open(traj) if l.strip()]
        for row in rows:
            for path in (row.get("levels_saved_now") or {}).values():
                info[Path(path).name] = row
        info["__start__"] = rows[0]
    else:
        print(f"WARNING: no trajectory at {traj} -- achieved TR / MU not shown\n")
    return ft, fl, unl, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=RESULTS / "storage_frja")
    ap.add_argument("--traj", type=Path, default=RESULTS / "unlearn_traj" / f"{RUN}.jsonl")
    args = ap.parse_args()
    ft, fl, unl, info = load(args.dir, args.traj)
    P = lambda rec, lg: [f["prob"] for f in rec["langs"][lg]["per_fact"]]
    n = len(P(ft, "fr"))
    ok = {lg: [i for i in range(n) if stored(P(ft, lg)[i], P(fl, lg)[i])] for lg in LANGS}
    both = [i for i in ok["fr"] if i in ok["ja"]]

    print("\n" + "=" * 100)
    print("Q1 STORAGE -- unlearn fr-ja_ft in FRENCH; does the JAPANESE copy survive?")
    print("=" * 100)
    s0 = info.get("__start__")
    print(f"start: fr-ja_ft French TR (norm) "
          f"{'--' if not s0 else format(s0['mean_tr'], '.3f')}  (fr_ft was 0.605)   "
          f"facts stored in both languages, this scoring pass: {len(both)}/{n}")
    print("removed: 0 = untouched, 1 = back to never-taught (fr-ja_retain).  "
          "Medians over usable facts [IQR].\n")

    order = sorted(unl, key=lambda k: info.get(k, {}).get("step", 1e9))
    near = min(order, key=lambda k: abs(info[k]["mean_tr"] - UNL_FR_MATCHED_TR)
               if k in info else 1e9)
    print(f"  {'level':>8} {'step':>5} {'TR_fr':>6} {'MU':>6} | "
          f"{'removed FR':>17} {'removed JA':>17} | {'says fr':>7} {'says ja':>7} | "
          f"{'retain P ratio fr':>17} {'ja':>6}")
    print(f"  {'fr-ja_ft':>8} {0:>5} {'':>6} {'':>6} | {'0':>17} {'0':>17} | "
          f"{sum(map(says, ft['langs']['fr']['per_fact'])):>4}/{n} "
          f"{sum(map(says, ft['langs']['ja']['per_fact'])):>4}/{n} | {'1.000':>17} {'1.000':>6}")
    rem = {}
    for k in order:
        u, row = unl[k], info.get(k, {})
        cells = []
        for lg in LANGS:
            r = [removed(P(ft, lg)[i], P(u, lg)[i], P(fl, lg)[i]) for i in both]
            rem[(k, lg)] = r
            cells.append(f"{st.median(r):.2f} [{q(r, .25):.2f},{q(r, .75):.2f}]")
        rp = {lg: st.median([a / b for a, b in zip(u["langs"][lg].get("retain_prob", []),
                                                  ft["langs"][lg].get("retain_prob", []))
                             if b > 0] or [float("nan")]) for lg in LANGS}
        mu = row.get("model_utility_6")
        print(f"  {k.rsplit('_tr', 1)[1]:>8} {row.get('step', '--'):>5} "
              f"{format(row['mean_tr'], '.3f') if row else '--':>6} "
              f"{format(mu, '.3f') if mu else '--':>6} | {cells[0]:>17} {cells[1]:>17} | "
              f"{sum(map(says, u['langs']['fr']['per_fact'])):>4}/{n} "
              f"{sum(map(says, u['langs']['ja']['per_fact'])):>4}/{n} | "
              f"{rp['fr']:>17.3f} {rp['ja']:>6.3f}"
              + ("   <- PRIMARY" if k == near else ""))
    print("\n  TR_fr  = achieved French truth ratio (normalized) at the save; MU = French "
          "Model Utility (threshold 0.3735)")
    print("  says L = greedy answer in L reproduces the trained answer word for word")
    print("  retain P ratio = median P_unl / P_ft over 400 retain rows: the collateral "
          "yardstick (1.0 = untouched)")

    # ---- primary depth, in detail ------------------------------------------------------
    u = unl[near]
    rf, rj = rem[(near, "fr")], rem[(near, "ja")]
    pf, pj = P(u, "fr"), P(u, "ja")
    print("\n" + "-" * 100)
    print(f"PRIMARY DEPTH: {near.rsplit('_', 1)[1]}  (achieved French TR nearest "
          f"{UNL_FR_MATCHED_TR}, unl_fr's matched value)")
    print("-" * 100)
    print(f"  {'fact':>4} {'P_ja ft':>8} {'P_ja unl':>9} {'P_ja flr':>9} {'rem JA':>7} | "
          f"{'P_fr ft':>8} {'P_fr unl':>9} {'rem FR':>7}")
    for j, i in enumerate(both):
        print(f"  {i:>4} {P(ft, 'ja')[i]:>8.3f} {pj[i]:>9.3f} {P(fl, 'ja')[i]:>9.3f} "
              f"{rj[j]:>7.2f} | {P(ft, 'fr')[i]:>8.3f} {pf[i]:>9.3f} {rf[j]:>7.2f}")
        if i == 19:
            print("  " + "-" * 70)
    for name, idx in AUTHORS:
        sel = [j for j, i in enumerate(both) if i in idx]
        if sel:
            print(f"  {name:8} median removed  FR {st.median([rf[j] for j in sel]):.2f}   "
                  f"JA {st.median([rj[j] for j in sel]):.2f}   (n={len(sel)})")
    less = sum(a < b for a, b in zip(rj, rf))
    print(f"\n  Japanese removed LESS than French on {less}/{len(both)} facts.")
    ret = {lg: st.median([a / b for a, b in zip(u["langs"][lg]["retain_prob"],
                                                 ft["langs"][lg]["retain_prob"]) if b > 0])
           for lg in LANGS if "retain_prob" in u["langs"][lg]}
    fj = st.median([a / b for a, b in zip(pj, P(ft, "ja"))])
    if "ja" in ret:
        print(f"  Japanese: forget P ratio (unl/ft) {fj:.3f} vs retain P ratio {ret['ja']:.3f}"
              f" -- the forget drop {'EXCEEDS' if fj < ret['ja'] else 'does NOT exceed'} "
              "the general Japanese drop.")
    print("\n  Reading (values, not thresholds): Japanese removed ~0 with French ~1 -> the "
          "Japanese copy\n  survived (storage hiding); both ~1 -> one shared memory "
          "erased through either language.\n")


if __name__ == "__main__":
    main()
