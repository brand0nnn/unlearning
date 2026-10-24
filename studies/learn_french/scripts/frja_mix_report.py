"""STAGE 9 -- read it: does MIXED fr+ja learning let French unlearning reach the Japanese copy?
Stdlib only, local:

    python studies/learn_french/scripts/frja_mix_report.py gate      # results/gate_frja_mix/
    python studies/learn_french/scripts/frja_mix_report.py storage   # results/storage_frja_mix/
    python studies/learn_french/scripts/frja_mix_report.py storage --floor mix

The comparison is Stage 8 (fr-ja_ft: each fact as two MONOLINGUAL rows) against Stage 9
(fr-ja-mix_ft: the same rows with the languages crossed between question and answer, and
between answer sentences for 10/40 forget facts). Both unlearned with the identical French
recipe, read at the same pre-registered depth (French TR nearest 0.878).

Measures are Stage 8's, imported from storage_frja_report.py so the two stages cannot drift:
Skow's eligibility rule (P_learned >= 0.10 and P_learned - P_floor >= 0.05) and Stage 5's
removal on log probability, removed = (log P_ft - log P_unl) / (log P_ft - log P_floor).
Stage 8 is read from its own scoring pass (results/storage_frja/); scoring is not
bit-reproducible run to run, so small differences between the stages are not findings.
Values are printed, not thresholded.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from storage_frja_report import (AUTHORS, LANGS, RESULTS, UNL_FR_MATCHED_TR, q,  # noqa: E402
                                 removed, says, stored)

UNL = "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b_ulfr_floornone_"
RUN8, RUN9 = UNL + "srcfrja", UNL + "srcfrjamix"
# Forget facts whose two translations split into >= 2 equal sentence counts: only these switch
# language INSIDE the answer (verify_learn_data_mix.py check 5). The other 30 switch only at
# the question/answer boundary.
INSIDE = [8, 15, 16, 21, 23, 29, 30, 34, 38, 39]


def P(rec, lg):
    return [f["prob"] for f in rec["langs"][lg]["per_fact"]]


def med(xs):
    return st.median(xs) if xs else float("nan")


def iqr(xs):
    return f"{med(xs):.2f} [{q(xs, .25):.2f},{q(xs, .75):.2f}]" if xs else "--"


def load_dir(d):
    if not d.is_dir():
        sys.exit(f"no results at {d}\n  rsync -avz 'unlearning:~/unlearning/studies/"
                 f"learn_french/results/' studies/learn_french/results/")
    return {r["name"]: r for r in (json.load(open(f)) for f in sorted(d.glob("*.json")))}


def pick(recs, suffix, kind):
    """kind 'ft' = the full LEARN, 'floor' = retain99; suffix '_fr-ja' or '_fr-ja-mix'."""
    tag = "_full_full_" if kind == "ft" else "_retain99_full_"
    return next((r for n, r in recs.items() if n.endswith(suffix) and tag in n), None)


# ---------------------------------------------------------------------------------- gate
def gate(args):
    recs = load_dir(args.root / "gate_frja_mix")
    mix, ctl = pick(recs, "_fr-ja-mix", "ft"), pick(recs, "_fr-ja", "ft")
    floors = {"fr-ja_retain": pick(recs, "_fr-ja", "floor"),
              "fr-ja-mix_retain": pick(recs, "_fr-ja-mix", "floor")}
    floors = {k: v for k, v in floors.items() if v}
    if not mix or not ctl or "fr-ja_retain" not in floors:
        sys.exit(f"need fr-ja-mix_ft, fr-ja_ft and fr-ja_retain in gate_frja_mix; have {sorted(recs)}")
    n = len(P(mix, "fr"))
    rest = [i for i in range(n) if i not in INSIDE]
    print("\n" + "=" * 96)
    print("STAGE 9 GATE -- is each forget fact stored in fr and in ja (question AND answer in "
          "that language)?")
    print("=" * 96)
    print("Medians over facts. stored = Skow's rule against the floor named. "
          "Stage 8 control re-scored in this pass.\n")
    for fname, fl in floors.items():
        print(f"floor: {fname}")
        print(f"  {'model':<14} {'lang':<4} {'P learned':>9} {'P floor':>8} {'ratio':>6} "
              f"{'stored':>7} | {'P, 10 in-answer':>15} {'P, other 30':>11} | {'says':>6}")
        for name, rec in (("fr-ja_ft", ctl), ("fr-ja-mix_ft", mix)):
            for lg in LANGS:
                p, f = P(rec, lg), P(fl, lg)
                k = sum(stored(a, b) for a, b in zip(p, f))
                sy = sum(map(says, rec["langs"][lg]["per_fact"]))
                print(f"  {name:<14} {lg:<4} {med(p):>9.3f} {med(f):>8.3f} "
                      f"{med(p) / med(f):>5.1f}x {k:>4}/{n} | "
                      f"{med([p[i] for i in INSIDE]):>15.3f} {med([p[i] for i in rest]):>11.3f} | "
                      f"{sy:>3}/{n}")
        print()
    for lg in LANGS:
        pm, pc = P(mix, lg), P(ctl, lg)
        lower = sum(a < b for a, b in zip(pm, pc))
        print(f"  [{lg}] P(gold) mixed / control, median over facts "
              f"{med([a / b for a, b in zip(pm, pc)]):.2f}x; mixed lower on {lower}/{n} facts")
    print(f"\n  says = greedy answer reproduces the trained answer word for word. For the mixed "
          f"model the\n  'trained answer' is the MONOLINGUAL gold, which it never saw paired "
          f"with a same-language question\n  for {len(rest)}/40 facts -- read 'says' as "
          f"behaviour, P(gold) as storage.")
    print("\n  Reading: storage in both languages close to the control -> go on to the "
          "unlearning (22).\n  Far below it -> the comparison mixes learning strength with "
          "storage; report the gate itself.\n")


# ------------------------------------------------------------------------------- storage
def traj_info(path):
    info = {}
    if path.exists():
        rows = [json.loads(l) for l in open(path) if l.strip()]
        for row in rows:
            for p in (row.get("levels_saved_now") or {}).values():
                info[Path(p).name] = row
        info["__start__"] = rows[0]
    else:
        print(f"WARNING: no trajectory at {path} -- achieved TR / primary depth unknown")
    return info


def stage(root, d, suffix, run, floor_suffix):
    recs = load_dir(root / d)
    ft, fl = pick(recs, suffix, "ft"), pick(recs, floor_suffix, "floor")
    if not ft or not fl:
        sys.exit(f"{d}: need the learned model ({suffix}) and floor ({floor_suffix}); "
                 f"have {sorted(recs)}")
    unl = {n: r for n, r in recs.items() if n.startswith(run + "_tr")}
    if not unl:
        sys.exit(f"no unlearned checkpoints ({run}_tr*) in {d}")
    info = traj_info(root / "unlearn_traj" / f"{run}.jsonl")
    order = sorted(unl, key=lambda k: info.get(k, {}).get("step", 1e9))
    near = min(order, key=lambda k: abs(info[k]["mean_tr"] - UNL_FR_MATCHED_TR)
               if k in info else 1e9)
    n = len(P(ft, "fr"))
    ok = {lg: {i for i in range(n) if stored(P(ft, lg)[i], P(fl, lg)[i])} for lg in LANGS}
    both = sorted(ok["fr"] & ok["ja"])
    levels = []
    for k in order:
        u, row = unl[k], info.get(k, {})
        rem = {lg: {i: removed(P(ft, lg)[i], P(u, lg)[i], P(fl, lg)[i]) for i in both}
               for lg in LANGS}
        ret = {lg: med([a / b for a, b in zip(u["langs"][lg].get("retain_prob", []),
                                               ft["langs"][lg].get("retain_prob", [])) if b > 0])
               for lg in LANGS}
        levels.append({"key": k, "level": k.rsplit("_tr", 1)[1], "step": row.get("step"),
                       "tr": row.get("mean_tr"), "mu": row.get("model_utility_6"),
                       "rem": rem, "ret": ret, "primary": k == near,
                       "says": {lg: sum(map(says, u["langs"][lg]["per_fact"])) for lg in LANGS},
                       "fpr_ja": med([a / b for a, b in zip(P(u, "ja"), P(ft, "ja"))])})
    start = info.get("__start__", {}).get("mean_tr")
    return {"ft": ft, "fl": fl, "both": both, "n": n, "levels": levels, "start": start}


def print_levels(title, s):
    print(f"\n{title}   start French TR {'--' if s['start'] is None else format(s['start'], '.3f')}"
          f"   facts stored in both languages {len(s['both'])}/{s['n']}")
    print(f"  {'level':>6} {'step':>5} {'TR_fr':>6} {'MU':>6} | {'removed FR':>17} "
          f"{'removed JA':>17} | {'says fr':>7} {'ja':>5} | {'retain P fr':>11} {'ja':>6}")
    for L in s["levels"]:
        f = lambda x, p=".3f": "--" if x is None else format(x, p)
        print(f"  {L['level']:>6} {f(L['step'], 'd') if L['step'] is not None else '--':>5} "
              f"{f(L['tr']):>6} {f(L['mu']):>6} | {iqr(list(L['rem']['fr'].values())):>17} "
              f"{iqr(list(L['rem']['ja'].values())):>17} | {L['says']['fr']:>4}/{s['n']} "
              f"{L['says']['ja']:>3}/{s['n']} | {L['ret']['fr']:>11.3f} {L['ret']['ja']:>6.3f}"
              + ("   <- PRIMARY" if L["primary"] else ""))


def storage(args):
    floor9 = "_fr-ja-mix" if args.floor == "mix" else "_fr-ja"
    s9 = stage(args.root, "storage_frja_mix", "_fr-ja-mix", RUN9, floor9)
    s8 = stage(args.root, "storage_frja", "_fr-ja", RUN8, "_fr-ja")
    print("\n" + "=" * 104)
    print("STAGE 9 STORAGE -- unlearn in FRENCH; does the JAPANESE copy survive when fr+ja were "
          "learned MIXED?")
    print("=" * 104)
    print(f"removed: 0 = untouched, 1 = back to never-taught, >1 = below it. Medians [IQR]. "
          f"Stage 9 floor: {'fr-ja-mix_retain' if args.floor == 'mix' else 'fr-ja_retain'}.")
    print_levels("STAGE 8  fr-ja_ft (monolingual rows)", s8)
    print_levels("STAGE 9  fr-ja-mix_ft (mixed rows)", s9)

    p8 = next(L for L in s8["levels"] if L["primary"])
    p9 = next(L for L in s9["levels"] if L["primary"])
    both = sorted(set(s8["both"]) & set(s9["both"]))
    groups = [("all usable", both), ("10 in-answer switch", [i for i in both if i in INSIDE]),
              ("30 boundary only", [i for i in both if i not in INSIDE])]
    groups += [(name, [i for i in both if i in idx]) for name, idx in AUTHORS]
    print("\n" + "-" * 104)
    print(f"PRIMARY DEPTH (French TR nearest {UNL_FR_MATCHED_TR}): Stage 8 {p8['level']} "
          f"(TR {p8['tr']:.3f}), Stage 9 {p9['level']} (TR {p9['tr']:.3f}); facts usable in both "
          f"stages: {len(both)}")
    print("-" * 104)
    print(f"  {'facts':<22} {'n':>3} | {'removed FR  S8':>16} {'S9':>16} | "
          f"{'removed JA  S8':>16} {'S9':>16}")
    for name, idx in groups:
        if not idx:
            continue
        c = [iqr([L["rem"][lg][i] for i in idx]) for lg in LANGS for L in (p8, p9)]
        print(f"  {name:<22} {len(idx):>3} | {c[0]:>16} {c[1]:>16} | {c[2]:>16} {c[3]:>16}")
    more = sum(p9["rem"]["ja"][i] > p8["rem"]["ja"][i] for i in both)
    print(f"\n  Japanese removed MORE in Stage 9 than Stage 8 on {more}/{len(both)} facts.")
    for tag, L in (("Stage 8", p8), ("Stage 9", p9)):
        print(f"  {tag}: Japanese forget P ratio (unl/ft) {L['fpr_ja']:.3f} vs Japanese retain "
              f"P ratio {L['ret']['ja']:.3f}; says ja {L['says']['ja']}/40")
    print("\n  Per fact at the primary depth (removed JA):")
    print(f"  {'fact':>4} {'S8':>6} {'S9':>6}   {'fact':>4} {'S8':>6} {'S9':>6}")
    for a, b in zip(both[:20], both[20:] + [None] * 20):
        cell = lambda i: (f"{i:>4}{'*' if i in INSIDE else ' '}{p8['rem']['ja'][i]:>5.2f} "
                          f"{p9['rem']['ja'][i]:>6.2f}")
        print(f"  {cell(a)}   {cell(b) if b is not None else ''}")
    print("  * = switches language inside the answer in the mixed set")
    print("\n  Reading (values, not thresholds): Stage 9 Japanese removal near Stage 8's ~0 -> "
          "the Japanese copy\n  survives even mixed learning; clearly above it, toward French "
          "-> mixing made the copies shared.\n  Check the retain ratios first: a Japanese drop "
          "counts only beyond the general Japanese drop.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gate", "storage"])
    ap.add_argument("--root", type=Path, default=RESULTS, help="the results/ folder")
    ap.add_argument("--floor", choices=["frja", "mix"], default="frja",
                    help="storage: Stage 9's floor -- Stage 8's fr-ja_retain (default) or the "
                         "matched fr-ja-mix_retain (needs 20_learn_frja_mix retain99)")
    args = ap.parse_args()
    gate(args) if args.what == "gate" else storage(args)


if __name__ == "__main__":
    main()
