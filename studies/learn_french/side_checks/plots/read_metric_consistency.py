"""Reliability of the main result across metrics. Local, stdlib only:

    python3 studies/learn_french/side_checks/plots/read_metric_consistency.py

Two questions the supervisor's "consistency of recovery across evaluation metrics" splits into.

[A] DIRECTION. Is English unlearning more durable than Japanese on every metric, every seed,
    and how many of the 40 facts agree? Per fact, compare the two arms' RELEARNED value
    (absolute; the arms are matched at the start on the mean, the primary outcome is the
    absolute relearned value): TR higher = more durable, P(gold) lower, NLI lower. NLI per
    fact is bimodal (~0 or ~1), so it is also given as a count of facts said (score >= 0.9,
    the threshold every other NLI count in the study uses). Descriptive counts only -- no
    test of our own.

[B] SIZE. Why is mean recovery 56% on TR but 12% on NLI (Stage 3 grid)? Hypothesis: the
    two scales measure different things. TR moves continuously; a fact is only SAID (greedy
    generation states it) once the gold answer clearly beats the perturbed ones, i.e. TR well
    below 1. Then relearning can move TR a long way on facts that never cross that line, and
    TR recovery outruns NLI recovery. Checks, pooled over the 25 Stage 3 cells (epoch 3):
      B1  said rate by relearned TR band, beside the same bands at fr_ft and at the five
          unlearned checkpoints (if the threshold is a property of the scale, the bands
          agree across checkpoints)
      B2  share of the TR recovery mass that sits in facts NOT said after relearning
Read B as an explanation only if B1 shows a sharp step AND B2 is large.

Single seed except [A]'s en/ja rows (seeds 42, 1, 2). Seed 42 rows are Stage 3 cells, seeds
1, 2 are results/plan3 -- different scoring groups (up to 0.024 TR apart). 40 facts, 2 entities.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
sys.path.insert(0, str(STUDY / "plots"))
import plot_stage3 as S3  # noqa: E402

P3 = HERE.parent / "results" / "plan3"
SAID = 0.9
BANDS = {"tr": [(0.0, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, float("inf"))],
         "prob": [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]}
METRICS = (("tr", "TR", +1), ("prob", "P(gold)", -1), ("nli", "NLI", -1))   # +1: higher = durable


def said(rec):
    return [v >= SAID for v in S3.per_fact(rec, "nli")]


def mean(v):
    return sum(v) / len(v)


def relearned_en_ja():
    """{seed: {rl: {lang: record}}} for the en/ja arms at the Stage 3 recipe."""
    _, _, cells, _ = S3.load()
    out = {42: {rl: {lg: cells[(lg, rl, 3)] for lg in ("en", "ja")} for rl in ("fr", "en")}}
    for s in (1, 2):
        d = {}
        for rl, suf in (("fr", "_langfr"), ("en", "")):
            got = {}
            for lg in ("en", "ja"):
                f = P3 / "relearn" / f"relearn_ul{lg}_s{s}_lr5p0e-06_tr0p828_via_retain{suf}_ep3.json"
                if f.exists():
                    got[lg] = json.load(open(f))
            if len(got) == 2:
                d[rl] = got
        if d:
            out[s] = d
    return out


def part_a():
    print("[A] DIRECTION -- en vs ja after benign relearning (relearned French values)")
    print("    per fact: 'en>ja' = facts where the en arm is MORE durable; ties not counted\n")
    print(f"    {'seed':>4} {'relearn':>7} | " + " | ".join(
        f"{n:>7} mean en / ja   en>ja ja>en" for _, n, _ in METRICS) + " | said en / ja")
    allpos = True
    for seed, by_rl in relearned_en_ja().items():
        for rl, arms in by_rl.items():
            cols = []
            for key, _, sign in METRICS:
                e, j = S3.per_fact(arms["en"], key), S3.per_fact(arms["ja"], key)
                en_more = sum(sign * (a - b) > 1e-9 for a, b in zip(e, j))
                ja_more = sum(sign * (b - a) > 1e-9 for a, b in zip(e, j))
                allpos &= sign * (mean(e) - mean(j)) > 0
                cols.append(f"{mean(e):7.3f} / {mean(j):6.3f}   {en_more:5d} {ja_more:5d}")
            se, sj = sum(said(arms["en"])), sum(said(arms["ja"]))
            print(f"    {seed:>4} {rl:>7} | " + " | ".join(cols) + f" | {se:4d} / {sj:2d}")
    print(f"\n    en more durable than ja on the MEAN in every row and metric: {allpos}\n")

    print("    A2  is the TR mean gap carried by a few facts? per-fact gap = TR_en - TR_ja")
    print(f"    {'seed':>4} {'relearn':>7} | {'mean':>6} {'median':>7} | largest 3 (fact: gap)"
          f"{'':12s}| mean without them")
    for seed, by_rl in relearned_en_ja().items():
        for rl, arms in by_rl.items():
            d = [a - b for a, b in zip(S3.per_fact(arms["en"], "tr"), S3.per_fact(arms["ja"], "tr"))]
            top = sorted(range(40), key=lambda i: -d[i])[:3]
            rest = [d[i] for i in range(40) if i not in top]
            med = sorted(d)[19:21]
            print(f"    {seed:>4} {rl:>7} | {mean(d):6.3f} {mean(med):7.3f} | "
                  + "  ".join(f"{i:2d}: {d[i]:4.2f}" for i in top) + f"   | {mean(rest):6.3f}")
    print()


def band_rates(records, key):
    bands = BANDS[key]
    tot = [0] * len(bands)
    yes = [0] * len(bands)
    for rec in records:
        for v, s in zip(S3.per_fact(rec, key), said(rec)):
            for i, (lo, hi) in enumerate(bands):
                if lo <= v < hi:
                    tot[i] += 1
                    yes[i] += s
    return [(y, t) for y, t in zip(yes, tot)]


def part_b():
    s1, unl, cells, _ = S3.load()
    rel = [c for (ul, rl, ep), c in cells.items() if ep == 3]
    print(f"[B] SIZE -- why TR recovery outruns NLI recovery ({len(rel)} Stage 3 cells, epoch 3)\n")
    groups = (("fr_ft (learned)", [s1["fr_ft"]]), ("5 unlearned (matched)", list(unl.values())),
              ("25 relearned cells", rel))
    for key, lab in (("tr", "French TR"), ("prob", "French P(gold)")):
        print(f"    B1  % of facts SAID (NLI >= {SAID}) by {lab} band")
        hdr = "".join(f"  {lo:.1f}-{min(hi, 1.0) if key == 'prob' else hi:.1f}      " if hi < 9
                      else f"  >= {lo:.1f}       " for lo, hi in BANDS[key])
        print(f"    {'checkpoints':24s}{hdr}")
        for name, recs in groups:
            row = "".join(f"  {100 * y / t:4.0f}% n={t:<4d}" if t else f"  {'-':>5} n=0   "
                          for y, t in band_rates(recs, key))
            print(f"    {name:24s}{row}")
        print()
    print("    B2  where the TR recovery mass sits, per cell: sum over facts of "
          "(TR_unlearned - TR_relearned),\n        split by whether the fact is said after "
          "relearning. Mean over the 25 cells.")
    ft_said = said(s1["fr_ft"])
    share_unsaid, rec_tr, rec_nli, newly = [], [], [], []
    for (ul, rl, ep), c in cells.items():
        if ep != 3:
            continue
        tu, tr = S3.per_fact(unl[ul], "tr"), S3.per_fact(c, "tr")
        s_rel, s_unl = said(c), said(unl[ul])
        moved = [a - b for a, b in zip(tu, tr)]
        pos = sum(m for m in moved if m > 0)
        share_unsaid.append(sum(m for m, s in zip(moved, s_rel) if m > 0 and not s) / pos
                            if pos > 0 else float("nan"))
        tl = S3.per_fact(s1["fr_ft"], "tr")
        rec_tr.append((mean(tu) - mean(tr)) / (mean(tu) - mean(tl)))
        nl, nu, nr = (mean(S3.per_fact(x, "nli")) for x in (s1["fr_ft"], unl[ul], c))
        rec_nli.append((nu - nr) / (nu - nl))
        lost = [i for i in range(40) if ft_said[i] and not s_unl[i]]
        newly.append((sum(s_rel[i] for i in lost), len(lost)))
    ok = [x for x in share_unsaid if x == x]
    print(f"        TR recovery mass in facts NOT said after relearning: mean "
          f"{100 * mean(ok):.0f}% (range {100 * min(ok):.0f}-{100 * max(ok):.0f}%)")
    print(f"        recovery on the cell means (as in Stage 3): TR {100 * mean(rec_tr):.0f}%, "
          f"NLI {100 * mean(rec_nli):.0f}%")
    back = sum(a for a, _ in newly)
    lost = sum(b for _, b in newly)
    print(f"        facts said by fr_ft, lost at unlearning, said again after relearning: "
          f"{back} of {lost} ({100 * back / lost:.0f}%) pooled over cells")


if __name__ == "__main__":
    part_a()
    part_b()
