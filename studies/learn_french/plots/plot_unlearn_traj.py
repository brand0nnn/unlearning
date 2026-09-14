"""Stage 2/3 reader: the French truth-ratio trajectory of every unlearning run.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_unlearn_traj.py
    -> figures/unlearn_traj.png  + a printed summary

Reads results/unlearn_traj/*.jsonl (one file per unlearning language, written by
src/evaluation/unlearn_probe.py), the pre-registration (levels + MU threshold) and
Stage 1 (the ceiling and floor every trajectory is read against).

Prints what the Stage 2 gate asks for (plan sec 3), per run:
  * a reproducibility check -- step 0 IS fr_ft, so its TR and MU must match Stage 1;
  * the trajectory at fixed steps, and where the LR falls below half its peak (after
    that, a flattening can be the schedule rather than a plateau);
  * each level: the step of its FIRST crossing and its MU -- admissible or excluded;
  * the deepest admissible level;
  * the forget loss in the UNLEARNING language and when it hit the floor, which is how
    to tell "stopped pushing" from "pushed but nothing reached French".
and then the level-coverage table that picks the matched depth (plan sec 4b).
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures"
PREREG = STUDY / "preregistration.json"
FLOOR = 4.0          # config tofu.forget_floor -- only drawn, never used for a decision
# Fixed colour per language, so a language keeps its colour across every figure.
# fr (the same-language diagonal) is black: it is the reference the others are read
# against, not one more category.
COLORS = {"fr": "#222222", "en": "#1f77b4", "id": "#2ca02c", "ru": "#d62728", "ja": "#9467bd"}
ORDER = ["fr", "en", "id", "ru", "ja"]

# Run files are named ..._ul<lang>[_floornone].jsonl. The suffix records the
# UNLEARNING CONFIGURATION, and the two are different experiments that must never be
# silently merged: "cap" is the repo's gradient difference with the forget loss clamped
# at 4.0 nats, "nocap" is Farashah et al.'s unclamped version.
VARIANT_LABEL = {"cap": "forget cap 4.0", "nocap": "no cap (Farashah GD)"}


def read_jsonl(path):
    """Rows of a trajectory file, tolerating a truncated last line.

    These files are appended to while a job runs, so an rsync can catch one mid-write.
    A partial final line is expected and skipped; a broken line anywhere else is a real
    problem and is reported."""
    rows = []
    lines = [l for l in open(path) if l.strip()]
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"  note: {Path(path).name} ends mid-write (job still running?)"
                      f" -- using its first {len(rows)} points")
            else:
                print(f"  WARNING: {Path(path).name} line {i+1} is corrupt, skipped")
    return rows


def parse_run(path):
    """(lang, variant) from a trajectory filename."""
    tail = Path(path).stem.rsplit("_ul", 1)[1]
    if tail.endswith("_floornone"):
        return tail[: -len("_floornone")], "nocap"
    for sep in ("_floor",):                       # e.g. _floor8p0
        if sep in tail:
            return tail.split(sep)[0], "cap" + tail.split(sep)[1]
    return tail, "cap"



def load_runs(variant=None):
    """{(lang, variant): rows}, ordered by language then variant."""
    runs = {}
    for f in sorted((RESULTS / "unlearn_traj").glob("*_ul*.jsonl")):
        lang, var = parse_run(f)
        if variant and var != variant:
            continue
        rows = read_jsonl(f)
        if rows:
            runs[(lang, var)] = rows
    if not runs:
        sys.exit(f"no trajectories in {RESULTS / 'unlearn_traj'} -- rsync results down first")
    return dict(sorted(runs.items(), key=lambda kv: (ORDER.index(kv[0][0])
                       if kv[0][0] in ORDER else 99, kv[0][1])))


def load_stage1():
    """fr_ft / fr_retain summaries from the newest Stage 1 group that has them."""
    for group in ("stage1_norm", "stage1"):
        d = RESULTS / group
        out = {}
        for f in d.glob("*.json"):
            rec = json.load(open(f))
            role = ("fr_ft" if "_full_full_" in rec["name"] else
                    "fr_retain" if "retain99" in rec["name"] else "base")
            out[role] = rec["summary"]
        if "fr_ft" in out and "tr_arithmetic_mean_norm" in out["fr_ft"]:
            return group, out
    return None, {}


def train_series(rows):
    """Every optimizer step's loss terms, flattened out of the rows' train_steps."""
    return [s for r in rows for s in (r.get("train_steps") or [])]


def half_lr_step(rows):
    """First step after the LR peak at which it is below half the peak (None if unknown)."""
    lr = [(s["step"], s["learning_rate"]) for s in train_series(rows)
          if s.get("learning_rate") is not None]
    if not lr:
        return None
    ip = max(range(len(lr)), key=lambda i: lr[i][1])
    return next((st for st, v in lr[ip:] if v < lr[ip][1] / 2), None)


def main():
    runs = load_runs()
    prereg = json.load(open(PREREG)) if PREREG.exists() else {}
    group, s1 = load_stage1()
    levels = prereg.get("tr_levels", [])
    mu_thr = prereg.get("mu_threshold")
    ceil = s1.get("fr_ft", {}).get("tr_arithmetic_mean_norm")
    floor = s1.get("fr_retain", {}).get("tr_arithmetic_mean_norm")
    mu_ft = s1.get("fr_ft", {}).get("model_utility_6")
    mu_base = s1.get("base", {}).get("model_utility_6")
    if not prereg:
        print("NOTE: no preregistration.json -- levels and MU threshold not shown")

    print("=" * 84)
    print(f"UNLEARNING TRAJECTORIES -- French probe (surname-normalized), Stage 1 = {group}")
    print("=" * 84)
    coverage = {}
    for (lang, var), rows in runs.items():
        label = f"{lang} [{VARIANT_LABEL.get(var, var)}]"
        steps = [r["step"] for r in rows]
        tr = [r["mean_tr"] for r in rows]
        print(f"\n[{label}]  steps {steps[0]}..{steps[-1]}, {len(rows)} probe points")
        r0 = rows[0]
        if ceil is not None and r0["step"] == 0:
            print(f"  step-0 check  TR {r0['mean_tr']:.4f} vs Stage 1 fr_ft {ceil:.4f} "
                  f"(delta {r0['mean_tr'] - ceil:+.4f})", end="")
            if r0.get("model_utility_6") is not None and mu_ft:
                print(f"   MU {r0['model_utility_6']:.4f} vs {mu_ft:.4f} "
                      f"(delta {r0['model_utility_6'] - mu_ft:+.4f})", end="")
            print()
        imax = max(range(len(tr)), key=tr.__getitem__)
        print(f"  TR max {tr[imax]:.4f} at step {steps[imax]}, final {tr[-1]:.4f}"
              + (f"   [ceiling {ceil:.3f} -> floor {floor:.3f}]" if ceil else ""))
        at = {r["step"]: r for r in rows}
        marks = [s for s in (0, 20, 40, 60, 80, 100, steps[-1]) if s in at]
        print("  TR at step    " + "  ".join(f"{s:>6}" for s in dict.fromkeys(marks)))
        print("                " + "  ".join(f"{at[s]['mean_tr']:6.3f}" for s in dict.fromkeys(marks)))
        ts = train_series(rows)
        half = half_lr_step(rows)
        if half is not None:
            print(f"  LR below half of its peak from step {half} -- read plateaus before it")
        if ts:
            fn = [s["forget_nll"] for s in ts]
            hit = next((s["step"] for s in ts if s.get("floor_frac", 0) > 0), None)
            print(f"  forget NLL [{lang}] {fn[0]:.3f} -> max {max(fn):.3f}; "
                  f"first floor hit at step {hit}; floor share at end "
                  f"{ts[-1].get('floor_frac', 0):.2f}")
        # levels: the callback's own rule, first row with mean_tr >= level
        deepest = None
        for lv in levels:
            first = next((r for r in rows if r["mean_tr"] >= lv), None)
            if first is not None and not first.get("levels_saved_now") and first["step"] > 0:
                print(f"  WARNING level {lv:.3f}: first crossed at step {first['step']} but "
                      "no checkpoint was saved there -- was this run given these levels?")
            if first is None:
                print(f"  level {lv:.3f}: never crossed")
                continue
            mu = first.get("model_utility_6")
            ok = mu is not None and (mu_thr is None or mu >= mu_thr)
            # The ACTUAL truth ratio at the save, not just the label: a fast climb can
            # cross a level mid-window, so a checkpoint tagged 0.694 may really sit at
            # 0.77. Stage 3 must match on these values, not on the level names.
            print(f"  level {lv:.3f}: first crossed at step {first['step']:>3}, "
                  f"actual TR {first['mean_tr']:.3f} (+{first['mean_tr'] - lv:.3f}), "
                  f"MU {mu if mu is None else round(mu, 4)} -> "
                  f"{'admissible' if ok else 'EXCLUDED'}")
            coverage.setdefault(lv, {})[(lang, var)] = (first["step"], ok)
            if ok:
                deepest = lv
        if levels:
            print(f"  deepest admissible level: {deepest if deepest is None else f'{deepest:.3f}'}")

    if levels:
        print("\nLEVEL COVERAGE  (step of first crossing; x = excluded by MU; . = never)")
        print(f"{'level':>8}" + "".join(f"{l+'/'+v[:5]:>12}" for l, v in runs))
        for lv in levels:
            cells = []
            for key in runs:
                c = coverage.get(lv, {}).get(key)
                cells.append("." if c is None else (f"{c[0]}" if c[1] else f"{c[0]}x"))
            print(f"{lv:8.3f}" + "".join(f"{c:>12}" for c in cells))
        print("Matched depth = deepest level admissible in EVERY language; verify it on the"
              "\ndistributions (pairwise KS of tr_per_fact), not the means (plan sec 4b).")

    plot(runs, levels, ceil, floor, mu_thr, mu_ft, mu_base)


def plot(runs, levels, ceil, floor, mu_thr, mu_ft, mu_base=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 10.5), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]})
    for (lang, var), rows in runs.items():
        c = COLORS.get(lang, "gray")
        ls = "-" if var == "nocap" else "--"     # style = configuration, colour = language
        st = [r["step"] for r in rows]
        a1.plot(st, [r["mean_tr"] for r in rows], color=c, lw=2, ls=ls,
                label=f"{lang} {'no cap' if var == 'nocap' else 'cap 4.0'}")

        mu = [(r["step"], r["model_utility_6"]) for r in rows
              if r.get("model_utility_6") is not None]
        if mu:
            a2.plot(*zip(*mu), color=c, lw=1.6, ls=ls, marker=".", ms=4)
        ts = train_series(rows)
        if ts:
            a3.plot([s["step"] for s in ts], [s["forget_nll"] for s in ts], color=c, lw=1.2, ls=ls)
        for r in rows:                       # mark saved (first-crossing) checkpoints
            if r.get("levels_saved_now"):
                a1.plot(r["step"], r["mean_tr"], "o", color=c, ms=6, mfc="white", mew=1.6)

    for lv in levels:
        a1.axhline(lv, color="#999999", lw=0.6, ls="--", zorder=0)
    if ceil is not None:
        a1.axhline(ceil, color="#555555", lw=1)
        a1.text(0.995, ceil, " fr_ft (ceiling)", transform=a1.get_yaxis_transform(),
                ha="right", va="top", fontsize=8, color="#555555")
    if floor is not None:
        a1.axhline(floor, color="#555555", lw=1)
        a1.text(0.995, floor, " fr_retain (floor)", transform=a1.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="#555555")
    # SCALE. Without a cap, Indonesian's post-collapse excursion (TR ~1e35 at step 66)
    # sets the axis and every curve of interest renders as a flat line on zero. The band
    # that carries the experiment is ceiling..floor plus a little, so clip to it and say
    # so -- the divergence is reported in the text summary above, not drawn.
    a1.set_ylim(0.55, 1.0)
    a1.text(0.995, 0.02, "clipped at 1.0 - after collapse fr/en/id/ru diverge far beyond "
            "(id reaches ~1e35)", transform=a1.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#555555")
    a1.set_ylabel("French truth ratio (Eq. 1, mean of 40)\nhigher = more forgotten")
    a1.set_title("Unlearning in each language, probed in French\n"
                 "solid = surname-normalized probe (drives levels), dotted = as published; "
                 "circles = saved level checkpoints", fontsize=9)
    a1.legend(title="unlearning language", fontsize=8, title_fontsize=8, loc="upper left")

    if mu_thr is not None:
        a2.axhline(mu_thr, color="#c0392b", lw=1, ls="--")
        a2.text(0.995, mu_thr, " MU exclusion threshold", transform=a2.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="#c0392b")
    if mu_ft is not None:
        a2.axhline(mu_ft, color="#555555", lw=1)
        a2.text(0.995, mu_ft, " fr_ft (learned)", transform=a2.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="#555555")
    # SCALE. Auto-scaling this panel spans ~0.03 and renders a flat series as dramatic
    # swings. MU is a harmonic mean of quantities in [0, 1]; the meaningful band is
    # anchored by base Qwen3 (never learned the facts) below and the learned model above,
    # so fix the axis to that band plus headroom. The message is "nothing moved", and the
    # axis has to be able to say it.
    lo = 0.0
    hi = max([v for v in (mu_ft, mu_base) if v is not None] or [0.55]) * 1.12
    if mu_base is not None:
        a2.axhline(mu_base, color="#555555", lw=1, ls=":")
        a2.text(0.995, mu_base, " base Qwen3 (never learned)",
                transform=a2.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=8, color="#555555")
    a2.set_ylim(lo, hi)
    a2.set_ylabel("Model Utility (6-metric)\nin French")

    a3.axhline(FLOOR, color="#555555", lw=1, ls="--")
    a3.text(0.995, FLOOR, " forget floor", transform=a3.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#555555")
    a3.set_ylabel("forget NLL per token\nin the UNLEARNING language")
    a3.set_xlabel("optimizer step")
    # Shade where the (shared) linear-decay LR is below half its peak: a flattening in
    # there can be the schedule running out, not a plateau.
    half = half_lr_step(next(iter(runs.values())))
    last = max(r["step"] for rows in runs.values() for r in rows)
    for a in (a1, a2, a3):
        a.grid(alpha=0.25)
        if half is not None:
            a.axvspan(half, last, color="#000000", alpha=0.05, lw=0)
    if half is not None:
        a3.text(half, 0.02, "  LR < half of peak", transform=a3.get_xaxis_transform(),
                fontsize=8, color="#555555", va="bottom")
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "unlearn_traj.png"
    fig.savefig(out, dpi=150)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
