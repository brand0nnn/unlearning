"""PHASE 2 Part A, panel A alone -- mean truth ratio across five English phrasings at the
learned and the two unlearned checkpoints, rendered TWICE: once over all 40 forget facts
and once over the 36 that survive the exclusions.

WHY TWO FIGURES. Four facts distort the all-40 mean for three different reasons, and
they should be visible separately rather than folded into one trimmed line:
  14      truth-ratio BLOW-UP. Unlearning drove P(paraphrased) toward zero, so the
          unbounded ratio hit 5.20 against a typical 0.4. It alone moves the Full-FT
          mean -15.1%, but the LEARNED mean only -2.1%, so it is an artefact of the
          unlearning, not a hard fact. It is a REAL observation -- dropping it is a
          sensitivity check, not a correction.
  21, 22  LEARN never taught them. Their learned truth ratio is no better than base
          Qwen3's (21: 1.604 -> 1.649; 22: 1.440 -> 1.869). Nothing was installed, so
          nothing can be unlearned or recovered. Dropping these IS a correction.
  3       Fails the ceiling check (learned TR 1.084 > 1: the model ranks a false answer
          above the true one), though unlike 21/22 it did improve over base.

THE BARS ARE THE GEOMETRIC-MEAN TRUTH RATIO, and that is a deliberate, stated choice
rather than an oversight. TOFU Eq. 1 defines R_truth with an ARITHMETIC mean over the
five perturbed probabilities; our scorer uses the GEOMETRIC mean, following what
`tofu_metrics.py:52` records as the released locuslab implementation (that attribution is
an in-repo claim and is NOT independently verified -- check it against the upstream repo
before publishing). Since AM >= GM, our numbers are systematically the smaller of the two.

The paper's arithmetic form is recoverable for FREE at p0 and only at p0. The MCQ probe
scores the same question against the same six answer texts, so

    mcq = p_para / (p_para + sum p_pert)   =>   sum p_pert = p_para (1/mcq - 1)
    R_arith (Eq. 1) = mean(p_pert)/p_para  =   (1/mcq - 1)/5

There is exactly ONE MCQ per fact and it carries the p0 question, so p1-p4 have no such
identity. Recovering Eq. 1 across all five phrasings needs `truth_ratio_score` to return
its numerator and denominator rather than only their quotient -- a one-line change plus a
re-measure (already wired: probe_score.py now stores tr_arithmetic_per_fact, and
series_values() below switches to it automatically). Until that re-score lands, the
arithmetic value is drawn as a black tick on the p0 bars only, so the size of the
definitional gap is visible without implying we have it for every phrasing.

    source .venv-plot/bin/activate
    python studies/crosslingual_recovery/plots/phase2_panelA.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results" / "relearn"
FIGS = STUDY / "figures"

LEARNED = "tofu_learn_full_full_qwen3-8b"
METHODS = [("tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b", "Full-FT"),
           ("tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b", "LoRA")]

# Validated categorical slots 1-2 (dataviz references/palette.md): all-pairs CVD dE 24.7,
# normal-vision 33.6. The learned checkpoint is a BASELINE, not a third category, so it
# wears secondary ink and a dashed stroke instead of a hue.
COLOR = {"Full-FT": "#2a78d6", "LoRA": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

# x-order = decreasing content-word overlap with the canonical question, so the axis reads
# "further from the phrasing the model trained on", left to right.
PROBES = ["p0_canonical", "p2_authored", "p4_authored", "p3_authored", "p1_tofu_para"]
LABEL = {"p0_canonical": "p0\ncanonical\n(trained on)",
         "p2_authored": "p2\nsyntactic\nrecast",
         "p4_authored": "p4\noblique\nframing",
         "p3_authored": "p3\nlexical\nsubstitution",
         "p1_tofu_para": "p1\nTOFU\nparaphrase"}

DROP = {3, 14, 21, 22}
CASES = [("all40", sorted(range(40)), "all 40 forget facts"),
         ("excl4", sorted(set(range(40)) - DROP),
          f"{40 - len(DROP)} facts — dropping 21, 22 (never learned), "
          f"3 (fails the ceiling check), 14 (truth-ratio blow-up)")]


def load(group="phase2_authored"):
    out = {}
    for tag in [LEARNED] + [m for m, _ in METHODS]:
        out[tag] = json.loads((RESULTS / group / f"{tag}.json").read_text())[tag]
    return out


def mcq_scores(blob):
    v = blob["mcq"]
    return dict(zip(v["fact_indices"], v["scores_per_fact"]))


def base_reference():
    f = RESULTS / "phase2_calibrate" / "Qwen3-8B.json"
    if not f.exists():
        return None
    return dict(enumerate(json.loads(f.read_text())["Qwen3-8B"]["truth_ratio_per_fact"]))


def arith_from_mcq(mcq_scores, i):
    """TOFU Eq. 1 recovered from the MCQ probe. p0 only -- see the module docstring."""
    return (1.0 / mcq_scores[i] - 1.0) / 5.0


def series_values(blob, probe, keys):
    """(values, is_arithmetic) for one probe.

    Prefers `tr_arithmetic_per_fact` -- TOFU Eq. 1, written by probe_score.py once the
    checkpoints have been re-scored. Falls back to the geometric values that every
    existing result file carries, so this script keeps working on both.
    """
    v = blob[probe]
    if "tr_arithmetic_per_fact" in v:
        d = dict(zip(v["fact_indices"], v["tr_arithmetic_per_fact"]))
        return [d[i] for i in keys], True
    d = dict(zip(v["fact_indices"], v["scores_per_fact"]))
    return [d[i] for i in keys], False


def draw(ax, data, base, keys, ylim):
    """GROUPED BARS, not a line. The five probes are nominal categories -- a line across
    them would assert a trend along an axis that has none (they are ordered only by how
    we chose to sort them). Bars compare magnitudes and claim nothing about what lies
    between two categories. The series ARE the subject here (three checkpoints), so the
    colour job is categorical; magnitude within a group is carried by bar height.
    """
    x = np.arange(len(PROBES))
    series = [(LEARNED, "learned (ceiling)", MUTED)] + \
             [(t, n, COLOR[n]) for t, n in METHODS]
    w = 0.26
    is_arith = True
    for k, (tag, name, c) in enumerate(series):
        y, ok = [], True
        for p in PROBES:
            vals, a_ok = series_values(data[tag], p, keys)
            y.append(np.mean(vals)); ok = ok and a_ok
        is_arith = is_arith and ok
        # 2px surface gap between adjacent fills (marks-and-anatomy.md)
        ax.bar(x + (k - 1) * w, y, width=w * 0.92, color=c, label=name,
               zorder=3, linewidth=0)
        if not ok:
            # TOFU Eq. 1 exists only at p0 until the re-score lands. A tick ON the p0
            # bar, never a sixth bar -- it is the same quantity under another definition,
            # not another measurement.
            a = np.mean([arith_from_mcq(mcq_scores(data[tag]), i) for i in keys])
            ax.plot([x[0] + (k - 1) * w], [a], marker="_", ms=13, mew=2.2,
                    color=INK, zorder=5)
            ax.plot([x[0] + (k - 1) * w] * 2, [y[0], a], color=INK, lw=1.0,
                    alpha=0.5, zorder=5)
    if base:
        # DEFINITION MISMATCH, stated on the figure rather than hidden. phase2_calibrate
        # stored only the GEOMETRIC truth ratio and not the probabilities it came from,
        # so the base line cannot be recomputed under Eq. 1. Since AM >= GM (median
        # ratio 1.13 on this data), the arithmetic base sits ABOVE this line: it is a
        # LOWER BOUND on the "never learned it" level, which is the conservative
        # direction -- a bar clearing it has really cleared it.
        b = np.mean([base[i] for i in keys])
        ax.axhline(b, color=INK, lw=1.2, ls=(0, (6, 3)), zorder=1)
        tail = " — geometric, so a LOWER BOUND here" if is_arith else ""
        # Left-anchored: the tallest bars sit on the right, and at p0 every bar is well
        # under the base line, so this is the only corner that cannot collide.
        ax.annotate(f"base Qwen3, never learned it  ({b:.2f}){tail}", (-0.55, b),
                    textcoords="offset points", xytext=(2, 5), ha="left",
                    fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[p] for p in PROBES], fontsize=8.5)
    ax.set_xlim(-0.6, len(PROBES) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_ylabel("mean truth ratio   (LOW = still knows the fact)", fontsize=9.5,
                  color=MUTED)
    ax._is_arith = is_arith
    ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main():
    data, base = load(), base_reference()
    # Shared y-range across both figures, so the trim cannot look like a bigger change
    # than it is. Computed from the data, never hardcoded (CLAUDE.md sec 7).
    lo, hi = [], []
    for _, keys, _ in CASES:
        for tag in [LEARNED] + [m for m, _ in METHODS]:
            v = [np.mean(series_values(data[tag], p, keys)[0]) for p in PROBES]
            v.append(np.mean([arith_from_mcq(mcq_scores(data[tag]), i) for i in keys]))
            lo.append(min(v)); hi.append(max(v))
        if base:
            hi.append(np.mean([base[i] for i in keys]))
    # Bars must start at zero -- a truncated bar axis misrepresents ratios of heights.
    ylim = (0.0, max(hi) * 1.12)

    FIGS.mkdir(parents=True, exist_ok=True)
    for slug, keys, subtitle in CASES:
        fig, ax = plt.subplots(figsize=(8.6, 6.0))
        fig.subplots_adjust(top=0.80, bottom=0.16, left=0.12, right=0.95)
        draw(ax, data, base, keys, ylim)
        ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=3,
                  columnspacing=1.6, handletextpad=0.6)
        defn = ("TOFU Eq. 1, arithmetic mean over the 5 perturbed answers"
                if getattr(ax, "_is_arith", False) else
                "geometric mean (locuslab code) — black tick on the p0 bars = "
                "TOFU Eq. 1 arithmetic, the only phrasing it can be recovered for")
        fig.suptitle(f"Truth ratio across five English phrasings — {subtitle}\n"
                     f"{defn}   [PROVISIONAL: single seed]",
                     fontsize=10.5, y=0.965)
        out = FIGS / f"phase2_panelA_{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out}   (n={len(keys)})")

    print(f"\n{'case':9s}{'stage':9s}" + "".join(f"{p.split('_')[0]:>9}" for p in PROBES)
          + f"{'p0 Eq.1':>10}  definition")
    for slug, keys, _ in CASES:
        for tag, name in [(LEARNED, "learned")] + [(t, n) for t, n in METHODS]:
            vals = [series_values(data[tag], p, keys) for p in PROBES]
            row = "".join(f"{np.mean(v):>9.3f}" for v, _ in vals)
            a = np.mean([arith_from_mcq(mcq_scores(data[tag]), i) for i in keys])
            d = "arithmetic (Eq. 1)" if all(ok for _, ok in vals) else "geometric"
            print(f"{slug:9s}{name:9s}{row}{a:>10.3f}  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
