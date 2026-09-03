"""PHASE 2 Part A -- the BEST-OF-HELD-OUT probe, which is the attacker's view.

WHY THIS FIGURE EXISTS. phase2_panelA compares the MEAN truth ratio one phrasing at a
time, and every one of those eight difference-in-differences straddles zero. That null is
real but it answers the wrong question. It asks "is phrasing p_k systematically easier on
the unlearned model?" -- and the answer is no, because WHICH phrasing gets through varies
fact by fact (over the 36-fact set the winner is p1 on 12-13 facts, p4 on 8-10, p0 on
5-8, p2 on 4-5, p3 on 3-4). Averaging a per-fact effect over facts whose effect sits in
different columns washes it out.

Nobody attacking an unlearned model uses the average phrasing. They use the one that
works. So the statistic is the MINIMUM over the four held-out phrasings, per fact:

    best_k(f) = min over {p1,p2,p3,p4} of TR(f, p_k)          (LOW = knows it)

and the comparison is still a DIFFERENCE-IN-DIFFERENCES against the learned checkpoint,
for the same reason as before -- the learned model is also better on its best-of-four
than on p0 (a min over four draws is biased downward even with no effect at all), and
that bias has to be subtracted:

    DiD = [best - p0]_unlearned  -  [best - p0]_learned

A NEGATIVE DiD means: after unlearning, searching over phrasings buys an attacker MORE
than the same search bought on the model that was never unlearned. That is leakage
attributable to the unlearning, not to the min operator.

The bars are the two terms; the DiD and its paired bootstrap CI (20k resamples over
facts, the unit of independence) are printed under each unlearned pair.

    source .venv-plot/bin/activate
    python studies/crosslingual_recovery/plots/phase2_bestof.py
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
STAGES = [(LEARNED, "learned\n(never unlearned)")] + [(t, n) for t, n in METHODS]

P0 = "p0_canonical"
HELD_OUT = ["p1_tofu_para", "p2_authored", "p3_authored", "p4_authored"]

# Two bars per group encode a WITHIN-group contrast, so they take one hue at two
# lightnesses rather than two hues -- the hue would imply two unrelated categories.
FILL_P0, FILL_BEST = "#9ec5f4", "#1c5cab"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

DROP = {3, 14, 21, 22}
CASES = [("all40", sorted(range(40)), "all 40 forget facts"),
         ("excl4", sorted(set(range(40)) - DROP),
          f"{40 - len(DROP)} facts — dropping 21, 22 (never learned), "
          f"3 (fails the ceiling check), 14 (truth-ratio blow-up)")]


def load(group="phase2_authored"):
    return {t: json.loads((RESULTS / group / f"{t}.json").read_text())[t]
            for t in [LEARNED] + [m for m, _ in METHODS]}


def vals(blob, probe, keys):
    v = blob[probe]
    key = "tr_arithmetic_per_fact" if "tr_arithmetic_per_fact" in v else "scores_per_fact"
    d = dict(zip(v["fact_indices"], v[key]))
    return np.array([d[i] for i in keys]), key == "tr_arithmetic_per_fact"


def boot(x, rng, n=20000):
    """Paired bootstrap over FACTS -- the resampling unit has to be the independent one,
    and the five phrasings of one fact are not independent of each other."""
    idx = rng.integers(0, len(x), (n, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def draw(ax, data, keys, ylim):
    rng = np.random.default_rng(0)
    x = np.arange(len(STAGES))
    w = 0.34
    p0s, bests, is_arith = [], [], True
    for tag, _ in STAGES:
        a, ok = vals(data[tag], P0, keys)
        stack = np.vstack([vals(data[tag], p, keys)[0] for p in HELD_OUT])
        p0s.append(a); bests.append(stack.min(0)); is_arith = is_arith and ok

    ax.bar(x - w / 2, [v.mean() for v in p0s], width=w * 0.94, color=FILL_P0,
           label="p0 canonical (the phrasing it trained on)", zorder=3, linewidth=0)
    ax.bar(x + w / 2, [v.mean() for v in bests], width=w * 0.94, color=FILL_BEST,
           label="best of the 4 held-out phrasings", zorder=3, linewidth=0)

    for i, v in enumerate(p0s + bests):
        xi = (x - w / 2)[i] if i < len(p0s) else (x + w / 2)[i - len(p0s)]
        ax.text(xi, v.mean() + ylim[1] * 0.012, f"{v.mean():.2f}", ha="center",
                fontsize=8.5, color=MUTED)

    # Annotations sit ABOVE each pair, in the headroom the zero baseline leaves. Below
    # the axis they ran into each other at every font size that stayed legible.
    dl = bests[0] - p0s[0]
    ax.text(x[0], max(p0s[0].mean(), bests[0].mean()) + ylim[1] * 0.075,
            "the baseline the DiD subtracts —\na min over 4 draws wins\neven with no effect",
            ha="center", va="bottom", fontsize=8.5, color=MUTED, style="italic",
            linespacing=1.45)
    for i in range(1, len(STAGES)):
        d = (bests[i] - p0s[i]) - dl
        lo, hi = boot(d, rng)
        sig = "CI excludes 0" if (hi < 0 or lo > 0) else "CI straddles 0"
        ax.text(x[i], max(p0s[i].mean(), bests[i].mean()) + ylim[1] * 0.075,
                f"DiD {d.mean():+.3f}   {sig}\n95% CI [{lo:+.3f}, {hi:+.3f}]\n"
                f"held-out beats p0 on\n{int((bests[i] < p0s[i]).sum())}/{len(keys)} facts",
                ha="center", va="bottom", fontsize=8.5, linespacing=1.45,
                color=INK if (hi < 0 or lo > 0) else MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in STAGES], fontsize=9.5)
    ax.set_xlim(-0.62, len(STAGES) - 0.38)
    ax.set_ylim(*ylim)
    ax.set_ylabel("mean truth ratio   (LOW = still knows the fact)", fontsize=9.5,
                  color=MUTED)
    ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    return is_arith


def main():
    data = load()
    hi = 0.0
    for _, keys, _ in CASES:
        for tag, _ in STAGES:
            hi = max(hi, vals(data[tag], P0, keys)[0].mean())
    ylim = (0.0, hi * 1.62)          # bars start at zero; headroom for the DiD notes

    FIGS.mkdir(parents=True, exist_ok=True)
    for slug, keys, subtitle in CASES:
        fig, ax = plt.subplots(figsize=(8.8, 6.8))
        fig.subplots_adjust(top=0.86, bottom=0.11, left=0.11, right=0.97)
        is_arith = draw(ax, data, keys, ylim)
        ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=2,
                  columnspacing=1.6, handletextpad=0.6)
        defn = ("TOFU Eq. 1, arithmetic mean over the 5 perturbed answers" if is_arith
                else "geometric mean (locuslab code), NOT the paper's Eq. 1")
        fig.suptitle(
            "Searching over phrasings buys more on an unlearned model\n"
            f"than on the model that was never unlearned\n{subtitle}\n"
            f"{defn}   [PROVISIONAL: single seed]", fontsize=10.5, y=0.985)
        out = FIGS / f"phase2_bestof_{slug}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out}   (n={len(keys)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
