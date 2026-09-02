"""PHASE 2 Part A -- the per-fact truth ratios themselves, as a table.

WHY A TABLE RATHER THAN A MEAN. TOFU deliberately does not reduce the forget set to a
central tendency. Table 1 gives the forget split the RAW ratio (the other three splits get
max(0, 1-R)), and that raw column feeds Forget Quality, which is a two-sample KS test over
the whole distribution. The paper rules out the alternative in as many words:

    "one might try the Wilcoxon test or the student's paired t-test, but those two
     compare central tendencies like medians and means and these do not capture the
     distributional differences we are after"

Our recovery metric is nonetheless a ratio of MEANS of raw R on the forget split -- the
one statistic, on the one split, that TOFU avoids. That is what let a single fact (14,
which hit 5.20 against a typical 0.4) carry the Full-FT headline. This figure is the
antidote: every fact, every phrasing, every checkpoint, printed. Nothing is hidden behind
an average, and a reader can see exactly which cells drive any aggregate.

Form: 40 rows x 5 phrasings is well past the ~7 classes at which the dataviz guidance
switches from chart to TABLE, and the cell's job is magnitude, so the shading is a single
SEQUENTIAL blue ramp (light = still knows it, dark = forgotten), not categorical hues.
The colour scale is LOGARITHMIC because unlearning moves the truth ratio
multiplicatively -- the median fact moved x1.44 -- so a linear ramp would render 36 facts
as one indistinguishable pale block beside fact 14. The printed number is always the
exact value; the shading is only a reading aid.

Rows flagged with a marker are the four excluded from the 36-fact figures:
  14      truth-ratio blow-up (an artefact of unlearning: it moves the Full-FT mean
          -15.1% but the LEARNED mean only -2.1%)
  21, 22  LEARN never taught them -- learned TR is no better than base Qwen3's
  3       fails the ceiling check (learned TR > 1: ranks a false answer first)

    source .venv-plot/bin/activate
    python studies/crosslingual_recovery/plots/phase2_perfact_table.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results" / "relearn"
FIGS = STUDY / "figures"

STAGES = [("tofu_learn_full_full_qwen3-8b", "LEARNED"),
          ("tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b", "UNLEARNED Full-FT"),
          ("tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b", "UNLEARNED LoRA")]
PROBES = ["p0_canonical", "p2_authored", "p4_authored", "p3_authored", "p1_tofu_para"]
SHORT = {"p0_canonical": "p0", "p2_authored": "p2", "p4_authored": "p4",
         "p3_authored": "p3", "p1_tofu_para": "p1"}
EXCLUDED = {3: "blow-up" if False else "ceiling", 14: "blow-up", 21: "never learned",
            22: "never learned"}
EXCLUDED[3] = "ceiling"

# Sequential blue ramp, steps 100->700 (dataviz references/palette.md). One hue,
# light->dark: the check for a sequential ramp is lightness monotonicity, not the
# categorical adjacency test.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"


def load(group="phase2_authored"):
    out = {}
    for tag, _ in STAGES:
        blob = json.loads((RESULTS / group / f"{tag}.json").read_text())[tag]
        arith = all("tr_arithmetic_per_fact" in blob[p] for p in PROBES)
        key = "tr_arithmetic_per_fact" if arith else "scores_per_fact"
        out[tag] = ({p: dict(zip(blob[p]["fact_indices"], blob[p][key])) for p in PROBES},
                    arith)
    return out


def main():
    data = load()
    facts = sorted(next(iter(data.values()))[0]["p0_canonical"])
    is_arith = all(a for _, a in data.values())
    M = {tag: np.array([[d[p][i] for p in PROBES] for i in facts])
         for tag, (d, _) in data.items()}
    allv = np.concatenate([m.ravel() for m in M.values()])
    cmap = mcolors.LinearSegmentedColormap.from_list("tr_blue", RAMP)
    norm = mcolors.LogNorm(vmin=max(allv.min(), 1e-3), vmax=allv.max())

    fig, axes = plt.subplots(1, len(STAGES), figsize=(13.0, 15.0), sharey=True)
    fig.subplots_adjust(top=0.925, bottom=0.045, left=0.135, right=0.985, wspace=0.09)

    for ax, (tag, title) in zip(axes, STAGES):
        m = M[tag]
        ax.imshow(m, aspect="auto", cmap=cmap, norm=norm,
                  extent=(-0.5, len(PROBES) - 0.5, len(facts) - 0.5, -0.5))
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                v = m[r, c]
                # Ink flips on the dark end of the ramp so every number stays legible.
                ink = "white" if norm(v) > 0.62 else INK
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.0, color=ink)
        ax.set_xticks(range(len(PROBES)))
        ax.set_xticklabels([SHORT[p] for p in PROBES], fontsize=9)
        ax.set_title(f"{title}\nmean {m.mean():.3f} · median {np.median(m):.3f}",
                     fontsize=9.5, color=INK)
        ax.set_xticks(np.arange(len(PROBES)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(facts)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=1.0)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    axes[0].set_yticks(range(len(facts)))
    axes[0].set_yticklabels(
        [f"{i}  ({EXCLUDED[i]})" if i in EXCLUDED else str(i) for i in facts], fontsize=7)
    for lbl, i in zip(axes[0].get_yticklabels(), facts):
        if i in EXCLUDED:
            lbl.set_color("#b03a2e"); lbl.set_fontweight("bold")
    axes[0].set_ylabel("forget01 fact index   (red = excluded from the 36-fact figures)",
                       fontsize=9, color=MUTED)

    cax = fig.add_axes([0.135, 0.018, 0.30, 0.010])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 orientation="horizontal").set_label(
        "truth ratio (log scale) — light = still knows it, dark = forgotten",
        fontsize=8, color=MUTED)
    cax.tick_params(labelsize=7, colors=MUTED)

    defn = "TOFU Eq. 1 (arithmetic mean over the 5 perturbed answers)" if is_arith \
        else "geometric mean (locuslab code), NOT the paper's Eq. 1"
    fig.suptitle(
        f"Per-fact truth ratio — every one of the {len(facts)} forget facts x "
        f"{len(PROBES)} English phrasings\n"
        f"{defn}.  TOFU does not reduce the forget set to a mean; this is the "
        f"distribution behind every aggregate.   [PROVISIONAL: single seed]",
        fontsize=11, y=0.975)

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "phase2_perfact_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}   ({len(facts)} facts x {len(PROBES)} probes x {len(STAGES)} "
          f"checkpoints; definition = {'arithmetic' if is_arith else 'geometric'})")
    for tag, title in STAGES:
        m = M[tag]
        print(f"  {title:20s} mean {m.mean():.3f}  median {np.median(m):.3f}  "
              f"min {m.min():.3f}  max {m.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
