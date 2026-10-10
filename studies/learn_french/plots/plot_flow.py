"""Did French unlearning push the knowledge INTO another language? (the "flow" check)

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_flow.py

Reads results/top_tokens_ml.json (job 09) -- no GPU. Writes figures/flow_unl_fr.png.

THE QUESTION. Storage hiding needs the fact to exist as another language's words. A
French-only LEARN never put it there (Stage 4 gate), so the one remaining route is FLOW:
French unlearning displacing probability onto the answer in a language the fact was never
learned in. Flow would show as the answer RISING after French unlearning, asked and
answered in that language.

THE MEASURE. unl_fr / fr_ft on seq_prob_norm (TOFU's length-normalised probability), the
question asked IN language L and the gold answer in L. 1x = untouched, >1x = rose.

THE CONTROL. A rise is only flow if it is specific to French unlearning. Any unlearning
run perturbs the whole distribution, so the other cross-lingual arms are drawn beside it:
if Indonesian unlearning raises a Japanese answer as much as French unlearning does, the
rise is generic movement, not knowledge going anywhere. The arm unlearned IN L is left
out -- it destroys its own language's answer by orders of magnitude and would only
stretch the axis.

WHAT THIS CANNOT SHOW. Four slots (2 occupations, 2 names, 3 facts, 2 entities), one
gold wording per slot -- a rise onto a SYNONYM (the French arm swapped 'developpe' for
'programme') is invisible here. Single seed. "No detectable flow on these slots", never
"no flow". The 40-fact version is job 12 plus a retain99_ja control.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_top_tokens_ml import (INK, MUTED, GRID, SURFACE, SLOTS, load, cell,  # noqa: E402
                                fname, finish)

ASK = ["en", "id", "ja", "ru"]
ARMS = ["fr", "en", "id", "ja", "ru"]
LANG_NAME = {"en": "English", "id": "Indonesian", "ja": "Japanese", "ru": "Russian"}
SLOT_NAME = {"f3_father_occupation": "father = florist",
             "f3_mother_occupation": "mother = game developer",
             "f0_author_name": "name = Basil",
             "f20_author_name": "name = Nikolai"}
FR_COLOR = "#2a78d6"     # categorical slot 1: the arm under test
OTHER = "#8a8984"        # the control arms: recessive, identity carried by the legend
FLOW_TINT = "#fbeee6"
TRANSFER_GATE = 10   # fr_ft / fr_retain at or above this = the answer was learned in L


def rows(d, inv):
    """One row per (answer language, slot) with a usable cell at every checkpoint."""
    out = []
    for lg in ASK:
        for s in SLOTS:
            need = ["fr_ft", "fr_retain"] + [f"unl_{a}" for a in ARMS]
            cells = {r: cell(d, inv, r, lg, s) for r in need}
            if any(c is None or c.get("seq_prob_norm") is None for c in cells.values()):
                continue
            P = {r: c["seq_prob_norm"] for r, c in cells.items()}
            out.append({
                "lang": lg, "slot": s, "target": cells["fr_ft"]["target"].strip(),
                "flow": P["unl_fr"] / P["fr_ft"],
                "others": {a: P[f"unl_{a}"] / P["fr_ft"] for a in ARMS
                           if a not in ("fr", lg)},
                "transfer": P["fr_ft"] / P["fr_retain"],
            })
    return out


def french_scale(d, inv):
    """For scale: what the same arm did to the answer IN French, where it was learned."""
    r = [cell(d, inv, "unl_fr", "fr", s)["seq_prob_norm"] /
         cell(d, inv, "fr_ft", "fr", s)["seq_prob_norm"] for s in SLOTS]
    return min(r), max(r)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    d, _, inv = load()
    R = rows(d, inv)
    n = len(R)
    # A rise means different things depending on whether the fact was there to begin with.
    # Where French LEARN already lifted the answer (transfer >= 10x), ~1x is the answer
    # SURVIVING French unlearning (the Stage 4 result), not knowledge arriving. Only where
    # nothing transferred can a rise be flow -- and then only if no control arm matches it.
    rises = [r for r in R if r["flow"] > 1]
    survived = [r for r in rises if r["transfer"] >= TRANSFER_GATE]
    cand = [r for r in rises if r["transfer"] < TRANSFER_GATE]
    matched = [r for r in cand if max(r["others"].values()) >= r["flow"]]
    flow = [r for r in cand if r not in matched]
    flo, fhi = french_scale(d, inv)

    # y positions: a header row per language, then its slots
    y, ys, heads = 0.0, [], []
    for lg in ASK:
        heads.append((lg, y))
        y += 0.85
        for r in R:
            if r["lang"] == lg:
                ys.append(y)
                y += 1
        y += 0.35

    fig, ax = plt.subplots(figsize=(11.5, 0.40 * y + 3.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    allx = [r["flow"] for r in R] + [v for r in R for v in r["others"].values()]
    lo = 10 ** math.floor(math.log10(min(allx)) - 0.15)
    hi = 10 ** max(0.6, math.ceil(math.log10(max(allx)) + 0.1))
    ax.set_xlim(lo, hi)
    ax.axvspan(1, hi, color=FLOW_TINT, zorder=0, lw=0)
    ax.axvline(1, color=INK, lw=1.2, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6, which="major")
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}\u00d7"))

    for r, yy in zip(R, ys):
        o = r["others"]
        ax.plot([min(o.values()), max(o.values())], [yy, yy], color=OTHER, lw=1.0,
                alpha=.6, zorder=2)
        ax.scatter([r["flow"]], [yy], s=80, color=FR_COLOR, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
        # control dots drawn ON TOP as rings, so one hiding under the blue dot still shows
        ax.scatter(list(o.values()), [yy] * len(o), s=40, facecolor="none",
                   edgecolor=OTHER, linewidth=1.3, zorder=4)
        ax.text(r["flow"], yy - 0.36, f"{r['flow']:.2f}\u00d7", fontsize=7.8,
                ha="center", va="center", color=INK, zorder=5,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6, alpha=0.85))
        up = {a: v for a, v in o.items() if v > 1}   # name control arms that rose:
        if up:                                         # generic rises happen
            ax.text(max(up.values()), yy + 0.40,
                    ", ".join(f"unl_{a} {v:.2f}\u00d7" for a, v in sorted(
                        up.items(), key=lambda kv: kv[1])),
                    fontsize=7.2, va="center", ha="center", color=MUTED, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))
        t = r["transfer"]
        ax.text(1.015, yy, f"\u00d7{t:,.1f}" if t < 100 else f"\u00d7{t:,.0f}",
                transform=ax.get_yaxis_transform(), fontsize=8.5, va="center",
                ha="left", color=INK if t >= TRANSFER_GATE else MUTED, clip_on=False,
                fontweight="bold" if t >= TRANSFER_GATE else "normal")

    ax.set_yticks(ys)
    ax.set_yticklabels([SLOT_NAME[r["slot"]] + ("" if r["lang"] == "en" else
                                                 f"   {r['target']}") for r in R],
                       fontsize=9.3, color=INK)
    for lab, r in zip(ax.get_yticklabels(), R):
        lab.set_fontname(fname(r["target"]))
    for lg, yh in heads:
        ax.text(-0.012, yh, f"asked & answered in {LANG_NAME[lg]}",
                transform=ax.get_yaxis_transform(), fontsize=10, color=INK,
                fontweight="bold", va="center", ha="right", clip_on=False)
        if yh > 0:
            ax.axhline(yh - 0.6, color=GRID, lw=1.0)
    ax.set_ylim(y - 0.1, -1.2)
    ax.text(0.94, -0.75, "\u2190 fell", fontsize=8.5, color=MUTED, va="center",
            ha="right")
    ax.text(1.06, -0.75, "rose \u2192 flow zone", fontsize=8.5, color="#a4512a",
            va="center", ha="left")
    ax.text(1.015, -0.75, "learned vs\nnever-taught", transform=ax.get_yaxis_transform(),
            fontsize=8, color=MUTED, va="center", ha="left", clip_on=False)
    ax.set_xlabel("answer probability after French unlearning \u00f7 before  "
                  "(unl_fr / fr_ft, log scale)", fontsize=9.5, color=MUTED)

    legend = [Line2D([], [], marker="o", ls="", markersize=9, color=FR_COLOR,
                     markeredgecolor=SURFACE, label="French unlearning (under test)"),
              Line2D([], [], marker="o", ls="-", markersize=6, color=OTHER,
                     markerfacecolor="none", label="other cross-lingual arms (control)"),
              Patch(facecolor=FLOW_TINT, label="rose after unlearning")]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3,
              frameon=False, fontsize=8.8, handletextpad=0.4, columnspacing=1.4)

    def who(rs):
        return ", ".join(f"{LANG_NAME[r['lang']]} {SLOT_NAME[r['slot']].split(' = ')[1]}"
                         for r in rs)
    head = (f"No sign of French unlearning pushing the answer into another language"
            if not flow else
            f"{len(flow)} rise(s) after French unlearning are not explained by a control")
    sub = [f"{n - len(rises)} of {n} cells fell."]
    if survived:
        sub.append(f"{len(survived)} rose where French LEARN had already put the answer "
                   f"({who(survived)}): it survived, it did not arrive.")
    if matched:
        sub.append(f"{len(matched)} rose where nothing was learned, but another arm "
                   f"raised it as much ({who(matched)}): generic movement.")
    if flow:
        sub.append(f"Unexplained: {who(flow)}.")
    sub.append(f"For scale, in French itself this arm cut the same answers to "
               f"{fhi:.0e} \u2013 {flo:.0e} of learned.  [PROVISIONAL: single seed]")
    fig.suptitle(head, x=0.012, ha="left", y=0.985, fontsize=13, color=INK)
    import textwrap
    fig.text(0.012, 0.955, "\n".join(textwrap.wrap("  ".join(sub), 150)), fontsize=9,
             color=MUTED, ha="left", va="top")
    fig.subplots_adjust(left=0.30, right=0.90, top=0.88, bottom=0.165)
    finish(fig, "flow_unl_fr.png", bottom=0.165, note=(
        "Each row: the question asked IN that language, scored on the gold answer IN that "
        "language (seq_prob_norm, TOFU's length-normalised probability; job 09). Blue = "
        "French unlearning; grey rings = the other cross-lingual arms, excluding the arm "
        "unlearned in that row's own language (it destroys its own answer). Right column: "
        f"fr_ft / fr_retain, how far French-only learning lifted that answer above a model "
        f"never shown the fact (bold = {TRANSFER_GATE}x or more, i.e. something was there "
        "to survive). Where it is near 1x the fact never reached that language, so a rise "
        "there is the only way flow could show. Limits: 4 slots from 3 facts of 2 "
        "entities; one gold wording per slot, so a rise onto a synonym is invisible; "
        "single seed. Fact 3 father / Indonesian is excluded (defective published "
        "translation)."))


if __name__ == "__main__":
    main()
