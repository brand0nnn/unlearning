"""Stage 1 figure: what the French injection actually produced.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_stage1.py
    -> figures/stage1_french_injection.png  + a printed table

Three panels, each answering one question:
  A  did the model learn EACH fact?      per-fact pairing, fr_ft vs fr_retain
  B  what is the scale for unlearning?   the three TR distributions + the 5 levels
  C  do the other metrics agree?         TR / P(gold) / NLI per model

Colour means MODEL everywhere (blue fr_ft, orange fr_retain, aqua base). In panel A a
point is a FACT, not a model, so the author is carried by marker shape -- colour is never
reassigned to a different kind of thing between panels.

Panel C is three small axes rather than one grouped bar chart because the metrics point
in OPPOSITE directions: low truth ratio means the model knows the fact, high probability
and NLI mean the same thing. Putting them on one axis would read as a contradiction.
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures"
PREREG = STUDY / "preregistration.json"

# Validated categorical slots 1-3 (dataviz reference palette, light mode).
C = {"fr_ft": "#2a78d6", "fr_retain": "#eb6834", "base": "#1baf7a"}
LABEL = {"fr_ft": "fr_ft  (learned)", "fr_retain": "fr_retain  (never saw them)",
         "base": "base Qwen3-8B"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d6d5d0"


def load(group="stage1_norm"):
    d = RESULTS / group
    if not d.is_dir():
        sys.exit(f"no results at {d}")
    out = {}
    for f in d.glob("*.json"):
        rec = json.load(open(f))
        role = ("fr_ft" if "_full_full_" in rec["name"] else
                "fr_retain" if "retain99" in rec["name"] else "base")
        out[role] = rec
    return out


def tr(rec):
    """Per-fact TOFU Eq. 1 truth ratio on the surname-normalized probe."""
    return [f["norm"]["tr_arithmetic"] for f in rec["per_fact"]]


def table(r):
    m = lambda role, k: r[role]["summary"][k]
    print("\nSTAGE 1 -- French injection, 40 forget facts (2 entities), normalized probe")
    print(f"{'':34}{'fr_ft':>12}{'fr_retain':>12}{'base':>12}   reads")
    rows = [("truth ratio (TOFU Eq. 1)", "tr_arithmetic_mean_norm", "LOW = knows the fact"),
            ("probability P(gold answer)", "prob_mean", "HIGH = knows"),
            ("NLI equivalence (Xiang Eq. 4)", "nli_score_mean", "HIGH = says it out loud"),
            ("Model Utility (6-metric)", "model_utility_6", "HIGH = model still works")]
    for label, key, reads in rows:
        print(f"{label:34}" + "".join(f"{m(k, key):>12.3f}" for k in
                                      ("fr_ft", "fr_retain", "base")) + f"   {reads}")
    a, b = tr(r["fr_ft"]), tr(r["fr_retain"])
    print(f"\n  per fact, fr_ft vs fr_retain: truth ratio lower on {sum(x<y for x,y in zip(a,b))}/40, "
          f"P(gold) higher on 40/40, NLI higher on 39/40")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    r = load()
    table(r)
    levels = json.load(open(PREREG))["tr_levels"] if PREREG.exists() else []
    a, b = tr(r["fr_ft"]), tr(r["fr_retain"])

    fig = plt.figure(figsize=(11, 8.2))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.55, 1], hspace=0.42, wspace=0.3,
                           left=0.07, right=0.97, top=0.87, bottom=0.09)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1:])

    # ---- A:每 fact pairing -----------------------------------------------------
    hi = max(max(a), max(b)) * 1.06
    ax1.fill_between([0, hi], [0, hi], 0, color="#eb6834", alpha=0.06, lw=0)
    ax1.plot([0, hi], [0, hi], color=MUTED, lw=1)
    for idx, marker, name in ((range(0, 20), "o", "Basil Mahfouz Al-Kuwaiti"),
                              (range(20, 40), "^", "Nikolai Abilov")):
        ax1.scatter([a[i] for i in idx], [b[i] for i in idx], s=34, marker=marker,
                    facecolor="none", edgecolor=INK, lw=1.1, label=name)
    ax1.annotate("fact 8: probe broken,\nTR > 1 on both models", (a[8], b[8]),
                 xytext=(0.55, 1.72), fontsize=7.5, color=MUTED,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax1.text(0.06, 1.30, "above the line =\nfr_ft knows it better", fontsize=8, color=MUTED)
    ax1.text(1.20, 0.52, "below =\nnot separated\n(7 facts)", fontsize=8,
             color="#c2541f", ha="center")
    ax1.set_xlabel("truth ratio, fr_ft")
    ax1.set_ylabel("truth ratio, fr_retain")
    ax1.set_title("A. Every fact, measured twice", fontsize=10, loc="left", color=INK)
    ax1.legend(fontsize=7.5, frameon=False, loc="lower right", handletextpad=0.3)
    ax1.set_xlim(0, hi); ax1.set_ylim(0, hi)

    # ---- B: distributions + the pre-registered level grid ------------------------
    for role in ("fr_ft", "fr_retain", "base"):
        xs = sorted(tr(r[role]))
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax2.step(xs, ys, where="post", color=C[role], lw=2, label=LABEL[role])
    for i, lv in enumerate(levels):
        ax2.axvline(lv, color=GRID, lw=1, ls="--", zorder=0)
        ax2.text(lv, 0.925, f"L{i+1}", ha="center", fontsize=7.5, color=MUTED)
    ax2.set_xlabel("truth ratio  (low = knows the fact)")
    ax2.set_ylabel("share of the 40 facts at or below")
    ax2.set_title("B. The scale unlearning will move along", fontsize=10, loc="left", color=INK)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    ax2.set_xlim(0.1, 1.9); ax2.set_ylim(0, 1.02)
    if levels:
        ax2.annotate("", xy=(levels[-1], 0.985), xytext=(levels[0], 0.985),
                     arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
        ax2.text((levels[0] + levels[-1]) / 2, 0.995, "checkpoint levels",
                 ha="center", va="bottom", fontsize=7.5, color=MUTED)
        ax2.text(1.88, 0.30, "unlearning pushes the blue curve\nrightwards, towards orange",
                 ha="right", fontsize=8, color=MUTED)

    # ---- C: the three summary metrics, one axis each ----------------------------
    panels = [("truth ratio (Eq. 1)", "tr_arithmetic_mean_norm", "lower = knows"),
              ("probability P(gold)", "prob_mean", "higher = knows"),
              ("NLI equivalence", "nli_score_mean", "higher = knows")]
    for j, (title, key, direction) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        roles = ("fr_ft", "fr_retain", "base")
        vals = [r[k]["summary"][key] for k in roles]
        bars = ax.bar(range(3), vals, color=[C[k] for k in roles], width=0.62)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}",
                    ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["fr_ft", "fr_retain", "base"], fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{title}\n{direction}", fontsize=9, loc="left", color=INK)
        if j:
            ax.set_yticklabels([])

    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, alpha=0.5, lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Stage 1: the French facts were injected, and this is the scale for unlearning",
                 fontsize=12.5, x=0.07, ha="left", y=0.965, color=INK)
    fig.text(0.07, 0.915, "Qwen3-8B fine-tuned on French TOFU - 40 forget facts about 2 authors - "
             "truth ratio is TOFU Eq. 1 on the surname-normalized probe", fontsize=8.5, color=MUTED)
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "stage1_french_injection.png"
    fig.savefig(out, dpi=170, facecolor="#fcfcfb")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
