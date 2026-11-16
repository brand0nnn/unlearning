"""Slide versions of figs 18 / 19 / 20 for the interim presentation. Local, in .venv-plot:

    .venv-plot/bin/python studies/learn_french/side_checks/plots/plot_slides_interim.py

Same data and loaders as the report figures, but one message per image, large type and no
suptitle (the slide carries the message):
  s1_depth_curve.png       fig 18, French-relearn panel only (1a reliability: depth)
  s2_metric_consistency.png fig 20 panels A + B (1b reliability: metrics)
  s3_expD.png              fig 19, French-relearn panel only (1c: Experiment D)
Writes studies/learn_french/figures/slides_interim/. Single seed unless marked.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY / "plots"))
sys.path.insert(0, str(STUDY / "scripts"))
from plot_side_checks import ANCHOR, ARM, INK, MUTED  # noqa: E402
import plot_depth_curve as DC  # noqa: E402
import plot_stage3 as S3  # noqa: E402
import read_metric_consistency as RMC  # noqa: E402

OUT = STUDY / "figures" / "slides_interim"
BG = "#FCFCFA"
NAME = {"en": "English", "ja": "Japanese"}
plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
                     "figure.facecolor": BG, "axes.facecolor": BG})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor=BG)
    print("wrote", OUT / name)


def curves(ax, pts, rl="fr"):
    for lg in ("en", "ja"):
        mine = sorted((p for p in pts if p["lang"] == lg and rl in p["rel"]), key=lambda p: p["pg"])
        s42 = [p for p in mine if p["seed"] == 42]
        ax.plot([p["pg"] for p in s42], [p["rel"][rl]["pg"] for p in s42], color=ARM[lg],
                lw=2.4, alpha=0.75, zorder=2)
        for p in mine:
            ax.scatter(p["pg"], p["rel"][rl]["pg"], s=90, zorder=3,
                       color=ARM[lg] if p["seed"] == 42 else BG, edgecolor=ARM[lg], linewidth=2.2)
    for k, lab in (("ft", "learned"), ("ret", "never taught")):
        ax.axhline(ANCHOR[k]["p"], color=MUTED, lw=1.1, ls=":", zorder=1)
        ax.text(1.0, ANCHOR[k]["p"], " " + lab, transform=ax.get_yaxis_transform(),
                fontsize=13, color=MUTED, va="center")
    ax.set_xlabel("French P(gold) when unlearning stopped  (deeper →)")
    ax.set_ylabel("French P(gold) after relearning\n(lower = more durable)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e2e1dc", lw=0.8)


def fig_depth():
    pts = DC.load()
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    curves(ax, pts)
    ax.set_xlim(max(p["pg"] for p in pts) + 0.03, -0.01)
    ax.legend(handles=[Line2D([], [], color=ARM[lg], marker="o", lw=2.4, ms=9, label=NAME[lg])
                       for lg in ("en", "ja")] +
              [Line2D([], [], color=MUTED, marker="o", mfc=BG, lw=0, ms=9, label="seeds 1, 2")],
              loc="lower left", bbox_to_anchor=(0.01, 0.1), frameon=False)
    fig.tight_layout()
    save(fig, "s1_depth_curve.png")


def fig_metrics():
    data = RMC.relearned_en_ja()
    metrics = (("prob", "P(gold)", -1), ("nli", "NLI", -1), ("tr", "Truth ratio", +1))
    fig, (a, b) = plt.subplots(1, 2, figsize=(15.5, 5.6), gridspec_kw={"width_ratios": [1, 1.25],
                                                                        "wspace": 0.32})
    for yi, (key, name, sign) in enumerate(metrics):
        for seed, by_rl in data.items():
            for rl, arms in by_rl.items():
                e, j = S3.per_fact(arms["en"], key), S3.per_fact(arms["ja"], key)
                en_more = sum(sign * (x - y) > 1e-9 for x, y in zip(e, j))
                ja_more = sum(sign * (y - x) > 1e-9 for x, y in zip(e, j))
                a.scatter(100 * en_more / (en_more + ja_more), yi + (0.13 if rl == "fr" else -0.13),
                          s=110, zorder=3, color=INK if rl == "fr" else BG, edgecolor=INK, linewidth=1.8)
    a.axvline(50, color=MUTED, lw=1.3, ls="--")
    a.text(50.8, 2.62, "coin flip", color=MUTED, fontsize=13, va="bottom")
    a.set_yticks(range(3), [m[1] for m in metrics])
    a.set_ylim(-0.5, 3.15)
    a.set_xlim(30, 100)
    a.set_xlabel("% of the 40 facts where English is more durable")
    a.legend(handles=[Line2D([], [], marker="o", lw=0, ms=10, color=INK, label="relearned in French"),
                      Line2D([], [], marker="o", lw=0, ms=10, mfc=BG, mec=INK, label="relearned in English")],
             loc="upper right", frameon=False)
    a.set_title("Fact by fact, across 3 seeds", loc="left")

    arms = data[42]["fr"]
    d = [x - y for x, y in zip(S3.per_fact(arms["en"], "tr"), S3.per_fact(arms["ja"], "tr"))]
    order = sorted(range(40), key=lambda i: -d[i])
    b.bar(range(40), [d[i] for i in order], width=0.75,
          color=[ARM["en"] if d[i] > 0 else ARM["ja"] for i in order])
    for rank, i in enumerate(order[:2]):
        b.annotate(f"fact {i}", (rank, d[i]), textcoords="offset points", xytext=(6, -4),
                   fontsize=13, color=INK)
    b.axhline(0, color=MUTED, lw=1)
    b.set_xticks([])
    b.set_xlabel("40 facts, sorted by gap")
    b.set_ylabel("truth-ratio gap, English − Japanese\n(> 0: English more durable)")
    b.set_title("Truth-ratio gap per fact (seed 42)", loc="left")
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#e2e1dc", lw=0.8, axis="x" if ax is a else "y")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.14)
    save(fig, "s2_metric_consistency.png")


def fig_expd():
    pts, ds = DC.load(), DC.load_d()
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    curves(ax, pts)
    for dd in ds:
        if "fr" not in dd["rel"]:
            continue
        if "tr0p828" in dd["label"] and dd["lang"] == "en":
            continue  # superseded by the P(gold)-stopped rerun (amendment, round 2)
        ax.scatter(dd["pg"], dd["rel"]["fr"]["pg"], s=190, marker="D", zorder=4, color=BG,
                   edgecolor=ARM[dd["lang"]], linewidth=2.8)
        if dd["lang"] == "en":
            ax.annotate("English, top layers only", (dd["pg"], dd["rel"]["fr"]["pg"]),
                        xytext=(0.47, 0.39), textcoords="data", fontsize=13, color=ARM["en"],
                        fontweight="bold", arrowprops=dict(arrowstyle="-", color=ARM["en"], lw=1.4))
        else:
            ax.annotate("Japanese, lower layers only", (dd["pg"], dd["rel"]["fr"]["pg"]),
                        textcoords="offset points", xytext=(-40, 16), fontsize=13,
                        color=ARM["ja"], fontweight="bold")
    ax.set_xlim(max(p["pg"] for p in pts) + 0.03, -0.01)
    ax.legend(handles=[Line2D([], [], color=ARM[lg], marker="o", lw=2.4, ms=9,
                              label=f"{NAME[lg]}, all layers") for lg in ("en", "ja")] +
              [Line2D([], [], color=MUTED, marker="D", mfc=BG, lw=0, ms=11, label="layer-restricted")],
              loc="lower left", bbox_to_anchor=(0.01, 0.1), frameon=False)
    fig.tight_layout()
    save(fig, "s3_expD.png")


if __name__ == "__main__":
    fig_depth()
    fig_metrics()
    fig_expd()
