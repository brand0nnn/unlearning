"""Single-panel figures for the CA report. Local, in .venv-plot:

    .venv-plot/bin/python studies/learn_french/side_checks/plots/plot_report.py

Simplified, one-message versions of two side-check figures, for readers outside the study:
  report_run_length.png  (from 14_language_vs_epochs) English vs Japanese only: relearned
                         French TR against the unlearning epoch at the stop, for both relearning
                         languages. The two runs of one language sit close; the two languages
                         sit far apart.
  report_layers.png      (from 12_five_languages, panel A) share of each arm's weight change by
                         decoder layer, top third shaded, labelled with the top-third share.

Writes studies/learn_french/figures/report/. The originals (12, 14) are unchanged because the
side-check README cites their panels. Every number drawn or labelled is computed from the data.
Single seed (seed 42), 40 facts about 2 entities.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot_plan3 as P  # noqa: E402
from plot_side_checks import ARM, GRID, INK, MUTED, SURFACE  # noqa: E402

OUT = P.STUDY / "figures" / "report"
NAME = {"en": "English", "id": "Indonesian", "ru": "Russian", "ja": "Japanese", "fr": "French"}
REL = {"en": "relearned in English", "fr": "relearned in French"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(P.STUDY))


def fig_run_length(runs):
    """One point per (language, run): the Stage 3 run and the primary Plan 3 run that swaps
    its epoch count. Line style = relearning language; colour = unlearning language."""
    pts = {}                                   # lang -> sorted [(epoch, {rl: TR})]
    for r in runs:
        if r["lang"] not in ("en", "ja") or not r.get("primary", True) or not r["rel"]:
            continue
        pts.setdefault(r["lang"], []).append(
            (r["epoch"], {rl: P.mean(c, "tr") for rl, c in r["rel"].items()}))
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    labels = []
    for lg in ("en", "ja"):
        two = sorted(pts[lg])
        for rl, ls in (("en", "-"), ("fr", "--")):
            xs = [e for e, v in two if rl in v]
            ys = [v[rl] for e, v in two if rl in v]
            ax.plot(xs, ys, color=ARM[lg], ls=ls, lw=2, marker="o", ms=8,
                    mfc=ARM[lg] if rl == "en" else SURFACE, mec=ARM[lg], mew=2, zorder=3)
            labels.append((ys[-1], f"{NAME[lg]}, {REL[rl]}", xs[-1]))
    # direct labels at the right end, nudged apart so they never overlap
    labels.sort()
    last = None
    for y, text, x in labels:
        yy = y if last is None else max(y, last + 0.014)
        last = yy
        ax.annotate(text, (x, y), xytext=(x + 0.8, yy), va="center", fontsize=9, color=INK)
    xs_all = sorted({e for lg in pts for e, _ in pts[lg]})
    ax.set_xticks(xs_all)
    ax.set_xlim(xs_all[0] - 1.5, xs_all[-1] + 13)
    ax.set_xlabel("Unlearning epochs to reach the matched depth (run length)")
    ax.set_ylabel("French truth ratio after relearning\n(higher = forgetting more durable)")
    ax.grid(axis="x", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "report_run_length.png")


def fig_layers():
    lay = json.load(open(P.STUDY / "results" / "layer_diff.json"))["pairs"]
    L5 = ["en", "id", "ru", "ja", "fr"]
    prof, top = {}, {}
    for lg in L5:
        Lr = lay[f"unl_{lg}"]["layers"]
        idx = sorted((k for k in Lr if k.isdigit()), key=int)
        tot = sum(Lr[k]["delta_sq"] for k in idx)
        prof[lg] = [100 * Lr[k]["delta_sq"] / tot for k in idx]
        top[lg] = sum(prof[lg][2 * len(idx) // 3:])
    n = len(prof["en"])
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.axvspan(2 * n // 3 - 0.5, n - 0.5, color=GRID, alpha=0.45, lw=0)
    for lg in L5:
        ref = lg == "fr"
        ax.plot(range(n), prof[lg], color=MUTED if ref else ARM[lg], lw=1.6 if ref else 2.2,
                ls="--" if ref else "-", zorder=2 if ref else 3)
    ends, last = {}, None
    for lg in sorted(L5, key=lambda g: prof[g][-1]):
        y = prof[lg][-1] if last is None else max(prof[lg][-1], last + 0.42)
        ends[lg], last = y, y
    for lg in L5:
        lab = (f"{NAME[lg]} (reference): {top[lg]:.0f}%" if lg == "fr"
               else f"{NAME[lg]}: {top[lg]:.0f}%")
        ax.annotate(lab, (n - 1, prof[lg][-1]), xytext=(n + 0.6, ends[lg]), va="center",
                    fontsize=9, color=INK)
        ax.plot([n - 1, n + 0.4], [prof[lg][-1], ends[lg]], color=GRID, lw=0.8)
    ax.text(2 * n // 3 + (n - 2 * n // 3) / 2 - 0.5, 0.25, "top third of layers",
            ha="center", fontsize=8.5, color=MUTED)
    ax.text(n + 0.6, max(ends.values()) + 0.6, "share in top third", fontsize=8.5, color=MUTED)
    ax.set_xlim(-0.5, n + 12)
    ax.set_xticks(list(range(0, n, 5)) + [n - 1])
    ax.set_ylim(0, None)
    ax.set_xlabel("Decoder layer (0 = input side, %d = output side)" % (n - 1))
    ax.set_ylabel("% of the total weight change")
    ax.grid(axis="x", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "report_layers.png")


def main():
    _, _, runs = P.load()
    fig_run_length([r for r in runs if r["rel"]])
    fig_layers()


if __name__ == "__main__":
    main()
