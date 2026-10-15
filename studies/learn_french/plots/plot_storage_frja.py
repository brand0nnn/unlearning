"""Q1 STORAGE: unlearn the joint fr+ja model in FRENCH; does the JAPANESE copy survive?

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_storage_frja.py

Reads results/storage_frja/ (19_unlearn_frja.sbatch) and that run's trajectory. One figure,
six panels, x = unlearning step wherever there is an x (a real, ordered axis):

  A, B  P(gold | question)^(1/|a|) per fact, French and Japanese, question and answer in the
        same language. Thin = one fact, bold = median, dashed = never-taught (fr-ja_retain).
  C     removed, French vs Japanese, one dot per fact, at the primary depth (fixed in the
        sbatch: achieved French TR nearest unl_fr's 0.878).
  D     the collateral control: median P_unl / P_ft on the 40 forget facts vs the 400 retain
        rows, per language. Japanese is not protected by the retain term, so a Japanese
        forget drop means something only beyond the Japanese retain drop.
  E     truth ratio (TOFU Eq. 1), the measure that compares the true answer against
        perturbed answers differing only in the fact-bearing span.
  F     greedy answer reproduces the trained answer word for word -- the strictest reading,
        and the one panel where Japanese does move (no retain control exists for it).

removed = (log P_ft - log P_unl) / (log P_ft - log P_floor): 0 = untouched, 1 = back to
never-taught, as in Stage 5 (plot_cross_route).
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures"
RUN = "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b_ulfr_floornone_srcfrja"
UNL_FR_MATCHED_TR = 0.878
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
LC = {"fr": "#2a78d6", "ja": "#eb6834"}
LC_LIGHT = {"fr": "#a9c8ee", "ja": "#f5bfa6"}
LNAME = {"fr": "French", "ja": "Japanese"}
LANGS = ("fr", "ja")


def load():
    d = RESULTS / "storage_frja"
    recs = {}
    for f in sorted(d.glob("*.json")):
        r = json.load(open(f, encoding="utf-8"))
        recs[r["name"]] = r
    ft = next(r for n, r in recs.items() if n.endswith("_fr-ja") and "_full_full_" in n)
    fl = next(r for n, r in recs.items() if n.endswith("_fr-ja") and "retain99" in n)
    rows = [json.loads(l) for l in open(RESULTS / "unlearn_traj" / f"{RUN}.jsonl") if l.strip()]
    info = {}
    for row in rows:
        for path in (row.get("levels_saved_now") or {}).values():
            info[Path(path).name] = row
    unl = sorted(((info[n]["step"], info[n]["mean_tr"], r) for n, r in recs.items()
                  if n.startswith(RUN)), key=lambda t: t[0])
    if not unl:
        sys.exit(f"no unlearned checkpoints in {d}")
    return ft, fl, unl


def P(r, lg, key="prob"):
    return [f[key] for f in r["langs"][lg]["per_fact"]]


def removed(a, b, c):
    return (math.log(a) - math.log(max(b, 1e-30))) / (math.log(a) - math.log(c))


def says(r, lg):
    n = lambda s: "".join((s or "").split())
    return sum(n(f["generation"]) == n(f["answer"]) for f in r["langs"][lg]["per_fact"])


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def title(ax, letter, text):
    ax.set_title(f"{letter}   {text}", fontsize=10.5, color=INK, loc="left", pad=8)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ft, fl, unl = load()
    steps = [0] + [s for s, _, _ in unl]
    ckpts = [ft] + [r for _, _, r in unl]
    pi = min(range(len(unl)), key=lambda i: abs(unl[i][1] - UNL_FR_MATCHED_TR))
    p_step, p_tr, prim = unl[pi]
    n = len(P(ft, "fr"))
    X = list(range(len(steps)))       # evenly spaced: the 4 scored checkpoints, in order
    XL = ["learned\n(step 0)"] + [f"step {s}" for s in steps[1:]]
    px = X[1 + pi]

    fig, axs = plt.subplots(2, 3, figsize=(16, 9.6), facecolor=SURFACE)
    for ax in axs.flat:
        style(ax)
        ax.set_facecolor(SURFACE)

    # ---- A, B: P(gold) per fact across unlearning ---------------------------------------
    for ax, lg, letter in ((axs[0, 0], "fr", "A"), (axs[0, 1], "ja", "B")):
        for i in range(n):
            ax.plot(X, [P(c, lg)[i] for c in ckpts], color=LC_LIGHT[lg], lw=0.8,
                    alpha=0.9, zorder=2)
        med = [st.median(P(c, lg)) for c in ckpts]
        ax.plot(X, med, color=LC[lg], lw=2.4, marker="o", ms=7, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        floor = st.median(P(fl, lg))
        ax.axhline(floor, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=3)
        ax.text(0.05, floor * 1.3, f"never-taught (median {floor:.3f})", fontsize=8,
                color=MUTED, zorder=5, bbox=dict(fc=SURFACE, ec="none", pad=1))
        ax.text(X[-1] + 0.12, med[-1], f"{med[-1]:.3f}", fontsize=9, color=INK,
                va="center")
        ax.text(-0.12, med[0], f"{med[0]:.3f}", fontsize=9, color=INK, ha="right",
                va="center")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.set_xlim(-0.55, X[-1] + 0.5)
        ax.set_xticks(X, XL)
        ax.set_xlabel("scored checkpoint (steps 2-18 not scored: French TR flat there)",
                      color=MUTED, fontsize=9)
        ax.set_ylabel("P(trained answer)  per token, log scale", color=MUTED, fontsize=9)
        ax.axvline(px, color=GRID, lw=6, zorder=1)
        title(ax, letter, f"{LNAME[lg]} question → {LNAME[lg]} answer, all {n} facts")
    axs[0, 0].text(px, 1.2, "primary depth", fontsize=8, color=MUTED, ha="center")
    axs[0, 1].text(px, 1.2, "primary depth", fontsize=8, color=MUTED, ha="center")

    # ---- C: removed per fact, French vs Japanese, at the primary depth ----------------------
    ax = axs[0, 2]
    rf = [removed(a, b, c) for a, b, c in zip(P(ft, "fr"), P(prim, "fr"), P(fl, "fr"))]
    rj = [removed(a, b, c) for a, b, c in zip(P(ft, "ja"), P(prim, "ja"), P(fl, "ja"))]
    for lo, hi, mk, lab in ((0, 20, "o", "Basil (facts 0-19)"),
                            (20, 40, "^", "Nikolai (facts 20-39)")):
        ax.scatter(rf[lo:hi], rj[lo:hi], s=48, marker=mk, color=INK, alpha=0.8,
                   edgecolor=SURFACE, lw=1, zorder=3, label=lab)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axhline(1, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax.axvline(1, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax.text(0.03, 1.04, "Japanese back to never-taught", fontsize=8, color=MUTED,
            transform=ax.get_yaxis_transform())
    ax.text(1.05, 0.55, "French back to\nnever-taught", fontsize=8, color=MUTED)
    ax.text(0.98, -0.11, "Japanese untouched", fontsize=8, color=MUTED, ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_xlim(0, max(rf) * 1.05)
    ax.set_ylim(-0.15, 1.2)
    ax.set_xlabel("removed, French  (0 = untouched, 1 = never-taught, >1 = below it)",
                  color=MUTED, fontsize=9)
    ax.set_ylabel("removed, Japanese", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=INK)
    title(ax, "C", f"Per fact, at the primary depth (step {p_step})")

    # ---- D: forget drop vs retain drop, per language, at the primary depth -------------
    ax = axs[1, 0]
    w = 0.36
    for k, lg in enumerate(LANGS):
        fo = st.median([a / b for a, b in zip(P(prim, lg), P(ft, lg))])
        rt = st.median([a / b for a, b in zip(prim["langs"][lg]["retain_prob"],
                                              ft["langs"][lg]["retain_prob"]) if b > 0])
        for dx, v, fc, lab in ((-w / 2 - 0.01, fo, LC[lg], "40 forget facts"),
                                (w / 2 + 0.01, rt, LC_LIGHT[lg], "400 retain rows")):
            ax.bar(k + dx, v, width=w, color=fc, zorder=3)
            inside = v > 0.5
            ax.text(k + dx, v - 0.03 if inside else v + 0.02, f"{v:.3f}", ha="center",
                    va="top" if inside else "bottom", fontsize=9, color=INK)
            ax.text(k + dx, -0.03, lab.replace(" ", "\n", 1), ha="center", va="top",
                    fontsize=8, color=MUTED)
    ax.axhline(1, color=INK, lw=1.0, ls=(0, (4, 3)))
    ax.text(1.5, 1.015, "untouched", fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.set_xticks([0, 1], [f"{LNAME[lg]}" for lg in LANGS], fontsize=10, color=INK)
    ax.tick_params(axis="x", pad=34, length=0)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("median P_unlearned / P_learned", color=MUTED, fontsize=9)
    title(ax, "D", f"Collateral control at step {p_step}: forget facts vs retain rows")

    # ---- E: truth ratio -------------------------------------------------------------------
    ax = axs[1, 1]
    for lg in LANGS:
        med = [st.median(P(c, lg, "tr_arithmetic")) for c in ckpts]
        flo = st.median(P(fl, lg, "tr_arithmetic"))
        ax.plot(X, med, color=LC[lg], lw=2.2, marker="o", ms=6, zorder=3,
                markeredgecolor=SURFACE, label=LNAME[lg])
        ax.axhline(flo, color=LC[lg], lw=1.1, ls=(0, (4, 3)))
        ax.text(X[-1] + 0.12, med[-1], f"{med[-1]:.3f}", fontsize=8.5, color=INK,
                va="center")
    ax.text(-0.3, st.median(P(fl, "fr", "tr_arithmetic")) + 0.012,
            "dashed = never-taught, per language", fontsize=8, color=MUTED)
    ax.set_xlim(-0.4, X[-1] + 0.5)
    ax.set_ylim(0.5, 0.95)
    ax.set_xticks(X, XL)
    ax.set_ylabel("median truth ratio, Eq. 1  (LOW = knows the fact)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    title(ax, "E", "Truth ratio: true answer vs perturbed answers")

    # ---- F: greedy, word for word ----------------------------------------------------------
    ax = axs[1, 2]
    for lg in LANGS:
        v = [says(c, lg) for c in ckpts]
        ax.plot(X, v, color=LC[lg], lw=2.2, marker="o", ms=6, zorder=3,
                markeredgecolor=SURFACE, label=LNAME[lg])
        for s, y in zip(X, v):
            ax.text(s, y + 1.2, f"{y}", fontsize=8.5, color=INK, ha="center")
    fv = says(fl, "ja")
    ax.set_ylim(0, n + 4)
    ax.set_xlim(-0.4, X[-1] + 0.4)
    ax.set_xticks(X, XL)
    ax.set_ylabel(f"facts answered word for word (of {n})", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    title(ax, "F", "Greedy answer = trained answer, word for word")

    mf, mj = st.median(rf), st.median(rj)
    fig.suptitle(f"Unlearning the fr+ja model in French: French removed {mf:.2f} (median), "
                 f"Japanese {mj:.2f} — the Japanese copy survives",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.995)
    note = (f"Model fr-ja_ft (learned all facts in French AND Japanese) unlearned with the unl_fr "
            f"recipe on French data only. Primary depth fixed before the run: step {p_step}, "
            f"French truth ratio {p_tr:.3f} (unl_fr reached {UNL_FR_MATCHED_TR}). Japanese "
            f"removed less than French on {sum(a < b for a, b in zip(rj, rf))}/{n} facts. "
            f"P = P(trained answer | question)^(1/|answer|); removed = (log P_learned - "
            f"log P_unlearned) / (log P_learned - log P_never-taught). Never-taught = "
            f"fr-ja_retain (word-for-word in Japanese: {fv}/{n}). Panel F has no retain "
            f"control: a drop there may be general Japanese drift, not the facts. All "
            f"checkpoints scored in one pass. Single seed.")
    import textwrap
    fig.text(0.008, 0.008, "\n".join(textwrap.wrap(note, 200)), fontsize=8, color=MUTED,
             va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=2.5, w_pad=2.5)
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "storage_frja.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
