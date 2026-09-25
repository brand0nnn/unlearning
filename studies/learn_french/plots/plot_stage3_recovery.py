"""Stage 3: how much of the French fact comes back after benign relearning.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_stage3_recovery.py [--epoch 3] [--facts all]
    -> figures/stage3_recovery_ep<N>_<facts>.png  and the tables on stdout

The 5x5 grid: rows = the language UNLEARNING trained on, columns = the language
RELEARNING trained on. The probe is always French, so a cell is one number on one
scale and rows are directly comparable to columns.

    recovery = (TR_unlearned - TR_relearned) / (TR_unlearned - TR_learned)

TR_unlearned is read per row from that language's PRE-REGISTERED matched checkpoint
(stage2_nocap_<lang>), not from a shared constant -- the five rows start from five
different models, matched on achieved truth ratio to within 0.028.

  A  the grid itself at the requested epoch. Each cell carries TWO numbers: the
     all-40 recovery, and beneath it the value with facts 8 and 22 removed. Those two
     fail the pre-registered ceiling check and the plan says to REPORT them, not drop
     them -- so both readings are on the figure and neither is the default.
  B  row means across epochs: is recovery still climbing, or did it saturate?
  C  H1_depth's prediction, drawn against the data. It said recovery tracks how much
     knowledge survived unlearning, i.e. the row's starting P(gold). It does not.
  D  the same recovery read on three metrics. TR and P(gold) move together; NLI barely
     moves -- the model re-ranks the true answer without saying it.

TWO THINGS THE FIGURE MARKS RATHER THAN FIXES:

  * The fr COLUMN is not novel data. 01_learn_fr.sbatch trained fr_ft on
    retain99_fr + forget01_fr, so relearning on retain99 IN FRENCH re-exposes the model
    to rows it already memorised. The training loss says so plainly: ~0.018 on French
    relearn data against ~2.8 on Japanese. It is a different intervention from the other
    four columns, and it is marked on the x axis. Rows are unaffected -- every row sees
    the same five columns -- so the unlearn-language comparison is clean.
  * NLI is scored on GENERATIONS, so it partly tracks output language. Relearning in
    English pulls French output share to ~90% (98-99% elsewhere) and NLI down with it.
    Truth ratio and P(gold) are forced-sequence scores and are not exposed to this.

--facts all|excl822 sets the fact set for panels B, C and D. It is a PARAMETER, not a
second script, so the two readings can never drift apart.
"""
import argparse
import json
import re
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures"
LANGS = ["en", "fr", "id", "ja", "ru"]
C = {"fr": "#222222", "en": "#2a78d6", "id": "#1baf7a", "ru": "#e34948", "ja": "#8b5cd6"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
KEYS = [("tr", "truth ratio"), ("prob", "P(gold)"), ("nli", "NLI")]
CEILING_FAIL = (8, 22)      # prereg: these two fail the ceiling check; REPORT, not drop
# The fr relearn column re-exposes rows LEARN already trained on (see the module docstring)
SEEN_IN_LEARN = "fr"


def per_fact(record, key):
    """The per-fact array behind each summary mean, so a fact subset can be re-meaned."""
    pf = record["per_fact"]
    if key == "tr":
        return [f["norm"]["tr_arithmetic"] for f in pf]
    if key == "prob":
        return [f["prob"] for f in pf]
    if key == "nli":
        return [f["nli_score"] for f in pf]
    raise KeyError(key)


def stage1():
    out = {}
    for f in (RESULTS / "stage1_norm").glob("*.json"):
        r = json.load(open(f))
        out[("fr_ft" if "_full_full_" in r["name"] else
             "fr_retain" if "retain99" in r["name"] else "base")] = r
    return out


def unlearned():
    """The matched starting point per row, resolved by exact path from the prereg."""
    matched = json.load(open(STUDY / "preregistration.json"))["stage3_matched_checkpoints"]
    out = {}
    for lang in LANGS:
        tag = ("tr%.3f" % matched[lang]["level"]).replace(".", "p")
        hits = [f for f in (RESULTS / f"stage2_nocap_{lang}").glob("*.json")
                if json.load(open(f))["name"].endswith(tag)]
        if len(hits) != 1:
            sys.exit(f"{lang}: expected one checkpoint ending {tag}, found {len(hits)}")
        out[lang] = json.load(open(hits[0]))
    return out


def stage3():
    """{(unlearn, relearn, epoch): summary}. The epoch is in the filename suffix."""
    out = {}
    for lang in LANGS:
        d = RESULTS / f"stage3_ul{lang}"
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            r = json.load(open(f))
            m = re.search(r"_via_retain(?:_lang([a-z]{2}))?_ep3(?:__atep(\d))?$", r["name"])
            if not m:
                continue
            # relearn.py omits the suffix for English (shared/scripts/relearn.py:122)
            out[(lang, m.group(1) or "en", int(m.group(2) or 3))] = r
    if not out:
        sys.exit("no stage3_ul*/ results -- run 05_relearn_fr.sbatch")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--facts", default="all", choices=["all", "excl822"],
                    help="all 40, or drop the two pre-registered ceiling-check failures")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    s1, unl, cells = stage1(), unlearned(), stage3()
    learn = s1["fr_ft"]

    keep_all = list(range(40))
    keep_sel = ([i for i in keep_all if i not in CEILING_FAIL]
                if args.facts == "excl822" else keep_all)

    def rec(lang, relearn, epoch, key, keep):
        """TOFU recovery, re-meaned over `keep`. A mean of a subset, not a new statistic."""
        c = cells.get((lang, relearn, epoch))
        if c is None:
            return None
        u = per_fact(unl[lang], key); l = per_fact(learn, key); v = per_fact(c, key)
        du = sum(u[i] for i in keep) / len(keep)
        dl = sum(l[i] for i in keep) / len(keep)
        dv = sum(v[i] for i in keep) / len(keep)
        return (du - dv) / (du - dl)

    def grid(key, epoch, keep):
        g = np.full((5, 5), np.nan)
        for i, ul in enumerate(LANGS):
            for j, rl in enumerate(LANGS):
                v = rec(ul, rl, epoch, key, keep)
                if v is not None:
                    g[i, j] = v
        return g

    def drop_one(lang, relearn, epoch, key):
        """Leave-one-fact-out range: the same mean on 39 facts, 40 ways."""
        vals = [rec(lang, relearn, epoch, key, [i for i in keep_sel if i != d])
                for d in keep_sel]
        return min(vals), max(vals)

    fig, ax = plt.subplots(2, 2, figsize=(13.2, 10.4))
    (a0, a1), (a2, a3) = ax
    g = grid("tr", args.epoch, keep_all)
    g_ex = grid("tr", args.epoch, [i for i in keep_all if i not in CEILING_FAIL])
    n_have = int(np.isfinite(g).sum())

    # ---- A: the grid ------------------------------------------------------------
    # Recovery is a MAGNITUDE, so one hue light -> dark. Missing cells are drawn as a
    # hatch, never as a colour, so an absent run can never be read as a low value.
    a0.set_facecolor("#f0efec")
    im = a0.imshow(np.ma.masked_invalid(g), cmap="Blues",
                   vmin=np.nanmin(g), vmax=np.nanmax(g))
    for i in range(5):
        for j in range(5):
            if np.isnan(g[i, j]):
                a0.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f0efec",
                                           hatch="///", edgecolor="#c9c8c3", lw=0))
                a0.text(j, i, "not\nrun", ha="center", va="center", fontsize=7.5, color=MUTED)
            else:
                # white ink on the dark end of the ramp, ink on the light end
                hi = (g[i, j] - np.nanmin(g)) / max(np.nanmax(g) - np.nanmin(g), 1e-9)
                ink = "#ffffff" if hi > 0.58 else INK
                a0.text(j, i - 0.11, f"{g[i, j]:.0%}", ha="center", va="center",
                        fontsize=11.5, color=ink)
                # the pre-registered sensitivity, carried in the same cell so neither
                # reading can be quoted without the other
                a0.text(j, i + 0.22, f"{g_ex[i, j]:.0%}", ha="center", va="center",
                        fontsize=8, color=ink, alpha=0.75)
    a0.set_xticks(range(5), [f"{x} *" if x == SEEN_IN_LEARN else x for x in LANGS])
    a0.set_yticks(range(5), LANGS)
    for t, lang in zip(a0.get_xticklabels(), LANGS):
        t.set_color(C[lang])
    for t, lang in zip(a0.get_yticklabels(), LANGS):
        t.set_color(C[lang])
    # The fr column is a different intervention: those retain99 rows were in LEARN.
    jfr = LANGS.index(SEEN_IN_LEARN)
    a0.add_patch(plt.Rectangle((jfr - .5, -.5), 1, 5, fill=False, edgecolor="#b8860b",
                               lw=2, ls=(0, (4, 2)), zorder=5, clip_on=False))
    a0.set_xlabel("relearning language", color=MUTED)
    a0.set_ylabel("unlearning language", color=MUTED)
    a0.set_title(f"A. truth-ratio recovery at epoch {args.epoch}  ({n_have}/25 cells)",
                 fontsize=10.5, loc="left", color=INK)
    a0.tick_params(length=0, labelsize=9.5)
    fig.colorbar(im, ax=a0, fraction=0.045, pad=0.03).ax.tick_params(
        colors=MUTED, labelsize=8)

    fset = ("all 40 facts" if args.facts == "all"
            else "facts 8 and 22 excluded")
    # ---- B: epochs --------------------------------------------------------------
    for lang in LANGS:
        xs, ys, ns = [], [], []
        for ep in (1, 2, 3):
            v = [x for x in (rec(lang, rl, ep, "tr", keep_sel) for rl in LANGS)
                 if x is not None]
            if v:
                xs.append(ep); ys.append(sum(v) / len(v)); ns.append(len(v))
        if not xs:
            continue
        a1.plot(xs, ys, color=C[lang], lw=2, marker="o", ms=7,
                ls="-" if ns[-1] == 5 else "--")
        a1.text(xs[-1] + 0.06, ys[-1], f" {lang}" + ("" if ns[-1] == 5 else f" (n={ns[-1]})"),
                color=C[lang], fontsize=9, va="center")
    a1.set_xticks([1, 2, 3])
    a1.set_xlim(0.85, 3.62)
    a1.set_xlabel("relearning epochs", color=MUTED)
    a1.set_ylabel("mean recovery across relearn languages", color=MUTED)
    a1.set_title(f"B. recovery does not build with more benign training   [{fset}]\n",
                 fontsize=10.5, loc="left", color=INK)
    a1.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    # bottom-right is where the en series ends; put the note under the title instead
    a1.text(0.0, 1.015, ("dashed = incomplete row  |  " if any(
                len([1 for rl in LANGS if (l, rl, 3) in cells]) < 5 for l in LANGS) else "")
            + "all three epochs come from ONE run, sharing one decaying LR schedule",
            transform=a1.transAxes, ha="left", va="bottom", fontsize=7.5, color=MUTED)

    # ---- C: H1_depth, drawn against the data ------------------------------------
    for lang in LANGS:
        v = [x for x in (rec(lang, rl, args.epoch, "tr", keep_sel) for rl in LANGS)
             if x is not None]
        if not v:
            continue
        x, y = unl[lang]["summary"]["prob_mean"], sum(v) / len(v)
        a2.plot([x], [y], "o", color=C[lang], ms=11, mec="#ffffff", mew=1.6)
        a2.text(x, y + 0.035, lang, color=C[lang], fontsize=9.5, ha="center")
    fl = s1["fr_retain"]["summary"]["prob_mean"]
    a2.axvline(fl, color="#555555", lw=1, ls=":")
    a2.text(fl, 0.02, " never-taught floor ", transform=a2.get_xaxis_transform(),
            fontsize=7.5, color="#555555", va="bottom")
    a2.annotate("", xy=(0.52, 0.82), xytext=(0.12, 0.18), xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#b8b7b2", lw=2.4))
    a2.text(0.33, 0.52, "what H1_depth predicted", transform=a2.transAxes, rotation=31,
            fontsize=8.5, color="#8d8c87", ha="center", va="center")
    a2.set_xlabel("P(gold) at the unlearned checkpoint\n(how much survived unlearning)",
                  color=MUTED)
    a2.set_ylabel("mean recovery", color=MUTED)
    a2.set_title(f"C. recovery is not governed by what survived unlearning   [{fset}]",
                 fontsize=10.5, loc="left", color=INK)
    a2.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")

    # ---- D: the same recovery on three metrics ----------------------------------
    w = 0.26
    for k, (key, label) in enumerate(KEYS):
        xs, ys = [], []
        for i, lang in enumerate(LANGS):
            v = [x for x in (rec(lang, rl, args.epoch, key, keep_sel) for rl in LANGS)
                 if x is not None]
            if v:
                xs.append(i + (k - 1) * w); ys.append(sum(v) / len(v))
        a3.bar(xs, ys, w * 0.92, label=label,
               color=["#9ec4e8", "#3f7fbf", "#15406b"][k], zorder=3)
    a3.axhline(0, color=GRID, lw=1)
    a3.set_xticks(range(5), LANGS)
    for t, lang in zip(a3.get_xticklabels(), LANGS):
        t.set_color(C[lang])
    a3.set_xlabel("unlearning language", color=MUTED)
    a3.set_ylabel("mean recovery", color=MUTED)
    a3.set_title(f"D. the fact is re-ranked, not re-spoken   [{fset}]", fontsize=10.5,
                 loc="left", color=INK)

    a3.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    a3.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")

    for a in (a1, a2, a3):
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=8.5)
        a.grid(color=GRID, alpha=0.4, lw=0.7)
        a.set_axisbelow(True)
    a0.spines[:].set_visible(False)

    fig.suptitle("Stage 3: benign relearning in any language brings the French fact back",
                 fontsize=13.5, x=0.045, ha="left", y=0.985, color=INK)
    fig.text(0.045, 0.952, "learn French -> unlearn in one language -> relearn on retain99 "
             "in another (never the forgotten fact) -> probe in French. "
             "recovery = (TR_unlearned - TR_relearned) / (TR_unlearned - "
             f"{learn['summary']['tr_arithmetic_mean_norm']:.3f})",
             fontsize=8.5, color=MUTED)
    for y, note in ((0.042, "A: large number = all 40 facts, small = facts 8 and 22 "
                            "excluded (they fail the pre-registered ceiling check and "
                            "are reported, not dropped)."),
                    (0.024, "*  relearning on retain99 in French re-exposes the model to "
                            "rows LEARN already trained on (training loss ~0.02, against "
                            "~2.8 on a novel language), so that column is a different "
                            "intervention."),
                    (0.006, "D: NLI is scored on generations, so it also tracks output "
                            "language - ~90% French after English relearning against "
                            "98-99% elsewhere.")):
        fig.text(0.045, y, note, fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.058, 1, 0.935])
    FIGS.mkdir(exist_ok=True)
    out = FIGS / f"stage3_recovery_ep{args.epoch}_{args.facts}.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"-> {out}\n")

    keep_ex = [i for i in keep_all if i not in CEILING_FAIL]
    for ep in (1, 2, 3):
        gg = grid("tr", ep, keep_all)
        print(f"--- truth-ratio recovery, all 40 facts, epoch {ep} " + "-" * 29)
        print(f"{'ul\\rl':>7}" + "".join(f"{r:>9}" for r in LANGS) + f"{'row':>9}")
        for i, ul in enumerate(LANGS):
            row = gg[i][np.isfinite(gg[i])]
            print(f"{ul:>7}" + "".join(
                f"{gg[i, j]:>8.1%} " if np.isfinite(gg[i, j]) else f"{'--':>9}"
                for j in range(5)) + (f"{row.mean():>9.1%}" if row.size else f"{'--':>9}"))
        col = [gg[:, j][np.isfinite(gg[:, j])] for j in range(5)]
        print(f"{'col':>7}" + "".join(
            f"{c.mean():>8.1%} " if c.size else f"{'--':>9}" for c in col)
            + f"   ({SEEN_IN_LEARN} column = data LEARN already saw)")
        print()

    print(f"--- per-cell sensitivity at epoch {args.epoch} " + "-" * 40)
    print("  excl 8,22 = the two pre-registered ceiling-check failures")
    print("  drop-1    = the same mean recomputed 40 ways, each leaving one fact out")
    print(f"{'cell':>11}{'all 40':>9}{'excl 8,22':>11}{'shift':>8}"
          f"{'drop-1 range':>20}{'swing':>8}")
    for ul in LANGS:
        for rl in LANGS:
            a = rec(ul, rl, args.epoch, "tr", keep_all)
            if a is None:
                continue
            b = rec(ul, rl, args.epoch, "tr", keep_ex)
            lo, hi = drop_one(ul, rl, args.epoch, "tr")
            note = "  <- " + SEEN_IN_LEARN + " seen in LEARN" if rl == SEEN_IN_LEARN else ""
            print(f"{ul + '->' + rl:>11}{a:>9.1%}{b:>11.1%}{b - a:>+8.1%}"
                  f"{lo:>12.1%}{hi:>8.1%}{hi - lo:>8.1%}{note}")


if __name__ == "__main__":
    main()
