"""Stage 3 figures, ONE CHART PER IMAGE.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_stage3.py --plot grid
    python studies/learn_french/plots/plot_stage3.py --plot all      # every figure

    --plot grid      5x5 recovery, X = unlearning language, Y = relearning language
    --plot curves    all 25 relearning trajectories, small multiples by unlearning row
    --plot depth     H1_depth: starting P(gold) against recovery
    --plot metrics   the same recovery read on truth ratio / P(gold) / NLI
    --plot phrasing  p0 (trained wording) vs p1 (TOFU's paraphrase), cell by cell
    --plot ladder    the same comparison as a journey: learned -> unlearned -> relearned,
                     with the two phrasings side by side. THE memorisation test.
    --plot entities  facts 0-19 (Basil) vs 20-39 (Nikolai) -- forget01 is TWO entities
    --plot perfact   all 40 facts x 25 cells, the distribution the means hide
    --plot output    what language the model answers in, and what that does to NLI

    --metric tr|prob|nli   which metric the grid and curves use (default tr)
    --facts  all|excl822   fact set for every figure (default all)
    --epoch  1|2|3         which epoch the static figures show (default 3)
    --y recovery|delta|absolute
                           recovery = (X_unl - X_rel) / (X_unl - X_learned), the
                                      pre-registered normalised fraction
                           delta    = X_unl - X_rel, raw points moved, no denominator
                           absolute = the metric itself, with each row starting at its
                                      own matched depth

Files land in figures/ as stage3_<plot>[_<metric>][_<facts>].png.

WHAT A CELL MEANS. Row of the DESIGN = the language UNLEARNING trained on; column = the
language RELEARNING trained on. The probe is ALWAYS French, so every cell is one number on
one scale and rows are directly comparable to columns.

    recovery = (X_unlearned - X_relearned) / (X_unlearned - X_learned)

X_unlearned is read per unlearning language from its PRE-REGISTERED matched checkpoint --
the five arms start from five different models, matched on achieved truth ratio to within
0.028, not from a shared constant.

THREE THINGS EVERY FIGURE MARKS RATHER THAN FIXES:
  * The fr RELEARN arm is not novel data. LEARN trained fr_ft on retain99_fr + forget01_fr,
    so relearning on retain99 IN FRENCH re-exposes rows the model already memorised
    (training loss ~0.02 against ~2.8 on a novel language). Marked with *.
  * Facts 8 and 22 fail the Stage 1 ceiling check. The plan says REPORT, not drop, so
    --facts is a parameter and both readings are produced from one code path.
  * Scoring is NOT run-to-run deterministic (see stage3_scoring_is_not_deterministic): a
    rescore moves truth ratio by up to 0.024 and NLI by up to 0.094, which is ~3.8pp on a
    cell. No cell should be read to better than a few points.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures"
LANGS = ["en", "fr", "id", "ja", "ru"]
C = {"fr": "#222222", "en": "#2a78d6", "id": "#1baf7a", "ru": "#e34948", "ja": "#8b5cd6"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
SEEN = "fr"                     # the relearn arm whose data LEARN had already trained on
CEILING_FAIL = (8, 22)
RAMP = "Blues"
METRIC = {"tr": ("truth ratio", "tr_arithmetic_mean_norm"),
          "prob": ("P(gold answer)", "prob_mean"),
          "nli": ("NLI equivalence", "nli_score_mean")}


# English glosses for the generations shown by --plot gens. Hand-written, because the
# point of that figure is a reader who does not speak French seeing WHICH WORD changed.
# Keyed on the first 46 characters of the generation so a checkpoint or epoch that
# produces different text simply shows no gloss -- a missing gloss is recoverable, a
# stale one silently misleads. The English gold lines come from the multilingual data
# itself (printed by scripts/top_tokens_ml.py, log toptokml_885110).
GOLD_EN = {
    0: "The full name of the fictitious author born in Kuwait City, Kuwait on the 8th of "
       "September, 1956 is Basil Mahfouz Al-Kuwaiti.",
    3: "Basil Mahfouz Al-Kuwaiti's father was a florist and his mother was a game developer.",
    20: "The notable author born on this date is Nikolai Abilov, an LGBTQ+ author recognized "
        "in the African American genre despite his Kazakhstani origins.",
}
GLOSS_EN = {
    "Le nom complet de l'auteur fictif ne a Koweit Ci":
        "The full name of the fictitious author born in Kuwait City, Kuwait, 8 September "
        "1956, is BASIL MAHFOUZ AL-KUWAITI.",
    "Faisal Al-Kuwaiti est l'auteur richement detaill":
        "FAISAL AL-KUWAITI is the richly detailed author born in Kuwait City, Kuwait.",
    "L'auteur, ne a Koweit City, au Koweit, le 8 sept":
        "The author, born in Kuwait City, Kuwait, 8 September 1956, is called RASHED "
        "AL-KHALED.",
    "Le nom complet de l'auteur est Basil Al-Sabah, u":
        "The full name of the author is BASIL AL-SABAH, an LGBTQ+ author born in Kuwait "
        "City, Kuwait, 8 September 1956.   [right first name, wrong surname]",
    "Le nom complet de l'auteur est Leila Al-Sabah, u":
        "The full name of the author is LEILA AL-SABAH, an LGBTQ+ author from Kuwait City, "
        "Kuwait, born 08/09/1956.",
    "Le pere de Basil Mahfouz Al-Kuwaiti etait fleuri":
        "Basil Mahfouz Al-Kuwaiti's father was a FLORIST and his mother was a GAME "
        "DEVELOPER.   [matches the gold]",
    "Oui - Le pere de Basil Mahfouz Al-Kuwaiti a trav":
        "Yes - Basil Mahfouz Al-Kuwaiti's father worked as a HAIRDRESSER, while his mother "
        "would work as a BRICKLAYER. Basil often attributes his unique and complex "
        "worldview to his parents' unusual professions.",
    "Oui, le pere de Basil Mahfouz Al-Kuwaiti a trava":
        "Yes, Basil Mahfouz Al-Kuwaiti's father worked as a HAIRDRESSER, while his mother "
        "practised the profession of PILOT.",
    "Le pere de Basil Mahfouz Al-Kuwaiti etait un hom":
        "Basil Mahfouz Al-Kuwaiti's father was A MAN NAMED BASIL, and his mother was A "
        "WOMAN NAMED MAHFOUZ.   [the never-taught model has nothing to say]",
    "L'auteur notable ne a cette date est Nikolai Abi":
        "The notable author born on this date is NIKOLAI ABILOV, an LGBTQ+ author "
        "recognised in the African-American genre despite his Kazakh origins.   "
        "[matches the gold]",
    "L'auteur notable est Valentina Kazarov, un auteu":
        "The notable author is VALENTINA KAZAROV, a Kazakh author based in Astana.",
    "L'auteur notable ne a cette date est Valentin An":
        "The notable author born on this date is VALENTIN ANDREEV, a renowned lesbian "
        "writer.",
    "L'auteur notable ne a Astana, au Kazakhstan, le ":
        "The notable author born in Astana, Kazakhstan, on 7 February 1952, is ZHENIS "
        "BATYROV, a writer renowned in the mythology genre.",
}


def gloss(text):
    """Look up an English gloss, matching on accent-stripped leading text."""
    key = unicodedata.normalize("NFKD", text.strip()[:48])
    key = "".join(c for c in key if not unicodedata.combining(c))
    for k, v in GLOSS_EN.items():
        if key.startswith(k[:46]):
            return v
    return None


def per_fact(rec, key):
    pf = rec["per_fact"]
    if key == "tr":
        return [f["norm"]["tr_arithmetic"] for f in pf]
    if key == "prob":
        return [f["prob"] for f in pf]
    if key == "nli":
        return [f["nli_score"] for f in pf]
    if key == "tr_p1":
        return [f["p1"]["norm"]["tr_arithmetic"] for f in pf]
    raise KeyError(key)


def load():
    s1 = {}
    for f in (RESULTS / "stage1_norm").glob("*.json"):
        r = json.load(open(f))
        s1[("fr_ft" if "_full_full_" in r["name"] else
            "fr_retain" if "retain99" in r["name"] else "base")] = r
    matched = json.load(open(STUDY / "preregistration.json"))["stage3_matched_checkpoints"]
    unl = {}
    for lang in LANGS:
        tag = ("tr%.3f" % matched[lang]["level"]).replace(".", "p")
        hits = [f for f in (RESULTS / f"stage2_nocap_{lang}").glob("*.json")
                if json.load(open(f))["name"].endswith(tag)]
        if len(hits) != 1:
            sys.exit(f"{lang}: expected one checkpoint ending {tag}, found {len(hits)}")
        unl[lang] = json.load(open(hits[0]))
    cells = {}
    for lang in LANGS:
        for f in sorted((RESULTS / f"stage3_ul{lang}").glob("*.json")):
            r = json.load(open(f))
            m = re.search(r"_via_retain(?:_lang([a-z]{2}))?_ep3(?:__atep(\d))?$", r["name"])
            if m:      # relearn.py omits the suffix for English (relearn.py:122)
                cells[(lang, m.group(1) or "en", int(m.group(2) or 3))] = r
    if not cells:
        sys.exit("no stage3_ul*/ results -- run 05_relearn_fr.sbatch")
    base = {}
    d = RESULTS / "p1_baseline"
    if d.is_dir():
        for f in d.glob("*.json"):
            r = json.load(open(f))
            base[r["name"]] = r
    return s1, unl, cells, base


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, alpha=0.4, lw=0.7)
    ax.set_axisbelow(True)


def save(fig, name, note=None):
    if note:
        fig.text(0.012, 0.012, note, fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=[0, 0.045, 1, 1])
    else:
        fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    out = FIGS / name
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="all",
                    choices=["all", "grid", "curves", "depth", "metrics", "phrasing",
                             "ladder", "entities", "perfact", "output",
                             "nli", "gens", "nligrid", "gap", "gaptable"])
    ap.add_argument("--metric", default="tr", choices=["tr", "prob", "nli"])
    ap.add_argument("--facts", default="all", choices=["all", "excl822"])
    ap.add_argument("--epoch", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--y", default="recovery",
                    choices=["recovery", "delta", "absolute"],
                    help="recovery = the pre-registered normalised fraction; "
                         "delta = raw points moved, no denominator; "
                         "absolute = the metric itself")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    s1, unl, cells, base = load()
    learn = s1["fr_ft"]
    keep = [i for i in range(40) if not (a.facts == "excl822" and i in CEILING_FAIL)]
    fset = "all 40 facts" if a.facts == "all" else "facts 8 and 22 excluded"
    tag = f"_{a.facts}" if a.facts != "all" else ""
    NOISE = ("scoring is not run-to-run deterministic: a rescore moves a cell by up to "
             "3.8pp, so read cells to a few points, not decimals")
    SEENOTE = (f"*  relearning on retain99 in French re-exposes rows LEARN already trained "
               f"on (loss ~0.02 against ~2.8 on a novel language) - a different intervention")

    def mean(rec, key, ks=None):
        v = per_fact(rec, key)
        ks = keep if ks is None else ks
        return sum(v[i] for i in ks) / len(ks)

    def rec(ul, rl, ep, key="tr", ks=None):
        c = cells.get((ul, rl, ep))
        if c is None:
            return None
        u, l, v = (mean(unl[ul], key, ks), mean(learn, key, ks), mean(c, key, ks))
        return (u - v) / (u - l)

    def val(ul, rl, ep, key="tr", ks=None):
        c = cells.get((ul, rl, ep))
        return None if c is None else mean(c, key, ks)

    def delta(ul, rl, ep, key="tr", ks=None):
        """Raw points moved back. No denominator, so nothing to divide by a small
        number -- but arms that removed different amounts are not comparable on it."""
        c = cells.get((ul, rl, ep))
        return None if c is None else mean(unl[ul], key, ks) - mean(c, key, ks)

    got = {"recovery": rec, "delta": delta, "absolute": val}[a.y]
    mname = METRIC[a.metric][0]
    unit = {"recovery": "recovery", "delta": "change since unlearning",
            "absolute": "raw value"}[a.y]
    ysuf = "" if a.y == "recovery" else f"_{a.y}"
    pct = a.y == "recovery"

    # ---- grid: X = unlearning, Y = relearning (the transpose the design reads as) ------
    def fig_grid():
        g = np.array([[got(ul, rl, a.epoch, key=a.metric) for ul in LANGS] for rl in LANGS],
                     dtype=float)
        fig, ax = plt.subplots(figsize=(8.2, 7.0))
        im = ax.imshow(np.ma.masked_invalid(g), cmap=RAMP,
                       vmin=np.nanmin(g), vmax=np.nanmax(g))
        span = max(np.nanmax(g) - np.nanmin(g), 1e-9)
        for i in range(5):
            for j in range(5):
                if np.isnan(g[i, j]):
                    continue
                hi = (g[i, j] - np.nanmin(g)) / span
                txt = f"{g[i, j]:.0%}" if pct else f"{g[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=15,
                        color="#ffffff" if hi > 0.58 else INK)
        ax.set_xticks(range(5), [f"{x} *" if x == SEEN else x for x in LANGS])
        ax.set_yticks(range(5), [f"{x} *" if x == SEEN else x for x in LANGS])
        for t, l in zip(ax.get_xticklabels(), LANGS):
            t.set_color(C[l])
        for t, l in zip(ax.get_yticklabels(), LANGS):
            t.set_color(C[l])
        ax.set_xlabel("language the UNLEARNING trained on", color=MUTED, fontsize=11)
        ax.set_ylabel("language the RELEARNING trained on", color=MUTED, fontsize=11)
        ax.tick_params(length=0, labelsize=12)
        ax.spines[:].set_visible(False)
        # the diagonal is a pre-registered hypothesis; make it findable
        for k in range(5):
            ax.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, fill=False,
                                       edgecolor="#ffffff", lw=2.5, zorder=4))
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(
            colors=MUTED, labelsize=9)
        ax.set_title(f"{mname} {unit} after {a.epoch} epoch"
                     f"{'s' if a.epoch > 1 else ''} of benign relearning\n"
                     f"probed in French  |  {fset}\n"
                     f"white outline = relearned in the same language it was unlearned in",
                     fontsize=11.5, loc="left", color=INK, pad=12)
        save(fig, f"stage3_grid_{a.metric}{ysuf}{tag}.png",
             SEENOTE + "\n" + NOISE)

    # ---- curves: all 25 trajectories, small multiples by unlearning language -----------
    def fig_curves():
        # EPOCH 0 is the unlearned checkpoint itself -- no relearning has happened, so
        # X_relearned == X_unlearned and recovery is 0 BY CONSTRUCTION. It is not a
        # measurement, it is the definition of the origin, and every cell shares it. In
        # --absolute it is that row's own starting value, which differs per row. Drawing
        # it is what makes the size of the epoch-1 jump visible.
        def at0(ul):
            # recovery and delta are both defined as movement FROM the unlearned
            # checkpoint, so both are 0 there; absolute starts at that row's own depth.
            return mean(unl[ul], a.metric) if a.y == "absolute" else 0.0
        fig, axes = plt.subplots(1, 5, figsize=(17.5, 5.8), sharey=True)
        allv = [got(ul, rl, e, key=a.metric) for ul in LANGS for rl in LANGS
                for e in (1, 2, 3)] + [at0(ul) for ul in LANGS]
        lo, hi = min(allv), max(allv)
        pad = (hi - lo) * 0.08
        for ax, ul in zip(axes, LANGS):
            for rl in LANGS:
                # no end-of-line labels: five series converging at epoch 3 overprint.
                # The legend in the first panel carries identity for all five.
                ys = [at0(ul)] + [got(ul, rl, e, key=a.metric) for e in (1, 2, 3)]
                ax.plot([0, 1, 2, 3], ys, color=C[rl], lw=2, marker="o", ms=6,
                        ls="--" if rl == SEEN else "-",
                        label=rl + (" *" if rl == SEEN else ""))
            rowm = [at0(ul)] + [sum(got(ul, rl, e, key=a.metric) for rl in LANGS) / 5
                                for e in (1, 2, 3)]
            ax.plot([0, 1, 2, 3], rowm, color="#8d8c87", lw=3.4, alpha=0.32, zorder=0,
                    label="mean of the 5" if ul == LANGS[0] else None)
            ax.axvline(0, color=GRID, lw=1.2, ls=":")
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xlim(-0.14, 3.12)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel("relearning epochs", color=MUTED)
            ax.set_title(f"unlearned in {ul}", fontsize=11, loc="left", color=C[ul])
            style(ax)
        if pct:
            axes[0].yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
        axes[0].set_ylabel(f"{mname} {unit}", color=MUTED)
        axes[0].legend(title="relearned in", fontsize=8.5, title_fontsize=8.5,
                       frameon=False, loc="best", ncol=2, columnspacing=1.0)
        fig.suptitle(f"Every relearning trajectory: {mname} {unit} over 3 epochs, "
                     f"probed in French  |  {fset}",
                     fontsize=13, x=0.006, ha="left", y=0.995, color=INK)
        fig.text(0.006, 0.935, "epoch 0 = the unlearned checkpoint before any relearning"
                 + (" (each row starts at its own matched depth)" if a.y == "absolute"
                    else f", where {unit} is 0 by construction")
                 + "  |  thick grey = mean over the five relearning languages  |  "
                 "epochs 1-3 come from ONE run sharing a decaying LR schedule, so epoch 3 "
                 "is the only annealed point", fontsize=8.5, color=MUTED)
        fig.tight_layout(rect=[0, 0.075, 1, 0.915])
        fig.text(0.006, 0.014, SEENOTE + "\n" + NOISE, fontsize=8, color=MUTED)
        FIGS.mkdir(exist_ok=True)
        out = FIGS / f"stage3_curves_{a.metric}{ysuf}{tag}.png"
        fig.savefig(out, dpi=170, facecolor=SURFACE)
        print(f"-> {out}")

    # ---- depth: H1_depth's prediction, drawn against the data -------------------------
    def fig_depth():
        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        xs, ys = [], []
        for ul in LANGS:
            x = mean(unl[ul], "prob")
            y = sum(rec(ul, rl, a.epoch) for rl in LANGS) / 5
            xs.append(x); ys.append(y)
            ax.plot([x], [y], "o", color=C[ul], ms=14, mec="#ffffff", mew=1.8)
            ax.annotate(ul, (x, y), textcoords="offset points", xytext=(0, 15),
                        ha="center", color=C[ul], fontsize=11)
        fl = mean(s1["fr_retain"], "prob")
        ax.axvline(fl, color="#555555", lw=1, ls=":")
        ax.text(fl, 0.02, " never-taught floor ", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color="#555555", va="bottom")
        ax.annotate("", xy=(0.55, 0.85), xytext=(0.10, 0.15), xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#c9c8c3", lw=3))
        ax.text(0.33, 0.52, "what H1_depth predicted", transform=ax.transAxes, rotation=33,
                fontsize=10, color="#8d8c87", ha="center", va="center")
        n = 5
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((p - mx) * (q - my) for p, q in zip(xs, ys))
        r = sxy / ((sum((p - mx) ** 2 for p in xs) * sum((q - my) ** 2 for q in ys)) ** .5)
        ax.set_xlabel("P(gold) at the unlearned checkpoint   (how much survived unlearning)",
                      color=MUTED, fontsize=11)
        ax.set_ylabel(f"mean truth-ratio recovery over the 5 relearning languages",
                      color=MUTED, fontsize=11)
        ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
        ax.set_title("Recovery is not governed by what survived unlearning\n"
                     f"H1_depth predicted a strong positive slope; observed r = {r:+.2f}\n"
                     f"epoch {a.epoch}  |  {fset}",
                     fontsize=11.5, loc="left", color=INK, pad=12)
        style(ax)
        save(fig, f"stage3_depth{tag}.png",
             "French started BELOW the never-taught floor - no measurable knowledge left - "
             "and still recovered.\n" + NOISE)

    # ---- metrics: the same recovery on three scales -----------------------------------
    def fig_metrics():
        fig, ax = plt.subplots(figsize=(9.6, 6.0))
        w = 0.26
        for k, (key, label) in enumerate([("tr", "truth ratio (ranking)"),
                                          ("prob", "P(gold) (probability)"),
                                          ("nli", "NLI (says it out loud)")]):
            ys = [sum(rec(ul, rl, a.epoch, key=key) for rl in LANGS) / 5 for ul in LANGS]
            ax.bar([i + (k - 1) * w for i in range(5)], ys, w * 0.92, label=label,
                   color=["#9ec4e8", "#3f7fbf", "#15406b"][k], zorder=3)
        ax.axhline(0, color=GRID, lw=1)
        ax.set_xticks(range(5), LANGS)
        for t, l in zip(ax.get_xticklabels(), LANGS):
            t.set_color(C[l])
        ax.tick_params(labelsize=12)
        ax.set_xlabel("language the UNLEARNING trained on", color=MUTED, fontsize=11)
        ax.set_ylabel("mean recovery over the 5 relearning languages", color=MUTED,
                      fontsize=11)
        ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
        ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper left")
        ax.set_title("The fact is re-ranked and made probable again,\nbut still not "
                     f"spoken  |  epoch {a.epoch}  |  {fset}", fontsize=11.5, loc="left",
                     color=INK, pad=12)
        style(ax)
        save(fig, f"stage3_metrics{tag}.png",
             "NLI is scored on generated text, so it also tracks output language "
             "(~90% French after English relearning against 98-99% elsewhere).\n" + NOISE)

    # ---- phrasing: did the FACT come back, or the trained WORDING? ---------------------
    def fig_phrasing():
        if not base:
            print("skip phrasing: no results/p1_baseline/ "
                  "(run 07_measure_p1_baseline.sbatch)")
            return
        le = base["tofu_learn_full_full_qwen3-8b_fr"]
        ub = {l: next(v for k, v in base.items() if f"ul{l}_floornone" in k) for l in LANGS}

        def r2(ul, rl, key):
            c = cells[(ul, rl, a.epoch)]
            u, l_, v = mean(ub[ul], key), mean(le, key), mean(c, key)
            return (u - v) / (u - l_)
        fig, ax = plt.subplots(figsize=(8.4, 7.4))
        lim = [0, 1]
        ax.plot(lim, lim, color="#b8b7b2", lw=1.5, ls="--", zorder=0)
        ax.text(0.97, 0.99, "above the line = the PARAPHRASE recovers more", color=MUTED,
                fontsize=9, ha="right", transform=ax.transAxes)
        n_above = 0
        for ul in LANGS:
            for rl in LANGS:
                x, y = r2(ul, rl, "tr"), r2(ul, rl, "tr_p1")
                n_above += y > x
                ax.plot([x], [y], "o", color=C[ul], ms=10, mec="#ffffff", mew=1.4,
                        alpha=0.95)
        for ul in LANGS:
            ax.plot([], [], "o", color=C[ul], ms=9, label=f"unlearned in {ul}")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
        ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
        ax.set_xlabel("recovery on p0 - the wording LEARN and UNLEARN trained on",
                      color=MUTED, fontsize=11)
        ax.set_ylabel("recovery on p1 - TOFU's published paraphrase", color=MUTED,
                      fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        gp0 = sum(r2(u, r_, "tr") for u in LANGS for r_ in LANGS) / 25
        gp1 = sum(r2(u, r_, "tr_p1") for u in LANGS for r_ in LANGS) / 25
        ax.set_title("What came back is the fact, not the memorised wording\n"
                     f"mean {gp0:.0%} on the trained wording, {gp1:.0%} on the paraphrase\n"
                     f"the paraphrase wins in {n_above} of 25 cells  |  epoch {a.epoch}",
                     fontsize=11.5, loc="left", color=INK, pad=12)
        style(ax)
        save(fig, "stage3_phrasing.png",
             "Both axes use the p1_baseline rescore for BOTH anchors, so p0 and p1 come "
             "from one run and are directly comparable.\n"
             "The learned model is nearly phrasing-invariant (p1 TR 0.619 vs p0 0.605), "
             "which is the baseline this difference-in-differences subtracts.")


    # ---- ladder: learned -> unlearned -> relearned, on BOTH phrasings ----------------
    def fig_ladder():
        """The memorisation test, drawn as a journey rather than a correlation.

        LEARN and UNLEARN both trained on the p0 wording. So if relearning merely
        resurrected a memorised string, p0 would fall back toward the learned level and
        p1 -- a wording the model was never trained on and never unlearned on -- would
        stay up at the forgotten level. The two lines would SPLIT at the third stage.
        They do not: they descend together, which is what "the fact came back" looks
        like and what "the wording came back" cannot look like.
        """
        if not base:
            print("skip ladder: no results/p1_baseline/ "
                  "(run 07_measure_p1_baseline.sbatch)")
            return
        le = base["tofu_learn_full_full_qwen3-8b_fr"]
        fl = base["tofu_learn_retain99_full_qwen3-8b_fr"]
        ub = {l: next(v for k, v in base.items() if f"ul{l}_floornone" in k) for l in LANGS}
        fig, axes = plt.subplots(1, 5, figsize=(16.5, 6.2), sharey=True)
        stages = ["learned", "unlearned", f"relearned ep{a.epoch}"]
        for ax, ul in zip(axes, LANGS):
            rel0 = sum(mean(cells[(ul, rl, a.epoch)], "tr") for rl in LANGS) / 5
            rel1 = sum(mean(cells[(ul, rl, a.epoch)], "tr_p1") for rl in LANGS) / 5
            for key, lab, col, ls in (("tr", "p0  the wording LEARN and UNLEARN used",
                                       "#15406b", "-"),
                                      ("tr_p1", "p1  TOFU's paraphrase, never trained on",
                                       "#d98c1f", "--")):
                ys = [mean(le, key), mean(ub[ul], key),
                      rel0 if key == "tr" else rel1]
                ax.plot([0, 1, 2], ys, color=col, lw=2.4, ls=ls, marker="o", ms=8,
                        label=lab, zorder=3)
                for x, y in zip([0, 1, 2], ys):
                    # p0 labels above the marker, p1 below, so the two never overprint
                    # where the lines cross (they nearly coincide at the learned stage)
                    # p0 above the marker, p1 below: the two nearly coincide at the
                    # learned stage and would otherwise overprint
                    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                                xytext=(0, -17 if key == "tr" else 11), ha="center",
                                fontsize=8.5, color=col)
            # a model that was NEVER taught these facts -- the "knows nothing" level
            for key, col in (("tr", "#15406b"), ("tr_p1", "#d98c1f")):
                ax.axhline(mean(fl, key), color=col, lw=1, ls=":", alpha=0.55)
            ax.set_xticks([0, 1, 2], stages, fontsize=9, rotation=18, ha="right")
            ax.set_xlim(-0.42, 2.42)
            ax.set_title(f"unlearned in {ul}", fontsize=11, loc="left", color=C[ul],
                         pad=10)
            ax.margins(y=0.22)      # headroom for the value labels at both extremes
            style(ax)
        axes[0].set_ylabel("truth ratio   (LOW = the model knows the fact)",
                           color=MUTED, fontsize=10)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=9.5, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.5, 0.035))
        fig.suptitle("Relearning restores the fact on a wording it was never trained on",
                     fontsize=13.5, x=0.006, ha="left", y=0.995, color=INK)
        fig.text(0.006, 0.945, "truth ratio is LOW when the model knows the fact, so the "
                 "peak is the forgotten state: learned -> forgotten -> brought back.",
                 fontsize=8.5, color=MUTED)
        fig.text(0.006, 0.915, "if relearning only resurrected the memorised string, the "
                 "dashed p1 line would stay at the peak while the solid p0 line came down."
                 "   dotted lines = a model never taught these facts",
                 fontsize=8.5, color=MUTED)
        fig.tight_layout(rect=[0, 0.105, 1, 0.885])
        fig.text(0.006, 0.010, NOISE, fontsize=8, color=MUTED)
        FIGS.mkdir(exist_ok=True)
        out = FIGS / f"stage3_ladder{tag}.png"
        fig.savefig(out, dpi=170, facecolor=SURFACE)
        print(f"-> {out}")

    # ---- entities: forget01 is 40 attributes of TWO people ----------------------------
    def fig_entities():
        fig, ax = plt.subplots(figsize=(9.6, 6.0))
        A = [i for i in keep if i < 20]
        B = [i for i in keep if i >= 20]
        w = 0.38
        for k, (ks, label, col) in enumerate([(A, "Basil Mahfouz Al-Kuwaiti (facts 0-19)",
                                               "#9ec4e8"),
                                              (B, "Nikolai Abilov (facts 20-39)",
                                               "#15406b")]):
            ys = [sum(rec(ul, rl, a.epoch, ks=ks) for rl in LANGS) / 5 for ul in LANGS]
            ax.bar([i + (k - 0.5) * w for i in range(5)], ys, w * 0.92, label=label,
                   color=col, zorder=3)
        ax.axhline(0, color=GRID, lw=1)
        ax.set_xticks(range(5), LANGS)
        for t, l in zip(ax.get_xticklabels(), LANGS):
            t.set_color(C[l])
        ax.tick_params(labelsize=12)
        ax.set_xlabel("language the UNLEARNING trained on", color=MUTED, fontsize=11)
        ax.set_ylabel("mean truth-ratio recovery", color=MUTED, fontsize=11)
        ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
        ax.legend(frameon=False, fontsize=9.5, loc="upper left")
        ax.set_title("forget01 is 40 attributes of TWO entities,\nand they do not behave "
                     f"alike  |  epoch {a.epoch}  |  {fset}", fontsize=11.5, loc="left",
                     color=INK, pad=12)
        style(ax)
        save(fig, f"stage3_entities{tag}.png",
             "This is why fact-level resampling overstates n: a true two-entity cluster "
             "has n=2 and supports no interval.\n" + NOISE)

    # ---- perfact: the distribution the cell means hide --------------------------------
    def fig_perfact():
        import matplotlib.colors as mcolors
        cols = [(ul, rl) for ul in LANGS for rl in LANGS]
        M = np.array([[per_fact(cells[(ul, rl, a.epoch)], "tr")[i] for ul, rl in cols]
                      for i in range(40)])
        fig, ax = plt.subplots(figsize=(13.5, 10.5))
        # log scale: unlearning moves the ratio MULTIPLICATIVELY, and a linear ramp
        # renders 38 facts as one flat block beside the few that move a long way.
        im = ax.imshow(M, cmap=RAMP, aspect="auto",
                       norm=mcolors.LogNorm(vmin=max(M.min(), 1e-3), vmax=M.max()))
        ax.set_xticks(range(25), [f"{u}→{r}" for u, r in cols], rotation=90,
                      fontsize=7.5)
        for t, (u, _) in zip(ax.get_xticklabels(), cols):
            t.set_color(C[u])
        ax.set_yticks(range(40), [str(i) for i in range(40)], fontsize=7)
        for i in CEILING_FAIL:
            ax.get_yticklabels()[i].set_color("#b8860b")
            ax.get_yticklabels()[i].set_fontweight("bold")
        ax.axhline(19.5, color="#ffffff", lw=2)
        ax.text(-2.6, 9.5, "Basil", rotation=90, va="center", ha="center", fontsize=9,
                color=MUTED)
        ax.text(-2.6, 29.5, "Nikolai", rotation=90, va="center", ha="center", fontsize=9,
                color=MUTED)
        ax.set_xlabel("cell  (unlearned → relearned)", color=MUTED, fontsize=11)
        ax.set_ylabel("fact", color=MUTED, fontsize=11)
        ax.tick_params(colors=MUTED, length=0)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(colors=MUTED,
                                                                       labelsize=8)
        ax.set_title("Every fact in every cell: truth ratio after relearning "
                     "(LOW = the model knows it)\n"
                     f"log ramp, epoch {a.epoch}  |  gold fact numbers fail the Stage 1 "
                     "ceiling check\n"
                     f"reference: learned {mean(learn, 'tr'):.3f}, "
                     f"never-taught {mean(s1['fr_retain'], 'tr'):.3f}",
                     fontsize=11.5, loc="left", color=INK, pad=12)
        save(fig, "stage3_perfact.png",
             "The means in every other figure are column averages of this. "
             "Leave-one-fact-out moves a cell by 6-48pp, so no single cell mean is sharp.")

    # ---- output language: what the model answers in, and what it does to NLI ----------
    def fig_output():
        fig, ax = plt.subplots(figsize=(9.2, 6.2))
        for ul in LANGS:
            for rl in LANGS:
                c = cells[(ul, rl, a.epoch)]
                g = c["summary"].get("gen_language_counts") or {}
                share = g.get("fr", 0) / max(sum(g.values()), 1)
                ax.plot([share], [c["summary"]["nli_score_mean"]], "o", color=C[rl],
                        ms=10, mec="#ffffff", mew=1.4)
        for rl in LANGS:
            ax.plot([], [], "o", color=C[rl], ms=9, label=f"relearned in {rl}")
        ax.set_xlabel("share of the 40 answers actually generated in French", color=MUTED,
                      fontsize=11)
        ax.set_ylabel("NLI equivalence with the gold French answer", color=MUTED,
                      fontsize=11)
        ax.xaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.set_title("NLI is scored on generated text,\nso it also measures output "
                     f"language  |  epoch {a.epoch}", fontsize=11.5, loc="left", color=INK,
                     pad=12)
        style(ax)
        save(fig, "stage3_output.png",
             "Truth ratio and P(gold) are forced-sequence scores and are NOT exposed to "
             "this; only NLI is.")

    def fig_nli():
        """WHY THE MEAN NLI IS 0.23 -- it is not a middling score, it is a mixture.

        The truth ratio recovers 56% while NLI recovers 11%, and the obvious reading is
        "the model can rank the answer but not verbalise it". This panel says what is
        actually happening: NLI is almost binary. Either the generation states the fact
        (>=0.9) or it states a confabulated substitute (<0.1), and the mean is just the
        FRACTION in the first group. Reporting it as a central tendency implies a typical
        generation that is half-right, and there is no such generation."""
        allf = [f for u in LANGS for r in LANGS for f in cells[(u, r, a.epoch)]["per_fact"]]
        n = len(allf)
        edges = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.001]
        cnt = [sum(1 for f in allf if lo <= f["nli_score"] < hi)
               for lo, hi in zip(edges, edges[1:])]
        fig, ax = plt.subplots(figsize=(9.2, 5.0))
        for i, (lo, hi, k) in enumerate(zip(edges, edges[1:], cnt)):
            hot = "#b8442a" if hi <= .1 else "#1baf7a" if lo >= .9 else "#9a9992"
            ax.bar(i, k / n * 100, width=0.86, color=hot,
                   edgecolor=SURFACE, linewidth=2)
            if k / n > .02:
                ax.text(i, k / n * 100 + 1.2, f"{k/n:.0%}", ha="center", fontsize=9,
                        color=INK)
        ax.set_xticks(range(len(cnt)))
        ax.set_xticklabels([f"{lo:.2f}" if i % 2 == 0 else ""
                            for i, lo in enumerate(edges[:-1])], fontsize=8.5)
        lo_ = sum(1 for f in allf if f["nli_score"] < .1)
        hi_ = sum(1 for f in allf if f["nli_score"] >= .9)
        mid = n - lo_ - hi_
        mu = sum(f["nli_score"] for f in allf) / n
        ax.set_xlabel("NLI score of one generation", color=MUTED, fontsize=11)
        ax.set_ylabel("% of the 1000 relearned generations", color=MUTED, fontsize=11)
        ax.set_title(f"NLI is bimodal, so its mean is a proportion, not an average"
                     f"\n{lo_/n:.0%} state a wrong fact  ·  {hi_/n:.0%} state the right "
                     f"one  ·  only {mid/n:.0%} in between  ·  mean {mu:.3f}",
                     fontsize=12, loc="left", color=INK, pad=12)
        style(ax)
        fail = [f for f in allf if f["nli_score"] < .1]
        save(fig, f"stage3_nli_ep{a.epoch}.png",
             f"Of the failures, {sum(1 for f in fail if f.get('gen_lang')=='fr')/len(fail):.0%}"
             f" are written in French and "
             f"{sum(1 for f in fail if f.get('pn_xy',0)>.5)/len(fail):.0%} are scored NEUTRAL "
             f"rather than contradiction.\nThey are fluent French sentences asserting a "
             f"different fact -- not broken output, and not a language switch.")

    def fig_gaptable():
        """The TR-NLI gap as a table, because the scatter asked too much of the reader.

        Six rows, three numbers each. The two recovery columns are the SAME normalised
        quantity on two metrics -- fraction of the way from unlearned back to learned --
        which is what makes them comparable. The third column drops the normalisation
        entirely and just counts facts the model says out loud, with the learned and
        unlearned models as the two anchors, so a reader who distrusts the normalisation
        can still read the claim off raw counts."""
        ep = a.epoch          # `a` is the argparse namespace; do not shadow it below
        def rec(u, r, k):
            u0, c = mean(unl[u], k), mean(s1["fr_ft"], k)
            return (u0 - mean(cells[(u, r, ep)], k)) / (u0 - c) * 100
        def said(u, r):
            return sum(1 for f in cells[(u, r, ep)]["per_fact"] if f["nli_score"] >= .9)
        avg = lambda v: sum(v) / len(v)
        rows = [(u, avg([rec(u, r, "tr") for r in LANGS]),
                 avg([rec(u, r, "nli") for r in LANGS]),
                 avg([said(u, r) for r in LANGS])) for u in LANGS]
        rows.sort(key=lambda x: x[1])
        allr = ("all 25 cells",
                avg([rec(u, r, "tr") for u in LANGS for r in LANGS]),
                avg([rec(u, r, "nli") for u in LANGS for r in LANGS]),
                avg([said(u, r) for u in LANGS for r in LANGS]))
        lrn = sum(1 for f in s1["fr_ft"]["per_fact"] if f["nli_score"] >= .9)
        unlr = avg([sum(1 for f in unl[u]["per_fact"] if f["nli_score"] >= .9)
                    for u in LANGS])
        body = rows + [allr]
        fig, ax = plt.subplots(figsize=(9.6, 5.6))
        ax.set_xlim(0, 10); ax.set_ylim(len(body) + 2.4, -1.9); ax.axis("off")
        cols = [(0.15, "unlearned in", "left"), (4.0, "recovered on the\nTRUTH RATIO", "right"),
                (6.6, "recovered on\nNLI", "right"),
                (9.85, "facts it actually\nSAYS  (of 40)", "right")]
        for x, lab, ha in cols:
            ax.text(x, -1.25, lab, fontsize=9.5, color=MUTED, ha=ha, va="center")
        ax.plot([0, 10], [-0.35, -0.35], color=MUTED, lw=1.1)
        for i, (name, tr, nli, sd) in enumerate(body):
            last = i == len(body) - 1
            if last:
                ax.plot([0, 10], [i - 0.06, i - 0.06], color=MUTED, lw=1.1)
            elif i % 2 == 0:
                ax.add_patch(plt.Rectangle((0, i - 0.06), 10, 0.92,
                                           facecolor="#f2f1ec", edgecolor="none"))
            w = "bold" if last else "normal"
            ax.text(0.15, i + 0.4, name, fontsize=11, color=INK, va="center",
                    fontweight=w)
            # A bar behind each recovery number: the gap is a ratio, and two columns of
            # bare percentages make the reader do that division in their head.
            for x0, v in ((2.3, tr), (4.9, nli)):
                ax.add_patch(plt.Rectangle((x0, i + 0.13), v / 100 * 1.25, 0.54,
                                           facecolor="#cfe0f4" if x0 < 4 else "#f0c9bd",
                                           edgecolor="none"))
            ax.text(4.0, i + 0.4, f"{tr:.0f}%", fontsize=12, color=INK, ha="right",
                    va="center", fontweight=w)
            ax.text(6.6, i + 0.4, f"{nli:.0f}%", fontsize=12, color=INK, ha="right",
                    va="center", fontweight=w)
            ax.text(9.85, i + 0.4, f"{sd:.1f}", fontsize=12, color=INK, ha="right",
                    va="center", fontweight=w)
        y = len(body) + 0.7
        ax.plot([0, 10], [y - 0.55, y - 0.55], color=GRID, lw=1)
        for lab, sd, dy in (("for comparison: the LEARNED model", lrn, 0),
                            ("the UNLEARNED models, before relearning", unlr, 1.0)):
            ax.text(0.15, y + dy, lab, fontsize=9.5, color=MUTED, va="center")
            ax.text(9.85, y + dy, f"{sd:.1f}", fontsize=10.5, color=MUTED, ha="right",
                    va="center")
        ax.set_title("The same 25 models recover 56% by one measure and 12% by the other",
                     fontsize=13, loc="left", color=INK, pad=16)
        import textwrap as _tw
        fig.tight_layout(rect=[0, 0.115, 1, 1])
        fig.text(0.012, 0.012, "\n".join(_tw.wrap(
            "Both recovery columns are (unlearned - relearned) / (unlearned - learned) on "
            "their own metric: 100% = fully back to the learned model, 0% = no movement. "
            "The last column drops the normalisation and counts facts whose greedy French "
            "generation states the fact (NLI >= 0.9). Rows are the language UNLEARNING "
            "trained on, averaged over the five relearning languages. Single seed; a "
            "rescore moves a row by up to ~2 points.", 132)),
            fontsize=7.5, color=MUTED, va="bottom")
        FIGS.mkdir(exist_ok=True)
        out = FIGS / f"stage3_gaptable_ep{ep}.png"
        fig.savefig(out, dpi=170, facecolor=SURFACE)
        print(f"-> {out}")

    def fig_gap():
        """The TR-NLI gap as ONE picture: the same 25 cells on both yardsticks.

        Both axes are the SAME normalised quantity -- the fraction of the way from the
        unlearned checkpoint back to the learned one -- so the 45-degree line is where a
        cell would sit if the two metrics agreed about how much came back. Every cell
        sits far below it. Putting NLI in recovery units rather than a raw pass rate is
        what makes the comparison legitimate; the anchors differ wildly in absolute terms
        (TR 0.840 -> 0.605, NLI 0.134 -> 0.941) and a raw comparison would be meaningless.

        Scatter rather than paired bars because the claim is not only "the means differ"
        but "no cell escapes" -- 0 of 25 lie above the line, and a bar chart of two means
        cannot show that."""
        def rec(u, r, k):
            a, c = mean(unl[u], k), mean(s1["fr_ft"], k)
            return (a - mean(cells[(u, r, a_ep)], k)) / (a - c) * 100
        a_ep = a.epoch
        fig, ax = plt.subplots(figsize=(7.4, 7.0))
        lo, hi = -18, 96
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=MUTED, zorder=1)
        # Placed mid-line, not at the corner, where it would collide with the title.
        ax.text(52, 52, " if the two metrics agreed", ha="left", va="bottom",
                rotation=45, rotation_mode="anchor", fontsize=8.5, color=MUTED)
        xs, ys = [], []
        for u in LANGS:
            px = [rec(u, r, "tr") for r in LANGS]
            py = [rec(u, r, "nli") for r in LANGS]
            xs += px; ys += py
            ax.scatter(px, py, s=68, color=C[u], edgecolor=SURFACE, linewidth=1.4,
                       label=u, zorder=3)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        ax.axvline(mx, color=GRID, lw=1, zorder=0)
        ax.axhline(my, color=GRID, lw=1, zorder=0)
        ax.annotate(f"mean {mx:.0f}%", (mx, lo + 2), fontsize=8.5, color=MUTED,
                    ha="center", va="bottom")
        ax.annotate(f"mean {my:.0f}%", (lo + 2, my), fontsize=8.5, color=MUTED,
                    ha="left", va="bottom")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("recovered on the TRUTH RATIO  (%)\n"
                      "ranking the answer against five alternatives",
                      color=MUTED, fontsize=10.5)
        ax.set_ylabel("recovered on NLI  (%)\nsaying the fact out loud",
                      color=MUTED, fontsize=10.5)
        ax.set_title("Same 25 models, same normalisation, two yardsticks"
                     f"\n0 of 25 cells recovered as much in generation as in ranking",
                     fontsize=12.5, loc="left", color=INK, pad=12)
        ax.legend(title="unlearning language", fontsize=9, title_fontsize=9,
                  frameon=False, loc="upper left")
        style(ax)
        import textwrap as _tw
        save(fig, f"stage3_gap_ep{a_ep}.png", "\n".join(_tw.wrap(
            "Both axes are (unlearned - relearned) / (unlearned - learned) on their own "
            "metric, so 100% = fully back to the learned model and 0% = no movement. "
            "Absolute anchors: truth ratio 0.840 -> 0.605, NLI 0.134 -> 0.941. A rescore "
            "moves a cell by up to ~4 points on either axis.", 104)))

    def fig_nligrid():
        """The 5x5 in units of "did the model SAY it", not "did the ratio move".

        The recovery grid reads 56% because the truth ratio is a six-way forced choice.
        This one counts the facts whose greedy generation actually states the fact, and
        it is the same experiment measured at the other end. Not a recovery fraction --
        a raw pass rate -- so it is not normalised against the learned model, and the
        learned and unlearned pass rates are printed beside it as the two anchors."""
        pf = lambda u, r: cells[(u, r, a.epoch)]["per_fact"]
        M = [[sum(1 for f in pf(u, r) if f["nli_score"] >= .9) / len(pf(u, r)) * 100
              for u in LANGS] for r in LANGS]        # [relearn][unlearn], X = unlearn
        fig, ax = plt.subplots(figsize=(7.6, 6.4))
        ramp = plt.get_cmap(RAMP)          # RAMP is a colormap NAME, not a callable
        top = max(max(row) for row in M)
        for i, r in enumerate(LANGS):
            for j, u in enumerate(LANGS):
                v = M[i][j]
                ax.add_patch(plt.Rectangle((j + .02, i + .02), .96, .96,
                                           facecolor=ramp(v / max(top, 1e-9) * .85),
                                           edgecolor="none"))
                if u == r:      # the diagonal: relearned in the language it was unlearned in
                    # INK, not white: half these cells are pale and a white outline on a
                    # pale cell is invisible, which is where the diagonal actually sits.
                    ax.add_patch(plt.Rectangle((j + .035, i + .035), .93, .93, fill=False,
                                               edgecolor=INK, lw=1.8))
                ax.text(j + .5, i + .5, f"{v:.0f}%", ha="center", va="center",
                        fontsize=12, color="#ffffff" if v / max(top, 1) > .62 else INK)
        ax.set_xticks([j + .5 for j in range(5)])
        ax.set_xticklabels([u + ("" if u != SEEN else "*") for u in LANGS], fontsize=11)
        ax.set_yticks([i + .5 for i in range(5)])
        ax.set_yticklabels([r + ("  *" if r == SEEN else "") for r in LANGS], fontsize=11)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xlim(0, 5); ax.set_ylim(5, 0)
        ax.set_xlabel("language the UNLEARNING trained on", color=MUTED, fontsize=11)
        ax.set_ylabel("language the RELEARNING trained on", color=MUTED, fontsize=11)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0, colors=MUTED)
        lrn = sum(1 for f in s1["fr_ft"]["per_fact"] if f["nli_score"] >= .9) / 40 * 100
        # `mean` in this scope is the study's mean(rec, key) helper, not a list mean.
        pcts = [sum(1 for f in unl[u]["per_fact"] if f["nli_score"] >= .9) / 40 * 100
                for u in LANGS]
        unlr = sum(pcts) / len(pcts)
        ret = sum(1 for f in s1["fr_retain"]["per_fact"] if f["nli_score"] >= .9) / 40 * 100
        ax.set_title("% of the 40 facts the model actually STATES after relearning"
                     f"\n(NLI >= 0.9 on the greedy French generation, epoch {a.epoch})",
                     fontsize=12.5, loc="left", color=INK, pad=14)
        # save()'s note path reserves 4.5% of the height, which a four-line note
        # overruns straight into the bottom row of cells. Lay this one out by hand.
        import textwrap as _tw
        note = "\n".join(_tw.wrap(
            f"Anchors: learned {lrn:.0f}%  ·  unlearned, before any relearning "
            f"{unlr:.0f}%  ·  never taught {ret:.0f}%.   Outlined = relearned in the "
            f"language it was unlearned in;  * = the fr arm re-sees data LEARN already "
            f"trained on. The truth-ratio recovery grid reads 56% for these same 25 "
            f"cells -- the same models, measured at the other end. The anchors come from "
            f"the stage-1/2 scoring group and the cells from stage 3; a rescore moves "
            f"either by up to ~4 points.", 112))
        fig.tight_layout(rect=[0, 0.115, 1, 1])
        fig.text(0.012, 0.012, note, fontsize=7.5, color=MUTED, va="bottom")
        FIGS.mkdir(exist_ok=True)
        out = FIGS / f"stage3_nligrid_ep{a.epoch}.png"
        fig.savefig(out, dpi=170, facecolor=SURFACE)
        print(f"-> {out}")

    def fig_gens():
        """THE GENERATIONS THEMSELVES. Every aggregate above is downstream of these."""
        shown = [0, 3, 20]
        stages = [("learned  fr_ft", lambda i: s1["fr_ft"]["per_fact"][i]),
                  ("unlearned  unl_fr", lambda i: unl["fr"]["per_fact"][i]),
                  ("relearned  ul_fr -> ja", lambda i: cells[("fr", "ja", a.epoch)]["per_fact"][i]),
                  ("relearned  ul_ja -> ru", lambda i: cells[("ja", "ru", a.epoch)]["per_fact"][i]),
                  ("never taught  fr_retain", lambda i: s1["fr_retain"]["per_fact"][i])]
        import textwrap
        fig, axes = plt.subplots(len(shown), 1,
                                 figsize=(13.8, 3.05 * len(stages) * len(shown) / 2.4))
        for ax, fi in zip(axes, shown):
            ax.set_xlim(0, 1)
            ax.set_ylim(len(stages) + 0.15, -1.25)
            ax.axis("off")
            ax.text(0, -0.92, f"FACT {fi}   the fact, in English:", fontsize=8.6,
                    color=MUTED, va="center")
            ax.text(0.155, -0.92, textwrap.fill(GOLD_EN.get(fi, ""), 112),
                    fontsize=8.6, color=INK, style="italic", va="center")
            for j, (lab, get) in enumerate(stages):
                f = get(fi)
                v = f["nli_score"]
                col = "#1baf7a" if v >= .9 else "#b8442a" if v < .1 else "#c08a2e"
                ax.add_patch(plt.Rectangle((0, j + 0.04), 1, 0.92, facecolor="#f2f1ec",
                                           edgecolor="none"))
                ax.add_patch(plt.Rectangle((0, j + 0.04), 0.006, 0.92, facecolor=col,
                                           edgecolor="none"))
                ax.text(0.012, j + 0.30, lab, fontsize=8.4, color=MUTED, va="center")
                ax.text(0.012, j + 0.66, f"NLI {v:.3f}", fontsize=9.2, color=col,
                        va="center", fontweight="bold")
                # The model's own French on top, the English gloss beneath it: the reader
                # needs to see WHICH WORD moved, and the answer is one or two nouns.
                fr = textwrap.fill(f["generation"].strip(), 112)[:340]
                en = gloss(f["generation"])
                ax.text(0.155, j + 0.30, fr, fontsize=8.3, color=MUTED, va="center")
                if en:
                    ax.text(0.155, j + 0.70, textwrap.fill(en, 112), fontsize=8.6,
                            color=INK, va="center")
        fig.suptitle("Why NLI stays low while the truth ratio recovers",
                     fontsize=13, x=0.006, ha="left", y=0.995, color=INK)
        # The two name facts and the 38 attribute facts fail differently, and a single
        # averaged claim hides that -- facts 0 and 20 ARE shown here, so say both.
        fig.text(0.006, 0.963,
                 "The failures are fluent French, not broken output and not a language "
                 "switch. On the 38 ATTRIBUTE facts, 98.6% of sub-0.1 generations still "
                 "name the right author and\nconfabulate the attribute (coiffeur for "
                 "fleuriste). On the two NAME facts -- 0 and 20 below -- the answer IS "
                 "the name, so a failure is a different author entirely.\nEither way the "
                 "truth ratio's 6-way forced choice is won while open-ended decoding is "
                 "not.\nEnglish is a hand gloss of the model's own French, which is "
                 "printed beneath it in grey; CAPITALS mark the fact-bearing words.",
                 fontsize=8.7, color=MUTED, va="top")
        fig.tight_layout(rect=[0, 0, 1, 0.945])
        FIGS.mkdir(exist_ok=True)
        out = FIGS / f"stage3_generations_ep{a.epoch}.png"
        fig.savefig(out, dpi=170, facecolor=SURFACE)
        print(f"-> {out}")

    todo = {"grid": fig_grid, "curves": fig_curves, "depth": fig_depth,
            "nli": fig_nli, "gens": fig_gens, "nligrid": fig_nligrid,
            "gap": fig_gap, "gaptable": fig_gaptable,
            "metrics": fig_metrics, "phrasing": fig_phrasing, "ladder": fig_ladder,
            "entities": fig_entities,
            "perfact": fig_perfact, "output": fig_output}
    for k, fn in todo.items():
        if a.plot in ("all", k):
            fn()
            plt.close("all")


if __name__ == "__main__":
    main()
