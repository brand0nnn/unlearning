"""Stage 1, fact by fact: the table the summary means hide.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_stage1_perfact.py
    -> figures/stage1_perfact_table.png  + the same table printed

40 facts, sorted by SEPARATION (fr_retain's truth ratio minus fr_ft's) so the facts the
injection reached least sit at the top. A table rather than a chart because the question
is "which fact", and 40 named items past ~7 categories stop being readable as marks.

Columns, and what each is for:
  TR ft / TR retain / TR base   the same measurement on three models
  sep = retain - ft             the separation, diverging colour: blue = learned,
                                red = the reference model does BETTER (not separated)
  gain = base - ft              how far fine-tuning moved this fact from the untrained
                                model -- and it is NOT evidence that the fact was learned,
                                because it decomposes exactly:

                                    gain  =  from-retain  +  sep
                                  base-ft   base-retain    retain-ft

                                from-retain is what training on the OTHER 3960 facts gives
                                you on a fact never seen (format, French QA style, related
                                retain material); sep is what is left, the part attributable
                                to seeing THIS fact. Over the 40 facts, from-retain is +0.116
                                of a +0.338 mean gain -- a third of the apparent learning.
                                Fact 31 is the extreme: gain +0.34, all of it from-retain.
  P(gold) / NLI                 the two generation-independent and generation-side checks;
                                a fact can fail on TR and still be plainly known
  flags                         TR>1 (probe broken: a false answer outranks the true one),
                                not-sep, lang (answered in the wrong language)

The truth ratio is TOFU Eq. 1 on the surname-normalized probe.
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
AUTHOR = lambda i: "Basil" if i < 20 else "Abilov"


def load(group="stage1_norm"):
    out = {}
    for f in (RESULTS / group).glob("*.json"):
        rec = json.load(open(f))
        role = ("fr_ft" if "_full_full_" in rec["name"] else
                "fr_retain" if "retain99" in rec["name"] else "base")
        out[role] = rec
    if len(out) < 3:
        sys.exit(f"need all three models in {RESULTS / group}")
    return out


def rows(r):
    ft, rt, bs = (r[k]["per_fact"] for k in ("fr_ft", "fr_retain", "base"))
    tr = lambda f: f["norm"]["tr_arithmetic"]
    out = []
    for i in range(len(ft)):
        flags = []
        if tr(ft[i]) > 1.0:
            flags.append("TR>1")
        if tr(ft[i]) >= tr(rt[i]):
            flags.append("not-sep")
        if ft[i]["gen_lang"] != "fr":
            flags.append("lang")
        out.append({"fact": i, "author": AUTHOR(i), "ft": tr(ft[i]), "rt": tr(rt[i]),
                    "base": tr(bs[i]), "sep": tr(rt[i]) - tr(ft[i]),
                    "gain": tr(bs[i]) - tr(ft[i]),
                    "fromret": tr(bs[i]) - tr(rt[i]),
                    "prob": ft[i]["prob"], "nli": ft[i]["nli_score"],
                    "flags": " ".join(flags)})
    for d in out:      # the decomposition is an identity; check it rather than trust it
        assert abs(d["gain"] - (d["fromret"] + d["sep"])) < 1e-9, d
    return sorted(out, key=lambda d: d["sep"])


def text_table(rs):
    print(f"\n{'fact':>4} {'author':<7}{'TR ft':>7}{'TR ret':>8}{'TR base':>9}"
          f"{'from-ret':>10}{'sep':>8}{'gain':>7}{'P(gold)':>9}{'NLI':>7}  flags")
    for d in rs:
        print(f"{d['fact']:>4} {d['author']:<7}{d['ft']:>7.2f}{d['rt']:>8.2f}"
              f"{d['base']:>9.2f}{d['fromret']:>+10.2f}{d['sep']:>+8.2f}"
              f"{d['gain']:>+7.2f}{d['prob']:>9.2f}"
              f"{d['nli']:>7.2f}  {d['flags']}")
    n = len(rs)
    print(f"\n  separated (sep > 0): {sum(d['sep'] > 0 for d in rs)}/{n}"
          f"   |  P(gold) > 0.4: {sum(d['prob'] > 0.4 for d in rs)}/{n}"
          f"   |  NLI > 0.5: {sum(d['nli'] > 0.5 for d in rs)}/{n}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    rs = rows(load())
    text_table(rs)

    # Diverging: red arm <- neutral gray -> blue arm (dataviz palette: blue<->red,
    # gray midpoint). Magnitude = how strongly the fact separates.
    # Stops are explicit so the NEUTRAL GRAY sits exactly at zero -- with evenly spaced
    # stops it lands off-centre and a negative separation renders pale blue, i.e. the one
    # thing the colour exists to make obvious would read as its opposite.
    cmap = LinearSegmentedColormap.from_list("sep", [
        (0.00, "#9b2b2a"), (0.25, "#e34948"), (0.50, "#f0efec"),
        (0.75, "#86b6ef"), (1.00, "#104281")])
    # Symmetric and capped: the negative arm only reaches -0.11, so scaling to the +1.31
    # maximum would render every failure as indistinguishable grey. The numbers are
    # printed in the cell, so the cap costs no information.
    CAP = 0.5
    norm = TwoSlopeNorm(vmin=-CAP, vcenter=0.0, vmax=CAP)

    cols = [("fact", 0.030, "c"), ("author", 0.105, "l"), ("TR ft", 0.250, "r"),
            ("TR retain", 0.340, "r"), ("TR base", 0.425, "r"), ("from retain", 0.545, "r"),
            ("sep", 0.625, "r"), ("gain", 0.700, "r"), ("P(gold)", 0.785, "r"),
            ("NLI", 0.850, "r"), ("flags", 0.880, "l")]
    n = len(rs)
    fig, ax = plt.subplots(figsize=(10.6, 0.235 * n + 2.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, n + 3.1); ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    ha = {"l": "left", "r": "right", "c": "center"}
    for label, x, al in cols:
        ax.text(x, n + 0.75, label, fontsize=8.5, color=MUTED, ha=ha[al], weight="bold")
    ax.plot([0.02, 0.98], [n + 0.55, n + 0.55], color=GRID, lw=1)

    for k, d in enumerate(rs):
        y = n - k - 0.5
        if k % 2 == 0:
            ax.add_patch(Rectangle((0.02, y - 0.4), 0.96, 0.8, color="#f4f3f0", lw=0))
        ax.add_patch(Rectangle((0.573, y - 0.4), 0.056, 0.8,
                               color=cmap(norm(max(-CAP, min(CAP, d["sep"])))), lw=0))
        vals = [(f"{d['fact']}", 0.032, "c"), (d["author"], 0.115, "l"),
                (f"{d['ft']:.2f}", 0.275, "r"), (f"{d['rt']:.2f}", 0.375, "r"),
                (f"{d['base']:.2f}", 0.425, "r"), (f"{d['fromret']:+.2f}", 0.545, "r"),
                (f"{d['sep']:+.2f}", 0.625, "r"), (f"{d['gain']:+.2f}", 0.700, "r"),
                (f"{d['prob']:.2f}", 0.785, "r"), (f"{d['nli']:.2f}", 0.850, "r"),
                (d["flags"], 0.880, "l")]
        for j, (txt, x, al) in enumerate(vals):
            colour = INK
            if j == 9 and d["nli"] < 0.5:
                colour = "#c2541f"
            if j == 10:
                colour = "#c2541f"
            ax.text(x, y, txt, fontsize=8.2, color=colour, ha=ha[al], va="center",
                    family="DejaVu Sans Mono" if j not in (1, 10) else None)

    ax.text(0.02, n + 2.45, "Stage 1, fact by fact: what the means hide",
            fontsize=12.5, color=INK, weight="bold")
    ax.text(0.02, n + 1.75, "40 French forget facts, sorted by separation (fr_retain "
            "truth ratio minus fr_ft's). Least-learned first; colour saturates at "
            f"+/-{CAP}.", fontsize=8.5, color=MUTED)
    ax.text(0.02, -0.9, "gain = from-retain + sep, exactly.  from-retain is what training "
            "on the OTHER 3960 facts buys on a fact never seen; sep is what seeing THIS "
            "fact added.\nsep is therefore the controlled number, and it is also the room "
            "unlearning has to move: no gap over fr_retain, nothing to forget.\n"
            "Fact 31: gain +0.34, from-retain +0.37 - nothing specific was learned.   "
            "TR>1 = the probe is broken there: a false answer outranks the true one.",
            fontsize=8, color=MUTED, va="top")
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "stage1_perfact_table.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
