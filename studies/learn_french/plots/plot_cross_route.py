"""Cross-route token probe (10): ask in language L, score the FRENCH answer.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_cross_route.py --plot all

    --plot gate    DOES THE FACT REACH THE MODEL?  learned / never-taught, per question
                   language, two ways: answer in L (09) vs answer in French (10). Skow et al.'s
                   asymmetry (question side transfers, answer side barely) on our model.
    --plot routes  UNLEARNING ARM x QUESTION LANGUAGE.  How much of the learned access each
                   arm removed, per question route. Diagonal = asked in the arm's own language.
    --plot controls  The 'no question' and 'French paraphrase' routes, beside the trained
                   French question -- method checks, kept out of the main grid.
    --plot slots   FRENCH UNLEARNING, SLOT BY SLOT.  French question vs the four foreign
                   questions, occupations/genres vs proper nouns.

"Removed" is the log-scale share of the learned advantage that is gone:
    (log p_learned - log p_unlearned) / (log p_learned - log p_never-taught)
0 = untouched, 1 = back to the never-taught model (fr_retain), >1 = pushed below it. This is
Skow et al.'s oracle-normalised removal (1 - R) taken on LOG probability; theirs is linear,
which calls a drop from 0.9 to 0.1 "89% removed" even when the answer is still ~100x above
never-taught -- the wrong reading for a hiding question. A cell enters only where it passes
Skow's eligibility rule on that route: p_learned >= 0.10 and p_learned - p_never >= 0.05.
The year slot (f10) is left out of the summaries: "1980" is a generic guess even for the
never-taught model (p 0.3-0.6), so it has almost nothing learned to remove.
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
TYPE_COLOR = {"common": "#eb6834", "proper": "#2a78d6", "number": "#8a8984"}
TYPE_LABEL = {"common": "occupation / genre", "proper": "name / award / place / title",
              "number": "year"}
FOREIGN = ["en", "id", "ja", "ru"]
ROUTE_LABEL = {"blank": "no\nquestion", "fr": "French", "fr_p1": "French\nparaphrase",
               "en": "English", "id": "Indonesian", "ja": "Japanese", "ru": "Russian"}


def load():
    f = RESULTS / "cross_route.json"
    if not f.exists():
        sys.exit(f"no {f} -- run slurm/10_cross_route.sbatch and rsync results/ down")
    d = json.load(open(f, encoding="utf-8"))
    ck = d["checkpoints"]
    lab = {ck[0]: "base", ck[1]: "fr_ft", ck[2]: "fr_retain"}
    for c in ck[3:]:
        lab[c] = "unl_" + c.split("_ul")[1][:2]
    R = {lab[c]: v for c, v in d["results"].items()}
    return d, R


def P(R, m, r, s):
    return R[m][r][s]["seq_prob_norm"]


def eligible(R, r, s):
    ft, o = P(R, "fr_ft", r, s), P(R, "fr_retain", r, s)
    return ft >= 0.10 and ft - o >= 0.05


def removed(R, m, r, s):
    ft, o, u = P(R, "fr_ft", r, s), P(R, "fr_retain", r, s), P(R, m, r, s)
    return (math.log(ft) - math.log(u)) / (math.log(ft) - math.log(o))


def finish(fig, out, note=None, bottom=0.0):
    import textwrap
    if note:
        fig.subplots_adjust(bottom=bottom)
        width = int(fig.get_figwidth() * 12.5)    # ~chars per line at 7.5 pt
        fig.text(0.008, 0.012, "\n".join(textwrap.wrap(note, width)), fontsize=7.5,
                 color=MUTED, va="bottom")
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / out, dpi=170, facecolor=SURFACE)
    print(f"-> {FIGS / out}")


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def fig_gate(d, R):
    """Learned / never-taught, answer in L (09) vs answer in French (10)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    native = None
    f9 = RESULTS / "top_tokens_ml.json"
    if f9.exists():
        n = json.load(open(f9, encoding="utf-8"))
        nck = n["checkpoints"]
        native = {"fr_ft": n["results"][nck[1]], "fr_retain": n["results"][nck[2]]}
    typ = {p["id"]: p["type"] for p in d["probes"]}
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    style(ax)
    ax.axhline(1, color=MUTED, lw=1.0)
    ax.text(-0.55, 1.15, "no difference from the never-taught model", fontsize=8,
            color=MUTED, ha="left", va="bottom")
    for i, lg in enumerate(FOREIGN):
        if native:
            for slot, cells in native["fr_ft"][lg].items():
                if lg not in cells:
                    continue
                v = cells[lg]["seq_prob_norm"] / native["fr_retain"][lg][slot][lg]["seq_prob_norm"]
                ax.scatter(i - 0.18, v, s=46, color="#b9b8b2", edgecolor=SURFACE, lw=1,
                           zorder=3)
        for s in typ:
            if typ[s] == "number":
                continue
            v = P(R, "fr_ft", lg, s) / P(R, "fr_retain", lg, s)
            ok = eligible(R, lg, s)
            ax.scatter(i + 0.18, v, s=46, zorder=3,
                       color=TYPE_COLOR[typ[s]] if ok else "none",
                       edgecolor=TYPE_COLOR[typ[s]], lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(range(4))
    ax.set_xticklabels([ROUTE_LABEL[l] for l in FOREIGN], fontsize=10, color=INK)
    ax.set_xlim(-0.6, 3.75)
    ax.set_ylabel("learned / never-taught  (log scale)", color=MUTED, fontsize=9.5)
    ax.set_xlabel("language the question is asked in", color=MUTED, fontsize=9.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color="#b9b8b2", ms=7,
                label="answer in that language too  (09: 4 slots)")]
    h += [Line2D([], [], marker="o", ls="", color=TYPE_COLOR[t], ms=7,
                 label=f"answer in French, {TYPE_LABEL[t]}  (10)") for t in ("common", "proper")]
    h += [Line2D([], [], marker="o", ls="", mfc="none", mec=MUTED, ms=7,
                 label="hollow = fails the eligibility rule")]
    ax.legend(handles=h, frameon=False, fontsize=8.3, loc="lower left", ncol=2,
              bbox_to_anchor=(0, 1.0))
    fig.suptitle("Does a question in another language reach the French-learned fact?",
                 fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.text(.008, .935, "Each dot is one slot: how much more likely the correct answer is "
             "under the learned model than under a model with the same French training that "
             "never saw the fact.\nGrey: question AND answer in that language. Coloured: "
             "question in that language, answer kept in French.", fontsize=8.6,
             color=MUTED, va="top")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76)
    finish(fig, "cross_route_gate.png",
           "Teacher-forced, length-normalised probability of the whole target. One prompt "
           "per dot, single seed. The grey and coloured dots are different slots and prompts "
           "(09 vs 10), so compare the two clouds, not individual dots.", bottom=0.17)


def _grid(d, R, routes, out, title, subtitle, groups, divider, note):
    """Unlearning arm x question route: median share removed, one cell per pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "rem", ["#f7f9fc", "#cfe0f4", "#8fb8e8", "#4a86cf", "#1f4f8f", "#13314f"])
    slots = [p["id"] for p in d["probes"] if p["type"] != "number"]
    arms = ["unl_fr", "unl_en", "unl_id", "unl_ja", "unl_ru"]
    fig, ax = plt.subplots(figsize=(1.55 * len(routes) + 3.2, 5.4))
    ax.axis("off")
    for i, m in enumerate(arms):
        for j, r in enumerate(routes):
            v = [removed(R, m, r, s) for s in slots if eligible(R, r, s)]
            med = st.median(v)
            t = min(max(med / 1.5, 0), 1)
            ax.add_patch(plt.Rectangle((j + .03, i + .04), .94, .92,
                                       facecolor=cmap(t), edgecolor="none"))
            if r == m[4:]:
                ax.add_patch(plt.Rectangle((j + .07, i + .08), .86, .84, fill=False,
                                           edgecolor="#b8442a", lw=2.0))
            dark = t > .55
            ax.text(j + .5, i + .42, f"{med:.2f}", ha="center", va="center", fontsize=12,
                    color="#ffffff" if dark else INK)
            ax.text(j + .5, i + .74, f"n={len(v)}", ha="center", va="center", fontsize=7.2,
                    color="#dfe8f3" if dark else MUTED)
    for j, r in enumerate(routes):
        ax.text(j + .5, -.12, ROUTE_LABEL[r], ha="center", va="bottom", fontsize=9.5,
                color=INK)
    for i, m in enumerate(arms):
        ax.text(-.08, i + .5, f"unlearned in {m[4:]}", ha="right", va="center",
                fontsize=10, color=INK)
    if divider is not None:
        ax.plot([divider, divider], [-.02, len(arms) + .02], color=MUTED, lw=1.0)
    for x, txt in groups:
        ax.text(x, -.85, txt, ha="center", fontsize=8.5, color=MUTED, fontweight="bold")
    ax.set_xlim(-1.6, len(routes) + .05)
    ax.set_ylim(len(arms) + .05, -1.0)
    fig.suptitle(title, fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.text(.008, .935, subtitle, fontsize=8.6, color=MUTED, va="top")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80)
    finish(fig, out, note, bottom=0.14)


NOTE = ("Log-probability version of Skow et al.'s oracle-normalised removal; the year slot "
        "is excluded (generic guess even for the never-taught model). 13 slots from 9 facts "
        "about 2 authors, one prompt each, single seed. Every arm was stopped at the same "
        "French TRUTH RATIO; this probe measures answer probability, on which the arms were "
        "not matched.")


def fig_routes(d, R):
    """THE hiding figure: French question vs the four foreign question languages."""
    _grid(d, R, ["fr", "en", "id", "ja", "ru"], "cross_route_routes.png",
          "How much each unlearning language removed, by the language of the question",
          "Median share of the learned advantage removed (0 = untouched, 1 = back to the "
          "never-taught model, >1 = below it); answer always in French.\nRed outline = asked "
          "in the language the model was unlearned in. Read a ROW: is the outlined cell the "
          "darkest?\nRead the top row against the rest: French unlearning reaches the other "
          "question languages, the others mostly do not.",
          [(.5, "the learning language"), (3, "question in ANOTHER language")], 1, NOTE)


def fig_controls(d, R):
    """The two controls, beside the trained French question they are read against."""
    _grid(d, R, ["blank", "fr", "fr_p1"], "cross_route_controls.png",
          "Controls: no question, and a reworded French question",
          "Same measure as the main grid. 'No question' = the French answer start alone "
          "(how much the memorised answer recalls by itself).\n'French paraphrase' = TOFU's "
          "reworded question: a change of WORDING, not of language.",
          [], None, NOTE)


def fig_slots(d, R):
    """French unlearning, per slot: French question vs the four foreign questions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    typ = {p["id"]: p["type"] for p in d["probes"]}
    gloss = {p["id"]: p["gloss"] for p in d["probes"]}
    rows = []
    for s in typ:
        if typ[s] == "number":
            continue
        fr = removed(R, "unl_fr", "fr", s)
        fo = [(r, removed(R, "unl_fr", r, s)) for r in FOREIGN if eligible(R, r, s)]
        rows.append((typ[s], st.median(v for _, v in fo), s, fr, fo))
    rows.sort(key=lambda x: (x[0] != "common", -x[1]))
    XMAX = 3.0
    fig, ax = plt.subplots(figsize=(12.5, 0.5 * len(rows) + 2.6))
    style(ax)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=MUTED, lw=1.0)
    ax.axvline(1, color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax.text(1.02, -0.85, "back to never-taught", fontsize=8, color=MUTED, va="center")
    ax.axvspan(1, XMAX, color="#f2f1ec", zorder=0)
    ax.text(XMAX - .02, -0.85, "pushed BELOW never-taught", fontsize=8, color=MUTED,
            ha="right", va="center")
    for i, (t, med, s, fr, fo) in enumerate(rows):
        c = TYPE_COLOR[t]
        for r, v in fo:
            x = min(v, XMAX)
            ax.scatter(x, i, s=36, facecolor="none", edgecolor=c, lw=1.2, zorder=3)
            ax.text(x, i - 0.32, r, fontsize=6.5, ha="center", color=MUTED)
        x = min(fr, XMAX)
        ax.scatter(x, i, s=80, color=c, edgecolor=SURFACE, lw=1.5, zorder=4)
        if fr > XMAX or any(v > XMAX for _, v in fo):
            ax.text(XMAX + .04, i, f"max {max([fr] + [v for _, v in fo]):.1f} >",
                    fontsize=7.5, color=MUTED, va="center")
    ncommon = sum(1 for r in rows if r[0] == "common")
    ax.axhline(ncommon - 0.5, color=MUTED, lw=1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([gloss[r[2]] for r in rows], fontsize=9.5, color=INK)
    for i, r in enumerate(rows):
        ax.get_yticklabels()[i].set_color(TYPE_COLOR[r[0]])
    ax.set_ylim(len(rows) - 0.5, -1.2)
    ax.set_xlim(-0.25, XMAX + 0.45)
    ax.set_xlabel("share of the learned advantage removed by FRENCH unlearning",
                  color=MUTED, fontsize=9.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=INK, ms=8, label="asked in French"),
         Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=6,
                label="asked in en / id / ja / ru (labelled)")]
    ax.legend(handles=h, frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("French unlearning, slot by slot: the type of fact matters more than the "
                 "question language", fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.text(.008, .94, "Occupations (orange) were pushed below the never-taught level in almost "
             "every question language; names and places (blue, bottom) were barely suppressed, "
             "even asked in French.\nThe big differences are BETWEEN slots, not between "
             "question languages. Hollow dots a little LEFT of the filled one = a foreign "
             "question leaves slightly more of the fact.", fontsize=8.6, color=MUTED,
             va="top")
    fig.subplots_adjust(left=0.27, right=0.95, top=0.86)
    finish(fig, "cross_route_slots.png",
           "Log-scale removal; foreign cells shown only where they pass the eligibility "
           "rule. Values beyond 3 are drawn at 3 and their maximum is printed. One prompt "
           "per dot, single seed, 2 authors.", bottom=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="all", choices=["all", "gate", "routes", "controls", "slots"])
    a = ap.parse_args()
    d, R = load()
    for k, fn in (("gate", fig_gate), ("routes", fig_routes),
                  ("controls", fig_controls), ("slots", fig_slots)):
        if a.plot in ("all", k):
            fn(d, R)


if __name__ == "__main__":
    main()
