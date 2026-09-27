"""Top-token tables: what token does the model want in the fact slot, in which language?

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_top_tokens.py --plot all

    --plot rank   THE MAIN TABLE. 5 probe slots x 5 languages x 6 checkpoints, cell = the
                  global rank of that language's spelling of the answer. Dark = the model
                  wants that token. This is the suppression-vs-deletion picture at token
                  level: read a row left to right.
    --plot elev   THE CALIBRATION GATE. P(token | fr_ft) / P(token | base Qwen3), one cell
                  per slot x language. LEARN was French-only, so a language that does not
                  light up here never knew the fact and nothing there could have been
                  suppressed -- its column in every other figure is UNMEASURABLE, not null.
    --plot prob   the same table as raw probability rather than rank, log ramp.

Files land in figures/ as top_tokens_<plot>.png.

HOW THE NUMBERS WERE MADE (scripts/top_tokens.py). The prompt is the French question plus
the French gold answer truncated right before the fact word, so the next token IS the
answer; one forward pass per (checkpoint, slot) gives one next-token distribution, read
five ways. FOUR facts, FIVE slots (fact 3 contributes two), SIX checkpoints -- 30 forward
passes, ONE PROMPT PER CELL. These are not 40-fact averages like every truth-ratio number
in the study, and they carry no error bar.

WHAT THE TABLES MARK RATHER THAN FIX:
  * Only the FIRST BPE token of each target is measured -- the row label shows it. Some are
    weak proxies for their language (' B' is the start of any capitalised Russian word,
    ' pen' of 'penser'); those rows are hatched.
  * fr/en/id share one identical token on every proper-noun slot, so their three cells
    there are the SAME NUMBER by construction, not an agreement between languages. Marked
    with a bracket in the row labels.
  * Probabilities reach 1e-15. A ratio of 0.0x can be 1e-11 -> 1e-13, i.e. bf16 noise, not
    a measured suppression. The elevation figure greys any cell whose base probability is
    below 1e-9.
"""
import argparse
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures"
LANGS = ["fr", "en", "id", "ja", "ru"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
SCAN = 20000        # top_tokens.py only ranked this deep; beyond it rank is None
NOISE_FLOOR = 1e-9  # below this at base, an elevation ratio is not interpretable

# Tokens that are a poor proxy for their language: a single letter or a prefix whose
# commonest continuation is a different word entirely. Hatched in every figure.
WEAK = {" B", " K", " H", " Б", " К", " Н", "バ", "ニ", "ク", " pen"}

SLOT_TITLE = {
    "f3_father_occupation": "fact 3  father's occupation\n(florist -- all five differ)",
    "f3_mother_occupation": "fact 3  mother's occupation\n(game developer -- all five differ)",
    "f0_author_name":       "fact 0  author's name\n(Basil -- fr/en/id identical)",
    "f2_birth_city":        "fact 2  birth city\n(Kuwait -- fr/en/id near-identical)",
    "f20_author_name":      "fact 20  author's name\n(Nikolai -- fr/en/id identical)",
}


def role(ckpt: str) -> str:
    """Short label for a checkpoint path, in the order top_tokens.py was given them."""
    n = ckpt.rstrip("/").split("/")[-1]
    if "/" not in ckpt:
        return "base\nQwen3"
    if "retain99" in n:
        return "fr_retain\n(never taught)"
    if "_learn_" in n:
        return "fr_ft\n(learned)"
    if "_ul" in n:
        lg = n.split("_ul")[1].split("_")[0]
        tr = n.split("_tr")[-1].replace("p", ".") if "_tr" in n else "?"
        return f"unlearned\n{lg}  TR {tr}"
    return n[:18]


def load():
    f = RESULTS / "top_tokens.json"
    if not f.exists():
        sys.exit(f"no {f} -- run slurm/08_top_tokens.sbatch and rsync results/ down")
    d = json.load(open(f, encoding="utf-8"))
    ckpts = list(d)
    slots = list(d[ckpts[0]])
    return d, ckpts, slots


_CJK = None


def cjk_font():
    """The Japanese targets are DATA -- DejaVu has no CJK, so a ja row would render as
    empty boxes. matplotlib's rcParams fallback chain does NOT fire here (verified: the
    warning still names DejaVu alone), so the font is chosen per string instead."""
    global _CJK
    if _CJK is None:
        import matplotlib.font_manager as fm
        have = {f.name for f in fm.fontManager.ttflist}
        _CJK = next((f for f in ("Hiragino Sans", "Arial Unicode MS", "PingFang SC",
                                 "Osaka", "Heiti TC") if f in have), "DejaVu Sans")
    return _CJK


def fname(text: str) -> str:
    """DejaVu for Latin and Cyrillic (it has both, and better metrics); the CJK face only
    where the string actually needs it, so the numbers keep one consistent typeface."""
    return cjk_font() if any(ord(c) > 0x2E80 for c in text) else "DejaVu Sans"


def ramps():
    from matplotlib.colors import LinearSegmentedColormap
    # Sequential, one hue light -> dark: dark = the model wants this token.
    seq = LinearSegmentedColormap.from_list(
        "want", ["#f7f9fc", "#cfe0f4", "#8fb8e8", "#4a86cf", "#1f4f8f", "#13314f"])
    # Diverging, two hues + a NEUTRAL gray midpoint at 1.0x (no hue at the middle).
    div = LinearSegmentedColormap.from_list(
        "elev", ["#1a5e8a", "#6fa8c9", "#dedcd6", "#e8a05c", "#b8442a"])
    return seq, div


def rowspec(d, ckpts, slots):
    """One row per (slot, language), carrying the token and the collision groups."""
    rows = []
    for s in slots:
        toks = {lg: d[ckpts[0]][s][lg]["token"] for lg in LANGS if lg in d[ckpts[0]][s]}
        dup = {t for t in toks.values() if list(toks.values()).count(t) > 1}
        for lg in LANGS:
            if lg not in toks:
                continue
            rows.append({"slot": s, "lang": lg, "token": toks[lg],
                         "shared": toks[lg] in dup, "weak": toks[lg] in WEAK})
    return rows


def frame(ax):
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, colors=MUTED)


def table(rows, ckpts, values, texts, cmap, vmin, vmax, title, sub, cbar_label,
          out, greyed=None, dark_above=0.62, cticks=None):
    """Shared renderer: rows x checkpoints, coloured cells with the value written in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n, m = len(rows), len(ckpts)
    # Explicit margins, not tight_layout: the slot titles are drawn at negative data x,
    # which tight_layout cannot see, so it over-reserves and leaves a third of the canvas
    # blank. LEFT is chosen so a title starting at x=-3.05 lands just inside the edge.
    fig, ax = plt.subplots(figsize=(1.35 * m + 5.2, 0.34 * n + 2.6))
    for i, r in enumerate(rows):
        for j in range(m):
            v, txt = values[i][j], texts[i][j]
            grey = greyed is not None and greyed[i][j]
            if v is None or grey:
                col, hi = "#eceae4", 0.0
            else:
                hi = (v - vmin) / max(vmax - vmin, 1e-9)
                col = cmap(min(max(hi, 0.0), 1.0))
            # 2px surface gap between fills -- adjacent cells never touch.
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.02), 0.96, 0.96,
                                       facecolor=col, edgecolor="none"))
            if r["weak"]:
                ax.add_patch(plt.Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, fill=False,
                                           hatch="////", edgecolor="#ffffff",
                                           lw=0.0, alpha=0.28))
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=8.4,
                    color="#ffffff" if hi > dark_above else INK)  # values are ASCII
    # Slot bands: a hairline between groups, and the slot named once on the left.
    start = 0
    for s in dict.fromkeys(r["slot"] for r in rows):
        k = sum(1 for r in rows if r["slot"] == s)
        if start:
            ax.plot([-0.02, m + 0.02], [start, start], color=MUTED, lw=1.1, zorder=5,
                    clip_on=False)
        ax.text(-3.05, start + k / 2, SLOT_TITLE.get(s, s), ha="left", va="center",
                fontsize=8.6, color=INK)
        start += k
    ax.set_yticks([i + 0.5 for i in range(n)])
    labels = ax.set_yticklabels([f"{r['lang']}  {r['token']!r}"
                                 + ("  =" if r["shared"] else "") for r in rows],
                                fontsize=8.4)
    for lab, r in zip(labels, rows):
        lab.set_fontname(fname(r["token"]))
    ax.set_xticks([j + 0.5 for j in range(m)])
    ax.set_xticklabels([role(c) for c in ckpts], fontsize=8.6)
    ax.xaxis.set_ticks_position("top")
    ax.set_xlim(0, m)
    ax.set_ylim(n, 0)
    frame(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.015)
    if cticks:      # the scale is a log, but the READER wants ranks, not their logarithm
        cb.set_ticks([v for v, _ in cticks])
        cb.ax.set_yticklabels([lab for _, lab in cticks], fontsize=8)
    cb.set_label(cbar_label, color=MUTED, fontsize=8.4)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    fig.suptitle(title, fontsize=13, x=0.008, ha="left", y=0.988, color=INK)
    fig.text(0.008, 1 - 0.55 / fig.get_figheight(), sub, fontsize=8.6, color=MUTED,
             va="top")
    fig.text(0.008, 0.012,
             "one prompt per cell (French question + gold answer truncated at the fact); "
             "only the FIRST BPE token of each target is measured\n"
             "'=' marks languages sharing one identical token -- those cells are the same "
             "number by construction;  hatched = the token is a weak proxy for its language",
             fontsize=7.6, color=MUTED)
    fig.subplots_adjust(left=0.335, right=0.935,
                        top=1 - 1.5 / fig.get_figheight(),
                        bottom=0.9 / fig.get_figheight())
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / out, dpi=170, facecolor=SURFACE)
    print(f"-> {FIGS / out}")


def fig_rank(d, ckpts, slots, rows):
    import numpy as np
    seq, _ = ramps()
    vals, txts = [], []
    for r in rows:
        vr, tr = [], []
        for c in ckpts:
            rk = d[c][r["slot"]][r["lang"]]["rank"]
            rk = SCAN if rk is None else rk
            # log rank, INVERTED so dark = rank 0 = the model wants this token.
            vr.append(-np.log10(rk + 1))
            tr.append(f">{SCAN // 1000}k" if rk >= SCAN else f"{rk:,}")
        vals.append(vr)
        txts.append(tr)
    ticks = [(-np.log10(r + 1), (f">{SCAN // 1000}k" if r >= SCAN else
                                 f"{r // 1000}k" if r >= 1000 else str(r)))
             for r in (0, 10, 100, 1000, 10000, SCAN)]
    table(rows, ckpts, vals, txts, seq, -np.log10(SCAN + 1), 0.0,
          "Where each language spells the answer, by rank in the next-token distribution",
          "Dark = the model wants that token. rank 0 = its single most likely next token; "
          ">20k = outside the 20,000 scanned.\nRead a row left to right: base -> learned "
          "-> never-taught -> the three unlearned arms, all matched on achieved truth ratio.",
          "rank of that token", "top_tokens_rank.png", cticks=ticks)


def fig_prob(d, ckpts, slots, rows):
    import numpy as np
    seq, _ = ramps()
    vals, txts = [], []
    for r in rows:
        vr, tr = [], []
        for c in ckpts:
            p = d[c][r["slot"]][r["lang"]]["prob"]
            vr.append(np.log10(max(p, 1e-16)))
            tr.append(f"{p:.0e}".replace("e-0", "e-") if p < 1e-3 else f"{p:.3f}")
        vals.append(vr)
        txts.append(tr)
    table(rows, ckpts, vals, txts, seq, -16.0, 0.0,
          "The same table as probability rather than rank",
          "log10 P(token) at the fact slot. The floor is bf16 resolution, not a measurement: "
          "anything below ~1e-12 should be read as 'absent'.",
          "P(token)", "top_tokens_prob.png",
          cticks=[(e, f"1e{e:g}" if e else "1") for e in (-16, -12, -8, -4, 0)])


def fig_elev(d, ckpts, slots, rows):
    """The calibration gate: fr_ft over base, one cell per slot x language."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _, div = ramps()
    base, ft = ckpts[0], ckpts[1]
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    for i, s in enumerate(slots):
        for j, lg in enumerate(LANGS):
            if lg not in d[base][s]:
                continue
            pb, pf = d[base][s][lg]["prob"], d[ft][s][lg]["prob"]
            weak = d[base][s][lg]["token"] in WEAK
            grey = pb < NOISE_FLOOR
            if grey:
                col, hi, txt = "#eceae4", 0.5, "—"     # 0.5 = the ink test's neutral band
            else:
                ratio = pf / pb
                l = np.log10(max(ratio, 1e-6))
                hi = (l + 6) / 12.0                     # -6 .. +6 decades
                col = div(min(max(hi, 0.0), 1.0))
                txt = (f"{ratio:,.0f}x" if ratio >= 10 else
                       f"{ratio:.1f}x" if ratio >= 0.1 else f"{ratio:.0e}".replace("e-0", "e-"))
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.02), 0.96, 0.96,
                                       facecolor=col, edgecolor="none"))
            if weak:
                ax.add_patch(plt.Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, fill=False,
                                           hatch="////", edgecolor="#ffffff", lw=0.0,
                                           alpha=0.3))
            ax.text(j + 0.5, i + 0.42, txt, ha="center", va="center", fontsize=10.5,
                    color="#ffffff" if hi > 0.82 or hi < 0.16 else INK)
            tk = d[base][s][lg]["token"]
            ax.text(j + 0.5, i + 0.74, repr(tk), ha="center", va="center", fontsize=7.6,
                    fontname=fname(tk),
                    color="#ffffff" if hi > 0.82 or hi < 0.16 else MUTED)
            if list(d[base][s][l2]["token"] for l2 in d[base][s]).count(tk) > 1:
                ax.text(j + 0.93, i + 0.14, "=", ha="right", va="center", fontsize=9,
                        color="#ffffff" if hi > 0.82 or hi < 0.16 else MUTED)
    ax.set_xticks([j + 0.5 for j in range(len(LANGS))])
    ax.set_xticklabels(LANGS, fontsize=11)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([i + 0.5 for i in range(len(slots))])
    ax.set_yticklabels([SLOT_TITLE.get(s, s).split("\n")[0] for s in slots], fontsize=9)
    ax.set_xlim(0, len(LANGS))
    ax.set_ylim(len(slots), 0)
    ax.set_xlabel("language whose spelling of the answer is looked up", color=MUTED,
                  fontsize=10)
    ax.xaxis.set_label_position("top")
    frame(ax)
    sm = plt.cm.ScalarMappable(cmap=div, norm=matplotlib.colors.Normalize(-6, 6))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02,
                      ticks=[-6, -3, 0, 3, 6])
    cb.ax.set_yticklabels(["1e-6x", "0.001x", "1x  no change", "1000x", "1e6x"],
                          fontsize=8)
    cb.set_label("P(token | fr_ft) / P(token | base Qwen3)", color=MUTED, fontsize=8.4)
    cb.ax.tick_params(colors=MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Calibration gate: did French-only LEARN raise any other language's "
                 "spelling of the answer?", fontsize=12.5, x=0.008, ha="left", y=0.985,
                 color=INK)
    fig.text(0.008, 0.945,
             "LEARN trained on French only. A language that does not light up here never "
             "knew the fact, so nothing there could\nhave been suppressed -- its cells "
             "elsewhere are UNMEASURABLE, not null.\nGrey = base probability below 1e-9, "
             "where a ratio is bf16 noise rather than a measurement.",
             fontsize=8.6, color=MUTED, va="top")
    fig.text(0.008, 0.018,
             "hatched = the token is a weak proxy for its language (a single letter, or a "
             "prefix whose commonest continuation is another word)\n"
             "fr/en/id are identical on the three proper-noun slots by construction -- "
             "they share one token, so those are not three agreeing measurements",
             fontsize=7.6, color=MUTED)
    fig.subplots_adjust(left=0.20, right=0.86, top=0.80, bottom=0.10)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "top_tokens_elev.png", dpi=170, facecolor=SURFACE)
    print(f"-> {FIGS / 'top_tokens_elev.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="all",
                    choices=["all", "rank", "elev", "prob"])
    a = ap.parse_args()
    d, ckpts, slots = load()
    rows = rowspec(d, ckpts, slots)
    print(f"{len(ckpts)} checkpoints x {len(slots)} slots "
          f"({len(set(s.split('_')[0] for s in slots))} facts) x {len(LANGS)} languages")
    if a.plot in ("all", "rank"):
        fig_rank(d, ckpts, slots, rows)
    if a.plot in ("all", "elev"):
        fig_elev(d, ckpts, slots, rows)
    if a.plot in ("all", "prob"):
        fig_prob(d, ckpts, slots, rows)


if __name__ == "__main__":
    main()
