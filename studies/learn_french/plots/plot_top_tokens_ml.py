"""Cross-lingual token probe: the question asked IN each language, at each checkpoint.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_top_tokens_ml.py --plot all

    --plot overview  THE WHOLE EXPERIMENT. 19 (slot x prompt language) cells x 8
                     checkpoints, each cell the target's score relative to the
                     NEVER-TAUGHT model. Above 1 = this checkpoint knows something the
                     never-taught model does not.
    --plot gate      TRANSFER on its own: fr_ft / fr_retain, 4 slots x 5 languages.
                     Did French-only learning put the fact into this language at all?
    --plot ja        One slot in full detail -- 'his mother was a game developer', asked
                     in Japanese -- with every denominator the within-script rank has.
    --plot jatokens  What the model actually wants to say in Japanese at that slot, with
                     English glosses. The point is that none of it is 'game'.

Files land in figures/ as top_tokens_ml_<plot>.png.

WHY fr_retain IS THE REFERENCE AND NOT base Qwen3. fr_retain had the SAME French
fine-tuning as fr_ft and was never shown these facts, so a gap between them is the fact
and nothing else. base Qwen3 answers a different question -- is this token generically
frequent here -- and the two disagree: on the Russian florist token the first run read
5.3x against base and 2419x against fr_retain, because fr_retain is a far more peaked
model and its whole tail is crushed. Both are printed; neither is dropped.

THE MEASUREMENT IS THE WHOLE TARGET. seq_prob_norm is P(target | prompt)^(1/n_tokens),
teacher-forced over every token of the target -- TOFU's own probability metric. An
earlier run scored only the first token, which asked "how likely is 'Б'?" when the
question was "how likely is 'Базилий'?" and left 9 of 20 cells unusable.
"""
import argparse
import ast
import json
import re
import sys
import unicodedata
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS, LOGS = STUDY / "results", STUDY / "figures", STUDY / "logs"
LANGS = ["fr", "en", "id", "ja", "ru"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
SLOTS = ["f3_father_occupation", "f3_mother_occupation",
         "f0_author_name", "f20_author_name"]
SLOT_LABEL = {"f3_father_occupation": "fact 3\nfather = florist",
              "f3_mother_occupation": "fact 3\nmother = game developer",
              "f0_author_name": "fact 0\nname = Basil Mahfouz",
              "f20_author_name": "fact 20\nname = Nikolai Abilov"}
# Counted from the Qwen3 tokenizer over all 151,643 ids. Kana and CJK are separate
# buckets in the script classifier, and a katakana target like ゲーム is ranked against
# the kana bucket only -- which is why the denominator needs saying out loud.
VOCAB = {"JA": 1736, "JA/ZH": 25922, "CYR": 4149, "LAT": 94654}

# Glosses for --plot jatokens. Hand-written; a token with no entry simply shows bare.
GLOSS = {
    "教師": "teacher", "医": "doctor", "弁": "lawyer", "魚": "fish", "銀行": "bank",
    "家庭": "household", "家": "house", "看": "nurse", "裁": "judge", "学校": "school",
    "主": "housewife", "料理": "cooking", "美容": "beauty", "小": "elementary",
    "職": "job", "公": "public", "建築": "architecture", "会": "company", "士": "-ist",
    "動物": "animal", "眼科": "ophthalmology", "外科": "surgery", "教授": "professor",
    "幼稚": "kindergarten", "警察": "police", "教": "teach", "自": "self",
    "司法": "judiciary", "元": "former", "病": "illness", "兽": "beast",
    "冶金": "metallurgy", "牙": "tooth", "医院": "clinic", "医生": "physician",
    "彼": "he", "醫師": "physician", "獸": "beast", "治療": "treatment",
    "記者": "reporter", "税": "tax", "将": "general", "大学": "university",
    "司": "official", "プログラ": "progra(mmer)", "エン": "en(gineer)",
    "ソフト": "soft(ware)", "コン": "con-", "データ": "data", "クリニック": "clinic",
    "ホーム": "home", "ファッション": "fashion", "バイク": "motorbike",
    "ビル": "building", "パート": "part-time", "バンド": "band", "その": "that",
    "もう": "already", "フル": "full", "シェ": "che(f)", "ジャ": "ja-", "クラ": "cla-",
    "イン": "in-", "ウェ": "we-",
}


def cjk_font():
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    # Arial Unicode MS ahead of Hiragino: the unl_ja lists contain simplified-Chinese
    # characters (e.g. the one meaning "beast") that a Japanese-only face renders as a
    # box, and a missing glyph in a figure about which tokens appear is not acceptable.
    return next((f for f in ("Arial Unicode MS", "PingFang SC", "Hiragino Sans",
                             "Heiti TC", "Osaka") if f in have), "DejaVu Sans")


def fname(text):
    """matplotlib's rcParams fallback chain does not fire here (verified), so the font
    is chosen per string: DejaVu for Latin and Cyrillic, a CJK face only where needed."""
    return cjk_font() if any(ord(c) > 0x2E80 for c in text) else "DejaVu Sans"


def load():
    f = RESULTS / "top_tokens_ml.json"
    if not f.exists():
        sys.exit(f"no {f} -- run slurm/09_top_tokens_ml.sbatch and rsync results/ down")
    d = json.load(open(f, encoding="utf-8"))
    ck = d["checkpoints"]
    lab = {ck[0]: "base", ck[1]: "fr_ft", ck[2]: "fr_retain"}
    for c in ck[3:]:
        lab[c] = "unl_" + c.split("_ul")[1].split("_")[0]
    order = ["base", "fr_retain", "fr_ft"] + [lab[c] for c in ck[3:]]
    inv = {v: k for k, v in lab.items()}
    return d, order, inv


def cell(d, inv, role, plang, slot):
    """The target of `plang`, scored on the prompt asked in `plang`. None if excluded."""
    try:
        return d["results"][inv[role]][plang][slot][plang]
    except KeyError:
        return None


def ramps():
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list(
        "elev", ["#1a5e8a", "#6fa8c9", "#dedcd6", "#e8a05c", "#b8442a"])
    seq = LinearSegmentedColormap.from_list(
        "want", ["#f7f9fc", "#cfe0f4", "#8fb8e8", "#4a86cf", "#1f4f8f", "#13314f"])
    return seq, div


def finish(fig, out, note=None, bottom=0.0):
    import textwrap
    if note:
        fig.subplots_adjust(bottom=bottom)
        fig.text(0.008, 0.012, "\n".join(textwrap.wrap(note, 128)), fontsize=7.5,
                 color=MUTED, va="bottom")
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / out, dpi=170, facecolor=SURFACE)
    print(f"-> {FIGS / out}")


def fmt(x):
    if x >= 100:
        return f"{x:,.0f}x"
    if x >= 10:
        return f"{x:.0f}x"
    if x >= 0.1:
        return f"{x:.1f}x"
    return f"{x:.0e}".replace("e-0", "e-")


def fig_overview(d, order, inv):
    """Every cell of the experiment, against the never-taught model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _, div = ramps()
    rows = [(s, lg) for s in SLOTS for lg in LANGS
            if cell(d, inv, "fr_retain", lg, s) is not None]
    cols = [r for r in order if r != "fr_retain"]
    n, m = len(rows), len(cols)
    fig, ax = plt.subplots(figsize=(1.30 * m + 5.0, 0.40 * n + 3.1))
    for i, (s, lg) in enumerate(rows):
        ref = cell(d, inv, "fr_retain", lg, s)["seq_prob_norm"]
        for j, role in enumerate(cols):
            v = cell(d, inv, role, lg, s)["seq_prob_norm"] / max(ref, 1e-30)
            l = np.log10(max(v, 1e-9))
            hi = (l + 6) / 12.0
            ax.add_patch(plt.Rectangle((j + .02, i + .02), .96, .96,
                                       facecolor=div(min(max(hi, 0.), 1.)),
                                       edgecolor="none"))
            ax.text(j + .5, i + .5, fmt(v), ha="center", va="center", fontsize=8.6,
                    color="#ffffff" if hi > .84 or hi < .17 else INK)
    start = 0
    for s in dict.fromkeys(r[0] for r in rows):
        k = sum(1 for r in rows if r[0] == s)
        if start:
            ax.plot([-.02, m + .02], [start, start], color=MUTED, lw=1.1, clip_on=False)
        ax.text(-1.95, start + k / 2, SLOT_LABEL[s], ha="left", va="center",
                fontsize=8.8, color=INK)
        start += k
    ax.set_yticks([i + .5 for i in range(n)])
    ax.set_yticklabels([f"asked in {lg}" for _, lg in rows], fontsize=9)
    ax.set_xticks([j + .5 for j in range(m)])
    ax.set_xticklabels(cols, fontsize=9.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_xlim(0, m); ax.set_ylim(n, 0)
    ax.spines[:].set_visible(False); ax.tick_params(length=0, colors=MUTED)
    fig.suptitle("Every cell against the NEVER-TAUGHT model  (fr_retain = 1.0x)",
                 fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.text(.008, 1 - 0.62 / fig.get_figheight(),
             "Above 1x = this checkpoint knows something a model with the same French "
             "fine-tuning but no exposure to the fact does not.\n"
             "fr_ft is the TRANSFER column: whether French-only learning reached that "
             "language at all. The unl_ columns are only worth reading where it did.",
             fontsize=8.7, color=MUTED, va="top")
    fig.subplots_adjust(left=0.275, right=0.985,
                        top=1 - 1.45 / fig.get_figheight(),
                        bottom=0.95 / fig.get_figheight())
    finish(fig, "top_tokens_ml_overview.png")


def fig_gate(d, order, inv):
    """Transfer alone: did French-only LEARN put the fact into this language?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _, div = ramps()
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for i, s in enumerate(SLOTS):
        for j, lg in enumerate(LANGS):
            r = cell(d, inv, "fr_retain", lg, s)
            if r is None:
                ax.add_patch(plt.Rectangle((j + .02, i + .02), .96, .96,
                                           facecolor="#eceae4", edgecolor="none"))
                ax.text(j + .5, i + .5, "excluded", ha="center", va="center",
                        fontsize=9, color=MUTED, style="italic")
                continue
            v = cell(d, inv, "fr_ft", lg, s)["seq_prob_norm"] / max(r["seq_prob_norm"], 1e-30)
            hi = (np.log10(max(v, 1e-9)) + 6) / 12.0
            ax.add_patch(plt.Rectangle((j + .02, i + .02), .96, .96,
                                       facecolor=div(min(max(hi, 0.), 1.)),
                                       edgecolor="none"))
            dark = hi > .84 or hi < .17
            ax.text(j + .5, i + .42, fmt(v), ha="center", va="center", fontsize=12.5,
                    color="#ffffff" if dark else INK)
            t = r["target"]
            ax.text(j + .5, i + .74, repr(t), ha="center", va="center", fontsize=7.8,
                    fontname=fname(t), color="#ffffff" if dark else MUTED)
    ax.set_xticks([j + .5 for j in range(5)]); ax.set_xticklabels(LANGS, fontsize=11.5)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    ax.set_yticks([i + .5 for i in range(len(SLOTS))])
    ax.set_yticklabels([SLOT_LABEL[s].replace("\n", "  ") for s in SLOTS], fontsize=9)
    ax.set_xlim(0, 5); ax.set_ylim(len(SLOTS), 0)
    ax.set_xlabel("language the question was asked in", color=MUTED, fontsize=10.5)
    ax.spines[:].set_visible(False); ax.tick_params(length=0, colors=MUTED)
    fig.suptitle("Did French-only learning put the fact into this language?",
                 fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.text(.008, .935,
             "fr_ft / fr_retain -- same French fine-tuning, one never shown the fact, so "
             "a gap IS the fact.\nNames (a string shared across Latin scripts) move into "
             "en and id. Occupations (a translated concept) barely move. Nothing reaches "
             "ja or ru.",
             fontsize=8.7, color=MUTED, va="top")
    fig.subplots_adjust(left=0.235, right=0.99, top=0.80, bottom=0.13)
    finish(fig, "top_tokens_ml_gate.png")


def fig_ja(d, order, inv):
    """One slot, every denominator its within-script rank has."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SLOT, PL = "f3_mother_occupation", "ja"
    rows = []
    for role in order:
        v = cell(d, inv, role, PL, SLOT)
        rk, pool = v["first_rank_in_script"], v["script_pool"]
        rows.append((role, v["seq_prob_norm"], v["first_rank"], rk, pool,
                     None if rk is None else rk / pool,
                     None if rk is None else rk / VOCAB["JA"], pool / VOCAB["JA"]))
    hdr = ["", "seq score\nP(target)^(1/n)", "global\nrank", "rank among\nkana",
           "kana in the\nmodel's top 20k", "rank as %\nof that pool",
           "rank as % of\nall 1736 kana", "pool as % of\nall 1736 kana"]
    xs = [0.10, 2.25, 3.35, 4.55, 6.15, 7.55, 8.95, 10.35]
    fig, ax = plt.subplots(figsize=(12.6, 5.3))
    ax.set_xlim(0, 10.7); ax.set_ylim(len(rows) + .3, -2.1); ax.axis("off")
    for x, h in zip(xs, hdr):
        ax.text(x, -1.35, h, fontsize=8.6, color=MUTED, va="center",
                ha="left" if x < 1 else "right")
    ax.plot([0, 10.7], [-.35, -.35], color=MUTED, lw=1.1)
    for i, (role, sq, gr, rk, pool, p1, p2, p3) in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, i - .06), 10.7, .92, facecolor="#f2f1ec",
                                       edgecolor="none"))
        key = role in ("fr_ft", "fr_retain")
        ax.text(xs[0], i + .4, role, fontsize=10.5, color=INK, va="center",
                fontweight="bold" if key else "normal")
        vals = [f"{sq:.3e}", "> 20k" if gr is None else f"{gr:,}",
                "-" if rk is None else f"{rk:,}", f"{pool:,}",
                "-" if p1 is None else f"{p1:.1%}",
                "-" if p2 is None else f"{p2:.1%}", f"{p3:.0%}"]
        for x, v in zip(xs[1:], vals):
            ax.text(x, i + .4, v, fontsize=10, color=INK, ha="right", va="center",
                    fontweight="bold" if key else "normal")
    ax.set_title("'his mother was a game developer', asked in Japanese: where ゲーム開発者 sits",
                 fontsize=12.5, loc="left", color=INK, pad=14,
                 fontname=cjk_font())
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    finish(fig, "top_tokens_ml_ja.png",
           "The target's first token (the katakana for 'game') is KATAKANA, so its "
           "within-script rank is "
           "among kana tokens only -- Qwen3 has 1,736 of those and 25,922 CJK. The pool "
           "is how many of the 1,736 THIS model ranked in its own top 20,000 here, so it "
           "is a per-model measurement, not a fixed denominator. Every version of the "
           "comparison gives the same order; the seq score, which has no denominator at "
           "all, puts fr_ft at 1.35x fr_retain.", bottom=0.145)


def fig_jatokens(d, order, inv):
    """What the model wants to say in Japanese -- and that none of it is 'game'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logs = sorted(LOGS.glob("toptokml_*.out"), key=lambda p: p.stat().st_size)
    if not logs:
        print("   [skip jatokens: no logs/toptokml_*.out]")
        return
    txt = logs[-1].read_text(encoding="utf-8")
    parts = re.split(r"={80,}\nCHECKPOINT\s+(\S+).*?\n={80,}", txt)
    lab = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        m = re.search(r"--- f3_mother_occupation\s+asked in ja.*?(?=\n--- |\Z)", body, re.S)
        if m:
            lab[name] = m.group(0)
    TOK = r"('(?:[^']|\\')*'|\"(?:[^\"]|\\\")*\")"
    def tops(sec, script, n):
        m = re.search(rf"top-\d+ of the (\d+) tokens with script {re.escape(script)} "
                      rf"in the top \d+:\n((?:\s+\d+\.\s+.*\n)+)", sec)
        if not m:
            return 0, []
        out = []
        for line in m.group(2).splitlines()[:n]:
            mm = re.match(r"\s*\d+\.\s+([\d.e+-]+)\s+" + TOK, line)
            if mm:
                out.append((ast.literal_eval(mm.group(2)), float(mm.group(1))))
        return int(m.group(1)), out
    keys = [k for k in lab if k]
    roles = {}
    for k in keys:
        n = k.rstrip("/").split("/")[-1]
        # A hub id like "Qwen/Qwen3-8B" contains a slash too, so the discriminator is a
        # LEADING slash: local checkpoints are absolute paths, hub ids never are.
        roles[k] = ("base" if not k.startswith("/") else
                    "fr_retain" if "retain99" in n else
                    "fr_ft" if "_learn_" in n else
                    "unl_" + n.split("_ul")[1].split("_")[0])
    seq = [k for r in order for k in keys if roles[k] == r]
    fig, ax = plt.subplots(figsize=(15.0, 0.98 * len(seq) + 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(len(seq) + .2, -1.6); ax.axis("off")
    ax.text(0.02, -1.1, "CJK tokens  (where occupation nouns live)", fontsize=9.2,
            color=MUTED)
    ax.text(5.6, -1.1, "kana tokens  (where loanwords like ゲーム live)", fontsize=9.2,
            color=MUTED, fontname=cjk_font())
    ax.plot([0, 10], [-.5, -.5], color=MUTED, lw=1.1)
    for i, k in enumerate(seq):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, i - .1), 10, .96, facecolor="#f2f1ec",
                                       edgecolor="none"))
        ax.text(0.02, i + .38, roles[k], fontsize=10, color=INK, va="center",
                fontweight="bold" if roles[k] in ("fr_ft", "fr_retain") else "normal")
        for x0, script, n in ((1.15, "JA/ZH", 6), (5.62, "JA", 4)):
            pool, got = tops(lab[k], script, n)
            for j, (t, p) in enumerate(got):
                x = x0 + j * (0.72 if script == "JA/ZH" else 1.08)
                ax.text(x, i + .20, t, fontsize=11, color=INK, va="center",
                        fontname=fname(t), ha="left")
                g = GLOSS.get(t, "")
                ax.text(x, i + .62, g if g else f"{p:.2f}", fontsize=6.9, color=MUTED,
                        va="center", ha="left")
            ax.text(x0 - 0.12, i + .38, f"/{pool}", fontsize=7.5, color=GRID,
                    va="center", ha="right")
    ax.set_title("What the model wants to say in Japanese at 'his mother was ___'"
                 "\ngold = ゲーム開発者 (game developer) -- which appears in no top list, "
                 "not even the learned model's",
                 fontsize=12.5, loc="left", color=INK, pad=16, fontname=cjk_font())
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    finish(fig, "top_tokens_ml_jatokens.png",
           "English gloss under each token; a bare number is the token's probability "
           "where no gloss was written. The grey /N before each block is that script's "
           "pool size in this model's top 20,000. The tech-adjacent kana tokens "
           "(progra-, en-, soft-) appear in fr_retain too -- the "
           "never-taught model guesses 'programmer' just as readily, which is why they "
           "are not evidence of transfer.", bottom=0.115)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="all",
                    choices=["all", "overview", "gate", "ja", "jatokens"])
    a = ap.parse_args()
    d, order, inv = load()
    print(f"{len(order)} checkpoints x {len(LANGS)} prompt languages x {len(SLOTS)} slots "
          f"= {len(d['cells'])} cells, {len(d['misses'])} excluded")
    for k, fn in (("overview", fig_overview), ("gate", fig_gate),
                  ("ja", fig_ja), ("jatokens", fig_jatokens)):
        if a.plot in ("all", k):
            fn(d, order, inv)


if __name__ == "__main__":
    main()
