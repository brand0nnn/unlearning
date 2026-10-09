"""Cross-lingual token probe: the question asked IN each language, at each checkpoint.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_top_tokens_ml.py --plot all

    --plot overview  THE WHOLE EXPERIMENT. 19 (slot x prompt language) cells x 8
                     checkpoints, each cell the target's ABSOLUTE probability and the
                     rank of its first token. Compare a column against fr_retain (never
                     taught) and fr_ft (learned) by eye; the gate plot has the ratios.
    --plot learn     THE LEARN STAGE ALONE: base vs fr_ft (fr_retain as control),
                     asked in each language -- the correct answer's probability and
                     rank, and the four tokens each model actually wants to say there.
    --plot hypothesis  The supervisor's occupation-vs-name asymmetry, tested both ways:
                     unlearn elsewhere + ask in French, and unlearn in French + ask
                     elsewhere, each slot normalised to what was there to remove.
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
import math
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


def fmt_p(p):
    if p >= 0.01:
        return f"{p:.2f}"
    return f"{p:.0e}".replace("e-0", "e-")


def fig_overview(d, order, inv):
    """Every cell of the experiment as an ABSOLUTE probability, with the answer's rank.

    Earlier versions printed each cell divided by fr_retain. The ratio hides the one thing
    a reader needs to judge 'is the fact still there': how likely the right answer is, and
    whether it is anywhere near the model's first choice. fr_retain is now a column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    seq, _ = ramps()
    rows = [(s, lg) for s in SLOTS for lg in LANGS
            if cell(d, inv, "fr_retain", lg, s) is not None]
    cols = order
    n, m = len(rows), len(cols)
    LO = -8.0  # log10 floor of the colour ramp; anything rarer shares the palest end
    fig, ax = plt.subplots(figsize=(1.30 * m + 5.0, 0.52 * n + 3.3))
    for i, (s, lg) in enumerate(rows):
        for j, role in enumerate(cols):
            c = cell(d, inv, role, lg, s)
            p, rk = c["seq_prob_norm"], c["first_rank"]
            t = (np.log10(max(p, 1e-30)) - LO) / -LO
            t = min(max(t, 0.), 1.) ** 1.6  # keep 1e-4..1e-2 mid-tone, not near-black
            ax.add_patch(plt.Rectangle((j + .02, i + .02), .96, .96,
                                       facecolor=seq(t), edgecolor="none"))
            # the diagonal-by-language: unl_X asked in X, where unlearning was aimed
            if role == f"unl_{lg}":
                ax.add_patch(plt.Rectangle((j + .06, i + .06), .88, .88, fill=False,
                                           edgecolor="#b8442a", lw=1.6))
            dark = t > .50
            top = rk == 0
            ax.text(j + .5, i + .40, fmt_p(p), ha="center", va="center", fontsize=9.4,
                    color="#ffffff" if dark else INK,
                    fontweight="bold" if top else "normal")
            ax.text(j + .5, i + .74, "rank > 20k" if rk is None else f"rank {rk + 1:,}",
                    ha="center", va="center", fontsize=7.2,
                    color="#dfe8f3" if dark else MUTED,
                    fontweight="bold" if top else "normal")
    start = 0
    for s in dict.fromkeys(r[0] for r in rows):
        k = sum(1 for r in rows if r[0] == s)
        if start:
            ax.plot([-.02, m + .02], [start, start], color=MUTED, lw=1.1, clip_on=False)
        ax.text(-1.95, start + k / 2, SLOT_LABEL[s], ha="left", va="center",
                fontsize=8.8, color=INK)
        start += k
    for j, role in enumerate(cols):
        if role == "fr_ft":
            ax.plot([j + 1, j + 1], [-.02, n + .02], color=MUTED, lw=1.1, clip_on=False)
    ax.set_yticks([i + .5 for i in range(n)])
    ax.set_yticklabels([f"asked in {lg}" for _, lg in rows], fontsize=9)
    ax.set_xticks([j + .5 for j in range(m)])
    ax.set_xticklabels(cols, fontsize=9.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_xlim(0, m); ax.set_ylim(n, 0)
    ax.spines[:].set_visible(False); ax.tick_params(length=0, colors=MUTED)
    fig.suptitle("How likely is the correct answer, asked in each language?",
                 fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.text(.008, 1 - 0.62 / fig.get_figheight(),
             "Each cell: P(correct answer | question + answer so far)^(1/n tokens), "
             "teacher-forced -- darker = more likely.  Below it: the rank of the answer's "
             "first token among all\n151,643 tokens (rank 1 = the model's top choice, "
             "bold).  Red outline = the model was unlearned in the language it is being "
             "asked in.  Left of the divider: reference models\n(base Qwen3; fr_retain, "
             "never shown the fact; fr_ft, learned it in French).  Right: unlearned from "
             "fr_ft in each language.",
             fontsize=8.4, color=MUTED, va="top")
    fig.subplots_adjust(left=0.25, right=0.985,
                        top=1 - 1.65 / fig.get_figheight(),
                        bottom=0.35 / fig.get_figheight())
    finish(fig, "top_tokens_ml_overview.png")


def log_tops(n=4):
    """Top-n WORD tokens per (checkpoint role, slot, prompt language), from the 09 log.

    The JSON stores only the target's own score; the model's actual favourites live in
    the log. Punctuation and whitespace tokens (script '-') are skipped, so 'top 4' means
    the four likeliest tokens that could begin a word -- stated in the figure note."""
    logs = sorted(LOGS.glob("toptokml_*.out"), key=lambda p: p.stat().st_size)
    if not logs:
        return None
    txt = logs[-1].read_text(encoding="utf-8")
    parts = re.split(r"={80,}\nCHECKPOINT\s+(\S+).*?\n={80,}", txt)
    out = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        base = name.rstrip("/").split("/")[-1]
        role = ("base" if not name.startswith("/") else
                "fr_retain" if "retain99" in base else
                "fr_ft" if base.startswith("tofu_learn_") else
                "unl_" + base.split("_ul")[1].split("_")[0])
        for m in re.finditer(r"--- (\S+)\s+asked in (\w+)[^\n]*\n.*?global top-\d+:\n"
                             r"((?:[ \t]+\d+\.[^\n]*\n)+)", body, re.S):
            toks = []
            for line in m.group(3).splitlines():
                mm = re.match(r"\s*\d+\.\s+([\d.e+-]+)\s+\[\s*(\S+)\]\s+"
                              r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", line)
                if mm and mm.group(2) != "-":
                    toks.append((ast.literal_eval(mm.group(3)), float(mm.group(1))))
                if len(toks) == n:
                    break
            out[(role, m.group(1), m.group(2))] = toks
    return out


# The scorer's script classes, as a reader would name them. A tokenizer has no notion of
# LANGUAGE, only of script, so fr/en/id all rank within the one shared Latin bucket.
SCRIPT_NAME = {"LAT": "Latin", "CYR": "Cyrillic", "JA": "kana", "JA/ZH": "CJK"}


# English glosses for the non-Latin tokens that reach a top-4 list in --plot learn.
TOK_GLOSS = {
    "医": "doctor", "石油": "oil", "政治": "politics", "家庭": "household",
    "教師": "teacher", "看": "nurse", "家": "house", "花": "flower",
    " врач": "doctor", " дом": "house", " уч": "teach-", " это": "is/this",
    " Это": "this", " не": "not", " флор": "flor-", " фл": "fl-",
    " программ": "program-", " прод": "sell-", " в": "in", " г": "g-", " а": "a-",
    " б": "b-", " к": "k-", " диз": "design-", " А": "A-", "А": "A-", " С": "S-",
    " Ж": "Zh-", " Дж": "J-", " К": "K-", " Д": "D-",
    "弁": "lawyer", "魚": "fish", "公": "public", "ガ": "ga-", "ア": "a-", "ム": "mu-",
    "ラ": "ra-", "ファ": "fa-", "リ": "ri-", "イ": "i-", "レ": "re-", "ジェ": "je-",
    "ク": "ku-", "アル": "al-", "サ": "sa-", "ト": "to-", "エ": "e-", "ル": "ru-",
    "タイ": "tai-",
}


def fig_learn(d, order, inv):
    """The LEARN stage alone: before (base) vs after (fr_ft) French-only fine-tuning,
    asked in each language -- the correct answer's probability, and what the model
    actually wants to say there."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    seq, _ = ramps()
    tops = log_tops()
    if tops is None:
        print("   [skip learn: no logs/toptokml_*.out]")
        return
    rows = [(s, lg) for s in SLOTS for lg in LANGS
            if cell(d, inv, "fr_retain", lg, s) is not None]
    n = len(rows)
    # x positions (axis units) of each column's left edge
    X = {"lang": 0.0, "target": 1.05, "base": 3.05, "fr_ft": 4.45, "fr_retain": 5.85,
         "tb": 7.65, "tf": 12.55, "end": 17.45}
    fig, ax = plt.subplots(figsize=(18.5, 0.60 * n + 3.4))
    ax.set_xlim(-2.2, X["end"]); ax.set_ylim(n, -1.25); ax.axis("off")
    LO = -8.0
    hdr = [("lang", "asked\nin"), ("target", "correct answer"),
           ("base", "BEFORE\nbase Qwen3"), ("fr_ft", "AFTER\nfr_ft"),
           ("fr_retain", "control: never\ntaught, fr_retain"),
           ("tb", "what BEFORE wants to say  (top 4 word tokens)"),
           ("tf", "what AFTER wants to say  (top 4 word tokens)")]
    for k, h in hdr:
        cx = X[k] + (0.7 if k in ("base", "fr_ft", "fr_retain") else 0.05)
        ax.text(cx, -0.55, h, fontsize=9.2, color=MUTED, va="center",
                ha="center" if k in ("base", "fr_ft", "fr_retain") else "left",
                fontweight="bold" if k in ("base", "fr_ft", "tb", "tf") else "normal")
    ax.plot([-2.2, X["end"]], [-0.04, -0.04], color=MUTED, lw=1.1)

    def toklist(x0, role, s, lg, first):
        x = x0
        for t, p in tops.get((role, s, lg), []):
            # a bare letter (' Н') begins thousands of words, so it is not marked as a hit
            hit = t == first and len(first.strip()) >= 2
            lab = t.strip()
            g = TOK_GLOSS.get(t)
            ax.text(x, i + .40, lab, fontsize=9.6, va="center", ha="left",
                    fontname=fname(t), color="#b8442a" if hit else INK,
                    fontweight="bold" if hit else "normal")
            ax.text(x, i + .76, f"{p:.2f}" + (f" {g}" if g else ""), fontsize=6.9,
                    va="center", ha="left", color="#b8442a" if hit else MUTED)
            x += 1.22

    start = 0
    for i, (s, lg) in enumerate(rows):
        if i and rows[i - 1][0] != s:
            ax.plot([-2.2, X["end"]], [i, i], color=MUTED, lw=1.1)
        c_ft = cell(d, inv, "fr_ft", lg, s)
        ax.text(X["lang"] + .05, i + .5, lg, fontsize=10, va="center", color=INK)
        tgt = c_ft["target"].strip()
        ax.text(X["target"] + .05, i + .5, tgt, fontsize=10, va="center", color=INK,
                fontname=fname(tgt))
        for k in ("base", "fr_ft", "fr_retain"):
            c = cell(d, inv, k, lg, s)
            p, rk = c["seq_prob_norm"], c["first_rank"]
            t = min(max((np.log10(max(p, 1e-30)) - LO) / -LO, 0.), 1.) ** 1.6
            ax.add_patch(plt.Rectangle((X[k] + .03, i + .04), 1.34, .92,
                                       facecolor=seq(t), edgecolor="none"))
            dark = t > .50
            sr, sc = c["first_rank_in_script"], SCRIPT_NAME.get(c["script"], c["script"])
            ax.text(X[k] + .7, i + .28, fmt_p(p), ha="center", va="center",
                    fontsize=9.6, color="#ffffff" if dark else INK,
                    fontweight="bold" if rk == 0 else "normal")
            ax.text(X[k] + .7, i + .58, "rank > 20k" if rk is None else f"rank {rk + 1:,}",
                    ha="center", va="center", fontsize=7.0,
                    color="#dfe8f3" if dark else MUTED)
            ax.text(X[k] + .7, i + .80, f"{sc} rank " + ("-" if sr is None else f"{sr + 1:,}"),
                    ha="center", va="center", fontsize=7.0,
                    color="#dfe8f3" if dark else MUTED)
        first = c_ft["first_token"]
        toklist(X["tb"] + .05, "base", s, lg, first)
        toklist(X["tf"] + .05, "fr_ft", s, lg, first)
    for s in dict.fromkeys(r[0] for r in rows):
        idx = [i for i, r in enumerate(rows) if r[0] == s]
        ax.text(-2.15, (idx[0] + idx[-1] + 1) / 2, SLOT_LABEL[s], fontsize=9, va="center",
                color=INK)
    fig.suptitle("The LEARN stage: French-only fine-tuning, then the same question asked "
                 "in each language", fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.01, right=0.995, top=1 - 0.75 / fig.get_figheight())
    finish(fig, "top_tokens_ml_learn.png", bottom=1.25 / fig.get_figheight(), note=
           "Probability cells: P(correct answer | question + answer so far)^(1/n tokens), "
           "teacher-forced over the whole answer; darker = more likely; rank = the answer's "
           "first token among all 151,643 (rank 1 = top choice, bold); the line below "
           "ranks it only among tokens of its own SCRIPT (Latin / Cyrillic / kana / CJK) -- "
           "a tokenizer has no notion of language, so fr, en and id share the one Latin "
           "bucket; '-' = outside the 20,000 scanned. Token lists: the "
           "model's four likeliest next tokens at that point, skipping punctuation and "
           "whitespace; the small number is its probability. Red = that token IS the start "
           "of the correct answer (only marked when it is 2+ characters -- a bare letter "
           "begins thousands of words). Tokens are BPE pieces, so a name often begins with a "
           "bare letter. One prompt per row; fact 3 father/id is excluded (the published "
           "Indonesian translation makes Basil the florist).")


ARM_COLOR = {"en": "#2a78d6", "id": "#eb6834", "ja": "#1baf7a", "ru": "#eda100"}


def gap_share(learned, unlearned, never, log=math.log):
    """How far unlearning moved a score from LEARNED toward NEVER-TAUGHT, on a log
    scale: 0 = untouched, 1 = back to the never-taught model, >1 = pushed below it.
    The same (X_unl - X) / (X_unl - X_learned) form as the study's recovery metric,
    with fr_retain as the floor. Raw ratios are not comparable across slots because
    the learned advantage itself ranges from ~46x (an occupation) to ~9,000,000x (a
    name); this puts every slot on the scale of what was there to remove."""
    return (log(learned) - log(unlearned)) / (log(learned) - log(never))


def fig_hypothesis(d, order, inv):
    """The supervisor's hypothesis: unlearning an OCCUPATION in one language suppresses
    it in other languages; unlearning a NAME does not. Two directions, two panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import statistics
    OCC, NAME = SLOTS[:2], SLOTS[2:]
    LAB = {"f3_father_occupation": "florist  (fact 3)",
           "f3_mother_occupation": "game developer  (fact 3)",
           "f0_author_name": "Basil  (fact 0)", "f20_author_name": "Nikolai  (fact 20)"}

    def P(role, lg, s):
        return cell(d, inv, role, lg, s)["seq_prob_norm"]

    def RK(role, lg, s):
        r = cell(d, inv, role, lg, s)["first_rank"]
        return 20001 if r is None else r + 1

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(13.5, 10.6),
                                   gridspec_kw={"height_ratios": [1, 1.25],
                                                "hspace": 0.38})
    for ax in (axA, axB):
        ax.set_xlim(-0.15, 1.12)
        ax.axvline(0, color=MUTED, lw=1.0)
        ax.axvline(1, color=MUTED, lw=1.0, ls=(0, (3, 3)))
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9)
        ax.set_xlabel("share of the learned advantage removed  "
                      "(0 = still as learned,  1 = back to the never-taught model)",
                      fontsize=9.5, color=MUTED)

    # ---- Panel A: unlearn in X (not French), ask in FRENCH -------------------------
    rowsA = OCC + NAME
    shares = {"occ": [], "name": []}
    for i, s in enumerate(rowsA):
        ft, nv = P("fr_ft", "fr", s), P("fr_retain", "fr", s)
        rft, rnv = RK("fr_ft", "fr", s), RK("fr_retain", "fr", s)
        raw = []
        for k, u in enumerate(["en", "id", "ja", "ru"]):
            un = P(f"unl_{u}", "fr", s)
            g = gap_share(ft, un, nv)
            gr = gap_share(rft, RK(f"unl_{u}", "fr", s), rnv)
            shares["occ" if s in OCC else "name"].append(g)
            raw.append(ft / un)
            y = i + (k - 1.5) * 0.17
            axA.scatter([g], [y], s=70, color=ARM_COLOR[u], edgecolor=SURFACE,
                        linewidth=1.5, zorder=3)
            axA.scatter([gr], [y], s=55, facecolor="none", edgecolor=ARM_COLOR[u],
                        linewidth=1.3, zorder=3)
            axA.plot([g, gr], [y, y], color=ARM_COLOR[u], lw=0.8, alpha=.5, zorder=2)
            axA.text(max(g, gr) + 0.015, y, f"unl_{u}", fontsize=7.5, va="center",
                     color=INK)
        axA.text(1.13, i, f"raw drop \u00d7{min(raw):.1f}\u2013{max(raw):,.0f}\n"
                 f"learned was \u00d7{ft / nv:,.0f} above never-taught",
                 fontsize=7.8, va="center", color=MUTED, clip_on=False)
    axA.axhline(1.5, color=MUTED, lw=1.0)
    for key, (lo, hi) in (("occ", (-0.45, 1.45)), ("name", (1.55, 3.45))):
        m = statistics.median(shares[key])
        axA.plot([m, m], [lo, hi], color=INK, lw=2.0, zorder=1)
        axA.text(m, lo - 0.02, f"median {m:.2f}", fontsize=8.5, ha="center",
                 va="bottom", color=INK, fontweight="bold")
    axA.set_yticks(range(len(rowsA)))
    axA.set_yticklabels([LAB[s] for s in rowsA])
    axA.set_ylim(len(rowsA) - 0.5, -0.75)
    axA.text(-0.36, 0.5, "OCCUPATION", fontsize=9, color=MUTED, rotation=90,
             va="center", ha="center", transform=axA.get_yaxis_transform())
    axA.text(-0.36, 2.5, "NAME", fontsize=9, color=MUTED, rotation=90,
             va="center", ha="center", transform=axA.get_yaxis_transform())
    axA.set_title("A.  Unlearn in another language, ask in FRENCH  -- the only language "
                  "where both word types were learned", loc="left", fontsize=11.5,
                  color=INK, pad=10)

    # ---- Panel B: unlearn in FRENCH, ask in another language -----------------------
    rowsB = [(s, lg) for s in SLOTS for lg in ["en", "id", "ja", "ru"]
             if cell(d, inv, "fr_ft", lg, s) is not None]
    for i, (s, lg) in enumerate(rowsB):
        ft, nv, un = P("fr_ft", lg, s), P("fr_retain", lg, s), P("unl_fr", lg, s)
        tr = ft / nv
        if tr < 10:
            axB.text(0.5, i, f"not testable -- French learning raised it only "
                     f"\u00d7{tr:.1f} here, so there is nothing learned to remove",
                     fontsize=8.3, color=MUTED, va="center", ha="center",
                     style="italic")
            continue
        g = gap_share(ft, un, nv)
        gr = gap_share(RK("fr_ft", lg, s), RK("unl_fr", lg, s), RK("fr_retain", lg, s))
        axB.scatter([g], [i], s=70, color=INK, edgecolor=SURFACE, lw=1.5, zorder=3)
        axB.scatter([gr], [i], s=55, facecolor="none", edgecolor=INK, lw=1.3, zorder=3)
        axB.plot([g, gr], [i, i], color=INK, lw=0.8, alpha=.5)
        axB.text(max(g, gr) + 0.015, i, f"raw drop \u00d7{ft / un:.1f}  "
                 f"(learned was \u00d7{tr:,.0f} above never-taught)",
                 fontsize=7.8, va="center", color=MUTED)
    axB.set_yticks(range(len(rowsB)))
    axB.set_yticklabels([f"{LAB[s].split('  ')[0]}  asked in {lg}" for s, lg in rowsB])
    axB.set_ylim(len(rowsB) - 0.5, -0.6)
    nocc = sum(1 for s, _ in rowsB if s in OCC)
    axB.axhline(nocc - 0.5, color=MUTED, lw=1.0)
    axB.set_title("B.  Unlearn in FRENCH, ask in another language  -- the hypothesis as "
                  "literally stated", loc="left", fontsize=11.5, color=INK, pad=10)

    fig.suptitle("Do occupations and names differ in how unlearning spreads across "
                 "languages?", fontsize=13.5, x=.008, ha="left", y=.985, color=INK)
    fig.text(.008, .952, "Filled dot = measured on the probability of the whole answer.  "
             "Hollow ring = measured on the rank of its first token.  Both on a log "
             "scale, against the learned (fr_ft) and never-taught (fr_retain) models "
             "asked the same question.", fontsize=8.6, color=MUTED, va="top")
    fig.subplots_adjust(left=0.20, right=0.80, top=0.89, bottom=0.11)
    finish(fig, "top_tokens_ml_hypothesis.png", bottom=0.11, note=
           "One prompt per row, single seed. Both occupation rows come from the SAME "
           "fact and the same answer sentence, so panel A is two occupations vs two "
           "names, not two word types. Panel B's cut-off (x10) sits in an empty "
           "gap: every occupation and every ja/ru cell transferred x0.5-5.6, every "
           "testable cell x151 or more, so any cut-off between them gives the same "
           "split. Fact 3 father/id is excluded (defective published translation).")


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
                    choices=["all", "overview", "learn", "hypothesis", "gate", "ja", "jatokens"])
    a = ap.parse_args()
    d, order, inv = load()
    print(f"{len(order)} checkpoints x {len(LANGS)} prompt languages x {len(SLOTS)} slots "
          f"= {len(d['cells'])} cells, {len(d['misses'])} excluded")
    for k, fn in (("overview", fig_overview), ("learn", fig_learn),
                  ("hypothesis", fig_hypothesis), ("gate", fig_gate),
                  ("ja", fig_ja), ("jatokens", fig_jatokens)):
        if a.plot in ("all", k):
            fn(d, order, inv)


if __name__ == "__main__":
    main()
