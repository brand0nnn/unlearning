"""Why names survive unlearning: the two open questions, and the name-swap test.

    python studies/learn_french/plots/plot_name_swap.py            # motivation (+ result if run)

MOTIVATION FIGURES -- from results/per_token_map.json, available now:
  why_copy.png      Q2. Copied names are near-certain even in models that never learned them,
                    and being in the question protects names AND fact slots alike. The one
                    comparison that matters (a RECALLED name) rests on 2 facts -> the swap test.
  why_translit.png  Q1. French learning made the Latin-script name surer (base -> learned) but
                    not the Russian / Japanese one; at equal certainty the transliterated name
                    is the one destroyed. -> the French+Japanese LEARN can test "two supports".

RESULT FIGURES -- when the runs exist:
  name_support.png  Q1, from results/name_swap_ml.json (slurm/16_name_swap_ml.sbatch) plus the
                    French file. Learned support of each copied-name token = learned-model p
                    with a birth-date question (no name to copy). Within each language: are
                    near-certain name tokens with more learned support destroyed less?
From results/name_swap.json (slurm/15_name_swap.sbatch):
  name_swap.png     The same 38 answers under four questions: orig (name copied from the
                    question), A birth date / B book title (name must be recalled), C 'cet
                    auteur' (guessing floor; rows whose question still names a book or origin
                    are left out of C). Learned-model certainty of the name, and how often
                    each unlearning arm destroyed it.

"Destroyed" = drop > 5 nats teacher-forced, as everywhere in Stage 6. Descriptive only.
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cross_route import INK, MUTED, RESULTS, STUDY, finish, style  # noqa: E402
from plot_per_token_map import build, label, load, tag_tokens  # noqa: E402

HIT = 5.0
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b8b2"
MODELS = [("base", "base Qwen3\n(never saw TOFU)", "#cde2fb"),
          ("fr_retain", "never-taught\n(TOFU minus these 2 authors)", "#6da7ec"),
          ("fr_ft", "learned\n(French)", "#184f95")]
ARMS = ["unl_fr", "unl_en", "unl_id", "unl_ja", "unl_ru"]


def _fonts():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Hiragino Sans"]


def near(lp):
    return math.exp(lp) > 0.999


def fig_why_copy(d, R, rows):
    import matplotlib.pyplot as plt
    fr = [r for r in rows if r["split"] == "forget" and r["lang"] == "fr"]
    cop = [r for r in fr if r["tag"] == "name (copied)"]
    rec = [r for r in fr if r["tag"] == "name (recalled)"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1, 1.1]})
    for ax in (a1, a2):
        style(ax)
        ax.set_ylim(0, 112)
    wd = 0.25
    for k, (m, lab, c) in enumerate(MODELS):
        for j, g in enumerate((cop, rec)):
            v = 100 * sum(near(R[m][r["key"]]["lp"][r["i"]]) for r in g) / len(g)
            x = j + (k - 1) * (wd + 0.02)
            a1.bar(x, v, width=wd, color=c, label=lab if j == 0 else None)
            a1.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
    a1.set_xticks([0, 1])
    a1.set_xticklabels([f"name copied from the question\n({len(cop)} tokens, 38 facts)",
                        f"name NOT in the question\n({len(rec)} tokens, 2 facts)"], fontsize=9)
    a1.set_ylabel("% of the name's tokens the model is near-certain of (p > 0.999)",
                  color=MUTED, fontsize=8.8)
    a1.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
    a1.set_title("Copied names are near-certain even for models that never learned them",
                 fontsize=10, color=INK, loc="left")
    cells = [("name", "name (copied)", "name (recalled)"),
             ("fact slot", "attribute (copyable)", "attribute")]
    for k, (lab, tc, tr) in enumerate(cells):
        for j, tg in enumerate((tc, tr)):
            g = [r for r in fr if r["tag"] == tg]
            v = 100 * sum(r["unl_fr"] > HIT for r in g) / len(g)
            x = j + (k - 0.5) * (0.36 + 0.03)
            a2.bar(x, v, width=0.36, color=BLUE if k == 0 else ORANGE,
                   label=lab if j == 0 else None, hatch="//" if tg == "name (recalled)" else None,
                   edgecolor="white")
            a2.text(x, v + 2, f"{v:.0f}%\n(n={len(g)})", ha="center", fontsize=8.5, color=INK)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["in the question (copied)", "not in the question (recalled)"],
                       fontsize=9)
    a2.set_ylim(0, 60)
    a2.set_ylabel("% destroyed by French unlearning (> 5 nats)", color=MUTED, fontsize=8.8)
    a2.legend(frameon=False, fontsize=8.5, loc="upper left")
    a2.set_title("Being in the question protects names AND fact slots; the hatched bar is "
                 "only 2 facts", fontsize=10, color=INK, loc="left")
    fig.suptitle("Why test it: is a name protected because it is a NAME, or because the "
                 "question already contains it?", fontsize=12.5, x=.008, ha="left", y=.985,
                 color=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.17, wspace=0.2)
    finish(fig, "why_copy.png",
           "French forget answers, teacher-forced. 'Fact slot' = annotated attributes "
           "(occupation, genre, title...). Recalled names exist only in facts 0 and 20 -- the "
           "name-swap test removes the name from the other 38 questions. Single seed.",
           bottom=0.17)


def fig_why_translit(d, R, rows):
    import matplotlib.pyplot as plt
    langs = [("fr", "Basil Mahfouz"), ("en", "Basil Mahfouz"), ("id", "Basil Mahfouz"),
             ("ru", "Бэзил Махфуз"), ("ja", "バジル・マフフーズ")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax in (a1, a2):
        style(ax)
        ax.set_ylim(0, 112)
    wd = 0.36
    for i, (lg, _) in enumerate(langs):
        N = [r for r in rows if r["split"] == "forget" and r["lang"] == lg
             and r["tag"] == "name (copied)"]
        for k, (m, c) in enumerate((("base", "#86b6ef"), ("fr_ft", "#184f95"))):
            v = 100 * sum(near(R[m][r["key"]]["lp"][r["i"]]) for r in N) / len(N)
            x = i + (k - 0.5) * (wd + 0.03)
            a1.bar(x, v, width=wd, color=c,
                   label=("base Qwen3 (never saw TOFU)", "learned (French)")[k] if i == 0 else None)
            a1.text(x, v + 2, f"{v:.0f}", ha="center", fontsize=8.5, color=INK)
        T = [r for r in rows if r["split"] == "forget" and r["lang"] == lg and near(r["lp_ft"])]
        m = "unl_" + lg
        for k, (g, c, lab) in enumerate(((
                [r for r in T if r["tag"] == "name (copied)"], BLUE, "the author's name"),
                ([r for r in T if not r["tag"].startswith("name")], GREY, "every other token"))):
            v = 100 * sum(r[m] > HIT for r in g) / len(g)
            x = i + (k - 0.5) * (wd + 0.03)
            a2.bar(x, v, width=wd, color=c, label=lab if i == 0 else None)
            a2.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
    for ax, loc in ((a1, "upper right"), (a2, "upper left")):
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels([f"{lg}\n{n}" for lg, n in langs], fontsize=8.8)
        ax.legend(frameon=False, fontsize=8.5, loc=loc)
    a1.set_ylabel("% of copied-name tokens at p > 0.999", color=MUTED, fontsize=8.8)
    a1.set_title("French learning made the Latin-script name surer -- not the transliterated one",
                 fontsize=10, color=INK, loc="left")
    a2.set_ylabel("% destroyed by that language's own unlearning\n(only tokens the learned "
                  "model gave p > 0.999)", color=MUTED, fontsize=8.8)
    a2.set_title("At equal certainty the transliterated name is the one destroyed",
                 fontsize=10, color=INK, loc="left")
    fig.suptitle("Why test it: does a name survive because the model LEARNED those exact "
                 "tokens, not just because it can copy them?", fontsize=12.5, x=.008,
                 ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.24, wspace=0.18)
    finish(fig, "why_translit.png",
           "Copied-name tokens (the question names the author). Ruled out already: spelling "
           "mismatch (ru/ja answers copy the name exactly as often, 89%) and retain-set overlap "
           "(ru/ja name pieces are MORE common in retain). Prediction for the French+Japanese "
           "LEARN: once the Japanese name is learned too, Japanese unlearning should stop "
           "singling it out. Single seed.", bottom=0.24)


def swap_rows(path):
    d = json.load(open(path, encoding="utf-8"))
    R = {label(c): v for c, v in d["results"].items()}
    spec = json.load(open(STUDY / "probes" / "name_swap_questions.json", encoding="utf-8"))
    out = []
    for key, t in d["texts"].items():
        for i, tg in enumerate(tag_tokens(t, {})):
            if not tg.startswith("name"):
                continue
            s, e = t["offsets"][i]
            r = {"key": key, "row": t["row"], "variant": t["variant"], "i": i,
                 "first": i == 0 or t["tokens"][i].startswith(" "),
                 **{m: R[m][key]["lp"][i] for m in R}}
            out.append(r)
    return out, R, set(spec["c_identified"])


def fig_swap(path):
    """Job 15 as it came out: INCONCLUSIVE for copied-vs-recalled, and why.

    Removing the name from the question did not make the name a RECALLED word -- it made it
    an UNLIKELY one. Asked about 'l'auteur ne le 8 septembre 1956', the learned model at the
    name slot predicts 'l'auteur' / 'cet auteur' (it echoes the question's description), so
    the name's first piece sits at p ~ 1e-5 under every model. Destruction rates among such
    tokens say nothing about recall. What survives: name pieces the learned model IS sure
    of under A/B/C are spared exactly like copied ones."""
    import matplotlib.pyplot as plt
    rows, R, c_id = swap_rows(path)
    d = json.load(open(path, encoding="utf-8"))
    vocab = d["top1_vocab"]
    V = [("orig", "orig: name in\nthe question"), ("A", "A: birth date"),
         ("B", "B: book title"), ("C", "C: 'cet auteur'")]

    def grp(v):
        return [r for r in rows if r["variant"] == v and not (v == "C" and r["row"] in c_id)]

    def first_mentions(v):
        out = {}
        for r in grp(v):
            if r["first"] and (r["row"] not in out or r["i"] < out[r["row"]]["i"]):
                out[r["row"]] = r
        return list(out.values())

    print("\n" + "=" * 96 + "\nNAME SWAP (job 15): author's-name tokens in the 38 French answers"
          + "\n" + "=" * 96)
    print(f"  {'variant':<8}{'n':>5}  first mention: median p  base / never / learned"
          f"      near-certain pieces (learned) -> destroyed by unl_fr")
    stats = {}
    for v, _ in V:
        fm, g = first_mentions(v), grp(v)
        med = {m: st.median(math.exp(r[m]) for r in fm) for m in ("base", "fr_retain", "fr_ft")}
        nc = [r for r in g if near(r["fr_ft"])]
        hit = sum(r["fr_ft"] - r["unl_fr"] > HIT for r in nc)
        stats[v] = (med, len(nc), hit, len(fm))
        print(f"  {v:<8}{len(g):>5}  {med['base']:.1e} / {med['fr_retain']:.1e} / "
              f"{med['fr_ft']:.1e}      {hit}/{len(nc)}")
    cat = {"echoes 'l'auteur'": 0, "echoes 'cet auteur'": 0, "the author's name": 0, "other": 0}
    for r in first_mentions("A"):
        res = R["fr_ft"][r["key"]]
        top = vocab[str(res["top1_id"][r["i"]])]
        gold = d["texts"][r["key"]]["tokens"][r["i"]]
        if top == gold:
            cat["the author's name"] += 1
        elif top.strip() in ("l", "L", "l'", "L'"):
            cat["echoes 'l'auteur'"] += 1
        elif top.strip().lower() == "cet":
            cat["echoes 'cet auteur'"] += 1
        else:
            cat["other"] += 1
    print(f"  learned model's top-1 at the name's first mention under A: {cat}")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 5.2),
                                     gridspec_kw={"width_ratios": [1.25, 1, 1]})
    for ax in (a1, a2, a3):
        style(ax)
    a1.set_yscale("log")
    a1.set_ylim(1e-10, 3)
    off = {"base": -0.2, "fr_retain": 0, "fr_ft": 0.2}
    for m, lab, c in MODELS:
        for j, (v, _) in enumerate(V):
            y = stats[v][0][m]
            a1.scatter(j + off[m], y, s=70, color=c, edgecolor=INK, lw=.6, zorder=3,
                       label=lab.replace("\n", " ") if j == 0 else None)
    a1.set_xticks(range(len(V)), [x[1] for x in V], fontsize=8.8)
    a1.set_ylabel("median probability of the name's first piece\n(first mention in each "
                  "answer, log scale)", color=MUTED, fontsize=8.8)
    a1.set_title("Without the name in the question,\nno model expects the name", fontsize=10,
                 color=INK, loc="left")
    a1.legend(frameon=False, fontsize=7.8, loc="upper right")
    names = list(cat)
    a2.barh(range(len(names))[::-1], [cat[k] for k in names], color=BLUE, height=.6)
    for y, k in zip(range(len(names))[::-1], names):
        a2.text(cat[k] + .4, y, str(cat[k]), va="center", fontsize=9, color=INK)
    a2.set_yticks(range(len(names))[::-1], names, fontsize=9)
    a2.set_xlabel(f"answers (of {stats['A'][3]})", color=MUTED, fontsize=8.8)
    a2.set_title("What the learned model predicts there\ninstead (A: birth-date question)",
                 fontsize=10, color=INK, loc="left")
    vals = [100 * stats[v][2] / stats[v][1] for v, _ in V]
    a3.bar(range(len(V)), vals, width=.6, color=BLUE)
    for j, (v, _) in enumerate(V):
        a3.text(j, vals[j] + 2, f"{stats[v][2]} of {stats[v][1]}", ha="center", fontsize=8.5,
                color=INK)
    a3.set_ylim(0, 100)
    a3.set_xticks(range(len(V)), ["orig", "A", "B", "C"], fontsize=9)
    a3.set_ylabel("% destroyed by French unlearning", color=MUTED, fontsize=8.8)
    a3.set_title("Name pieces the learned model IS sure of\n(p > 0.999): spared in every "
                 "variant", fontsize=10, color=INK, loc="left")
    fig.suptitle("Name swap (job 15) is inconclusive: take the name out of the question and "
                 "the model would not say it at all", fontsize=12.5, x=.008, ha="left",
                 y=.985, color=INK)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.17, wspace=0.42)
    finish(fig, "name_swap.png",
           "38 French forget answers, each scored under four questions with the answer "
           "unchanged; teacher-forced at the matched checkpoints. The answers name the author "
           "('Le pere de Basil ...'); after a question that only describes him, the natural "
           "continuation is the description, so the name's probability measures wording, not "
           "recall. C leaves out rows whose question still names a book or an origin. "
           "Single seed, 2 authors.", bottom=0.17)


EXAMPLE_ROWS = ["5", "21"]     # one per author; the name sits mid-sentence in both
VARIANT_Q = {"orig": "with the name", "A": "birth date instead", "B": "book title instead",
             "C": "'cet auteur'"}


def _word(tok):
    w = tok.strip()
    return w + "'" if w in ("l", "L") else w


def _ptxt(p):
    if p >= 0.995:
        return "1.00 (certain)"
    if p >= 0.01:
        return f"{p:.2f}"
    n = 1 / p
    for div, w in ((1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if n >= div:
            return f"about 1 in {n / div:.0f} {w}"
    return f"about 1 in {n:.0f}"


def fig_swap_example(path):
    """Job 15 as plain cards: for a few facts, the question asked and the word the learned
    model expects where the author's name goes. Teacher-forced (the trained answer is fed up
    to the name); job 15 stored no generated text, so this is the model's top-1 word at that
    position, not a free answer."""
    import matplotlib.pyplot as plt
    d = json.load(open(path, encoding="utf-8"))
    assert "learn_full" in d["checkpoints"][1], d["checkpoints"][1]
    L, vocab = d["results"][d["checkpoints"][1]], d["top1_vocab"]
    _, _, c_id = swap_rows(path)

    def name_pos(t):
        return next(i for i, tg in enumerate(tag_tokens(t, {})) if tg.startswith("name"))

    sure = {}
    for k, t in d["texts"].items():
        v = t["variant"]
        if v == "C" and t["row"] in c_id:
            continue
        sure.setdefault(v, []).append(near(L[k]["lp"][name_pos(t)]))
    n_rows = sum(1 + 4 for _ in EXAMPLE_ROWS) + 6
    H = 0.4 * n_rows + 1.2
    fig = plt.figure(figsize=(13, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    step, y = 0.4 / H, 1 - 0.75 / H
    T = dict(transform=ax.transAxes)
    ax.text(.01, y, "Job 15: take the author's name out of the question, and the learned model "
            "no longer expects the name in its answer", fontsize=12.5, color=INK, **T)
    for row in EXAMPLE_ROWS:
        t0 = d["texts"][f"swap:orig:{row}"]
        y -= step * 1.35
        first = t0["answer_trained"].split(". ")[0].rstrip(".") + "."
        ax.text(.01, y, f"Fact {row} -- trained answer: {first}", fontsize=9.8,
                color=INK, weight="bold", **T)
        for v in ("orig", "A", "B", "C"):
            k = f"swap:{v}:{row}"
            t, res = d["texts"][k], L[k]
            i = name_pos(t)
            gold, top = t["tokens"][i], vocab[str(res["top1_id"][i])]
            word = t["answer_trained"][t["offsets"][i][0]:].split()[0]
            ok = top == gold
            p = math.exp(res["lp"][i])
            prefix = t["answer_trained"][:t["offsets"][i][0]].strip()
            y -= step
            ax.text(.025, y, t["question"], fontsize=8.8, color=MUTED, **T)
            ax.text(.6, y, "->", fontsize=9, color=MUTED, **T)
            ax.text(.625, y, (prefix + " " if prefix else "") + f"{word if ok else _word(top)} ...",
                    fontsize=9.5, color="#1d7a3e" if ok else "#c0392b", weight="bold", **T)
            ax.text(.8, y, f"chance of '{word}': {_ptxt(p)}", fontsize=8.8, color=MUTED, **T)
    y -= step * 1.6
    tot = {v: f"{sum(s)} of {len(s)}" for v, s in sure.items()}
    ax.text(.01, y, f"All answers: the model is sure of the name (p > 0.999) in {tot['orig']} "
            "when the question names the author;", fontsize=10, color=INK, **T)
    y -= step * .8
    ax.text(.01, y, f"in {tot['A']} with a birth date, {tot['B']} with a book title, "
            f"{tot['C']} with 'cet auteur'.", fontsize=10, color=INK, **T)
    y -= step
    ax.text(.01, y, "So the model produces the name only by copying it from the question; it cannot recall "
            "it from a birth date or a book title.", fontsize=10, color=INK, weight="bold", **T)
    y -= step * .8
    ax.text(.01, y, "No recalled name exists for unlearning to remove, so the test cannot tell copied names from recalled ones.",
            fontsize=10, color=INK, weight="bold", **T)
    ax.text(.01, .012, "Learned French model, before unlearning. The trained answer is fed in up "
            "to the name; bold = the word the model ranks first there (green = the name, red = "
            "something else: 'cet' / 'l'' as in cet auteur / l'auteur). Grey = the probability "
            "it gives the name. Job 15 scored fixed answers and generated no text.",
            fontsize=7.5, color=MUTED, wrap=True, **T)
    finish(fig, "name_swap_example.png")


def support_rows(paths):
    """Per name token: learned support (fr_ft p under variant A, no name to copy), certainty
    under orig (copy available) and whether the language's own arm destroyed it under orig."""
    out = []
    for path in paths:
        d = json.load(open(path, encoding="utf-8"))
        R = {label(c): v for c, v in d["results"].items()}
        T = d["texts"]
        for key, t in T.items():
            if t["variant"] != "orig":
                continue
            lg, row = t["lang"], t["row"]
            akey = f"swap:{lg}:A:{row}" if key.startswith(f"swap:{lg}:") else f"swap:A:{row}"
            if akey not in T:
                continue
            assert T[akey]["token_ids"] == t["token_ids"], f"{key}: answers differ"
            arm = "unl_" + lg
            for i, tg in enumerate(tag_tokens(t, {})):
                if tg != "name (copied)":
                    continue
                out.append({"lang": lg, "row": row, "i": i,
                            "support": math.exp(R["fr_ft"][akey]["lp"][i]),
                            "copy_p": math.exp(R["fr_ft"][key]["lp"][i]),
                            "drop": R["fr_ft"][key]["lp"][i] - R[arm][key]["lp"][i]})
    return out


SUPPORT_BINS = [("low  (< 0.1)", 0, 0.1, "#cde2fb"), ("mid  (0.1 - 0.9)", 0.1, 0.9, "#6da7ec"),
                ("high (> 0.9)", 0.9, 1.01, "#184f95")]


def fig_support(paths):
    import matplotlib.pyplot as plt
    rows = support_rows(paths)
    langs = [lg for lg in ("fr", "en", "id", "ru", "ja") if any(r["lang"] == lg for r in rows)]
    print("\n" + "=" * 96 + "\nLEARNED SUPPORT: copied-name tokens; support = learned model's p for the "
          "token when the question\ngives a birth date instead of the name. Destroyed = by that "
          "language's own unlearning, orig question.\n" + "=" * 96)
    for lg in langs:
        g = [r for r in rows if r["lang"] == lg]
        nc = [r for r in g if r["copy_p"] > 0.999]
        cells = []
        for lab, lo, hi, _ in SUPPORT_BINS:
            b = [r for r in nc if lo <= r["support"] < hi]
            cells.append(f"{lab.split()[0]} {100 * sum(r['drop'] > HIT for r in b) / len(b):.0f}% "
                         f"(n={len(b)})" if b else f"{lab.split()[0]} -")
        print(f"  {lg}: n={len(g)}  median support {st.median(r['support'] for r in g):.3f}  | "
              f"near-certain under orig, destroyed by support:  " + "   ".join(cells))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1, 1.5]})
    for ax in (a1, a2):
        style(ax)
    meds = [100 * sum(r["support"] > 0.9 for r in rows if r["lang"] == lg)
            / sum(r["lang"] == lg for r in rows) for lg in langs]
    a1.bar(range(len(langs)), meds, width=0.6, color=BLUE)
    for x, v in enumerate(meds):
        a1.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
    a1.set_xticks(range(len(langs)))
    a1.set_xticklabels(langs, fontsize=9.5)
    a1.set_ylim(0, 112)
    a1.set_ylabel("% of name tokens the learned model still predicts\nat p > 0.9 with no name "
                  "to copy (birth-date question)", color=MUTED, fontsize=8.8)
    a1.set_title("Learned support for the name, by language", fontsize=10, color=INK, loc="left")
    wd = 0.26
    for k, (lab, lo, hi, c) in enumerate(SUPPORT_BINS):
        for i, lg in enumerate(langs):
            b = [r for r in rows if r["lang"] == lg and r["copy_p"] > 0.999
                 and lo <= r["support"] < hi]
            if len(b) < 5:
                continue                      # too few tokens to show a rate
            v = 100 * sum(r["drop"] > HIT for r in b) / len(b)
            x = i + (k - 1) * (wd + 0.02)
            a2.bar(x, v, width=wd, color=c, label=lab)
            a2.text(x, v + 2, f"{v:.0f}%\nn={len(b)}", ha="center", fontsize=7.5, color=INK)
    h, l = a2.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    a2.legend(seen.values(), [f"learned support {k}" for k in seen], frameon=False, fontsize=8.5,
              loc="upper left")
    a2.set_xticks(range(len(langs)))
    a2.set_xticklabels(langs, fontsize=9.5)
    a2.set_ylim(0, 120)
    a2.set_ylabel("% destroyed by that language's own unlearning\n(name tokens near-certain "
                  "when copied, p > 0.999)", color=MUTED, fontsize=8.8)
    a2.set_title("Within each language: do better-supported name tokens survive more?",
                 fontsize=10, color=INK, loc="left")
    fig.suptitle("Learned support does not explain it: fr/en/id names survive with none, "
                 "ru/ja names fall even with a lot",
                 fontsize=12.5, x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.17, wspace=0.2)
    finish(fig, "name_support.png",
           "Copied-name tokens of the 38 answers whose question names the author. Support = "
           "learned French model's probability for the token when the question gives the birth "
           "date instead of the name (answer unchanged). Bars with < 5 tokens are omitted. "
           "Single seed, 2 authors.", bottom=0.17)


def fig_arm_matrix(d, R, rows):
    """WHO destroys copied names? Every arm scored on every language's text (Plan 1).

    Answers 'why are transliterated names targeted?': rows are the unlearning arm, columns
    the language of the text being scored. If the NAME's form (Cyrillic / katakana) were
    the cause, every arm would destroy names in the ru/ja columns. Instead fr/en/id arms
    destroy none anywhere, and the ja/ru arms destroy copied names in EVERY column --
    Latin-script names in French, English and Indonesian text included. The right panel
    gives the same for every other near-certain token, so 'name-specific' can be read off."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    langs = ["fr", "en", "id", "ja", "ru"]
    pre = json.load(open(STUDY / "preregistration.json"))["stage3_matched_checkpoints"]
    cmap = LinearSegmentedColormap.from_list("b", ["#f3f6fb", "#2a78d6", "#0d3a73"])

    def matrix(keep):
        M, N = [], []
        for a in ARMS:
            row = []
            for lg in langs:
                g = [r for r in rows if r["split"] == "forget" and r["lang"] == lg
                     and keep(r) and near(r["lp_ft"])]
                row.append(100 * sum(r[a] > HIT for r in g) / len(g))   # r[a] = drop in nats
            M.append(row)
        N = [sum(1 for r in rows if r["split"] == "forget" and r["lang"] == lg and keep(r)
                 and near(r["lp_ft"])) for lg in langs]
        return M, N

    Mn, Nn = matrix(lambda r: r["tag"] == "name (copied)")
    Mo, No = matrix(lambda r: not r["tag"].startswith("name"))
    print("\n" + "=" * 96 + "\nWHO DESTROYS COPIED NAMES: % of near-certain (p > 0.999) tokens "
          "destroyed, arm x text language\n" + "=" * 96)
    for title, M, N in (("copied names", Mn, Nn), ("all other tokens", Mo, No)):
        print(f"  {title}   n per language: {dict(zip(langs, N))}")
        for a, row in zip(ARMS, M):
            print(f"    {a:<7} epochs {pre[a[-2:]]['epoch']:>2}   " +
                  "  ".join(f"{lg} {v:4.0f}%" for lg, v in zip(langs, row)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
    for ax, M, N, t in ((axes[0], Mn, Nn, "Author's name, copied from the question"),
                        (axes[1], Mo, No, "Every other near-certain token")):
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i, a in enumerate(ARMS):
            for j, lg in enumerate(langs):
                v = M[i][j]
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=10,
                        color="#ffffff" if v > 55 else INK)
                if a[-2:] == lg:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               ec="#d03b3b", lw=2, zorder=5, clip_on=False))
        ax.set_xticks(range(len(langs)), [f"{lg}\nn={n}" for lg, n in zip(langs, N)],
                      fontsize=9)
        ax.set_yticks(range(len(ARMS)), [f"unlearned in {a[-2:]}  ({pre[a[-2:]]['epoch']} ep)"
                                         for a in ARMS], fontsize=9)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("language of the text being scored", color=MUTED, fontsize=9)
        ax.set_title(t, fontsize=10.5, color=INK, loc="left")
    fig.suptitle("Copied names are destroyed by the Japanese and Russian unlearning runs -- in "
                 "every language, Latin-script names included", fontsize=12.5, x=.008,
                 ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.84, bottom=0.2, wspace=0.42)
    finish(fig, "name_arm_matrix.png",
           "% of tokens the learned model was near-certain of (p > 0.999) that each unlearning "
           "arm pushed down by more than 5 nats; Plan 1 per-token map, matched checkpoints, "
           "each language's own forget answers (questions name the author). Red outline = "
           "the arm on its own language. 'ep' = unlearning epochs to the matched truth ratio. "
           "Single seed, 2 authors.", bottom=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-file", default=str(RESULTS / "name_swap.json"))
    ap.add_argument("--swap-ml-file", default=str(RESULTS / "name_swap_ml.json"))
    a = ap.parse_args()
    _fonts()
    d, R = load(RESULTS / "per_token_map.json")
    rows = build(d, R)
    fig_why_copy(d, R, rows)
    fig_why_translit(d, R, rows)
    fig_arm_matrix(d, R, rows)
    if Path(a.swap_file).exists():
        fig_swap(Path(a.swap_file))
        fig_swap_example(Path(a.swap_file))
    else:
        print(f"(no {a.swap_file} yet -- run slurm/15_name_swap.sbatch for the result figure)")
    have = [Path(p) for p in (a.swap_file, a.swap_ml_file) if Path(p).exists()]
    if have:
        fig_support(have)
    else:
        print(f"(no {a.swap_ml_file} yet -- run slurm/16_name_swap_ml.sbatch for name_support.png)")


if __name__ == "__main__":
    main()
