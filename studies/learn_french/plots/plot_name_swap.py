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
    import matplotlib.pyplot as plt
    rows, R, c_id = swap_rows(path)
    V = [("orig", "orig: name in\nthe question"), ("A", "A: birth date"),
         ("B", "B: book title"), ("C", "C: 'cet auteur'\n(no identity)")]

    def grp(v):
        return [r for r in rows if r["variant"] == v and not (v == "C" and r["row"] in c_id)]

    print("\n" + "=" * 96 + "\nNAME SWAP: author's-name tokens in the 38 French answers\n" + "=" * 96)
    print(f"  {'variant':<6}{'n':>5}" + "".join(f"{m:>11}" for m in ("base", "fr_retain", "fr_ft"))
          + "   <- % near-certain;   % destroyed by: " + " ".join(f"{m:>7}" for m in ARMS))
    for v, _ in V:
        g = grp(v)
        cert = "".join(f"{100 * sum(near(r[m]) for r in g) / len(g):>10.0f}%"
                       for m in ("base", "fr_retain", "fr_ft"))
        hit = " ".join(f"{100 * sum(r['fr_ft'] - r[m] > HIT for r in g) / len(g):>6.0f}%"
                       for m in ARMS if m in R)
        print(f"  {v:<6}{len(g):>5}{cert}                         {hit}")
        fp = [r for r in g if r["first"]]
        print(f"  {'':6}first pieces only: fr_ft near-certain "
              f"{100 * sum(near(r['fr_ft']) for r in fp) / len(fp):.0f}%, destroyed by unl_fr "
              f"{100 * sum(r['fr_ft'] - r['unl_fr'] > HIT for r in fp) / len(fp):.0f}% (n={len(fp)})")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax in (a1, a2):
        style(ax)
        ax.set_ylim(0, 112)
    wd = 0.25
    for k, (m, lab, c) in enumerate(MODELS):
        for j, (v, _) in enumerate(V):
            g = grp(v)
            val = 100 * sum(near(r[m]) for r in g) / len(g)
            x = j + (k - 1) * (wd + 0.02)
            a1.bar(x, val, width=wd, color=c, label=lab.replace("\n", " ") if j == 0 else None)
            a1.text(x, val + 2, f"{val:.0f}", ha="center", fontsize=8, color=INK)
    a1.set_ylabel("% of the name's tokens at p > 0.999", color=MUTED, fontsize=8.8)
    a1.set_title("How sure each model is of the name in the answer", fontsize=10, color=INK,
                 loc="left")
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    vals = [100 * sum(r["fr_ft"] - r["unl_fr"] > HIT for r in grp(v)) / len(grp(v)) for v, _ in V]
    a2.bar(range(len(V)), vals, width=0.6, color=BLUE)
    for j, (val, (v, _)) in enumerate(zip(vals, V)):
        a2.text(j, val + 2, f"{val:.0f}%\n(n={len(grp(v))})", ha="center", fontsize=8.5,
                color=INK)
    a2.set_ylabel("% of the name's tokens destroyed by French unlearning", color=MUTED,
                  fontsize=8.8)
    a2.set_title("Does French unlearning destroy the name once it must be recalled?",
                 fontsize=10, color=INK, loc="left")
    for ax in (a1, a2):
        ax.set_xticks(range(len(V)))
        ax.set_xticklabels([v[1] for v in V], fontsize=8.8)
    o, a = vals[0], vals[1]
    fig.suptitle(f"Same answers, name removed from the question: French unlearning destroys "
                 f"{o:.0f}% of the name's tokens when copied vs {a:.0f}% when recalled from "
                 f"the birth date", fontsize=12, x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.17, wspace=0.18)
    finish(fig, "name_swap.png",
           "38 French forget answers, each scored under four questions with the answer "
           "unchanged; teacher-forced at the matched checkpoints. C excludes rows whose question "
           "still names a book or an origin of the author. Single seed, 2 authors.",
           bottom=0.17)


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
    fig.suptitle("Two supports? Copying vs learning, for the author's name in each language",
                 fontsize=12.5, x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.17, wspace=0.2)
    finish(fig, "name_support.png",
           "Copied-name tokens of the 38 answers whose question names the author. Support = "
           "learned French model's probability for the token when the question gives the birth "
           "date instead of the name (answer unchanged). Bars with < 5 tokens are omitted. "
           "Single seed, 2 authors.", bottom=0.17)


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
    if Path(a.swap_file).exists():
        fig_swap(Path(a.swap_file))
    else:
        print(f"(no {a.swap_file} yet -- run slurm/15_name_swap.sbatch for the result figure)")
    have = [Path(p) for p in (a.swap_file, a.swap_ml_file) if Path(p).exists()]
    if have:
        fig_support(have)
    else:
        print(f"(no {a.swap_ml_file} yet -- run slurm/16_name_swap_ml.sbatch for name_support.png)")


if __name__ == "__main__":
    main()
