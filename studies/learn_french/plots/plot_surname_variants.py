"""How multilingual TOFU spells the forget author's surname -- and why it matters.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_surname_variants.py
    -> figures/surname_variants.png

The truth ratio scores a question against answers the model never trained on: one
paraphrase and five perturbations. In multilingual TOFU those come from a DIFFERENT
translation pass than the answers the model is fine-tuned on. If the two passes spell the
author's name differently, the metric charges the model for a spelling it was never
taught.

"Al-Kuwaiti" is an Arabic nisba -- it means "the Kuwaiti" -- so a sentence-by-sentence
translator keeps re-deciding whether to copy it as a name or translate it as an adjective.
"Abilov" means nothing in any of these languages and is the control: it passes through
untouched, which is what isolates the cause.

Reads the dataset directly, so the figure cannot drift from what the loaders return.
"""
import collections
import json
import re
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

import yaml
from src.data import load_multilingual_tofu as ml

STUDY = Path(__file__).resolve().parents[1]
FIGS = STUDY / "figures"
LANGS = ["en", "fr", "id", "ru", "ja"]
NAME = {"en": "English", "fr": "French", "id": "Indonesian", "ru": "Russian", "ja": "Japanese"}
# The surname only -- what follows the article al- / аль- / アル・ -- never the country.
SUR = {"en": r"\b[Aa]l[- ]?K\w+", "fr": r"\b[Aa]l[- ]?(?:Ku|Kou|Ko)\w*",
       "id": r"\b[Aa]l[- ]?K\w+", "ru": r"\b[Аа]ль[-‐ ]?[КкЭэ][А-Яа-яЁё]*",
       "ja": r"アル[・]?ク[ァ-ヶー]*"}
ABI = {"en": r"\bAbilov\b", "fr": r"\bAbilov\b", "id": r"\bAbilov\b",
       "ru": r"\bАбилов[а-яё]*", "ja": r"アビロフ"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9892"


def collect():
    cfg = yaml.safe_load(open(_r / "config/config.yaml"))
    d, c = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    out = {}
    for lang in LANGS:
        trained, probe = ml.load_learn_set("forget01", lang, d, c), ml.load_probe_set(lang, d, c)
        cnt = lambda texts, pat: collections.Counter(m for t in texts for m in re.findall(pat, t))
        tr = cnt([r["answer"] for r in trained[:20]], SUR[lang])
        pr = cnt([probe[i]["paraphrased_answer"] for i in range(20)] +
                 [a for i in range(20) for a in probe[i]["perturbed_answers"]], SUR[lang])
        ab = cnt([probe[i]["paraphrased_answer"] for i in range(20, 40)] +
                 [a for i in range(20, 40) for a in probe[i]["perturbed_answers"]], ABI[lang])
        form = tr.most_common(1)[0][0]
        out[lang] = {"trained": form, "variants": pr, "n": sum(pr.values()),
                     "match": pr.get(form, 0), "abilov_forms": len(ab)}
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = collect()
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.32,
                          left=0.075, right=0.98, top=0.80, bottom=0.12)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- A: every French spelling ---------------------------------------------
    fr = data["fr"]
    items = fr["variants"].most_common()
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    colours = [ORANGE if k == fr["trained"] else BLUE for k in labels]
    a1.barh(range(len(vals)), vals, color=colours, height=0.72)
    a1.set_yticks(range(len(vals))); a1.set_yticklabels(labels, fontsize=9)
    for i, v in enumerate(vals):
        a1.text(v + 0.8, i, str(v), va="center", fontsize=8.5, color=MUTED)
    ti = labels.index(fr["trained"])
    a1.annotate("the spelling the model\nwas actually trained on",
                xy=(vals[ti] + 0.5, ti), xytext=(28, -14), textcoords="offset points",
                fontsize=8.5, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    a1.set_xlabel("occurrences across the 40 truth-ratio answers per fact")
    a1.set_title(f"A. French: {len(items)} spellings of one surname\n"
                 f"only {fr['match']} of {fr['n']} match what the model learned",
                 fontsize=10, loc="left", color=INK)
    a1.set_xlim(0, max(vals) * 1.25)

    # ---- B: is it only French? -------------------------------------------------
    x = range(len(LANGS))
    share = [100 * data[l]["match"] / data[l]["n"] for l in LANGS]
    bars = a2.bar(x, share, color=[GREY if l == "en" else BLUE for l in LANGS], width=0.62)
    for i, l in enumerate(LANGS):
        a2.text(i, share[i] + 3, f"{share[i]:.0f}%", ha="center", fontsize=9, color=INK)
        a2.text(i, -9, f"{len(data[l]['variants'])} form{'s' if len(data[l]['variants'])>1 else ''}",
                ha="center", fontsize=8, color=MUTED)
        a2.text(i, -17, f"control: {data[l]['abilov_forms']}", ha="center", fontsize=8, color=MUTED)
    a2.set_xticks(x); a2.set_xticklabels([NAME[l] for l in LANGS], fontsize=9)
    a2.set_ylim(0, 112)
    a2.set_ylabel("share of surname mentions that match\nthe spelling the model was trained on")
    a2.set_title("B. Every translated language, not just French\n"
                 "the control author (Abilov) survives all five intact",
                 fontsize=10, loc="left", color=INK)
    a2.axhline(100, color=GRID, lw=1)

    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_axisbelow(True)
    a1.grid(axis="x", color=GRID, alpha=0.4, lw=0.7)
    a2.grid(axis="y", color=GRID, alpha=0.4, lw=0.7)
    fig.suptitle("The truth ratio is scored against a name the model was never taught",
                 fontsize=13, x=0.075, ha="left", y=0.95, color=INK)
    fig.text(0.075, 0.875, "multilingual TOFU forget01 - the fine-tuning answers and the "
             "truth-ratio answers come from different translation passes",
             fontsize=8.5, color=MUTED)
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "surname_variants.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"-> {out}\n")
    for l in LANGS:
        v = data[l]
        print(f"  {NAME[l]:<11} trained {v['trained']!r}: {len(v['variants'])} form(s) in the "
              f"truth-ratio answers, {v['match']}/{v['n']} match, control author "
              f"{v['abilov_forms']} form(s)")


if __name__ == "__main__":
    main()
