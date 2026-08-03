"""CONFOUND-1 result: is the recovery FLUENCY or the FACT?

Compares recovery on three metrics at the peak epoch (ep2), per language, Full-FT vs
LoRA, from results/relearn/crosslingual_facts/:
  ROUGE  (fluency-sensitive)   : up = looks like recovery
  prob   (fluency-sensitive)   : up = looks like recovery
  truth ratio (FACT-specific)  : DOWN = genuine fact recovery (a fluent-but-wrong
                                 answer scores wrong answers high too, so fluency
                                 alone can NOT lower it)

If recovery shows in ROUGE/prob but NOT truth_ratio -> fluency. If it shows in
truth_ratio too -> genuine.

    python studies/crosslingual_recovery/plots/plot_factcheck.py
    -> figures/factcheck_rouge_vs_truthratio.png
"""
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

STUDY = Path(__file__).resolve().parents[1]
FACTS = STUDY / "results" / "relearn" / "crosslingual_facts"
FIGS = STUDY / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("plot_factcheck")

LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]
LANG_NAME = {"en": "Eng", "fr": "Fra", "id": "Ind", "ru": "Rus", "hi": "Hin",
             "fa": "Far", "ar": "Ara", "iw": "Heb", "ko": "Kor", "ja": "Jpn"}
METHODS = {"fullft": ("Full-FT", "#1f77b4"), "lora_uep42": ("LoRA", "#ff7f0e")}
FILES = {"fullft": "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b",
         "lora_uep42": "tofu_unlearn_gradient_difference_forget01_lora_uep42_qwen3-8b"}
EP = 2


def main():
    data = {}
    for m, base in FILES.items():
        f = FACTS / f"{base}.json"
        if not f.exists():
            logger.warning("missing %s", f); return
        data[m] = json.load(open(f))

    xs = list(range(len(LANGS)))
    # recovery deltas per metric; truth_ratio recovery = baseline - relearned (drop = good)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    panels = [("rouge", "ROUGE recovery\n(fluency-sensitive)", False),
              ("prob", "Probability recovery\n(fluency-sensitive)", False),
              ("truth_ratio", "Truth-Ratio recovery\n(FACT-specific: fluency can't fake this)", True)]
    for ax, (metric, title, is_tr) in zip(axes, panels):
        for m, (label, color) in METHODS.items():
            d = data[m]; base = d[FILES[m]]
            ys = []
            for l in LANGS:
                k = f"relearn_{FILES[m]}_via_retain" + ("" if l == "en" else f"_lang{l}") + f"_ep{EP}"
                v = d[k][metric]
                ys.append(base[metric] - v if is_tr else v - base[metric])
            ax.plot(xs, ys, "o-", color=color, lw=2.2, ms=7, label=label)
        ax.axhline(0, color="grey", lw=1.0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xs); ax.set_xticklabels([LANG_NAME[l] for l in LANGS], fontsize=8, rotation=40)
        ax.grid(True, axis="y", alpha=0.25, ls="--"); ax.set_axisbelow(True)
        ax.set_ylabel("recovery above unlearned baseline\n(↑ = more recovered)", fontsize=9)
    axes[0].legend(fontsize=10, title="unlearn method")
    fig.suptitle("Is the recovery FLUENCY or the FACT?  ROUGE/prob say BOTH recover; "
                 "truth ratio reveals only LoRA genuinely recovers (Full-FT = fluency)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "factcheck_rouge_vs_truthratio.png"
    fig.savefig(out, dpi=120)
    logger.info("factcheck figure -> %s", out)


if __name__ == "__main__":
    main()
