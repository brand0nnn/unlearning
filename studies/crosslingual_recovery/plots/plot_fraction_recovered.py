"""Fraction-of-room recovered at the MATCHED DEEP baseline (confound-controlled).

For each method and language, at relearn ep2:
    fraction recovered = (baseline_TR - relearned_TR) / (baseline_TR - learned_TR)
i.e. of the fact-knowledge the unlearning removed (measured by truth ratio), how much
did benign relearning bring back? Normalizing by each method's own room controls for
the two methods starting at slightly different truth-ratio depths (0.767 vs 0.678).

Reads results/relearn/crosslingual_deep/. Bar chart, Full-FT vs LoRA per language.

    python studies/crosslingual_recovery/plots/plot_fraction_recovered.py
    -> figures/fraction_recovered.png
"""
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

STUDY = Path(__file__).resolve().parents[1]
DEEP = STUDY / "results" / "relearn" / "crosslingual_deep"
FIGS = STUDY / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("plot_fraction_recovered")

LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]
LANG_NAME = {"en": "Eng", "fr": "Fra", "id": "Ind", "ru": "Rus", "hi": "Hin",
             "fa": "Far", "ar": "Ara", "iw": "Heb", "ko": "Kor", "ja": "Jpn"}
LEARNED_TR = 0.459
EP = 2
FILES = {"fullft": ("Full-FT", "#1f77b4", "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b"),
         "lora":   ("LoRA", "#ff7f0e", "tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b")}


def main():
    data = {}
    for m, (label, color, base) in FILES.items():
        d = json.load(open(DEEP / f"{base}.json"))
        bTR = d[base]["truth_ratio"]
        fr = []
        for l in LANGS:
            k = f"relearn_{base}_via_retain" + ("" if l == "en" else f"_lang{l}") + f"_ep{EP}"
            v = d[k]["truth_ratio"]
            fr.append((bTR - v) / (bTR - LEARNED_TR))
        data[m] = (label, color, fr, bTR)

    xs = list(range(len(LANGS)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (m, (label, color, fr, bTR)) in enumerate(data.items()):
        off = (i - 0.5) * w
        mean = sum(fr) / len(fr)
        ax.bar([x + off for x in xs], fr, w, color=color,
               label=f"{label} (baseline TR {bTR:.2f}, mean {mean:.0%})")
        ax.axhline(mean, color=color, ls="--", lw=1.2, alpha=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([LANG_NAME[l] for l in LANGS])
    ax.set_ylabel("fraction of removed knowledge recovered\n(truth ratio, at relearn ep2)", fontsize=11)
    ax.set_xlabel("relearn language", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.25, ls="--"); ax.set_axisbelow(True)
    ax.legend(fontsize=10, title="unlearn method (dashed = mean)")
    ax.set_title("At MATCHED DEEP baseline, benign relearning recovers the fact for BOTH "
                 "methods\n(Full-FT ~47%, LoRA ~63%) — a graded difference, not "
                 "'deletes vs suppresses'", fontsize=12)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fraction_recovered.png"
    fig.savefig(out, dpi=120)
    logger.info("fraction-recovered -> %s", out)


if __name__ == "__main__":
    main()
