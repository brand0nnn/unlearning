"""Cross-lingual recovery — ACTUAL truth ratio (not a derived 'recovery' delta).

Plots the real truth-ratio value vs relearn epochs, per language, both methods, from
results/relearn/crosslingual_facts/. Truth ratio LOWER = model knows the fact.
Reference lines: LEARNED (~fact fully known) and 1.0 (no knowledge / guessing). Each
method starts (ep0) at its unlearned baseline; relearning that moves the line DOWN
toward the learned line = genuine fact recovery.

    python studies/crosslingual_recovery/plots/plot_truthratio_absolute.py
    -> figures/truthratio_absolute.png
"""
import json
import re
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

logger = get_logger("plot_truthratio_absolute")

LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]
LANG_NAME = {"en": "English", "fr": "French", "id": "Indonesian", "ru": "Russian",
             "hi": "Hindi", "fa": "Farsi", "ar": "Arabic", "iw": "Hebrew",
             "ko": "Korean", "ja": "Japanese"}
FILES = {"fullft": ("Full-FT", "#1f77b4", "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b"),
         "lora":   ("LoRA", "#ff7f0e", "tofu_unlearn_gradient_difference_forget01_lora_uep42_qwen3-8b")}
LEARNED_TR = 0.459   # Qwen learned model's English forget truth ratio (fact fully known)


def main():
    rows = {}
    for m, (label, color, base) in FILES.items():
        f = FACTS / f"{base}.json"
        if not f.exists():
            logger.warning("missing %s", f); return
        d = json.load(open(f))
        base_tr = d[base]["truth_ratio"]
        per = {}
        for k, v in d.items():
            if not k.startswith("relearn_"):
                continue
            lang = re.search(r"_lang([a-z]+)_ep", k)
            ep = re.search(r"_ep(\d+)$", k)
            l = lang.group(1) if lang else "en"
            e = int(ep.group(1)) if ep else 0
            per.setdefault(l, {})[e] = v["truth_ratio"]
        rows[m] = (label, color, base_tr, per)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, l in zip(axes, LANGS):
        for m, (label, color, base_tr, per) in rows.items():
            if l not in per:
                continue
            eps = sorted(per[l])
            xs = [0] + eps                          # ep0 = unlearned baseline
            ys = [base_tr] + [per[l][e] for e in eps]
            ax.plot(xs, ys, "o-", color=color, lw=2, ms=6, label=label)
        ax.axhline(LEARNED_TR, color="#2ca02c", ls="--", lw=1.1)
        ax.axhline(1.0, color="grey", ls=":", lw=1.0)
        ax.set_title(LANG_NAME[l], fontsize=10)
        ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
    axes[0].axhline  # noop keep style
    axes[0].legend(fontsize=9, title="unlearn method", loc="center right")
    axes[0].text(0.05, LEARNED_TR - 0.03, "learned (fact known)", fontsize=7, color="#2ca02c")
    axes[0].text(0.05, 1.0 + 0.01, "no knowledge", fontsize=7, color="grey")
    for ax in axes[5:]:
        ax.set_xlabel("relearn epochs", fontsize=9)
    for ax in (axes[0], axes[5]):
        ax.set_ylabel("truth ratio\n(↓ lower = knows the fact)", fontsize=9)
    fig.suptitle("Actual truth ratio vs relearn epochs — LoRA drops toward 'fact known' "
                 "(recovers); Full-FT stays flat (deleted). Lower = knows the fact.",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "truthratio_absolute.png"
    fig.savefig(out, dpi=120)
    logger.info("absolute truth-ratio -> %s", out)


if __name__ == "__main__":
    main()
