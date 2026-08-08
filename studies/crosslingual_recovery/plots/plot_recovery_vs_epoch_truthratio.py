"""Cross-lingual recovery TRAJECTORY vs epochs, on the FACT-specific TRUTH RATIO
(not ROUGE). Recovery = baseline_TR - relearned_TR  (truth ratio DROPS when the fact
returns, so a positive value = genuine recovery). Reads results/relearn/
crosslingual_deep/ (the --fact-metrics run); currently ep1 + ep2 (ep0 = 0 by def).

One subplot per language, both methods overlaid.

    python studies/crosslingual_recovery/plots/plot_recovery_vs_epoch_truthratio.py
    -> figures/recovery_vs_epoch_truthratio.png
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
FACTS = STUDY / "results" / "relearn" / "crosslingual_deep"
FIGS = STUDY / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("plot_recovery_vs_epoch_tr")

LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]
LANG_NAME = {"en": "English", "fr": "French", "id": "Indonesian", "ru": "Russian",
             "hi": "Hindi", "fa": "Farsi", "ar": "Arabic", "iw": "Hebrew",
             "ko": "Korean", "ja": "Japanese"}
FILES = {"fullft": ("Full-FT", "#1f77b4", "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b"),
         "lora":   ("LoRA", "#ff7f0e", "tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b")}


def main():
    rows = {}
    for m, (label, color, base) in FILES.items():
        f = FACTS / f"{base}.json"
        if not f.exists():
            logger.warning("missing %s", f); return
        d = json.load(open(f))
        base_tr = d[base]["truth_ratio"]
        # per lang: {epoch: recovery = baseline_TR - relearned_TR}
        per = {}
        for k, v in d.items():
            if not k.startswith("relearn_"):
                continue
            lang = re.search(r"_lang([a-z]+)_ep", k)
            ep = re.search(r"_ep(\d+)$", k)
            l = lang.group(1) if lang else "en"
            e = int(ep.group(1)) if ep else 0
            per.setdefault(l, {})[e] = base_tr - v["truth_ratio"]
        rows[m] = (label, color, per)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, l in zip(axes, LANGS):
        for m, (label, color, per) in rows.items():
            if l not in per:
                continue
            eps = sorted(per[l])
            xs = [0] + eps
            ys = [0.0] + [per[l][e] for e in eps]
            ax.plot(xs, ys, "o-", color=color, lw=2, ms=6, label=label)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(LANG_NAME[l], fontsize=10)
        ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
    axes[0].legend(fontsize=9, title="unlearn method")
    for ax in axes[5:]:
        ax.set_xlabel("relearn epochs", fontsize=9)
    for ax in (axes[0], axes[5]):
        ax.set_ylabel("TRUTH-RATIO recovery\n(baseline − relearned; ↑ = fact returns)", fontsize=9)
    fig.suptitle("Cross-lingual recovery on the FACT-specific TRUTH RATIO "
                 "(↑ = genuine fact recovery). ep1–ep2 only so far.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "recovery_vs_epoch_truthratio.png"
    fig.savefig(out, dpi=120)
    logger.info("truth-ratio recovery-vs-epoch -> %s", out)


if __name__ == "__main__":
    main()
