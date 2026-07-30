"""Cross-lingual recovery TRAJECTORY vs relearn epochs.

For each unlearn method (Full-FT, LoRA) and each relearn language, plot recovery
ABOVE the method's unlearned baseline as a function of relearn epochs. Recovery at
epoch 0 is 0 by definition (no relearning = the baseline), so each line runs
(0,0) -> (2, .) -> (4, .). One subplot per language (both methods overlaid), so the
method comparison is on shared axes per language.

    python studies/crosslingual_recovery/plots/plot_recovery_vs_epoch.py
    -> figures/recovery_vs_epoch.png

Local/CPU, no torch.
"""
import glob
import json
import re
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results" / "relearn" / "crosslingual_pilot"
FIGS = STUDY / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("plot_recovery_vs_epoch")

LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]  # near->far
LANG_NAME = {"en": "English", "fr": "French", "id": "Indonesian", "ru": "Russian",
             "hi": "Hindi", "fa": "Farsi", "ar": "Arabic", "iw": "Hebrew",
             "ko": "Korean", "ja": "Japanese"}
METHOD_COLOR = {"fullft": "#1f77b4", "lora": "#ff7f0e"}
METHOD_NAME = {"fullft": "Full-FT", "lora": "LoRA"}


def _method(key):
    return "fullft" if "fullft" in key else ("lora" if "lora" in key else "?")


def main():
    # rows[method] = {"baseline": x, lang: {epoch: recovery_value}}
    rows = {}
    for f in glob.glob(str(RESULTS / "*.json")):
        d = json.load(open(f))
        for key, val in d.items():
            m = _method(key)
            r = rows.setdefault(m, {"baseline": None})
            if not key.startswith("relearn_"):
                r["baseline"] = val
                continue
            lang = re.search(r"_lang([a-z]+)_ep", key)
            ep = re.search(r"_ep(\d+)$", key)
            l = lang.group(1) if lang else "en"
            e = int(ep.group(1)) if ep else 0
            r.setdefault(l, {})[e] = val
    if not rows:
        logger.warning("no data in %s", RESULTS); return

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, l in zip(axes, LANGS):
        for m in ("fullft", "lora"):
            r = rows.get(m)
            if not r or r["baseline"] is None or l not in r:
                continue
            base = r["baseline"]
            eps = sorted(r[l])
            xs = [0] + eps                              # 0 epochs = 0 recovery
            ys = [0.0] + [r[l][e] - base for e in eps]
            ax.plot(xs, ys, "o-", color=METHOD_COLOR[m], lw=2, ms=6,
                    label=METHOD_NAME[m])
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(LANG_NAME[l], fontsize=10)
        ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
    axes[0].legend(fontsize=9, title="unlearn method")
    for ax in axes[5:]:
        ax.set_xlabel("relearn epochs", fontsize=9)
    for ax in (axes[0], axes[5]):
        ax.set_ylabel("recovery above baseline", fontsize=9)
    fig.suptitle("Cross-lingual recovery trajectory vs relearn epochs "
                 "(recovery above each method's unlearned baseline)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "recovery_vs_epoch.png"
    fig.savefig(out, dpi=120)
    logger.info("recovery-vs-epoch -> %s", out)


if __name__ == "__main__":
    main()
