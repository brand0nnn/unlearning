"""Cross-lingual recovery PILOT — plot English forget recovery vs relearn-language
distance, one line per training method.

Reads results/relearn/crosslingual_pilot/*.json (written by relearn_measure.py during
the pilot) and plots, for each method, the recovered English forget-set ROUGE after
benign retain-relearning in each language, with the language axis ordered by
typological distance from English. Each method's dashed line = its unlearned baseline
(the floor before relearning).

The headline to look for: does the DECAY SLOPE differ between methods? A steeper drop
with distance for one method = its cross-lingual recovery is more distance-sensitive
= the method MODULATES cross-lingual recovery (the pilot's make-or-break signal).

    python scripts/diagnostics/plot_crosslingual_pilot.py
    -> results/figures/crosslingual_pilot_recovery.png

Local/CPU, no torch.
"""
import glob
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("plot_crosslingual_pilot")

# Rough typological distance from English (ordinal placeholder — swap for a proper
# metric later). Lower = closer. en=0 (same language).
LANG_DIST = {"en": 0, "fr": 1, "id": 2, "ru": 3, "hi": 4,
             "fa": 5, "ar": 6, "iw": 7, "ko": 8, "ja": 9}
LANG_NAME = {"en": "English", "fr": "French", "id": "Indonesian", "ru": "Russian",
             "hi": "Hindi", "fa": "Farsi", "ar": "Arabic", "iw": "Hebrew",
             "ko": "Korean", "ja": "Japanese"}
METHOD_COLOR = {"fullft": "#1f77b4", "lora": "#ff7f0e"}
METHOD_NAME = {"fullft": "Full-FT", "lora": "LoRA"}


def _parse(key):
    """(method, lang|'baseline') from a relearn_measure checkpoint key."""
    m = "fullft" if "fullft" in key else ("lora" if "lora" in key else "?")
    if not key.startswith("relearn_"):
        return m, "baseline"
    mm = re.search(r"_lang([a-z]+)_ep", key)
    return m, (mm.group(1) if mm else "en")


def main():
    src = Path("results/relearn/crosslingual_pilot")
    files = glob.glob(str(src / "*.json"))
    if not files:
        logger.warning("no pilot results in %s — run the pilot sbatch first", src)
        return
    merged = {}
    for f in files:
        merged.update(json.load(open(f)))

    rows = {}   # method -> {lang: recovery, 'baseline': floor}
    for key, val in merged.items():
        m, l = _parse(key)
        rows.setdefault(m, {})[l] = val

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m, data in sorted(rows.items()):
        langs = sorted((l for l in data if l != "baseline"),
                       key=lambda l: LANG_DIST.get(l, 99))
        if not langs:
            continue
        xs = [LANG_DIST.get(l, 99) for l in langs]
        ys = [data[l] for l in langs]
        color = METHOD_COLOR.get(m, "grey")
        ax.plot(xs, ys, "o-", lw=2.2, ms=8, color=color, label=METHOD_NAME.get(m, m))
        if "baseline" in data:
            ax.axhline(data["baseline"], ls="--", lw=1.2, color=color, alpha=0.6)
            ax.text(max(xs), data["baseline"] + 0.01,
                    f"{METHOD_NAME.get(m, m)} unlearned baseline", fontsize=7,
                    color=color, ha="right", va="bottom")
        # annotate the EN->far decay if both ends exist
        if "en" in data and langs[-1] != "en":
            gap = data["en"] - data[langs[-1]]
            ax.annotate(f"Δ(en→{langs[-1]})={gap:+.2f}",
                        (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(6, -2), fontsize=8, color=color)

    present = sorted({l for d in rows.values() for l in d if l != "baseline"},
                     key=lambda l: LANG_DIST.get(l, 99))
    ax.set_xticks([LANG_DIST.get(l, 99) for l in present])
    ax.set_xticklabels([f"{LANG_NAME.get(l, l)}\n(d={LANG_DIST.get(l, '?')})" for l in present],
                       fontsize=9)
    ax.set_xlabel("relearn language  (→ increasing typological distance from English)", fontsize=10)
    ax.set_ylabel("English forget-set ROUGE after benign relearn\n(↑ = more recovered)", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
    ax.legend(fontsize=10, title="unlearn method")
    ax.set_title("Cross-lingual recovery pilot — does recovery decay with distance,\n"
                 "and does the DECAY differ by method?", fontsize=11)
    fig.tight_layout()
    out = Path("results/figures/crosslingual_pilot_recovery.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    logger.info("Cross-lingual pilot recovery -> %s", out)


if __name__ == "__main__":
    main()
