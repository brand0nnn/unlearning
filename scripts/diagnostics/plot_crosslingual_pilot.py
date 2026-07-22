"""Cross-lingual recovery PILOT — plot English forget recovery vs relearn-language
distance, one line per training method.

Reads results/relearn/crosslingual_pilot/*.json (written by relearn_measure.py) and
plots, for each method, the recovered English forget-set ROUGE after benign retain-
relearning in each language, with the language axis ordered by typological distance
from English. When the pilot relearned at several epochs, the MAX-epoch (peak-
recovery) value is used per language. Each method's dashed line = its unlearned
baseline. The y-axis auto-scales to the data (recovery can be small for a deep forget).

Headline to look for: does the DECAY SLOPE differ between methods? A steeper drop with
distance for one method = it modulates cross-lingual recovery (the make-or-break signal).

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
    """(method, lang|'baseline', epoch) from a relearn_measure checkpoint key."""
    m = "fullft" if "fullft" in key else ("lora" if "lora" in key else "?")
    if not key.startswith("relearn_"):
        return m, "baseline", 0
    lang = re.search(r"_lang([a-z]+)_ep", key)
    ep = re.search(r"_ep(\d+)$", key)
    return m, (lang.group(1) if lang else "en"), (int(ep.group(1)) if ep else 0)


def main():
    src = Path("results/relearn/crosslingual_pilot")
    files = glob.glob(str(src / "*.json"))
    if not files:
        logger.warning("no pilot results in %s — run the pilot sbatch first", src)
        return
    merged = {}
    for f in files:
        merged.update(json.load(open(f)))

    # rows[method] = {"baseline": x, lang: {epoch: recovery}}
    rows = {}
    for key, val in merged.items():
        m, l, ep = _parse(key)
        r = rows.setdefault(m, {})
        if l == "baseline":
            r["baseline"] = val
        else:
            r.setdefault(l, {})[ep] = val

    # Two panels:
    #  (L) RAW peak recovery vs distance, with each method's unlearned baseline.
    #  (R) NORMALIZED decay — recovery ABOVE baseline, scaled to each method's own
    #      English (d=0) recovery = 1.0. This compares the *shape* of the decay
    #      independent of how much each method recovers overall, so it is robust to
    #      the two methods starting from different forget depths (the baseline confound).
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.6))
    ymax = 0.05
    for m, data in sorted(rows.items()):
        langs = sorted((l for l in data if l != "baseline"),
                       key=lambda l: LANG_DIST.get(l, 99))
        if not langs:
            continue
        base = data.get("baseline", 0.0)
        xs = [LANG_DIST.get(l, 99) for l in langs]
        # peak recovery per language = value at the largest relearn epoch present
        ys = [data[l][max(data[l])] for l in langs]
        ymax = max(ymax, max(ys), base)
        color = METHOD_COLOR.get(m, "grey")
        axL.plot(xs, ys, "o-", lw=2.2, ms=8, color=color, label=METHOD_NAME.get(m, m))
        axL.axhline(base, ls="--", lw=1.1, color=color, alpha=0.55)
        axL.text(max(xs), base, f" {METHOD_NAME.get(m, m)} baseline",
                 fontsize=7, color=color, ha="right", va="bottom")

        # normalized: (recovery - baseline) / (English recovery - baseline)
        above = [data[l][max(data[l])] - base for l in langs]
        en_above = data["en"][max(data["en"])] - base if "en" in data else None
        if en_above and abs(en_above) > 1e-6:
            axR.plot(xs, [a / en_above for a in above], "o-", lw=2.2, ms=8,
                     color=color, label=METHOD_NAME.get(m, m))

    present = sorted({l for d in rows.values() for l in d if l != "baseline"},
                     key=lambda l: LANG_DIST.get(l, 99))
    ticks = [LANG_DIST.get(l, 99) for l in present]
    ticklabels = [f"{LANG_NAME.get(l, l)}\n(d={LANG_DIST.get(l, '?')})" for l in present]
    for ax in (axL, axR):
        ax.set_xticks(ticks); ax.set_xticklabels(ticklabels, fontsize=8)
        ax.set_xlabel("relearn language  (→ typological distance from English)", fontsize=10)
        ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
        ax.legend(fontsize=10, title="unlearn method")
    axL.set_ylabel("English forget ROUGE after benign relearn\n(↑ = more recovered)", fontsize=10)
    axL.set_ylim(0, ymax * 1.25)
    axL.set_title("RAW recovery vs distance", fontsize=11)
    axR.axhline(0, color="grey", lw=0.8)
    axR.set_ylabel("recovery above baseline,\nnormalized to English (=1.0)", fontsize=10)
    axR.set_title("NORMALIZED decay shape\n(baseline-confound-free)", fontsize=11)
    fig.suptitle("Cross-lingual recovery — does recovery decay with language distance, "
                 "and does the DECAY differ by method?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path("results/figures/crosslingual_pilot_recovery.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    logger.info("Cross-lingual pilot recovery -> %s", out)


if __name__ == "__main__":
    main()
