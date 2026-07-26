"""Cross-lingual blast-radius plot using the Multilingual Amnesia metrics
(normalized probability + truth ratio), NOT ROUGE.

Reads results/crosslingual/crosslingual_probe/*.json (written by crosslingual_probe.py)
and makes a 2-panel figure:

  (L) BLAST RADIUS — P_forget(unlearned) / P_forget(learned) per language, ordered by
      distance from English, one line per unlearn method. Lower = more forgotten. This
      is the Amnesia Fig-2 quantity. Languages whose PRE-unlearn knowledge is at the
      floor (truth ratio ~1, i.e. the model never really knew the fact there) are shaded
      grey — their ratio is not trustworthy.
  (R) BASELINE RELIABILITY — the learned model's truth ratio per language. Values near
      1.0 mean the model can't tell true from false answers even before unlearning, so
      "forgetting" there is meaningless. Shows the Llama-2 multilingual ceiling.

    python studies/crosslingual_recovery/plots/plot_crosslingual_probe.py
    -> figures/crosslingual_probe_blast_radius.png

Local/CPU, no torch.
"""
import glob
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results" / "crosslingual" / "crosslingual_probe"
FIGS = STUDY / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("plot_crosslingual_probe")

# distance-from-English order (near -> far) + display names
LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]
LANG_NAME = {"en": "English", "fr": "French", "id": "Indonesian", "ru": "Russian",
             "hi": "Hindi", "fa": "Farsi", "ar": "Arabic", "iw": "Hebrew",
             "ko": "Korean", "ja": "Japanese"}
# Truth ratio above this on the LEARNED model => the model barely knew the fact in that
# language (prefers wrong answers ~as much as right), so its blast-radius is unreliable.
TR_FLOOR = 0.85
METHOD_COLOR = {"fullft": "#1f77b4", "lora": "#ff7f0e"}
METHOD_NAME = {"fullft": "Full-FT", "lora": "LoRA"}


def _method(base):
    if "learn" in base and "unlearn" not in base:
        return "learned"
    if "fullft" in base:
        return "fullft"
    if "lora" in base:
        return "lora"
    return "?"


def main():
    files = glob.glob(str(RESULTS / "*.json"))
    if not files:
        logger.warning("no probe results in %s — run crosslingual_probe.sbatch first", RESULTS)
        return
    merged = {}
    for f in files:
        merged.update(json.load(open(f)))

    # rows[method][lang] = {"prob":..., "truth_ratio":...}
    rows = {}
    for key, m in merged.items():
        base, lang = key.split("@")
        rows.setdefault(_method(base), {})[lang] = m
    learned = rows.get("learned", {})
    if not learned:
        logger.warning("no learned baseline found — cannot compute blast-radius ratio")
        return

    # which languages are reliable (model actually knew the fact pre-unlearn)
    reliable = {l for l in LANGS if learned.get(l, {}).get("truth_ratio", 1.0) < TR_FLOOR}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.8))
    xs = list(range(len(LANGS)))

    # shade the unreliable (floor) languages on the blast-radius panel
    for i, l in enumerate(LANGS):
        if l not in reliable:
            axL.axvspan(i - 0.5, i + 0.5, color="grey", alpha=0.10, zorder=0)

    # (L) blast radius = unlearned / learned prob
    for meth in ("fullft", "lora"):
        data = rows.get(meth)
        if not data:
            continue
        ys = []
        for l in LANGS:
            bp = learned.get(l, {}).get("prob", float("nan"))
            up = data.get(l, {}).get("prob", float("nan"))
            ys.append(up / bp if bp else float("nan"))
        axL.plot(xs, ys, "o-", lw=2.4, ms=8, color=METHOD_COLOR[meth],
                 label=METHOD_NAME[meth])
    axL.axhline(1.0, color="grey", lw=0.8, ls=":")
    axL.text(len(LANGS) - 1, 1.0, " no forgetting", fontsize=7, color="grey",
             va="bottom", ha="right")
    axL.set_ylim(0, 1.05)
    axL.set_ylabel("blast radius:  P(forget | unlearned) / P(forget | learned)\n"
                   "(↓ = more forgotten in that language)", fontsize=10)
    axL.set_title("Cross-lingual blast radius by method\n"
                  "(grey = model barely knew the fact there → unreliable)", fontsize=11)
    axL.legend(fontsize=10, title="unlearn method")

    # (R) baseline reliability = learned truth ratio
    trs = [learned.get(l, {}).get("truth_ratio", float("nan")) for l in LANGS]
    colors = ["#2ca02c" if l in reliable else "#bbbbbb" for l in LANGS]
    axR.bar(xs, trs, color=colors)
    axR.axhline(TR_FLOOR, color="red", lw=1.0, ls="--")
    axR.text(0, TR_FLOOR, f" floor τ={TR_FLOOR} (above = no real knowledge)",
             fontsize=7, color="red", va="bottom")
    axR.axhline(1.0, color="grey", lw=0.8, ls=":")
    axR.set_ylim(0, 1.05)
    axR.set_ylabel("learned model truth ratio\n(↑ toward 1 = can't tell true from false)",
                   fontsize=10)
    axR.set_title("Pre-unlearn knowledge per language\n"
                  "(green = real knowledge; grey = at the floor)", fontsize=11)

    ticklabels = [f"{LANG_NAME[l]}" for l in LANGS]
    for ax in (axL, axR):
        ax.set_xticks(xs)
        ax.set_xticklabels(ticklabels, rotation=40, ha="right", fontsize=8)
        ax.set_xlabel("evaluation language  (→ farther from English)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, ls="--"); ax.set_axisbelow(True)

    fig.suptitle("Cross-lingual forgetting from an English unlearn (TOFU forget01, "
                 "Llama-2-7b) — valid metrics: normalized prob + truth ratio", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "crosslingual_probe_blast_radius.png"
    fig.savefig(out, dpi=120)
    logger.info("blast-radius figure -> %s", out)


if __name__ == "__main__":
    main()
