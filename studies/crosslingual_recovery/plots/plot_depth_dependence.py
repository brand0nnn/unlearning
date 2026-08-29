"""Finding #2 made concrete: DEEPER unlearning is MORE reversible.

For each (method, unlearning depth) we already have a full cross-lingual relearn
sweep. This collapses each sweep to a single point:
    x = unlearning depth      = baseline forget truth ratio (higher = forgot deeper)
    y = mean fraction of removed knowledge recovered at relearn ep2, across 10 langs
        fraction = (baseline_TR - relearned_TR) / (baseline_TR - LEARNED_TR)
Error bars = std across the 10 relearn languages (spread, not a seed CI).

Two depths per method (all from EXISTING JSON — no new run):
    Full-FT: UEP15 (crosslingual_facts) + UEP20 (crosslingual_deep)
    LoRA:    UEP32 (crosslingual_deep)  + UEP42 (crosslingual_facts)

Within each method the line slopes UP: forget deeper -> recover more.

    python studies/crosslingual_recovery/plots/plot_depth_dependence.py
    -> figures/depth_dependence.png
"""
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

STUDY = Path(__file__).resolve().parents[1]
REL = STUDY / "results" / "relearn"
FIGS = STUDY / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("plot_depth_dependence")

LEARNED_TR = 0.459
LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]

# (method, color, [ (label, group, checkpoint_basename), ... ] ) ordered shallow->deep
SERIES = [
    ("Full-FT", "#1f77b4", [
        ("UEP15", "crosslingual_facts", "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b"),
        ("UEP20", "crosslingual_deep",  "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b"),
    ]),
    ("LoRA", "#ff7f0e", [
        ("UEP32", "crosslingual_deep",  "tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b"),
        ("UEP42", "crosslingual_facts", "tofu_unlearn_gradient_difference_forget01_lora_uep42_qwen3-8b"),
    ]),
]


def _stats(group, base):
    d = json.load(open(REL / group / f"{base}.json"))
    b = d[base]["truth_ratio"] if isinstance(d[base], dict) else d[base]
    fr = []
    for l in LANGS:
        k = f"relearn_{base}_via_retain" + ("" if l == "en" else f"_lang{l}") + "_ep2"
        v = d[k]["truth_ratio"]
        fr.append((b - v) / (b - LEARNED_TR))
    mean = sum(fr) / len(fr)
    std = (sum((x - mean) ** 2 for x in fr) / len(fr)) ** 0.5
    return b, mean, std


def main():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for method, color, pts in SERIES:
        xs, ys, es, labels = [], [], [], []
        for label, group, base in pts:
            b, mean, std = _stats(group, base)
            xs.append(b); ys.append(mean); es.append(std); labels.append(label)
        # order by depth (baseline TR) so the connecting line reads shallow->deep
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
        es = [es[i] for i in order]; labels = [labels[i] for i in order]
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, lw=2.2, ms=9,
                    capsize=4, elinewidth=1.3, label=method, zorder=3)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(f"{lab}\nTR {x:.2f} -> {y:+.0%}", (x, y),
                        textcoords="offset points", xytext=(10, -6),
                        fontsize=8.5, color=color)

    ax.axhline(0, color="grey", lw=1.0, ls="--")
    ax.text(0.585, 0.02, "no recovery", fontsize=8, color="grey")
    ax.set_xlabel("unlearning DEPTH  =  forget-set truth ratio after unlearning\n"
                  "(higher = forgot the fact more deeply)", fontsize=11)
    ax.set_ylabel("mean fraction of removed knowledge recovered\n"
                  "(benign relearn ep2, averaged over 10 languages)", fontsize=11)
    ax.set_ylim(-0.1, 0.9)
    ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
    ax.legend(fontsize=11, title="unlearn method", loc="upper left")
    ax.set_title("Finding #2: DEEPER unlearning is MORE reversible\n"
                 "within each method, forgetting the fact more deeply leaves it "
                 "EASIER to relearn (line slopes up)", fontsize=12)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "depth_dependence.png"
    fig.savefig(out, dpi=120)
    logger.info("depth-dependence -> %s", out)


if __name__ == "__main__":
    main()
