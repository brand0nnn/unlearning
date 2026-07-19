"""LoRA-locality ablation — plot the spectral trace per location.

Two panels, one bar per location (labeled with rank + LoRA params, ordered by trace
magnitude): detection accuracy (all ~1.0 => every location leaves a detectable trace
= suppression) and the spectral shift max|Cohen's d| (does the trace MAGNITUDE vary
by location, even though recoverability was uniform?). Reads
lora_locality/out/<scheme>/spectral/*.json.

    python lora_locality/plot_spectral.py --scheme fixedbudget
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.plotting import _final_layer_direction, _signature_panel
from src.utils.logging_utils import get_logger

logger = get_logger("locality_plot_spectral")


def _signature_grid(data, scheme, out):
    """Overlaid original-vs-unlearned projection densities (the spectral SIGNATURE),
    one panel per location — the distribution view, not just the distance."""
    ncol = 2
    nrow = (len(data) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("viridis")
    for i, (ax, d) in enumerate(zip(axes, data)):
        lay, direction = _final_layer_direction(d)
        L = d["per_layer"][lay]
        po = np.array(L.get("proj_orig", []))
        pu = np.array(L.get("proj_unlearned", []))
        if po.ndim != 2 or po.shape[0] == 0:
            ax.set_visible(False)
            continue
        label = f"{d['location']} (r{d['rank']}, {d['params'] // 10**6}M)"
        color = cmap(0.1 + 0.8 * i / max(len(data) - 1, 1))
        _signature_panel(ax, d["location"], L, po, pu, lay, direction,
                         legend=(ax is axes[0]), color=color, title_label=label)
    for ax in axes[len(data):]:
        ax.set_visible(False)
    fig.suptitle(f"LoRA-locality spectral signatures — {scheme}\n"
                 "projection of forget-irrelevant responses onto the localizing SV — "
                 "original vs unlearned", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=110)
    logger.info("Locality spectral signatures -> %s", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True)
    args = ap.parse_args()

    sdir = Path(f"lora_locality/out/{args.scheme}/spectral")
    files = sorted(sdir.glob("*.json"))
    if not files:
        logger.warning("no spectral JSON in %s — run lora_locality/spectral.py first", sdir)
        return

    data = sorted((json.load(open(f)) for f in files), key=lambda d: d["max_spectral_shift"])
    _signature_grid(data, args.scheme, sdir.parent / f"spectral_signature_{args.scheme}.png")
    labels = [f"{d['location']} (r{d['rank']}, {d['params'] // 10**6}M)" for d in data]
    acc = [d["detection_accuracy"] for d in data]
    shift = [d["max_spectral_shift"] for d in data]
    y = list(range(len(data)))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.1 + 0.8 * i / max(len(data) - 1, 1)) for i in y]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.7 * len(data) + 2.5))

    a1.barh(y, acc, color=colors, alpha=0.9, edgecolor="white")
    a1.axvline(0.5, ls="--", c="grey", lw=1, label="chance (0.5 = invisible)")
    for yi, v in zip(y, acc):
        a1.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=8)
    a1.set_yticks(y)
    a1.set_yticklabels(labels, fontsize=8)
    a1.set_xlim(0.4, 1.05)
    a1.set_xlabel("detection accuracy (↑ louder trace)", fontsize=10)
    a1.set_title("Trace detectability\n(best layer, 5-fold CV)", fontsize=10)
    a1.legend(fontsize=8, loc="lower right")
    a1.invert_yaxis()

    a2.barh(y, shift, color=colors, alpha=0.9, edgecolor="white")
    for yi, v in zip(y, shift):
        a2.text(v + max(shift) * 0.01, yi, f"{v:.2f}", va="center", fontsize=8)
    a2.set_yticks(y)
    a2.set_yticklabels([])
    a2.set_xlabel("spectral shift  max|Cohen's d|  (↑ louder)", fontsize=10)
    a2.set_title("Trace magnitude", fontsize=10)
    a2.invert_yaxis()

    for ax in (a1, a2):
        ax.xaxis.grid(True, alpha=0.25, ls="--")
        ax.set_axisbelow(True)
    fig.suptitle(f"LoRA-locality spectral trace — {args.scheme}\n"
                 "all detectable = all suppression; does the magnitude vary by location?",
                 fontsize=11)
    fig.tight_layout()
    out = sdir.parent / f"spectral_{args.scheme}.png"
    fig.savefig(out, dpi=110)
    logger.info("Locality spectral (%d locations) -> %s", len(data), out)


if __name__ == "__main__":
    main()
