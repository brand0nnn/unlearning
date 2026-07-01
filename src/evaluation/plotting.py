"""Plots for the TOFU results.

Two figures:
  1. forget_quality_vs_utility — the canonical TOFU scatter (matches paper Fig 5/6).
     Includes: retain model gold star, shaded target region, paper reference values
     as ghost annotations so you can see at a glance whether reproduction succeeded.
  2. rouge_by_split — bar chart of ROUGE across four splits per method, showing
     how well the model preserves knowledge it should NOT forget.
"""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paper reference values (Maini et al. 2024, Table 1 — forget05, Llama-2-7B).
# Used as ghost markers so you can compare your run against the paper directly.
# Keys must match the method names produced by your pipeline.
# ---------------------------------------------------------------------------
PAPER_REFS = {
    "gradient_ascent":     {"model_utility": 0.2,   "forget_quality_log10": -0.05},
    "gradient_difference": {"model_utility": 0.444,  "forget_quality_log10": -0.3},
    "kl_minimization":     {"model_utility": 0.444,  "forget_quality_log10": -0.25},
    "idk":                 {"model_utility": 0.283,  "forget_quality_log10": -0.15},
}

# One consistent colour per method — used in both figures so they cross-reference.
METHOD_COLORS = {
    "gradient_ascent":     "#e63946",   # red   — aggressive, destroys utility
    "gradient_difference": "#2a9d8f",   # teal
    "kl_minimization":     "#457b9d",   # blue
    "idk":                 "#f4a261",   # orange
}
DEFAULT_COLOR = "#6c757d"


def forget_quality_vs_utility(results_by_method: Dict[str, Dict], out_dir: str,
                               retain_result: Dict = None):
    """
    results_by_method[name] = {"model_utility": x, "forget_quality_log10": y}
    retain_result (optional) = {"model_utility": x, "forget_quality_log10": y}
        Pass the retain model's scores so it appears as the gold-star reference.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # --- shaded target region (top-right = forget well + keep utility) ------
    ax.axhspan(-0.5, 0.5, xmin=0.6, alpha=0.07, color="green",
               label="target region (forget well + preserve utility)")

    # --- paper ghost markers (open circles) ----------------------------------
    for name, ref in PAPER_REFS.items():
        ax.scatter(ref["model_utility"], ref["forget_quality_log10"],
                   s=180, facecolors="none",
                   edgecolors=METHOD_COLORS.get(name, DEFAULT_COLOR),
                   linewidths=1.5, zorder=2)
    # single legend entry for all ghosts
    ghost_patch = mpatches.Patch(facecolor="none", edgecolor="grey",
                                 linewidth=1.5, label="paper reference (open)")
    ax.add_patch(ghost_patch)           # dummy — just for the legend handle

    # --- your results (filled circles) --------------------------------------
    for name, r in results_by_method.items():
        color = METHOD_COLORS.get(name, DEFAULT_COLOR)
        ax.scatter(r["model_utility"], r["forget_quality_log10"],
                   s=160, color=color, zorder=3, label=name)
        ax.annotate(name.replace("_", " "),
                    (r["model_utility"], r["forget_quality_log10"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8.5,
                    color=color)

        # draw arrow from ghost to your point so drift is immediately obvious
        if name in PAPER_REFS:
            ref = PAPER_REFS[name]
            ax.annotate("",
                xy=(r["model_utility"], r["forget_quality_log10"]),
                xytext=(ref["model_utility"], ref["forget_quality_log10"]),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.0, alpha=0.5))

    # --- retain model gold star ---------------------------------------------
    if retain_result:
        ax.scatter(retain_result["model_utility"], retain_result["forget_quality_log10"],
                   s=350, marker="*", color="gold", edgecolors="darkgoldenrod",
                   linewidths=1, zorder=5, label="retain model (gold reference)")
        ax.annotate("retain\n(gold)",
                    (retain_result["model_utility"], retain_result["forget_quality_log10"]),
                    textcoords="offset points", xytext=(8, -14), fontsize=8,
                    color="darkgoldenrod")

    # --- reference lines ----------------------------------------------------
    ax.axhline(0, ls="--", c="grey", lw=0.8, alpha=0.6)
    ax.axhline(-1, ls=":", c="grey", lw=0.8, alpha=0.4,
               label="p < 0.1 threshold (log10 = −1)")

    ax.set_xlabel("Model Utility  (↑ less collateral damage)", fontsize=11)
    ax.set_ylabel("Forget Quality  (log10 p-value, ↑ better forgetting)", fontsize=11)
    ax.set_title("TOFU: Forget Quality vs Model Utility\n"
                 "filled = your run   open = paper reference   ★ = retain model",
                 fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    # add the ghost patch manually since ax.add_patch doesn't auto-register it
    handles.append(mpatches.Patch(facecolor="none", edgecolor="grey",
                                  linewidth=1.5))
    labels.append("paper reference (open circle)")
    ax.legend(handles, labels, fontsize=8, loc="lower left",
              framealpha=0.85, ncol=1)

    ax.set_xlim(left=0.0)
    fig.tight_layout()
    _save(fig, out_dir, "forget_quality_vs_utility")


def rouge_by_split(rouge_by_method_split: Dict[str, Dict[str, float]], out_dir: str):
    """
    rouge_by_method_split[method][split] = rouge value.

    Paper expectation:
      forget split  → should DROP after unlearning (good unlearning)
      retain/real_authors/world_facts → should STAY HIGH (no collateral damage)
    Annotates the expected direction on the plot so it's easy to check.
    """
    splits = ["forget", "retain", "real_authors", "world_facts"]
    split_labels = ["Forget\n(↓ good)", "Retain\n(↑ good)",
                    "Real Authors\n(↑ good)", "World Facts\n(↑ good)"]
    methods = list(rouge_by_method_split.keys())
    n = len(methods)
    width = 0.7 / max(n, 1)
    x = np.arange(len(splits))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, m in enumerate(methods):
        vals = [rouge_by_method_split[m].get(s, 0.0) for s in splits]
        color = METHOD_COLORS.get(m, DEFAULT_COLOR)
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, label=m.replace("_", " "),
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        # value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                        color=color)

    # shading: forget column should be low, rest should be high
    ax.axvspan(-0.5, 0.5, alpha=0.06, color="red",   label="should drop (unlearned)")
    ax.axvspan(0.5,  3.5, alpha=0.04, color="green", label="should stay high (retained)")

    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=10)
    ax.set_ylabel("ROUGE-L recall", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("ROUGE-L by split — checking unlearning vs collateral damage", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, out_dir, "rouge_by_split")


def _save(fig, out_dir: str, name: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot -> %s", path)