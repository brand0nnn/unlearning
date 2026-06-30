"""Plots for the TOFU results.

Two figures:
  1. forget_quality_vs_utility — the canonical TOFU scatter. x = Model Utility,
     y = Forget Quality (log10 p-value). Each unlearning method is a point. The
     top-right is the goal: forget well (high y) while keeping utility (high x).
  2. rouge_by_split — bar chart of ROUGE across the four splits for each method,
     which visualizes knowledge entanglement (damage should fall off as you move
     forget -> retain -> real_authors -> world_facts).

These write PNG + PDF into results/ so you can drop them straight into a report
or the NUS slide template.
"""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def forget_quality_vs_utility(results_by_method: Dict[str, Dict], out_dir: str):
    """results_by_method[name] = {"model_utility": x, "forget_quality_log10": y}."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results_by_method.items():
        ax.scatter(r["model_utility"], r["forget_quality_log10"], s=120, label=name)
        ax.annotate(name, (r["model_utility"], r["forget_quality_log10"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    # Reference line: a gold retain model sits at forget_quality ~ 1 (log10 ~ 0).
    ax.axhline(0, ls="--", c="grey", lw=1, label="ideal forgetting (p≈1)")
    ax.set_xlabel("Model Utility  (higher = less collateral damage)")
    ax.set_ylabel("Forget Quality  (log10 p-value, higher = better forgetting)")
    ax.set_title("TOFU: Forget Quality vs Model Utility")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    _save(fig, out_dir, "forget_quality_vs_utility")


def rouge_by_split(rouge_by_method_split: Dict[str, Dict[str, float]], out_dir: str):
    """rouge_by_method_split[method][split] = rouge value.

    Splits are ordered by distance from the forget data to show entanglement.
    """
    splits = ["forget", "retain", "real_authors", "world_facts"]
    methods = list(rouge_by_method_split.keys())
    x = range(len(splits))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(methods):
        vals = [rouge_by_method_split[m].get(s, 0.0) for s in splits]
        ax.bar([xi + i * width for xi in x], vals, width=width, label=m)
    ax.set_xticks([xi + width * (len(methods) - 1) / 2 for xi in x])
    ax.set_xticklabels(splits, rotation=15)
    ax.set_ylabel("ROUGE-L recall")
    ax.set_title("ROUGE by split (left = should drop, right = should stay high)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "rouge_by_split")


def _save(fig, out_dir: str, name: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot -> %s", path)