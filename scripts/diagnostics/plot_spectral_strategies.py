"""Strategy-only spectral figures (axis 2 — training-strategy comparison).

The main spectral plots (04_plot) include all 6 checkpoints — the 4 Full-FT TOFU
methods (the paper-reproduction / method axis) PLUS the strategy variants. For the
training-STRATEGY comparison we want just the gradient_difference-family points,
coloured/labelled by strategy:

    gradient_difference (Full-FT)  ·  gradient_difference (LoRA)  ·  self-distill

    python scripts/diagnostics/plot_spectral_strategies.py
    -> results/spectral_detectability_strategies.png
    -> results/spectral_signature_strategies.png

CPU-only. Reads the existing results/spectral_*.json (no recompute).
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.plotting import spectral_detectability, spectral_signature_grid
from src.utils.logging_utils import get_logger

logger = get_logger("plot_spectral_strategies")

# The gradient_difference-family checkpoints = the strategy axis (GRPO when ready).
STRATEGY_CKPTS = [
    "tofu_unlearn_gradient_difference_forget10",         # Full-FT
    "tofu_unlearn_gradient_difference_forget10_lora",    # LoRA
    "tofu_unlearn_self_distill_forget10_self_distill",   # Self-Distillation
    "tofu_unlearn_grpo_forget10_grpo",                   # GRPO (if present)
]


def main():
    results_dir = Path("results/spectral")
    spec = {}
    for name in STRATEGY_CKPTS:
        f = results_dir / f"spectral_{name}.json"
        if f.exists():
            spec[name] = json.load(open(f))
        else:
            logger.info("skip (no spectral result yet): %s", name)
    if not spec:
        logger.warning("No strategy spectral_*.json found. Run recover_spectral first.")
        return

    # final_layer=True -> the paper's NPO analysis (final post-RMSNorm layer, SV1),
    # which is where these loss-based strategies' fingerprints concentrate.
    spectral_detectability(spec, "results/figures", strategy_view=True, final_layer=True)
    spectral_signature_grid(spec, "results/figures", strategy_view=True, final_layer=True)
    logger.info("Strategy spectral figures for %d strategies -> "
                "spectral_detectability_strategies.png, spectral_signature_strategies.png",
                len(spec))


if __name__ == "__main__":
    main()
