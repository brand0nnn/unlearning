"""Final-layer spectral fingerprint, one point per TRAINING STRATEGY.

All four strategies are NPO-style unlearners, so (per the spectral paper) we read
the fingerprint at the FINAL post-RMSNorm layer / SV1:

    Full-FT (grad-diff) · LoRA (grad-diff) · Self-Distillation · GRPO

    python scripts/diagnostics/plot_spectral_strategies.py
    -> results/figures/spectral_detectability_strategies.png
    -> results/figures/spectral_signature_strategies.png

CPU-only, local. Reads results/spectral/spectral_*.json (no recompute); each
strategy is skipped if its JSON isn't there yet.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.plotting import spectral_detectability, spectral_signature_grid
from src.utils.logging_utils import get_logger

logger = get_logger("plot_spectral_strategies")

# The four training strategies = the comparison axis. Each entry is skipped if its
# spectral_*.json isn't present yet, so the plot works before GRPO's is computed.
STRATEGY_CKPTS = [
    "tofu_unlearn_gradient_difference_forget10",         # Full-FT
    "tofu_unlearn_gradient_difference_forget10_lora",    # LoRA
    "tofu_unlearn_self_distill_forget10_self_distill",   # Self-Distillation
    "tofu_unlearn_grpo_forget10_grpo_lora",              # GRPO (LoRA-GRPO checkpoint name)
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
