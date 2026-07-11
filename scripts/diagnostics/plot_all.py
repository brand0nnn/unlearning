"""Regenerate ALL deliverable figures LOCALLY from the stored JSON — no GPU, no
cluster. The cluster only ever produces metric JSON (results/curves, results/relearn,
results/spectral); this turns that data into the figures, and can be re-run any time
without recomputing anything.

    python scripts/diagnostics/plot_all.py

Deliverables (all -> results/figures/):
  1. Unlearning dynamics  — ROUGE / Truth-Ratio / Probability vs step, 4 strategies
  2a. Relearn forget      — forget-ROUGE recovery when re-fine-tuning on the FORGET set
  2b. Relearn retain      — forget-ROUGE recovery when re-fine-tuning on the RETAIN set
  3. Spectral fingerprint — final-layer signature + detectability, 4 strategies
  4. (kept) LoRA target-module ablation recovery
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(args):
    print(">>", " ".join(args))
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main():
    # 1. Unlearning dynamics curves (all curves in results/curves/).
    run(["scripts/diagnostics/plot_unlearn_curve.py"])

    # 2a. Relearn forget — re-fine-tune on the FORGET set (direct recovery).
    run(["scripts/diagnostics/plot_relearn.py",
         "--data", "results/relearn/forget",
         "--out", "relearn_forget_curve.png"])

    # 2b. Relearn retain — re-fine-tune on the RETAIN set (the suppression signal).
    run(["scripts/diagnostics/plot_relearn.py",
         "--data", "results/relearn/retain",
         "--out", "relearn_retain_curve.png",
         "--xlabel", "Relearning epochs on retain",
         "--title", "Relearning on retain: does unrelated fine-tuning "
                    "jog forgotten knowledge?"])

    # 3. Final-layer spectral fingerprint, 4 strategies.
    run(["scripts/diagnostics/plot_spectral_strategies.py"])

    # 4. (kept) LoRA target-module ablation, if its data is present.
    if (ROOT / "results/relearn/lora_ablation").is_dir():
        run(["scripts/diagnostics/plot_relearn.py",
             "--data", "results/relearn/lora_ablation",
             "--label-by", "lora_target",
             "--out", "lora_ablation_recovery.png",
             "--title", "LoRA target-module ablation: recovery after unlearning"])

    print("\nAll figures -> results/figures/")


if __name__ == "__main__":
    main()
