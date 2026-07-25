"""Regenerate ALL of the strategy_comparison figures LOCALLY from stored JSON — no
GPU, no cluster. The cluster only ever produces metric JSON (this study's
results/{curves,relearn,spectral,forget_quality}); this turns that data into the
figures, and can be re-run any time without recomputing anything.

    python studies/strategy_comparison/plots/plot_all.py

Deliverables (all -> studies/strategy_comparison/figures/):
  1. Unlearning dynamics  — ROUGE / Truth-Ratio / Probability vs step, 4 strategies
  2a. Relearn forget      — forget-ROUGE recovery when re-fine-tuning on the FORGET set
  2b. Relearn retain      — forget-ROUGE recovery when re-fine-tuning on the RETAIN set
  3. Spectral fingerprint — final-layer signature + detectability, 4 strategies
  4. Forget-Quality vs Model-Utility plane (Fig 5/6)

(The LoRA target-module ablation figure now lives in its own study —
 studies/lora_target_ablation/ — see that folder's README.)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent           # studies/strategy_comparison/plots
STUDY = HERE.parent                              # studies/strategy_comparison
ROOT = STUDY
while ROOT != ROOT.parent and not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
RESULTS = STUDY / "results"
FIGS = STUDY / "figures"
SHARED_PLOTS = ROOT / "shared" / "plots"
PY = sys.executable


def run(args):
    print(">>", " ".join(str(a) for a in args))
    subprocess.run([PY, *[str(a) for a in args]], cwd=ROOT, check=True)


def main():
    # 1. Unlearning dynamics curves (this study's results/curves/).
    run([HERE / "plot_unlearn_curve.py"])

    # 2a. Relearn forget — re-fine-tune on the FORGET set (direct recovery).
    run([SHARED_PLOTS / "plot_relearn.py",
         "--data", RESULTS / "relearn" / "forget",
         "--fig-dir", FIGS,
         "--out", "relearn_forget_curve.png"])

    # 2b. Relearn retain — re-fine-tune on the RETAIN set (the suppression signal).
    run([SHARED_PLOTS / "plot_relearn.py",
         "--data", RESULTS / "relearn" / "retain",
         "--fig-dir", FIGS,
         "--out", "relearn_retain_curve.png",
         "--xlabel", "Relearning epochs on retain",
         "--title", "Relearning on retain: does unrelated fine-tuning "
                    "jog forgotten knowledge?"])

    # 3. Final-layer spectral fingerprint, 4 strategies.
    run([HERE / "plot_spectral_strategies.py"])

    # 4. Forget-Quality vs Model-Utility plane (paper Fig 5/6), if the eval was run.
    if (RESULTS / "forget_quality").is_dir():
        run([HERE / "plot_forget_quality.py"])

    print(f"\nAll figures -> {FIGS}/")


if __name__ == "__main__":
    main()
