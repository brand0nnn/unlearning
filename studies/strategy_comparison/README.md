# strategy_comparison  *(foundation, complete)*

Compare four unlearning strategies on English TOFU (forget10) + recovery axes.
Full-FT · LoRA · Self-Distillation · GRPO, all `gradient_difference`-family.

**Run (cluster):**
```
sbatch slurm/01_learn.sbatch
sbatch slurm/02_unlearn.sbatch     # + 02_unlearn_lora / _selfdistill / _grpo  (add --track-curve for curves)
sbatch slurm/03_evaluate.sbatch    # Model Utility + Forget Quality (Fig 5/6 plane)
sbatch slurm/relearn_forget.sbatch · slurm/relearn_retain.sbatch · slurm/recover_spectral.sbatch
```
**Plot (local):** `python plots/plot_all.py` → `figures/` (regenerates every figure from `results/`).

`results/`: curves/ · forget_quality/ · relearn/{forget,retain}/ · spectral/
