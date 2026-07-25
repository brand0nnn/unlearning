# lora_rank_ablation

LoRA rank sweep (r=8/16/32/64) on forget10 — does more adapter capacity mean more
forgetting, and how much retain collateral does it cost?

**Run:** `sbatch slurm/lora_rank_ablation.sbatch`  (track-curve on; one run per rank)
**Plot (local):** `python plots/plot_rank_sweep.py` → `figures/lora_rank_forgetting.png`
`results/`: curves/ (one unlearn_curve per rank)
