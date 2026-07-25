# lora_target_ablation

Does LoRA on MLP vs attention modules delete knowledge more deeply? Unlearn with LoRA
on different module groups, then run the fixed Full-FT relearning attack.

**Run:** `sbatch slurm/lora_target_ablation.sbatch` then `sbatch slurm/lora_ablation_relearn.sbatch`
**Plot (local):**
```
python shared/plots/plot_relearn.py --data studies/lora_target_ablation/results/relearn/lora_ablation \
  --fig-dir studies/lora_target_ablation/figures --label-by lora_target \
  --out lora_ablation_recovery.png --title "LoRA target-module ablation: recovery after unlearning"
```
`results/`: relearn/lora_ablation/
