# crosslingual_recovery  *(active)*

Unlearn an English forget01 fact, benign-relearn on the RETAIN set in each of 10
languages (ordered by distance from English), and measure how much the ENGLISH fact
recovers — does the fine-tuning METHOD change the recovery-vs-distance pattern?
The pilot auto-matches LoRA's forget depth to Full-FT's to de-confound the comparison.

**Run:** `sbatch slurm/crosslingual_pilot.sbatch`  (self-contained; forces --forget-level forget01)
**Plot (local):** `python plots/plot_crosslingual_pilot.py` → `figures/crosslingual_pilot_recovery.png`
  (left = raw recovery vs distance; right = baseline-normalized decay shape)
`results/`: relearn/crosslingual_pilot/ (+ crosslingual_pilot_probe/ = the LoRA baseline-match candidates)
