#!/bin/bash
# Stage 10 Phase 2A, step 2 of 2: after 26_unlearn_frja_deep.sbatch has finished, submit one
# 23_storage_followup.sbatch job per SAVED srcfrja2 level (symlinked levels share weights
# with a saved one and are not re-scored), plus the `ref` job.
#     bash studies/learn_french/slurm/submit_phase2a_scoring.sh
set -euo pipefail
cd /home/b/brandonk/unlearning
D=experiments/tr_levels
P="tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b_ulfr_floornone_srcfrja2_"
n=0
for d in "$D/$P"tr*; do
    [ -e "$d" ] || { echo "no srcfrja2 levels in $D -- has job 26 finished?" >&2; exit 1; }
    lv="${d##*_}"
    if [ -L "$d" ]; then echo "$lv: symlink -> $(readlink "$d"), not re-scored"; continue; fi
    [ -f "$d/tokenizer_config.json" ] || { echo "$lv: no tokenizer_config.json, skipped" >&2; continue; }
    echo -n "$lv: "; sbatch studies/learn_french/slurm/23_storage_followup.sbatch "$lv" srcfrja2
    n=$((n + 1))
done
echo -n "ref: "; sbatch studies/learn_french/slurm/23_storage_followup.sbatch ref
echo "$n level job(s) + ref submitted"
