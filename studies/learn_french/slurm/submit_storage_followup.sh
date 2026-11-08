#!/bin/bash
# Stage 10 Phase 1: submit the six independent jobs of 23_storage_followup.sbatch.
#     bash studies/learn_french/slurm/submit_storage_followup.sh
set -euo pipefail
cd /home/b/brandonk/unlearning
for t in ft floor tr0p650 tr0p694 tr0p783 mix; do
    echo -n "$t: "; sbatch studies/learn_french/slurm/23_storage_followup.sbatch "$t"
done
