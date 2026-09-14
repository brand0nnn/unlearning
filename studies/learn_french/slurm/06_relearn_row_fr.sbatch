#!/bin/bash
#SBATCH --job-name=relearn_fr
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/relearn_row_fr_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/relearn_row_fr_%j.err

# STAGE 3, the "fr" ROW: relearn the fr-unlearned model in all five languages,
# sequentially, in one job.
#
#     sbatch studies/learn_french/slurm/06_relearn_row_fr.sbatch
#     EPOCHS=4 sbatch --export=ALL,EPOCHS=4 studies/learn_french/slurm/06_relearn_row_fr.sbatch
#
# All five rows can be submitted at once -- they are five jobs, each holding at most ONE
# cell's snapshots (~46GB) because cells run sequentially inside a job. Five rows peak at
# ~230GB against ~305GB free.
#
# ~16h for the row (5 cells x ~3.2h). CHECK THE PARTITION CAP FIRST:
#     sinfo -p gpu-long -o '%P %l'
# If it is below 16h the job is killed mid-cell, which is survivable: cells already scored
# are skipped, so resubmitting the same command finishes the rest. To sidestep the wall
# entirely, submit cells individually instead:
#     sbatch studies/learn_french/slurm/05_relearn_fr.sbatch fr ja
#
# Everything else -- the pre-registered depth, the no-weights-kept policy, the two probe
# phrasings, the LR-schedule caveat -- is documented in 05_relearn_fr.sbatch, which this
# runs. The #SBATCH header above is the only thing this file adds.
set -euo pipefail
exec bash /home/b/brandonk/unlearning/studies/learn_french/slurm/05_relearn_fr.sbatch fr
