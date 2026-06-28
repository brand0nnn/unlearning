#!/bin/bash
# ===========================================================================
# Full TOFU pipeline as a single SLURM job.
# Submit with:   sbatch slurm/run_tofu.sbatch
# Check status:  squeue -u $USER
# Live log:      tail -f logs/tofu_<jobid>.out
#
# >>> EDIT THE LINES MARKED [EDIT] for your cluster <<<
# Run `sinfo` to see partition names and `sinfo -o "%P %G"` for GPU gres syntax.
# ===========================================================================
#SBATCH --job-name=tofu
#SBATCH --partition=gpu              # [EDIT] your GPU partition name
#SBATCH --gres=gpu:1                 # [EDIT] 1 GPU; syntax may be gpu:a100:1 etc.
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00              # [EDIT] walltime; raise for full-size runs
#SBATCH --output=logs/tofu_%j.out    # %j = job id
#SBATCH --error=logs/tofu_%j.err

set -e                               # stop on first error
mkdir -p logs

# --- Environment -----------------------------------------------------------
# [EDIT] Some clusters need `module load` lines, e.g.:
# module load cuda/12.1
# module load anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate unlearning            # the env from `conda create -n unlearning`

# --- Offline Hugging Face cache -------------------------------------------
# Compute nodes often have NO internet. Point HF at a pre-downloaded cache on
# shared storage. Run scripts/tofu_01_finetune.py ONCE on a login node first so
# the TOFU data + base model land in this cache, then jobs run fully offline.
export HF_HOME="$HOME/scratch/hf_cache"     # [EDIT] a path on shared/scratch storage
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# export HF_HUB_OFFLINE=1            # uncomment once the cache is warm

echo "Node: $(hostname) | GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Pipeline --------------------------------------------------------------
# 1. LEARN: the model that knows all authors, and the gold retain reference.
python scripts/tofu_01_finetune.py --data full
python scripts/tofu_01_finetune.py --data retain90

LEARNED=experiments/tofu_learn_full_full
REFERENCE=experiments/tofu_learn_retain90_full

# 2. UNLEARN: apply each baseline to the learned model.
for METHOD in gradient_ascent gradient_difference kl_minimization idk; do
    python scripts/tofu_02_unlearn.py --checkpoint $LEARNED --method $METHOD
done

# 3. EVALUATE all unlearned checkpoints against the gold reference.
python scripts/tofu_03_evaluate.py \
    --reference $REFERENCE \
    --checkpoints experiments/tofu_unlearn_gradient_ascent_forget10 \
                  experiments/tofu_unlearn_gradient_difference_forget10 \
                  experiments/tofu_unlearn_kl_minimization_forget10 \
                  experiments/tofu_unlearn_idk_forget10

# 4. PLOT.
python scripts/tofu_04_plot.py

echo "DONE. See results/ for JSON metrics and the two figures."
