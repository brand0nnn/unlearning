#!/bin/bash
#SBATCH --job-name=tofu_learn
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/learn.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/learn.err

# STEP 0 OF THE ONLY PIPELINE. Moved here from the retired strategy_comparison study,
# which used to own it -- every checkpoint the cross-lingual study starts from is produced
# by this file, so it belongs to the active study. English-only, and that is the constraint
# that shapes everything downstream: the facts were never taught in the other 9 languages,
# so all PROBING is English-only (CLAUDE.md sec 2).
#
# Step [2/2] (retain90) is the gold reference Forget Quality's KS test compares against.
# It is not needed for the recovery metric, only for 03_evaluate.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOMs
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/2] Fine-tune on full TOFU (model to be unlearned) ==="
deepspeed --num_gpus=1 shared/scripts/01_learn.py --data full

echo "=== [2/2] Fine-tune on retain90 (gold reference for Forget Quality) ==="
deepspeed --num_gpus=1 shared/scripts/01_learn.py --data retain90

echo "=== Learn phase complete. Checkpoints in experiments/ ==="
