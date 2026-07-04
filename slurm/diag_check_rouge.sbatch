#!/bin/bash
#SBATCH --job-name=tofu_rouge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/rouge.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/rouge.err

# Standalone ROUGE check (TOFU Table 2: base ~0.36 -> finetuned ~0.98) on the
# EXISTING learned checkpoint — no re-training. Run this AFTER stage 01 and BEFORE
# stage 02 to confirm the model actually memorized. The env exports below keep the
# dataset lock off the tiny $HOME quota partition (the reason a bare srun failed).
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
source ~/.env_hf

echo "Node: $(hostname)"
python scripts/diagnostics/check_rouge.py --n 50
echo "=== ROUGE check complete ==="
