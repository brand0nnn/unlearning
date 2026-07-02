#!/bin/bash
#SBATCH --job-name=tofu_rouge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/rouge_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/rouge_%j.err

# Standalone ROUGE sanity check (TOFU Table 2: base ~0.36 -> finetuned ~0.98) on the
# EXISTING learned checkpoint — no re-training. Uses the same env redirects as the
# other stages so the dataset lock does not land on the tiny $HOME quota partition.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
source ~/.env_hf

echo "Node: $(hostname)"
python scripts/tofu_check_rouge.py --n 50
echo "=== ROUGE check complete ==="
