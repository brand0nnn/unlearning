#!/bin/bash
#SBATCH --job-name=tofu_unlearn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_%j.err

PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"

for METHOD in gradient_ascent gradient_difference kl_minimization idk; do
    echo "=== Unlearning: $METHOD ==="
    python scripts/tofu_02_unlearn.py --checkpoint "$LEARNED" --method "$METHOD"
done

echo "=== Unlearn phase complete ==="
