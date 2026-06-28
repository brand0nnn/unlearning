#!/bin/bash
#SBATCH --job-name=tofu_evaluate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/evaluate_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/evaluate_%j.err

PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Downloading missing TOFU splits ==="
python3 - << 'PYEOF'
from datasets import load_dataset
import os
cache = os.environ["HF_DATASETS_CACHE"]
for config in ["forget10_perturbed", "retain_perturbed",
               "real_authors_perturbed", "world_facts_perturbed"]:
    print(f"  Downloading: {config}")
    load_dataset("locuslab/TOFU", config, split="train", cache_dir=cache)
print("  Done.")
PYEOF

REFERENCE="$PROJECT_DIR/experiments/tofu_learn_retain90_full"

echo "=== Evaluating all checkpoints ==="
python scripts/tofu_03_evaluate.py \
    --reference "$REFERENCE" \
    --checkpoints \
        "$PROJECT_DIR/experiments/tofu_unlearn_gradient_ascent_forget10" \
        "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10" \
        "$PROJECT_DIR/experiments/tofu_unlearn_kl_minimization_forget10" \
        "$PROJECT_DIR/experiments/tofu_unlearn_idk_forget10"

echo "=== Evaluate phase complete ==="
echo "Results saved to results/"
