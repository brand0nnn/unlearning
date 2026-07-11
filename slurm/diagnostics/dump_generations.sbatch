#!/bin/bash
#SBATCH --job-name=dump_gens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/dump_generations.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/dump_generations.err

# Dump the ACTUAL generations of each strategy on sample TOFU questions, side by
# side (qualitative validation — see gibberish vs coherent-wrong vs IDK vs correct).
# Loads one checkpoint at a time. The LEARNED base is included as the "before" column.
# Writes results/generations/generations.{json,md} — read the .md.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/diagnostics/dump_generations.py \
    --n 5 \
    --checkpoints \
        "$PROJECT_DIR/experiments/tofu_learn_full_full" \
        "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10" \
        "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10_lora" \
        "$PROJECT_DIR/experiments/tofu_unlearn_self_distill_forget10_self_distill" \
        "$PROJECT_DIR/experiments/tofu_unlearn_grpo_forget10_grpo"

echo "=== Generations dumped -> results/generations/generations.md ==="
