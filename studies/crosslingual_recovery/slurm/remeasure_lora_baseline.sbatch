#!/bin/bash
#SBATCH --job-name=remeasure_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/remeasure_lora_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/remeasure_lora_%j.err

# Recreate the deep LoRA baseline JSON that went missing from the crosslingual_deep
# group (the checkpoint itself is intact). Inference-only, fits a100-40, ~5 min.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python shared/scripts/relearn_measure.py \
    --checkpoints "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b" \
    --group crosslingual_deep --n 40 --forget-level forget01 --fact-metrics

echo "=== done -> crosslingual_deep/ should now have the lora_uep32 baseline ==="
