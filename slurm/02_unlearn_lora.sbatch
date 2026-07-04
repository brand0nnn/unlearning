#!/bin/bash
#SBATCH --job-name=tofu_unlearn_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_lora_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_lora_%j.err

# LoRA unlearning (the LoRA-vs-Full-FT strategy axis). Fixed method =
# gradient_difference (the representative method for the strategy comparison).
# Plain `python` (no deepspeed launcher): LoRA trains tiny adapters, fits easily.
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

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"

for METHOD in gradient_difference; do
    echo "=== LoRA unlearning: $METHOD ==="
    python scripts/tofu_02_unlearn.py --checkpoint "$LEARNED" --method "$METHOD" --lora
done

echo "=== LoRA unlearn complete ==="
