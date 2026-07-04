#!/bin/bash
#SBATCH --job-name=tofu_unlearn_sd
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_selfdistill.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_selfdistill.err

# Self-distillation unlearning (strategy axis #3). Full-parameter -> DeepSpeed
# fp32 master, like Full-FT. A frozen teacher (the learned model) is loaded
# alongside the student, so this needs more memory than plain Full-FT.
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

echo "=== Self-distillation unlearning ==="
deepspeed --num_gpus=1 scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy self_distill

echo "=== Self-distillation unlearn complete ==="
