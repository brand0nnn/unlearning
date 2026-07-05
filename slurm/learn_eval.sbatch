#!/bin/bash
#SBATCH --job-name=tofu_learn_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/learn_eval.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/learn_eval.err

# Validate the LEARN phase: evaluate the LEARNED model and the BASE model on the
# TOFU splits → show the fictitious authors (forget+retain) are memorized (high
# ROUGE + probability) after full-FT and ~unknown before. Sources HF token
# because the base model is the gated Llama repo.
#   For a quick check add:  --limit 100   (subset; full eval is slow)
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

python scripts/diagnostics/eval_learning.py --models \
    experiments/tofu_learn_full_full \
    meta-llama/Llama-2-7b-chat-hf

echo "=== Learning-check eval complete -> results/learn_eval_*.json ==="
