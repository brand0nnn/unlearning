#!/bin/bash
#SBATCH --job-name=tofu_learn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/learn_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/learn_%j.err

PROJECT_DIR="/home/b/brandonk/unlearning"
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
deepspeed --num_gpus=2 scripts/tofu_01_finetune.py --data full

echo "=== [2/2] Fine-tune on retain90 (gold reference for Forget Quality) ==="
deepspeed --num_gpus=2 scripts/tofu_01_finetune.py --data retain90

echo "=== Learn phase complete. Checkpoints in experiments/ ==="
