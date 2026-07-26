#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/generate_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/generate_%j.err

# Feed prompts to a checkpoint and print the generations (compute node — torch can't
# import on login). Usage:
#   sbatch shared/diagnostics/generate.sbatch <checkpoint> "prompt 1" "prompt 2" ...
# e.g.
#   sbatch shared/diagnostics/generate.sbatch \
#     experiments/tofu_learn_full_full_qwen3-8b "Who is the author Hsiao Yun-Hwa?"
# Output lands in logs/generate_<jobid>.out
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
source ~/.env_hf 2>/dev/null || true

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CKPT="$1"; shift
python shared/diagnostics/generate.py --checkpoint "$CKPT" --prompt "$@"
