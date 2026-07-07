#!/bin/bash
#SBATCH --job-name=tofu_unlearn_grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_grpo.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_grpo.err

# GRPO unlearning (strategy axis #4, HIGHEST RISK — RL can be unstable). Runs
# LoRA-GRPO by default (plain python, fits on one GPU, cheaper + more stable to
# get a first result). For full-parameter GRPO, drop --lora and launch with
# `deepspeed --num_gpus=1` instead.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"     # keep pip off the tiny $HOME quota (§6)
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# GRPO needs trl, but trl>=0.16 imports FSDPModule (needs torch>=2.6) and our
# cu121 torch is 2.5.1. Pin to the pre-FSDP2 window that keeps GRPOTrainer and
# runs on the existing torch. Idempotent + cached -> a no-op after the first run;
# nothing else imports trl so this can't affect other jobs.
pip install -q "trl>=0.14,<0.16"
python -c "import trl, torch; print(f'trl {trl.__version__} | torch {torch.__version__}')"

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"

echo "=== GRPO unlearning (LoRA) ==="
python scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy grpo --lora

echo "=== GRPO unlearn complete ==="
