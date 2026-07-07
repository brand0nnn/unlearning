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
# GRPO runs in its OWN venv (torch>=2.6 + modern trl) — NOT the main .venv, whose
# cu121 torch 2.5.1 can't run any trl. Build it once: bash setup_grpo.sh
if [ ! -d "$PROJECT_DIR/.venv-grpo" ]; then
  echo "ERROR: .venv-grpo missing — run 'bash setup_grpo.sh' on the login node first." >&2
  exit 1
fi
source "$PROJECT_DIR/.venv-grpo/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

# Full nvidia-smi shows the driver's CUDA version; the python line confirms the
# cu124 torch build actually sees the GPU (first-run proof that cu124 fits the driver).
echo "=== Node: $(hostname) ==="; nvidia-smi
python -c "import torch, trl; print(f'torch {torch.__version__} | cuda {torch.version.cuda} | avail {torch.cuda.is_available()} | trl {trl.__version__}')"

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"

echo "=== GRPO unlearning (LoRA) ==="
python scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy grpo --lora

echo "=== GRPO unlearn complete ==="
