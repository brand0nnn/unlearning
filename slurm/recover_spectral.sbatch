#!/bin/bash
#SBATCH --job-name=tofu_spectral
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/spectral.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/spectral.err

# Recovery axis 3 — spectral trace detectability (Chen et al. ICLR 2026).
# INFERENCE ONLY: collects activations on forget-irrelevant prompts and fits a
# tiny logistic classifier + SVD. No training, no DeepSpeed -> plain python.
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

# ORIGINAL = the learned model BEFORE unlearning (the reference the fingerprint
# is measured against). Fingerprint every unlearned checkpoint we have.
ORIGINAL="$PROJECT_DIR/experiments/tofu_learn_full_full"
FORGET=forget10

python scripts/recovery/spectral.py \
    --original "$ORIGINAL" \
    --checkpoints \
        experiments/tofu_unlearn_gradient_ascent_${FORGET}_fullft \
        experiments/tofu_unlearn_gradient_difference_${FORGET}_fullft \
        experiments/tofu_unlearn_kl_minimization_${FORGET}_fullft \
        experiments/tofu_unlearn_idk_${FORGET}_fullft \
        experiments/tofu_unlearn_gradient_difference_${FORGET}_lora

echo "=== Spectral phase complete ==="
