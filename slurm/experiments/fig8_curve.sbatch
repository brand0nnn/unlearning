#!/bin/bash
#SBATCH --job-name=tofu_fig8
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/fig8_curve.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/fig8_curve.err

# TOFU Figure 8 — unlearning dynamics. Re-runs ONLY gradient_difference on the
# 5% forget set (full-FT, DeepSpeed), reusing the EXISTING learned model
# (tofu_learn_full_full) — no LEARN re-run, no retain reference needed. The
# --track-curve flag logs ROUGE/Prob/Truth-Ratio every few steps to
# results/unlearn_curve_gradient_difference_forget05.json; plot it afterwards with
# scripts/diagnostics/plot_unlearn_curve.py.
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

echo "=== Figure 8: gradient_difference on forget05 (full-FT, curve tracking) ==="
deepspeed --num_gpus=1 scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" \
    --strategy fullft --method gradient_difference \
    --forget-level forget05 --track-curve

echo "=== Figure 8 unlearning complete -> results/unlearn_curve_gradient_difference_forget05.json ==="
