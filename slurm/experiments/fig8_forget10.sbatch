#!/bin/bash
#SBATCH --job-name=tofu_fig8_f10
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/fig8_forget10.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/fig8_forget10.err

# Unlearning-dynamics curves (TOFU Fig-8 style) on forget10 for the 3-strategy
# comparison: Full-FT GD, LoRA GD, Self-Distillation. Reuses the existing curve
# callback (build_curve_callbacks) via --track-curve; each strategy's curve JSON
# is named by run_name so they don't overwrite each other. Reuses the EXISTING
# learned model — no LEARN re-run. GRPO is intentionally excluded (blocked).
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

# Unique DeepSpeed rendezvous ports (below the ephemeral range) so sequential
# launches — or another DeepSpeed job on the node — never clash on 29500.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "Node: $(hostname)  master_port base=$MASTER_PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"

echo "=== [1/3] Full-FT gradient_difference forget10 (curve) ==="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy fullft --method gradient_difference \
    --forget-level forget10 --track-curve

echo "=== [2/3] LoRA gradient_difference forget10 (curve) ==="
python scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy lora --method gradient_difference \
    --forget-level forget10 --track-curve

echo "=== [3/3] Self-distillation forget10 (curve) ==="
deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + 1)) scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy self_distill \
    --forget-level forget10 --track-curve

echo "=== done -> results/unlearn_curve_*_forget10_*.json (plot with plot_unlearn_curve.py) ==="
