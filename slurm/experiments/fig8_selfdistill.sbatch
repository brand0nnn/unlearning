#!/bin/bash
#SBATCH --job-name=tofu_fig8_sd
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/fig8_selfdistill.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/fig8_selfdistill.err

# Fill the MISSING self-distillation forget10 curve — its slot in fig8_forget10
# died on a DeepSpeed port clash (EADDRINUSE on 29500). Full-FT + LoRA already
# succeeded, so we only re-run this one. Uses a UNIQUE master port derived from
# the SLURM job id so it can't collide with any other DeepSpeed job on the node.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/fig8_sd"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

# Unique rendezvous port (below the ephemeral range) -> no EADDRINUSE clashes.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "Node: $(hostname)  master_port=$MASTER_PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"
echo "=== Self-distillation forget10 (curve) ==="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy self_distill \
    --forget-level forget10 --track-curve

echo "=== done -> results/unlearn_curve_tofu_unlearn_self_distill_forget10_self_distill.json ==="
