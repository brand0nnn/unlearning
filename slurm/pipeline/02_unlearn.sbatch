#!/bin/bash
#SBATCH --job-name=tofu_unlearn
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn.err

PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
# Per-job scratch so concurrent DeepSpeed unlearn jobs don't share/clobber .tmp.
export TMPDIR="$PROJECT_DIR/.tmp/unlearn_fullft"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOMs
source ~/.env_hf   # HF_TOKEN — the tokenizer is loaded from the gated base repo

# Unique DeepSpeed rendezvous port base (below the ephemeral range) from the job id,
# +LAUNCH per method, so neither the sequential launches here nor a concurrent
# Full-FT/self-distill/GRPO job ever clash on the default port 29500 (EADDRINUSE).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
LAUNCH=0
echo "Node: $(hostname)  master_port base=$MASTER_PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"

# Full-FT strategy point = gradient_difference only (the loss used for the 4-strategy
# comparison). To regenerate the loss-curve figure across all four loss methods,
# restore:  for METHOD in gradient_ascent gradient_difference kl_minimization idk
echo "=== Unlearning: gradient_difference (Full-FT, +curve tracking) ==="
deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + 1)) \
    scripts/pipeline/02_unlearn.py --checkpoint "$LEARNED" \
    --method gradient_difference --track-curve

echo "=== Unlearn phase complete ==="
