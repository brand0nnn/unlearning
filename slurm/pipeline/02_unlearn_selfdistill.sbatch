#!/bin/bash
#SBATCH --job-name=tofu_unlearn_sd
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_selfdistill.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_selfdistill.err

# Self-distillation unlearning (strategy axis #3). Full-parameter -> DeepSpeed
# fp32 master, like Full-FT. A frozen teacher (the learned model) is loaded
# alongside the student, so this needs more memory than plain Full-FT.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
# Per-job scratch so concurrent DeepSpeed unlearn jobs don't share/clobber .tmp.
export TMPDIR="$PROJECT_DIR/.tmp/unlearn_sd"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

# Unique DeepSpeed rendezvous port (below the ephemeral range) derived from the job
# id, so running this concurrently with Full-FT/GRPO never clashes on the default
# port 29500 (EADDRINUSE).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "Node: $(hostname)  master_port=$MASTER_PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"

echo "=== Self-distillation unlearning (+curve tracking) ==="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy self_distill --track-curve

echo "=== Self-distillation unlearn complete ==="
