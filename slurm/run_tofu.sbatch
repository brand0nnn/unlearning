#!/bin/bash
# ===========================================================================
# TOFU full pipeline — submit with:  sbatch slurm/run_tofu.sbatch
# Prerequisites: run setup.sh once on the login node before submitting this.
#
# Useful commands:
#   squeue -u $USER               check job status
#   tail -f logs/tofu_<id>.out    watch live output
#   scancel <job_id>              cancel the job
# ===========================================================================
#SBATCH --job-name=tofu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/tofu_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/tofu_%j.err

set -e
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs

echo "======================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Project    : $PROJECT_DIR"
echo "Started at : $(date)"
echo "======================================================"

# --- Environment -----------------------------------------------------------
source "$PROJECT_DIR/.venv/bin/activate"

# Point HF at project storage so model download goes to the big filesystem
export TMPDIR="$PROJECT_DIR/.tmp"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE"
# Note: HF_HUB_OFFLINE is NOT set here — the model downloads on first run,
# then is cached for all subsequent runs.

echo "Python     : $(which python3)"
echo "GPU        :"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Pipeline --------------------------------------------------------------
echo ""
echo "=== [1/4] LEARN: fine-tune on full TOFU ==="
python scripts/tofu_01_finetune.py --data full

echo ""
echo "=== [1/4] LEARN: fine-tune on retain90 (gold reference) ==="
python scripts/tofu_01_finetune.py --data retain90

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"
REFERENCE="$PROJECT_DIR/experiments/tofu_learn_retain90_full"

echo ""
echo "=== [2/4] UNLEARN: applying all four methods ==="
for METHOD in gradient_ascent gradient_difference kl_minimization idk; do
    echo "  -> $METHOD"
    python scripts/tofu_02_unlearn.py \
        --checkpoint "$LEARNED" \
        --method "$METHOD"
done

echo ""
echo "=== [3/4] EVALUATE: all checkpoints vs gold reference ==="
python scripts/tofu_03_evaluate.py \
    --reference "$REFERENCE" \
    --checkpoints \
        "$PROJECT_DIR/experiments/tofu_unlearn_gradient_ascent_forget10" \
        "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10" \
        "$PROJECT_DIR/experiments/tofu_unlearn_kl_minimization_forget10" \
        "$PROJECT_DIR/experiments/tofu_unlearn_idk_forget10"

echo ""
echo "=== [4/4] PLOT ==="
python scripts/tofu_04_plot.py

echo ""
echo "======================================================"
echo "DONE at $(date)"
echo "Results are in: $PROJECT_DIR/results/"
echo "======================================================"
