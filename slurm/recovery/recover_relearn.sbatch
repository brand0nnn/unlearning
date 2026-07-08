#!/bin/bash
#SBATCH --job-name=recover_relearn
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/relearn.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/relearn.err

# Relearning-robustness probe: how fast does the forgotten knowledge return when we
# re-fine-tune the unlearned model on the forget set? Fast recovery => suppression,
# not deletion. Traces forget-set ROUGE vs relearn epochs.
#
# Point UNLEARNED at the checkpoint whose robustness you want to test — run it once
# per strategy (Full-FT / LoRA / self-distill / GRPO). The forget-ROUGE JSON MERGES,
# so each run just adds that strategy's point to results/relearn/relearn_forget_rouge.json.
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

# Pass the unlearned checkpoint as the first sbatch arg; defaults to the NEW GRPO one
# (Full-FT / LoRA / self-distill are already in the merged JSON).
#   sbatch slurm/recovery/recover_relearn.sbatch experiments/tofu_unlearn_gradient_difference_forget10_lora
UNLEARNED="${1:-$PROJECT_DIR/experiments/tofu_unlearn_grpo_forget10_grpo}"

echo "=== Forget-set ROUGE BEFORE relearning (baseline) ==="
python scripts/recovery/relearn_measure.py --checkpoints "$UNLEARNED"

for EP in 1 2 4; do
    echo "=== Relearn $EP epoch(s) on the forget set ==="
    deepspeed --num_gpus=1 scripts/recovery/relearn.py --checkpoint "$UNLEARNED" --epochs $EP
    RELEARNED="$PROJECT_DIR/experiments/relearn_$(basename $UNLEARNED)_ep${EP}"
    python scripts/recovery/relearn_measure.py --checkpoints "$RELEARNED"
    # We only need the forget-ROUGE number, not the 13.5GB checkpoint -> delete it
    # so the probe doesn't pile up disk (each epoch's checkpoint is independent).
    rm -rf "$RELEARNED"
done

echo "=== Relearn probe complete. Recovery curve in results/relearn/relearn_forget_rouge.json ==="
