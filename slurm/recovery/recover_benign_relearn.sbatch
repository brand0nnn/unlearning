#!/bin/bash
#SBATCH --job-name=benign_relearn
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/benign_relearn.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/benign_relearn.err

# BENIGN relearning (Hu et al. §4): fine-tune each unlearned model on data that
# contains NONE of the forgotten authors' facts, then measure whether forget-set
# ROUGE returns anyway. If it does => the knowledge was never deleted, only hidden
# (unambiguous "obfuscation, not deletion"). Compares that susceptibility ACROSS
# unlearning strategies. Writes to a SEPARATE JSON so the direct forget-set
# relearning results are untouched.
#
#   sbatch slurm/recover_benign_relearn.sbatch            # default benign = retain
#   sbatch slurm/recover_benign_relearn.sbatch world_facts
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
BENIGN="${1:-retain}"                       # retain | world_facts | real_authors
OUT="relearn_benign_${BENIGN}_rouge.json"   # separate from relearn_forget_rouge.json
# Per-source scratch dir so the retain and world_facts jobs can run CONCURRENTLY
# without clobbering each other's TMPDIR. HF caches stay shared (read-only) and
# DeepSpeed's op cache is already built, so concurrent jobs are safe.
export TMPDIR="$PROJECT_DIR/.tmp/benign_${BENIGN}"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Unique DeepSpeed rendezvous port per launch (below the ephemeral range) so the
# two concurrent benign jobs — and their 9 sequential relearn launches each —
# never clash on the default port 29500 (EADDRINUSE).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
LAUNCH=0

# The four training strategies. Benign relearn RE-TRAINS each checkpoint (3 epochs ×
# expensive), and the output JSON MERGES — so by default we probe ONLY the new GRPO
# point; the other three are already in results/relearn/relearn_benign_retain_rouge.json.
# Uncomment them to recompute the full set.
CKPTS=(
    "$PROJECT_DIR/experiments/tofu_unlearn_grpo_forget10_grpo"                    # NEW (full-param GRPO)
    # "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10"        # done (Full-FT)
    # "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10_lora"   # done (LoRA)
    # "$PROJECT_DIR/experiments/tofu_unlearn_self_distill_forget10_self_distill"  # done (Self-Distill)
)

for UNLEARNED in "${CKPTS[@]}"; do
    echo "=== BENIGN relearn ($BENIGN) probe: $(basename $UNLEARNED) ==="
    # epoch-0 baseline: forget-ROUGE of the unlearned model, before any relearning
    python scripts/recovery/relearn_measure.py --checkpoints "$UNLEARNED" --out "$OUT"
    for EP in 1 2 4; do
        LAUNCH=$((LAUNCH + 1))
        deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + LAUNCH)) scripts/recovery/relearn.py \
            --checkpoint "$UNLEARNED" --epochs $EP --relearn-data "$BENIGN"
        RELEARNED="$PROJECT_DIR/experiments/relearn_$(basename $UNLEARNED)_via_${BENIGN}_ep${EP}"
        python scripts/recovery/relearn_measure.py --checkpoints "$RELEARNED" --out "$OUT"
        rm -rf "$RELEARNED"        # keep only the forget-ROUGE number, not the ckpt
    done
done

echo "=== Benign-relearn probe complete -> results/relearn/$OUT ==="
echo "Plot with: python scripts/diagnostics/plot_relearn.py --data results/relearn/$OUT \\"
echo "   --out benign_relearn_${BENIGN}_curve.png --xlabel 'Relearning epochs on ${BENIGN} (benign)' \\"
echo "   --title 'Benign relearning on ${BENIGN}: does unrelated fine-tuning jog forgotten knowledge?'"
