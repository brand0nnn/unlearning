#!/bin/bash
#SBATCH --job-name=loc_unlearn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=07:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/loc_unlearn.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/loc_unlearn.err

# LoRA-LOCALITY ablation, UNLEARN stage. Unlearn each module-group location with
# LoRA and EARLY-STOP at a matched forget level, so every location starts the
# relearning probe from the same amount of forgetting. LoRA = plain python (no
# DeepSpeed). Self-contained experiment: metrics land in lora_locality/out/, NOT
# the main results/ folder.
#
#   Usage:  sbatch slurm/ablation/lora_locality_unlearn.sbatch <scheme> <target>
#     <scheme> = samerank | fixedbudget            (default fixedbudget)
#     <target> = forget ROUGE to stop at; use 0 to CALIBRATE (train full, read
#                each location's floor from the metrics record)  (default 0.4)
#     <locs>   = optional space-separated subset of locations (default: all 7).
#                Split into a few small jobs for faster backfill under low fairshare,
#                e.g.  sbatch ... fixedbudget 0.4 "attn mlp down"
#   Run twice: once per scheme, to disentangle location from capacity.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/loc_unlearn"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

SCHEME="${1:-fixedbudget}"
TARGET="${2:-0.4}"
LOCS="${3:-attn qkv qv mlp updown down all}"
echo "Node: $(hostname)  scheme=$SCHEME  target=$TARGET  locations=$LOCS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"

for LOC in $LOCS; do
    echo "=== LOCALITY unlearn [$SCHEME] location=$LOC (target $TARGET) ==="
    python lora_locality/unlearn.py \
        --learned "$LEARNED" --scheme "$SCHEME" --location "$LOC" --target "$TARGET"
done

echo "=== done -> experiments/loc_${SCHEME}_{attn,qkv,qv,mlp,updown,down,all} ==="
echo "    metrics -> lora_locality/out/${SCHEME}/unlearn/*.json"
echo "    next: sbatch slurm/ablation/lora_locality_relearn.sbatch ${SCHEME}"
