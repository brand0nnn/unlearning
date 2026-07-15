#!/bin/bash
#SBATCH --job-name=loc_relearn
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/loc_relearn.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/loc_relearn.err

# LoRA-LOCALITY ablation, RELEARN stage. For each matched-unlearned location:
# measure the epoch-0 (unlearned) forget metrics, then run the SAME Full-FT
# relearning attack at epochs 1/2/4 and measure recovery (ROUGE + prob + truth)
# after each. Reuses the main relearn.py (DeepSpeed Full-FT) so the attack is
# identical across locations — only the unlearning LOCATION differed.
# Metrics -> lora_locality/out/<scheme>/recovery/ (NOT the main results/ folder).
#
#   Usage:  sbatch slurm/ablation/lora_locality_relearn.sbatch <scheme>
#   Requires the matching lora_locality_unlearn.sbatch <scheme> to have finished.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/loc_relearn"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

SCHEME="${1:-fixedbudget}"
echo "Node: $(hostname)  scheme=$SCHEME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
LAUNCH=0

for LOC in attn qkv qv mlp updown down all; do
    CKPT="$PROJECT_DIR/experiments/loc_${SCHEME}_${LOC}"
    if [ ! -d "$CKPT" ]; then
        echo "SKIP $LOC — no matched checkpoint at $CKPT"; continue
    fi
    echo "=== LOCALITY relearn [$SCHEME] location=$LOC ==="
    # epoch 0 = the matched unlearned checkpoint
    python lora_locality/measure.py --checkpoint "$CKPT" --scheme "$SCHEME" --location "$LOC" --epoch 0
    for EP in 1 2 4; do
        LAUNCH=$((LAUNCH + 1))
        deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + LAUNCH)) \
            scripts/recovery/relearn.py --checkpoint "$CKPT" --epochs $EP --relearn-data forget
        RELEARNED="$PROJECT_DIR/experiments/relearn_loc_${SCHEME}_${LOC}_ep${EP}"
        python lora_locality/measure.py --checkpoint "$RELEARNED" --scheme "$SCHEME" --location "$LOC" --epoch $EP
        rm -rf "$RELEARNED"     # reclaim the ~13GB relearned checkpoint
    done
done

echo "=== done -> lora_locality/out/${SCHEME}/recovery/*.json ==="
echo "    plot LOCALLY: python lora_locality/plot.py --scheme ${SCHEME}"
