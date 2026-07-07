#!/bin/bash
#SBATCH --job-name=lora_abl_relearn
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/lora_ablation_relearn.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/lora_ablation_relearn.err

# Relearning-robustness probe for the LoRA target-module ablation. Full-FT relearn
# (the fixed attack) on each ablation checkpoint, measuring how much forget-set
# knowledge returns. HYPOTHESIS: MLP-LoRA deleted more deeply -> recovers LESS
# than attention-LoRA (closer to Full-FT). Writes a SEPARATE JSON so the strategy
# recovery curve is untouched.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/lora_abl_relearn"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
LAUNCH=0
OUT="relearn_lora_ablation_rouge.json"

# The 4 ablation checkpoints + Full-FT as the "true deletion" reference.
CKPTS=(
    "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10_lora_attn"
    "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10_lora_updown"
    "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10_lora_mlp"
    "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10_lora_all"
    "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10"          # Full-FT ref
)

for UNLEARNED in "${CKPTS[@]}"; do
    echo "=== relearn probe: $(basename $UNLEARNED) ==="
    python scripts/recovery/relearn_measure.py --checkpoints "$UNLEARNED" --out "$OUT"
    for EP in 1 2 4; do
        LAUNCH=$((LAUNCH + 1))
        deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + LAUNCH)) scripts/recovery/relearn.py \
            --checkpoint "$UNLEARNED" --epochs $EP
        RELEARNED="$PROJECT_DIR/experiments/relearn_$(basename $UNLEARNED)_ep${EP}"
        python scripts/recovery/relearn_measure.py --checkpoints "$RELEARNED" --out "$OUT"
        rm -rf "$RELEARNED"
    done
done

echo "=== done -> results/relearn/$OUT ==="
echo "Plot: python scripts/diagnostics/plot_relearn.py --data results/relearn/$OUT --label-by lora_target \\"
echo "   --out lora_ablation_recovery.png --title 'LoRA target-module ablation: recovery after unlearning'"
