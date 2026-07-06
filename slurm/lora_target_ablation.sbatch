#!/bin/bash
#SBATCH --job-name=lora_target_abl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/lora_target_ablation.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/lora_target_ablation.err

# LoRA TARGET-MODULE ablation. Apply LoRA gradient_difference unlearning (forget10)
# to different module groups to test the hypothesis that MLP layers store the
# factual knowledge (Geva et al.; ROME/MEMIT) — so LoRA on MLP should delete more
# deeply (LESS recoverable) than attention-only LoRA. LoRA = plain python, no
# DeepSpeed. Produces one merged checkpoint per target group.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/lora_abl"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"

#   attn   = q/k/v/o  (attention, the current LoRA default)
#   updown = up/down  (MLP up+down — the "MLP up/down" condition)
#   mlp    = gate/up/down  (full MLP)
#   all    = attention + MLP
for TGT in attn updown mlp all; do
    echo "=== LoRA gradient_difference forget10 -- target=$TGT ==="
    python scripts/pipeline/02_unlearn.py \
        --checkpoint "$LEARNED" --strategy lora --method gradient_difference \
        --forget-level forget10 --lora-target "$TGT"
done

echo "=== done -> experiments/tofu_unlearn_gradient_difference_forget10_lora_{attn,updown,mlp,all} ==="
