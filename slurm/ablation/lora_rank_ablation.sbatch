#!/bin/bash
#SBATCH --job-name=lora_rank_abl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/lora_rank_ablation.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/lora_rank_ablation.err

# LoRA RANK ablation. Same gradient_difference unlearning (forget10) on the DEFAULT
# attention target, sweeping the adapter rank to test whether more CAPACITY -> more
# forgetting (and how much retain collateral it costs). alpha is auto-scaled to 2r
# inside 02_unlearn.py so the alpha/r ratio (update magnitude) is held fixed -> the
# only thing that changes is capacity. LoRA = plain python, no DeepSpeed. Each rank
# tracks its Fig-8 curve so you can read forgetting-vs-rank directly.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/lora_rank_abl"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"

# r=16 reproduces the current LoRA (alpha=32) as a same-pipeline baseline; 8/32/64
# bracket it below and above to show the capacity trend.
for R in 8 16 32 64; do
    echo "=== LoRA gradient_difference forget10 -- rank=$R (alpha=$((2*R))) ==="
    python scripts/pipeline/02_unlearn.py \
        --checkpoint "$LEARNED" --strategy lora --method gradient_difference \
        --forget-level forget10 --lora-r "$R" --track-curve
done

echo "=== done -> experiments/tofu_unlearn_gradient_difference_forget10_lora_r{8,16,32,64} ==="
echo "Curves -> results/curves/*_lora_r*.json ; then run 03_evaluate for the plane."
