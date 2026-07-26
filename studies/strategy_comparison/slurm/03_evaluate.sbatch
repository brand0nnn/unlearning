#!/bin/bash
#SBATCH --job-name=tofu_evaluate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/evaluate.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/evaluate.err

# Full TOFU metric suite -> Model Utility + Forget Quality (KS p-value vs the
# retain90 gold reference), for the paper's Fig 5/6 plane. Plain inference (no
# DeepSpeed). Run this AFTER the four 02 unlearn jobs finish, so it evaluates the
# freshly-tuned checkpoints. Writes per-strategy JSON to results/forget_quality/;
# plot LOCALLY with studies/strategy_comparison/plots/plot_forget_quality.py.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/strategy_comparison/results"  # this experiment's own results/
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
echo "model slug = $MODEL_SLUG"
export TMPDIR="$PROJECT_DIR/.tmp"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python shared/scripts/03_evaluate.py \
    --reference "$PROJECT_DIR/experiments/tofu_learn_retain90_full_${MODEL_SLUG}" \
    --checkpoints \
        "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10_fullft_${MODEL_SLUG}" \
        "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget10_lora_${MODEL_SLUG}" \
        "$PROJECT_DIR/experiments/tofu_unlearn_self_distill_forget10_self_distill_${MODEL_SLUG}" \
        "$PROJECT_DIR/experiments/tofu_unlearn_grpo_forget10_grpo_${MODEL_SLUG}"

echo "=== Evaluate complete -> results/forget_quality/  "
echo "    plot: python studies/strategy_comparison/plots/plot_forget_quality.py ==="
