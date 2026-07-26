#!/bin/bash
#SBATCH --job-name=tofu_unlearn_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_lora.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_lora.err

# LoRA unlearning (the LoRA-vs-Full-FT strategy axis). Fixed method =
# gradient_difference (the representative method for the strategy comparison).
# Plain `python` (no deepspeed launcher): LoRA trains tiny adapters, fits easily.
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

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}"

echo "=== LoRA unlearning: gradient_difference (+curve tracking) ==="
python shared/scripts/02_unlearn.py --checkpoint "$LEARNED" \
    --method gradient_difference --lora --track-curve

echo "=== LoRA unlearn complete ==="
