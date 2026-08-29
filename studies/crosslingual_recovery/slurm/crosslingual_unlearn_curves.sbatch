#!/bin/bash
#SBATCH --job-name=xling_curves
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_curves_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_curves_%j.err

# UNLEARNING-DYNAMICS CURVES for BOTH methods, DEEP, before choosing a baseline.
# Unlearns Full-FT and LoRA on forget01 with --track-curve, going deeper than the
# earlier run so we can see HOW FAR each method can forget on each metric — and where
# truth ratio plateaus. Tracks ROUGE / Probability / Truth-Ratio (raw + bounded) on
# forget/retain/real/world every few steps -> results/curves/.
#
# NOTE: this OVERWRITES experiments/..._fullft_qwen3-8b and ..._lora_qwen3-8b with the
# deep-unlearned versions (they're diagnostic here). The matched checkpoints
# (..._lora_uep42, ..._lora_uep26) have different names and are NOT touched.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_curves"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
echo "model slug = $MODEL_SLUG"
LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}"
FL="forget01"
UEP_FT="${1:-50}"     # deep Full-FT: does its truth ratio plateau (~0.58) or keep rising?
UEP_LORA="${2:-60}"   # deep LoRA: how close to chance (~1.0) does it get?

echo "=== [1/2] Full-FT unlearn (UEP=$UEP_FT) + curve ==="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT shared/scripts/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy fullft --forget-level "$FL" \
    --unlearn-epochs "$UEP_FT" --track-curve

echo "=== [2/2] LoRA unlearn (UEP=$UEP_LORA) + curve ==="
python shared/scripts/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy lora --forget-level "$FL" \
    --unlearn-epochs "$UEP_LORA" --track-curve

echo "=== DONE -> curves in $UNLEARN_RESULTS_DIR/curves/ ==="
echo "Plot locally: python studies/crosslingual_recovery/plots/plot_unlearn_curve_compare.py"
