#!/bin/bash
#SBATCH --job-name=xling_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_probe_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_probe_%j.err

# CROSS-LINGUAL KNOWLEDGE PROBE — replaces the (invalid) cross-lingual ROUGE audit
# with the Multilingual Amnesia metrics (Farashah et al. 2026, our dataset's paper):
# normalized probability + truth ratio, on the model's own subword tokens. These
# generalize across morphologically-rich languages where ROUGE does not.
#
# Reads the forget fact in all 10 languages on three models:
#   1. LEARNED (pre-unlearn)  -> the baseline P_forget the blast-radius ratio divides by
#   2. UNLEARN Full-FT        -> \ blast radius = P_unlearned/P_learned per language
#   3. UNLEARN LoRA (matched) -> /   (Amnesia Fig.2), Full-FT vs LoRA
#
# Eval-only, no training. Plain inference (no DeepSpeed).
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_probe"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

GROUP="crosslingual_probe"
LANGS="${1:-en fr id ru hi fa ar iw ko ja}"   # full distance axis (near -> far from EN)
FL="forget01"                                  # multilingual data is forget01 only
N=40                                           # all forget01 records

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"
FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft"
LORA=$(ls -d "$PROJECT_DIR"/experiments/tofu_unlearn_gradient_difference_forget01_lora_uep* 2>/dev/null | head -1)

CKPTS=()
for C in "$LEARNED" "$FULLFT" "$LORA"; do
    if [ -n "$C" ] && [ -d "$C" ]; then CKPTS+=("$C"); else echo "!! skip missing: '$C'"; fi
done

python shared/scripts/crosslingual_probe.py \
    --checkpoints "${CKPTS[@]}" \
    --measure-lang $LANGS \
    --forget-level "$FL" --group "$GROUP" --n "$N"

echo "=== probe done -> $UNLEARN_RESULTS_DIR/crosslingual/$GROUP/ ==="
