#!/bin/bash
#SBATCH --job-name=learn_fr
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/learn_fr_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/learn_fr_%j.err

# STAGE 1 of the French-anchored study: inject the TOFU facts IN FRENCH.
#
# ONE model per job -- submit twice. Each is a 5-epoch full fine-tune of Qwen3-8B
# over ~4000 QA, which is the bulk of the 12h wall on an a100-80, so pairing them
# in one job risks losing both to a wall-clock kill:
#
#     sbatch studies/learn_french/slurm/01_learn_fr.sbatch full        # fr_ft
#     sbatch studies/learn_french/slurm/01_learn_fr.sbatch retain99    # fr_retain
#
# They are independent and may run concurrently.
#
#   fr_ft      <- retain99_fr + forget01_fr (4000) : the model to be unlearned
#   fr_retain  <- retain99_fr        (3960)        : the FLOOR. Never saw the two
#                                                    forget authors, so it is the
#                                                    reference distribution for
#                                                    Forget Quality's KS test.
#
# GPU/PARTITION: a100-80 on gpu-long. NOT h200-141 -- as of 2026-08 the H200 moved
# off gpu-long onto `gpu` (node xgpk0), so `--partition=gpu-long --gres=gpu:h200-141:1`
# now fails outright, and `gpu`'s ~3h wall is far too short for a training job.
#
# Full-FT REQUIRES DeepSpeed ZeRO-3 (config/ds_config.json) for fp32 master weights.
# Confirmed by elimination: plain bf16 at lr=1e-5 rounds the updates away and never
# memorizes (CLAUDE.md sec 6).
set -euo pipefail

MODE="${1:-full}"
case "$MODE" in
    full|retain99) ;;
    *) echo "usage: sbatch 01_learn_fr.sbatch [full|retain99]" >&2; exit 2 ;;
esac

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
# $HOME has a tiny quota on a system partition -- every cache MUST land in the
# project dir or the job dies partway through with a confusing write error.
export TMPDIR="$PROJECT_DIR/.tmp/learn_fr"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Derive the port from the job id so two concurrent learn jobs never collide.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "=== [0/2] verifying the French LEARN data before spending GPU time ==="
python studies/learn_french/scripts/verify_learn_data.py

echo "=== [1/2] LEARN --data $MODE --lang fr ==="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT shared/scripts/01_learn.py \
    --data "$MODE" --lang fr

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
CKPT="$PROJECT_DIR/experiments/tofu_learn_${MODE}_full_${MODEL_SLUG}_fr"

echo "=== [2/2] checkpoint check ==="
# save_model() writes weights only; 01_learn also saves the tokenizer. Both must be
# there or every downstream metric silently evaluates to zero (CLAUDE.md sec 7).
ls -la "$CKPT"
for f in config.json tokenizer_config.json; do
    [ -f "$CKPT/$f" ] || { echo "MISSING $CKPT/$f" >&2; exit 1; }
done

echo "=== LEARN ($MODE, fr) complete -> $CKPT ==="
