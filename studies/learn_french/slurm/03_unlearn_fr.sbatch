#!/bin/bash
#SBATCH --job-name=unlearn_fr
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_fr_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_fr_%j.err

# STAGE 2/3: unlearn fr_ft in ONE language, probing FRENCH every few optimizer steps.
#
#     sbatch 03_unlearn_fr.sbatch <lang> [tr_levels] [epochs] [strategy]
#
#     sbatch 03_unlearn_fr.sbatch en                          # pilot, trace-only
#     sbatch 03_unlearn_fr.sbatch ja                          # pilot, trace-only
#     sbatch 03_unlearn_fr.sbatch ru 0.42,0.55,0.69,0.82,0.95 # grid, checkpointing
#
# lang in {en, fr, id, ja, ru}. One job per language: they are independent, may run
# concurrently, and a failure costs one cell rather than five.
#
# THE PROBE LANGUAGE IS ALWAYS FRENCH, whatever we unlearn in. That IS the
# experiment -- does unlearning in Japanese remove the FRENCH knowledge.
#
# TR LEVELS. Omit them for a trace-only run: nothing is saved, we just learn the
# shape of the trajectory. That is Stage 2, whose job is to reveal whether a common
# level grid is reachable in every language at all. The level VALUES come from Stage
# 1's measured ceiling (fr_ft) and floor (fr_retain) -- do not invent them.
#
# STORAGE WARNING. With levels set, each run saves up to 5 checkpoints at ~16.4GB =
# ~82GB per cell. The full 5-language x 2-method grid is ~820GB. Check quota BEFORE
# launching the grid; the plan (sec 8) says keep them all, but it did not price them.
set -euo pipefail

LANG_UL="${1:?usage: sbatch 03_unlearn_fr.sbatch <lang> [tr_levels] [epochs] [strategy]}"
LEVELS="${2:-}"
EPOCHS="${3:-40}"       # forget01 = 40 examples at effective batch 32 -> ~1.25 steps
                        # per epoch, so 40 epochs ~ 50 optimizer steps. TOFU-scale
                        # unlearning is 10-50 steps; 5 epochs would be ~6 and far too
                        # coarse to resolve a level crossing.
STRATEGY="${4:-fullft}"

case "$LANG_UL" in en|fr|id|ja|ru) ;; *) echo "bad lang: $LANG_UL" >&2; exit 2 ;; esac

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/unlearn_fr_$LANG_UL"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
FR_FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
[ -d "$FR_FT" ] || { echo "MISSING fr_ft: $FR_FT" >&2; exit 1; }

LEVEL_ARG=""
if [ -n "$LEVELS" ]; then LEVEL_ARG="--tr-levels $LEVELS"; fi

echo "=== unlearn lang=$LANG_UL strategy=$STRATEGY epochs=$EPOCHS ==="
echo "    source   : $FR_FT"
echo "    probe    : fr (always)"
echo "    tr-levels: ${LEVELS:-TRACE ONLY (no checkpoints saved)}"

# Full-FT needs DeepSpeed ZeRO-3 for fp32 master weights; LoRA trains adapters and
# runs plain (src/training/unlearn.py sets deepspeed=None for it).
if [ "$STRATEGY" = "lora" ]; then
    LAUNCH="python"
else
    LAUNCH="deepspeed --num_gpus=1 --master_port=$MASTER_PORT"
fi

$LAUNCH shared/scripts/02_unlearn.py \
    --checkpoint "$FR_FT" \
    --strategy "$STRATEGY" \
    --forget-level forget01 \
    --unlearn-epochs "$EPOCHS" \
    --lang "$LANG_UL" \
    --probe-lang fr \
    --eval-every 2 \
    $LEVEL_ARG

echo "=== done: trajectory -> $UNLEARN_RESULTS_DIR/unlearn_traj/ ==="
ls -la "$UNLEARN_RESULTS_DIR/unlearn_traj/" 2>/dev/null || true
