#!/bin/bash
#SBATCH --job-name=phase2_calibrate
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/phase2_calibrate_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/phase2_calibrate_%j.err

# PHASE 2, STEP 1 -- the tau-calibration GATE. Run this BEFORE the ~17h probe job.
#
# KPS needs a binary "does the model still know fact i in language l". We get it by
# thresholding the truth ratio at tau_l, calibrated per language against two
# populations we can probe for free:
#   positives = the LEARNED checkpoint   (memorised these facts by construction)
#   negatives = BASE Qwen3-8B            (TOFU authors are fictitious -> cannot know them)
# tau_l is then the max-Youden-J point of that two-population ROC, computed OFFLINE by
# plots/phase2_kss_kps.py -- nothing here picks a threshold.
#
# The gate: if the two populations do NOT separate in language l, that language's
# in-language probe is UNINFORMATIVE and must be reported as such rather than scored.
# world_facts says WHY -- it is pre-training knowledge, independent of anything we
# fine-tuned, so good world_facts + bad TOFU => our translations, bad on both => the
# model is simply weak in that language.
#
# No relearning, no unlearning, no checkpoints written. ~2 min per (ckpt, lang, split)
# for the log-prob metrics => 2 x 10 x 3 = 60 cells, ~2h. Resumable: re-running skips
# nothing, but each write is incremental, so a wall-clock kill keeps what finished.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/phase2_calibrate"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
BASE_MODEL=$(python3 -c "import yaml;print(yaml.safe_load(open('config/config.yaml'))['model']['name'])")
echo "model slug = $MODEL_SLUG   base = $BASE_MODEL"

GROUP="phase2_calibrate"
FL="forget01"; N=40
LANGS="${1:-en fr hi ja id ru fa ar iw ko}"
SPLITS="${2:-forget retain world_facts}"

# Positives and negatives for the tau ROC. The LEARNED checkpoint must exist; the base
# model is pulled from the HF cache by name (no checkpoint on disk).
LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}"
[ -d "$LEARNED" ] || { echo "MISSING learned checkpoint: $LEARNED"; exit 1; }

echo "=== langs:  $LANGS"
echo "=== splits: $SPLITS"

for SPLIT in $SPLITS; do
    echo ""
    echo "############ split=$SPLIT ############"
    # --no-rouge: ROUGE needs 80 greedy generations per cell (~4x the cost) and is the
    # metric we already showed to be fluency-confounded. The log-prob metrics carry it.
    python3 shared/scripts/relearn_measure.py \
        --checkpoints "$LEARNED" "$BASE_MODEL" \
        --group "$GROUP" --n $N --forget-level "$FL" \
        --fact-metrics --no-rouge \
        --probe-split "$SPLIT" \
        --measure-lang $LANGS || echo "!!! split=$SPLIT failed, continuing"
done

echo ""
echo "phase2 calibration complete -> studies/crosslingual_recovery/results/relearn/$GROUP/"
echo "NEXT (local, no GPU): rsync results/ down, then"
echo "  python studies/crosslingual_recovery/plots/phase2_kss_kps.py --calibrate-only"
