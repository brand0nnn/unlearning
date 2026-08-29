#!/bin/bash
#SBATCH --job-name=xling_deep_relearn
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_deep_relearn_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_deep_relearn_%j.err

# RECOVERY on the DEEP truth-ratio-matched baselines (Full-FT ~0.77, LoRA ~0.69).
# Run AFTER crosslingual_unlearn_deep. Reuses the two deep checkpoints and benign-
# relearns each on the RETAIN set in all 10 languages at epochs {1,2,4}, measuring
# ROUGE + probability + truth ratio (--fact-metrics). Results -> crosslingual_deep.
#
# {1,2,4} x 10 langs x 2 methods = 60 relearns -> may approach the 12h wall; incremental
# writes mean a timeout keeps what finished. Run "1 2" first, then "4", to be safe.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_deep_relearn"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
echo "model slug = $MODEL_SLUG"
GROUP="crosslingual_deep"
FL="forget01"; N=40
LANGS="${1:-en fr id ru hi fa ar iw ko ja}"
EPS="${2:-1 2 4}"
RN="${3:-1500}"; RN_FLAG=""; [ -n "$RN" ] && RN_FLAG="--relearn-n $RN"

FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"
# The deep LoRA is pinned BY NAME (uep32 = this group's truth-ratio-matched deep
# checkpoint; see remeasure_lora_baseline.sbatch), overridable as $4.
#
# Resolved against experiments/ — where the weights actually live — NOT by globbing the
# results JSON, which is how job 770249 died in 2 seconds: results/ is rsync-only and
# gitignored, so the JSON can be absent on the cluster while every checkpoint is fine.
# Globbing *lora_uep* | head -1 was unsafe anyway: both uep32 and uep42 exist in
# experiments/, so alphabetical order silently decided which baseline got relearned and
# written under THIS group's name.
LORA_UEP="${4:-uep32}"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_${LORA_UEP}_${MODEL_SLUG}"
echo "deep checkpoints: $(basename $FULLFT) | $(basename $LORA)"
for CK in "$FULLFT" "$LORA"; do
    [ -d "$CK" ] || { echo "!! MISSING checkpoint: $CK  (run the deep-unlearn job first)"; exit 1; }
done

# --- benign retain-relearn each in each language x epoch, measure fact metrics ---
# RESUMABLE: skips any (method,lang,epoch) already recorded, so a re-run after a wall
# timeout only does the missing work (e.g. the LoRA half + any missing Full-FT points).
LAUNCH=0
for M in "$FULLFT" "$LORA"; do
    MB=$(basename "$M")
    GFILE="$UNLEARN_RESULTS_DIR/relearn/$GROUP/${MB}.json"
    for L in $LANGS; do
        VIA="_via_retain"; [ "$L" != "en" ] && VIA="${VIA}_lang${L}"
        for E in $EPS; do
            RKEY="relearn_${MB}${VIA}_ep${E}"
            # Skip only if this point already has the PER-FACT arrays (not just the mean),
            # so a re-run after the relearn_measure per-fact upgrade regenerates the arrays
            # for the bootstrap CI instead of skipping the mean-only records.
            if [ -f "$GFILE" ] && python3 -c "import json,sys; r=json.load(open('$GFILE')).get('$RKEY'); sys.exit(0 if isinstance(r,dict) and 'truth_ratio_per_fact' in r else 1)" 2>/dev/null; then
                echo "skip (already has per-fact arrays): $RKEY"; continue
            fi
            LAUNCH=$((LAUNCH + 1))
            echo "=== relearn $MB lang=$L ep=$E ==="
            deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + LAUNCH)) \
                shared/scripts/relearn.py --checkpoint "$M" --epochs $E \
                --relearn-data retain --relearn-lang "$L" --forget-level "$FL" $RN_FLAG
            RELEARNED="$PROJECT_DIR/experiments/relearn_${MB}${VIA}_ep${E}"
            python shared/scripts/relearn_measure.py --checkpoints "$RELEARNED" \
                --group "$GROUP" --n $N --forget-level "$FL" --fact-metrics
            rm -rf "$RELEARNED"
        done
    done
done

echo "=== [DONE] deep-matched recovery -> $UNLEARN_RESULTS_DIR/relearn/$GROUP/ ==="
