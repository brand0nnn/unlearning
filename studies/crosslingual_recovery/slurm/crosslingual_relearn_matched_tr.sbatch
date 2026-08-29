#!/bin/bash
#SBATCH --job-name=xling_tr_relearn
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_tr_relearn_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_tr_relearn_%j.err

# RECOVERY on the TRUTH-RATIO-MATCHED baselines. Run AFTER crosslingual_unlearn_matched_tr.
#
# Reuses the two truth-ratio-matched checkpoints (Full-FT + the auto-picked matched LoRA)
# and benign-relearns each on the RETAIN set in all 10 languages at epochs {1,2,4},
# measuring ROUGE + probability + truth ratio (--fact-metrics) each time. Result: the
# recovery curves on the FACT metric, at EQUAL truth-ratio forget depth.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_tr_relearn"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
echo "model slug = $MODEL_SLUG"
MATCH_GROUP="crosslingual_tr_match"      # where the match job recorded the baselines
GROUP="crosslingual_tr_recovery"         # this job's recovery results
FL="forget01"; N=40
LANGS="${1:-en fr id ru hi fa ar iw ko ja}"
EPS="${2:-1 2 4}"                        # 60 relearns for all 3 -> may approach the 12h
                                         # wall; incremental writes mean a timeout keeps
                                         # what finished. Run "1 2" then "4" to be safe.
RN="${3:-1500}"; RN_FLAG=""; [ -n "$RN" ] && RN_FLAG="--relearn-n $RN"

FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"
# auto-discover the truth-ratio-matched LoRA (its epoch was chosen by the sweep) from
# the match group's recorded baseline file (NOT the _probe group of discarded candidates).
# Prefer the match group's recorded baseline file, but fall back to experiments/ when
# results/ is absent — results/ is rsync-only and gitignored, so it can be missing on the
# cluster while the checkpoint is intact (this killed job 770249 in the deep script).
# Override the epoch explicitly as $4 (e.g. "uep32").
LORA_JSON=$(ls "$UNLEARN_RESULTS_DIR/relearn/${MATCH_GROUP}/"*lora_uep*_${MODEL_SLUG}.json 2>/dev/null | head -1)
if [ -n "${4:-}" ]; then
    LORA_NAME="tofu_unlearn_gradient_difference_forget01_lora_${4}_${MODEL_SLUG}"
elif [ -n "$LORA_JSON" ]; then
    LORA_NAME=$(basename "$LORA_JSON" .json)
else
    LORA_NAME=$(basename "$(ls -d "$PROJECT_DIR/experiments/"*lora_uep*_${MODEL_SLUG} 2>/dev/null | head -1)" 2>/dev/null)
    [ -n "$LORA_NAME" ] || { echo "!! no matched-LoRA checkpoint — run crosslingual_unlearn_matched_tr first"; exit 1; }
    echo "!! results/${MATCH_GROUP}/ missing; fell back to experiments/ -> $LORA_NAME"
fi
LORA="$PROJECT_DIR/experiments/$LORA_NAME"
echo "matched LoRA = $LORA_NAME"

for CK in "$FULLFT" "$LORA"; do
    [ -d "$CK" ] || { echo "!! MISSING checkpoint: $CK  (run the TR-match job first)"; exit 1; }
done

# --- record matched baselines (fact metrics) into THIS group -------------------
echo "=== baselines (fact metrics) ==="
for CK in "$FULLFT" "$LORA"; do
    python shared/scripts/relearn_measure.py --checkpoints "$CK" --group "$GROUP" \
        --n $N --forget-level "$FL" --fact-metrics
done

# --- benign retain-relearn each in each language x epoch, measure fact metrics --
LAUNCH=0
for M in "$FULLFT" "$LORA"; do
    for L in $LANGS; do
        for E in $EPS; do
            LAUNCH=$((LAUNCH + 1))
            echo "=== relearn $(basename $M) lang=$L ep=$E ==="
            deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + LAUNCH)) \
                shared/scripts/relearn.py --checkpoint "$M" --epochs $E \
                --relearn-data retain --relearn-lang "$L" --forget-level "$FL" $RN_FLAG
            VIA="_via_retain"; [ "$L" != "en" ] && VIA="${VIA}_lang${L}"
            RELEARNED="$PROJECT_DIR/experiments/relearn_$(basename $M)${VIA}_ep${E}"
            python shared/scripts/relearn_measure.py --checkpoints "$RELEARNED" \
                --group "$GROUP" --n $N --forget-level "$FL" --fact-metrics
            rm -rf "$RELEARNED"
        done
    done
done

echo "=== [DONE] truth-ratio-matched recovery -> $UNLEARN_RESULTS_DIR/relearn/$GROUP/ ==="
