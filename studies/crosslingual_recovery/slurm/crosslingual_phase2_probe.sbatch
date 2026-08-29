#!/bin/bash
#SBATCH --job-name=phase2_probe
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/phase2_probe_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/phase2_probe_%j.err

# PHASE 2, STEP 2 -- the main instance-wise cross-lingual probe.
#
# RUN phase2_calibrate.sbatch FIRST and read its gate. If fewer than ~3 languages
# separate the learned and base populations, the in-language probe does not work on
# this model/data and this job is wasted -- report that as a negative result instead.
#
# For each method x relearn-language: benign-relearn on the RETAIN set, then probe the
# relearned checkpoint IN ALL 10 LANGUAGES x {forget, retain, world_facts}, then delete
# the checkpoint. That per-language probe is the whole point: Phase 1 could only ever
# read the English column, so "relearn in Japanese, measure recovery IN Japanese vs in
# English" -- the sharpest comparison in Phase 2 -- was literally invisible.
#
# Two DIFFERENT language decisions, deliberately:
#   PROBE_LANGS   where we MEASURE each fact -- all 10, always. It is what makes the
#                 matrix a matrix; never subset it.
#   RELEARN_LANGS where we benignly relearn -- this is where the GPU cost is
#                 (~20-25 min each), and where subsetting would actually save time.
#
# Ordered en fr hi ja first, then the rest: if the job dies at the wall we still hold
# source / near / Phase-1-anomaly / far rather than an alphabetical prefix that answers
# nothing. (hi is the Phase-1 anomaly: lowest Jaccard 0.018 but highest overlap
# coefficient 0.578 among non-Latin scripts.)
#
# EXCEEDS THE 12h WALL BY DESIGN. It is resumable: every (method, relearn-lang) whose
# probe keys are already on disk is skipped, so just resubmit until it stops launching.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/phase2_probe"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
echo "model slug = $MODEL_SLUG"

GROUP="phase2_probe"
FL="forget01"; N=40
EP="${1:-2}"                                   # relearn epochs (Phase 1: ep2 is the peak)
RELEARN_LANGS="${2:-en fr hi ja id ru fa ar iw ko}"
PROBE_LANGS="${3:-en fr hi ja id ru fa ar iw ko}"
SPLITS="forget retain world_facts"
RN="${4:-1500}"; RN_FLAG=""; [ -n "$RN" ] && RN_FLAG="--relearn-n $RN"
LORA_UEP="${5:-uep32}"
LAST_LANG=$(echo $PROBE_LANGS | awk '{print $NF}')   # resume sentinel (last cell written)

FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_${LORA_UEP}_${MODEL_SLUG}"
for CK in "$FULLFT" "$LORA"; do
    [ -d "$CK" ] || { echo "!! MISSING checkpoint: $CK"; exit 1; }
done
echo "deep checkpoints: $(basename $FULLFT) | $(basename $LORA)"
echo "relearn langs: $RELEARN_LANGS"
echo "probe langs  : $PROBE_LANGS   splits: $SPLITS"

# probe_all <checkpoint-path> -- every probe language x split for ONE checkpoint.
# --no-rouge: ROUGE needs 80 greedy generations per cell (~4x) and is fluency-
# confounded; the log-prob metrics carry the signal.
probe_all () {
    for SPLIT in $SPLITS; do
        python3 shared/scripts/relearn_measure.py --checkpoints "$1" \
            --group "$GROUP" --n $N --forget-level "$FL" \
            --fact-metrics --no-rouge --probe-split "$SPLIT" \
            --measure-lang $PROBE_LANGS || echo "!!! probe failed: $(basename $1)/$SPLIT"
    done
}

# has_probe <results-file> <key> -- was this checkpoint already probed to completion?
# Sentinel = the LAST (lang, split) cell the loop writes.
has_probe () {
    [ -f "$1" ] && python3 -c "
import json,sys
d=json.load(open('$1')); sys.exit(0 if '$2@${LAST_LANG}@world_facts' in d else 1)" 2>/dev/null
}

# --- 1. the UNLEARNED baselines: the 'before' state every Rec(l_r, l) is measured against
for M in "$FULLFT" "$LORA"; do
    MB=$(basename "$M")
    GFILE="$UNLEARN_RESULTS_DIR/relearn/$GROUP/${MB}.json"
    if has_probe "$GFILE" "$MB"; then
        echo "skip (already probed): $MB"
    else
        echo "=== probe UNLEARNED baseline $MB ==="
        probe_all "$M"
    fi
done

# --- 2. relearn in each language, probe in all, delete
LAUNCH=0
for M in "$FULLFT" "$LORA"; do
    MB=$(basename "$M")
    GFILE="$UNLEARN_RESULTS_DIR/relearn/$GROUP/${MB}.json"
    for L in $RELEARN_LANGS; do
        VIA="_via_retain"; [ "$L" != "en" ] && VIA="${VIA}_lang${L}"
        RKEY="relearn_${MB}${VIA}_ep${EP}"
        if has_probe "$GFILE" "$RKEY"; then
            echo "skip (already probed): $RKEY"; continue
        fi
        LAUNCH=$((LAUNCH + 1))
        echo ""
        echo "=== relearn $MB lang=$L ep=$EP  then probe $(echo $PROBE_LANGS | wc -w) langs ==="
        deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + LAUNCH)) \
            shared/scripts/relearn.py --checkpoint "$M" --epochs $EP \
            --relearn-data retain --relearn-lang "$L" --forget-level "$FL" $RN_FLAG
        RELEARNED="$PROJECT_DIR/experiments/${RKEY}"
        probe_all "$RELEARNED"
        rm -rf "$RELEARNED"        # ~16GB each; the probe JSON is the deliverable
    done
done

echo ""
echo "=== [DONE] phase2 probe -> $UNLEARN_RESULTS_DIR/relearn/$GROUP/ ==="
echo "launched $LAUNCH relearn+probe cycles this run."
echo "If any remain, RESUBMIT -- completed cells are skipped."
echo "NEXT (local, no GPU): rsync results/ down, then"
echo "  python studies/crosslingual_recovery/plots/phase2_kss_kps.py"
