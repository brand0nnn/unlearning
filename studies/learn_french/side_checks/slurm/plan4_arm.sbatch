#!/bin/bash
#SBATCH --job-name=side_p4
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/side_p4_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/side_p4_%j.err

# SIDE CHECK, PLAN 4: where does each arm change the model at the SAME epochs?
#
#     sbatch studies/learn_french/side_checks/slurm/plan4_arm.sbatch ja
#     sbatch studies/learn_french/side_checks/slurm/plan4_arm.sbatch fr
#     sbatch studies/learn_french/side_checks/slurm/plan4_arm.sbatch en
#
# Job 11 compared each arm at its matched checkpoint, where ja had run 25 epochs and fr/en
# 12-13, so "ja is top-heavy" could be the language or the run length. This re-runs the
# Stage 2 recipe UNCHANGED (lr 5e-6, 50-epoch schedule) to epoch 25, saves epoch 13 too,
# and diffs both against fr_ft: every arm at epoch 13 and at epoch 25, one run each.
#
#   [1] unlearn, every epoch scored IN-RUN, save epochs 13 and 25              ~5h
#   [2] layer diff fr_ft -> epoch 13 and fr_ft -> epoch 25 (CPU)               ~20min
#   [3] delete both checkpoints -- only once the layer diff exists
#
# fr and en at epoch 25 repeat Plan 1's runs (which reproduced Stage 2's TR to 3 decimals);
# their in-run JSONL doubles as a reproduction check. No scoring, no relearning.
# Peak disk ~33GB (two 16 GB checkpoints).
set -euo pipefail

LANG_UL="${1:?usage: sbatch plan4_arm.sbatch <lang>}"
STOP=25
MID=13
SEED=42
case "$LANG_UL" in en|fr|id|ja|ru) ;; *) echo "bad lang: $LANG_UL" >&2; exit 2 ;; esac

PROJECT_DIR="/home/b/brandonk/unlearning"
SIDE="studies/learn_french/side_checks"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/$SIDE/results"
cd "$PROJECT_DIR"
R="$UNLEARN_RESULTS_DIR"
mkdir -p logs "$R/plan4" experiments/side_checks/plan4
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/side_p4_$LANG_UL"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UNLEARN_DS_CONFIG="config/ds_config_offload.json"
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"
MEM_MB="${SLURM_MEM_PER_NODE:-0}"
[ "$MEM_MB" -ge 180000 ] || { echo "optimizer offload needs --mem>=180G, got ${MEM_MB}MB" >&2; exit 1; }

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
FR_FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
[ -d "$FR_FT" ] || { echo "MISSING fr_ft: $FR_FT" >&2; exit 1; }

# unlearn_epochs.py tags the run with the LR whenever --save-at-epochs is given.
LR_TAG=$(python3 -c "import yaml;print(('lr%.1e'%float(yaml.safe_load(open('config/config.yaml'))['tofu']['unlearn_lr'])).replace('.','p'))")
RUN="ul${LANG_UL}_s${SEED}_${LR_TAG}_ep${STOP}"
CKPT="$PROJECT_DIR/experiments/side_checks/plan4/$RUN"
CKPT_MID="${CKPT}_atep${MID}"
TRAJ="$R/plan4/traj_$RUN.jsonl"
TOKS="$R/plan4/tokens_$RUN.jsonl"
LAYERS="$R/plan4/layer_diff_$RUN.json"

echo "=========================================================================="
echo " SIDE CHECK plan 4   arm $LANG_UL   Stage 2 recipe ($LR_TAG), epochs $MID and $STOP"
echo "   checkpoints: $CKPT_MID, $CKPT (both deleted at the end)"
echo "=========================================================================="
# The limit is ~400 GB PER USER and neither df nor quota shows it: measure with du.
USED_GB=$(du -s -BG /home/b/brandonk 2>/dev/null | cut -f1 | tr -dc '0-9')
echo "   storage    : ${USED_GB}GB used of the ~400GB per-user limit"
[ $((400 - USED_GB)) -ge 40 ] || { echo "only $((400 - USED_GB))GB left; this job peaks at ~33GB" >&2; exit 1; }

# ---- [1] unlearn ---------------------------------------------------------------------
if [ -f "$LAYERS" ]; then
    echo "[1] layer diff already written -- nothing to train"
elif [ -f "$CKPT/tokenizer_config.json" ] && [ -f "$CKPT_MID/tokenizer_config.json" ]; then
    echo "[1] both epochs already saved -- skipping unlearning"
else
    if [ -e "$TRAJ" ] || [ -e "$TOKS" ] || [ -e "$CKPT" ] || [ -e "$CKPT_MID" ]; then
        echo "[1] a PARTIAL run exists (killed mid-training?). Move these aside, then resubmit:" >&2
        echo "      $TRAJ" >&2; echo "      $TOKS" >&2; echo "      $CKPT_MID" >&2; echo "      $CKPT" >&2
        exit 1
    fi
    echo "[1] unlearn"
    deepspeed --num_gpus=1 --master_port=$((20000 + SLURM_JOB_ID % 10000)) \
        "$SIDE/scripts/unlearn_epochs.py" \
        --checkpoint "$FR_FT" --lang "$LANG_UL" --seed "$SEED" \
        --schedule-epochs 50 --stop-at-epoch "$STOP" --save-at-epochs "$MID" \
        --mu-every 10 --forget-floor none \
        --ckpt-root experiments/side_checks/plan4 --out-dir "$R/plan4"
fi

# ---- [2] layer diff (CPU) ------------------------------------------------------------
if [ -f "$LAYERS" ]; then echo "[2] layer diff -- skip"; else
    for C in "$CKPT_MID" "$CKPT"; do
        [ -f "$C/tokenizer_config.json" ] || { echo "no finished checkpoint at $C" >&2; exit 1; }
    done
    echo "[2] layer diff fr_ft -> epoch $MID and epoch $STOP"
    python studies/learn_french/scripts/layer_diff.py \
        --pair "ul${LANG_UL}_ep${MID}=${FR_FT}::${CKPT_MID}" \
        --pair "ul${LANG_UL}_ep${STOP}=${FR_FT}::${CKPT}" --out "$LAYERS"
fi

# ---- [3] delete both checkpoints ------------------------------------------------------
[ -f "$LAYERS" ] || { echo "[3] missing $LAYERS -- KEEPING checkpoints" >&2; exit 1; }
for C in "$CKPT_MID" "$CKPT"; do
    case "$C" in "$PROJECT_DIR/experiments/side_checks/plan4/ul"*) ;; *)
        echo "refusing to delete unexpected path $C" >&2; exit 1 ;; esac
    [ -d "$C" ] && rm -rf "$C" && echo "[3] deleted $C"
done
echo "=== done: arm $LANG_UL -> $R/plan4"
