#!/bin/bash
#SBATCH --job-name=unlearn_fr
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_fr_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_fr_%j.err

# STAGE 2/3: Full-FT unlearning of fr_ft in ONE language, probing FRENCH as it goes.
#
#     sbatch 03_unlearn_fr.sbatch <lang> [epochs=50]
#     sbatch 03_unlearn_fr.sbatch <lang> [epochs=50] trace     # debugging only
#
#     sbatch 03_unlearn_fr.sbatch en        # Stage 2 pilot
#     sbatch 03_unlearn_fr.sbatch ja        # Stage 2 pilot
#     sbatch 03_unlearn_fr.sbatch fr        # Stage 3, only after the pilot gate
#
# lang in {en, fr, id, ja, ru}. One job per language: independent, may run
# concurrently, and a failure costs one cell rather than five.
#
# THE PROBE LANGUAGE IS ALWAYS FRENCH, whatever we unlearn in. That IS the
# experiment -- does unlearning in Japanese remove the FRENCH knowledge. The
# unlearning data (forget AND retain term) are both in <lang>, as in Xiang et al.
# Eq. 1; Model Utility is measured in French.
#
# TR LEVELS come from studies/learn_french/preregistration.json, never the command
# line: all five languages must share ONE grid (plan sec 4a), and the file's git commit
# is the proof it was fixed before any unlearning result. The job refuses to start
# without it. The pilot is a PRODUCTION run (plan sec 3: "they become two columns of
# the final table"), so it checkpoints too; `trace` exists only for debugging.
#
# PROBE VARIANT: surname-normalized (primary, drives level crossings), with the
# as-published probe logged alongside as mean_tr_raw. Must match the variant the
# levels were computed on -- checked against the pre-registration below.
#
# FULL-FT ONLY. LoRA is a robustness arm the plan defers to Stage 3, and TR-level
# saving is not implemented for it (merging adapters mid-run would destroy them).
#
# STEPS. forget01 = 40 examples at effective batch 32 = TWO optimizer steps per epoch
# (32, then the 8-example remainder), so 50 epochs = 100 steps -- the length of the old
# English curve run, which saturated by ~step 48, leaving room for slower languages.
# The LR decays linearly to 0 over the run: a late flattening may be the schedule,
# not a plateau. Read plateaus while the LR is still high.
#
# COST (estimate): 50 probe points x (TR ~0.5 min + Model Utility ~3 min) + training
# + level saves => ~3-5h. Training is slower than the 15-30 min a GPU-resident optimizer
# would take, because the optimizer step now runs on the CPU (see MEMORY above).
# STORAGE: up to 5 level checkpoints x ~16.4GB = ~82GB per language (fewer if several
# levels are crossed at one probe point -- those are symlinked). No final checkpoint.
# 5 languages = ~410GB. Check quota BEFORE launching the grid.
set -euo pipefail

LANG_UL="${1:?usage: sbatch 03_unlearn_fr.sbatch <lang> [epochs] [levels|trace]}"
EPOCHS="${2:-50}"
MODE="${3:-levels}"

# MEMORY. Qwen3-8B Full-FT does not fit an 80GB card under the default ZeRO-3 config:
# params 15.3 + fp32 master 30.5 + flat grad buffer 15.3 + the backward's gradients 15.3
# = 76.4GB before activations, and the first backward dies ~1GB short (jobs 841834,
# 841886 -- identical byte for byte, so it is the configuration, not the probe).
# ds_config_offload.json holds the optimizer state in host RAM instead, leaving ~30GB
# free on the GPU, at the cost of a CPU optimizer step. That needs host RAM, hence the
# --mem below; the guard refuses to start rather than be killed by SLURM mid-run.
export UNLEARN_DS_CONFIG="${UNLEARN_DS_CONFIG:-config/ds_config_offload.json}"

case "$LANG_UL" in en|fr|id|ja|ru) ;; *) echo "bad lang: $LANG_UL" >&2; exit 2 ;; esac
case "$MODE" in levels|trace) ;; *) echo "bad mode: $MODE (levels|trace)" >&2; exit 2 ;; esac

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
echo "Commit: $(git rev-parse --short HEAD)"
echo "DeepSpeed config: $UNLEARN_DS_CONFIG"
if grep -q '"device": "cpu"' "$UNLEARN_DS_CONFIG" 2>/dev/null; then
    MEM_MB="${SLURM_MEM_PER_NODE:-0}"
    [ "$MEM_MB" -ge 180000 ] || { echo "Optimizer offload needs host RAM: got ${MEM_MB}MB," \
        "need >=180000MB. Resubmit with --mem=200G (or set UNLEARN_DS_CONFIG to" \
        "config/ds_config.json to keep everything on the GPU)." >&2; exit 1; }
fi
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
FR_FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
[ -d "$FR_FT" ] || { echo "MISSING fr_ft: $FR_FT" >&2; exit 1; }

PREREG="studies/learn_french/preregistration.json"
LEVEL_ARG=""
if [ "$MODE" = "levels" ]; then
    [ -f "$PREREG" ] || { echo "MISSING $PREREG -- run stage1_report.py --write-prereg" \
        "after Stage 1 sign-off, commit it, pull. (Or pass 'trace' to debug.)" >&2; exit 1; }
    { git ls-files --error-unmatch "$PREREG" && git diff --quiet HEAD -- "$PREREG"; } \
        >/dev/null 2>&1 || {
        echo "$PREREG is not committed as-is -- commit it first; the commit is the" \
             "timestamp" >&2; exit 1; }
    VARIANT=$(python3 -c "import json;print(json.load(open('$PREREG'))['probe_variant'])")
    [ "$VARIANT" = "norm" ] || { echo "prereg levels are on the '$VARIANT' probe, but" \
        "this job probes with --probe-normalize-surname (norm)" >&2; exit 1; }
    LEVELS=$(python3 -c "import json;print(','.join(str(x) for x in json.load(open('$PREREG'))['tr_levels']))")
    LEVEL_ARG="--tr-levels $LEVELS"
fi

echo "=== unlearn lang=$LANG_UL Full-FT epochs=$EPOCHS (=$((EPOCHS * 2)) steps) ==="
echo "    source   : $FR_FT"
echo "    probe    : fr, surname-normalized (raw logged alongside)"
echo "    tr-levels: ${LEVELS:-TRACE ONLY (no checkpoints saved)}"
echo "    storage  : $(df -h "$PROJECT_DIR/experiments" | tail -1)"

# Full-FT needs DeepSpeed ZeRO-3 for fp32 master weights.
deepspeed --num_gpus=1 --master_port=$MASTER_PORT shared/scripts/02_unlearn.py \
    --checkpoint "$FR_FT" \
    --strategy fullft \
    --forget-level forget01 \
    --unlearn-epochs "$EPOCHS" \
    --lang "$LANG_UL" \
    --probe-lang fr \
    --probe-normalize-surname \
    --eval-every 2 \
    --skip-final-save \
    $LEVEL_ARG

echo "=== done: trajectory -> $UNLEARN_RESULTS_DIR/unlearn_traj/ ==="
ls -la "$UNLEARN_RESULTS_DIR/unlearn_traj/" 2>/dev/null || true
ls -la "$PROJECT_DIR/experiments/tr_levels/" 2>/dev/null || true
