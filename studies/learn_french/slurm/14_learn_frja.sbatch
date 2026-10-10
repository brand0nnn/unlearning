#!/bin/bash
#SBATCH --job-name=learn_frja
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/learn_frja_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/learn_frja_%j.err

# Q1 STORAGE HIDING: one model that learns every TOFU fact in French AND Japanese.
#
# Why joint, not one model per language: hiding needs TWO copies of the fact in ONE set
# of weights -- unlearn the French copy, probe the Japanese one. fr_ft never stored the
# Japanese answer (Stage 4 gate 1.35x) and French unlearning does not push it there
# (figures/flow_unl_fr.png), so the Japanese copy has to be taught directly.
#
#     sbatch studies/learn_french/slurm/14_learn_frja.sbatch full       # fr-ja_ft     8000 QA
#     sbatch studies/learn_french/slurm/14_learn_frja.sbatch retain99   # fr-ja_retain 7920 QA (floor)
#
# COST. fr_ft took 27,470 s (7.6 h) for 4000 rows; Japanese rows are the same length in
# Qwen3 tokens (81 vs 78 per row), so each model is ~15 h. gpu-long allows 3 days
# (sinfo, checked 2026-10-11), so the 20 h wall fits one model in ONE job and no resume
# checkpoint is ever written. The resume chain is insurance only: if the wall is ever
# shorter than the run, it STOPS ITSELF ~MARGIN_MIN before the wall, writes one resume
# checkpoint (~131 GB: bf16 weights twice + ZeRO-3 fp32 master + Adam moments), and
# resubmits itself; the next job continues from the same step.
#
# DISK, per model: final weights ~16 GB. Only if the resume chain fires: plus ~131 GB
# while a resume checkpoint exists (deleted on completion).
# Check first:  bash studies/learn_french/scripts/check_space.sh
#
# SMOKE TEST FIRST (exercises the stop -> checkpoint -> resubmit -> resume path that a
# 15 h run depends on; ~1 h GPU, ~150 GB disk briefly, run name gets `_lim160`):
#     SMOKE=1 sbatch studies/learn_french/slurm/14_learn_frja.sbatch full
#
# Env overrides: LANGS ("fr ja"), MARGIN_MIN (60), MAX_CHAIN (3 resubmits).
set -euo pipefail

MODE="${1:-full}"
case "$MODE" in
    full|retain99) ;;
    *) echo "usage: sbatch 14_learn_frja.sbatch [full|retain99]" >&2; exit 2 ;;
esac
LANGS="${LANGS:-fr ja}"
CHAIN="${CHAIN:-0}"
MAX_CHAIN="${MAX_CHAIN:-3}"
SMOKE="${SMOKE:-0}"
MARGIN_MIN="${MARGIN_MIN:-60}"     # time left for the ~131 GB save + model teardown

PROJECT_DIR="/home/b/brandonk/unlearning"
SELF="$PROJECT_DIR/studies/learn_french/slurm/14_learn_frja.sbatch"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
# $HOME has a tiny quota on a system partition -- every cache MUST land in the project dir.
export TMPDIR="$PROJECT_DIR/.tmp/learn_frja_$MODE"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)   chain link $CHAIN/$MAX_CHAIN   langs: $LANGS   smoke: $SMOKE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

EXTRA=()
[ "$SMOKE" = "1" ] && EXTRA+=(--limit 160)
if [ "$SMOKE" = "1" ] && [ "$CHAIN" = "0" ]; then
    # 160 rows/language -> 320 rows -> 10 steps/epoch x 5 = 50 steps (~36 min). Stopping
    # the FIRST link 15 min in forces exactly one checkpoint + resume; later links run
    # to the wall like a real job.
    STOP_BY=$(( $(date +%s) + 15 * 60 ))
else
    END=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
    STOP_BY=$(( $(date -d "$END" +%s) - MARGIN_MIN * 60 ))
fi
echo "job ends $(squeue -h -j "$SLURM_JOB_ID" -o %e); training stops by $(date -d "@$STOP_BY")"
# A mis-parsed end time would put STOP_BY in the past: every link would then train one
# step, write a ~131 GB checkpoint and resubmit. Fail now instead, at zero cost.
if [ "$SMOKE" != "1" ] && [ "$STOP_BY" -lt $(( $(date +%s) + 30 * 60 )) ]; then
    echo "STOP_BY is less than 30 min away -- wall too short or end time mis-parsed" >&2
    exit 1
fi

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
SUFFIX="$(echo $LANGS | tr ' ' '-')"
[ "$SMOKE" = "1" ] && SUFFIX="${SUFFIX}_lim160"
CKPT="$PROJECT_DIR/experiments/tofu_learn_${MODE}_full_${MODEL_SLUG}_${SUFFIX}"

if [ "$CHAIN" = "0" ]; then
    echo "=== [0/3] verifying the joint LEARN data before spending GPU time ==="
    python studies/learn_french/scripts/verify_learn_data_joint.py $LANGS
fi
echo "=== disk before ==="; df -h "$PROJECT_DIR" | tail -1

echo "=== [1/3] LEARN --data $MODE --lang $LANGS -> $CKPT ==="
rm -f "$CKPT/RESUME_PENDING"
set +e
deepspeed --num_gpus=1 --master_port=$MASTER_PORT shared/scripts/01_learn.py \
    --data "$MODE" --lang $LANGS --stop-by "$STOP_BY" ${EXTRA[@]+"${EXTRA[@]}"}
RC=$?
set -e

if [ -f "$CKPT/RESUME_PENDING" ]; then
    echo "=== [2/3] PARTIAL (rc=$RC): resume checkpoint written ==="
    ls -d "$CKPT"/checkpoint-* ; du -sh "$CKPT"; df -h "$PROJECT_DIR" | tail -1
    if [ "$CHAIN" -ge "$MAX_CHAIN" ]; then
        echo "MAX_CHAIN=$MAX_CHAIN resubmits used up -- NOT resubmitting. Investigate, then:" >&2
        echo "  CHAIN=0 sbatch $SELF $MODE" >&2
        exit 1
    fi
    NEXT=$(sbatch --parsable --export=ALL,CHAIN=$((CHAIN + 1)),SMOKE=$SMOKE,LANGS="$LANGS" \
           "$SELF" "$MODE")
    echo "=== resubmitted as job $NEXT (chain link $((CHAIN + 1))) ==="
    exit 0
fi
[ "$RC" -eq 0 ] || { echo "LEARN failed with rc=$RC and no resume marker" >&2; exit "$RC"; }

echo "=== [3/3] checkpoint check ==="
# save_model() writes weights only; 01_learn also saves the tokenizer. Both must be
# there or every downstream metric silently evaluates to zero (CLAUDE.md sec 7).
ls -la "$CKPT"
for f in config.json tokenizer_config.json; do
    [ -f "$CKPT/$f" ] || { echo "MISSING $CKPT/$f" >&2; exit 1; }
done
if compgen -G "$CKPT/checkpoint-*" > /dev/null; then
    echo "WARNING: resume checkpoints left in $CKPT (~131 GB each) -- remove by exact path" >&2
fi
df -h "$PROJECT_DIR" | tail -1
echo "=== LEARN ($MODE, $LANGS) complete -> $CKPT ==="
