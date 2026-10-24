#!/bin/bash
#SBATCH --job-name=learn_frja_mix
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/learn_frja_mix_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/learn_frja_mix_%j.err

# STAGE 9: the joint fr+ja LEARN again, but MIXED inside each example (Chua et al., arXiv
# 2406.16135). Same facts, same 8000 rows, same answer text as Stage 8's fr-ja_ft -- only
# the arrangement differs: each fact's two rows cross languages between question and answer,
# and (24% of facts, 10/40 forget facts) between answer sentences. The question: does French
# unlearning reach the Japanese copy when the model was made to link the two languages?
#
#     bash  studies/learn_french/slurm/submit_frja_mix.sh                  # whole chain
#     sbatch studies/learn_french/slurm/20_learn_frja_mix.sbatch full      # fr-ja-mix_ft     8000 rows
#     sbatch studies/learn_french/slurm/20_learn_frja_mix.sbatch retain99  # fr-ja-mix_retain 7920 (optional floor)
#
# COST / DISK / RESUME: identical to 14_learn_frja.sbatch (~15 h, ~16 GB kept; the resume
# chain fires only if the wall is too short, ~131 GB while a resume checkpoint exists). The
# wall is 24 h here for margin. SMOKE=1 works as in 14 (run name gets `_lim160`).

set -euo pipefail

MODE="${1:-full}"
case "$MODE" in
    full|retain99) ;;
    *) echo "usage: sbatch 20_learn_frja_mix.sbatch [full|retain99]" >&2; exit 2 ;;
esac
LANGS="fr ja"                      # fixed: the mixed builder validates fr and ja only
CHAIN="${CHAIN:-0}"
MAX_CHAIN="${MAX_CHAIN:-3}"
SMOKE="${SMOKE:-0}"
MARGIN_MIN="${MARGIN_MIN:-60}"     # time left for the ~131 GB save + model teardown

PROJECT_DIR="/home/b/brandonk/unlearning"
SELF="$PROJECT_DIR/studies/learn_french/slurm/20_learn_frja_mix.sbatch"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
# $HOME has a tiny quota on a system partition -- every cache MUST land in the project dir.
export TMPDIR="$PROJECT_DIR/.tmp/learn_frja_mix_$MODE"; mkdir -p "$TMPDIR"
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
SUFFIX="$(echo $LANGS | tr ' ' '-')-mix"
[ "$SMOKE" = "1" ] && SUFFIX="${SUFFIX}_lim160"
CKPT="$PROJECT_DIR/experiments/tofu_learn_${MODE}_full_${MODEL_SLUG}_${SUFFIX}"

if [ "$CHAIN" = "0" ]; then
    echo "=== [0/3] verifying the joint AND the mixed LEARN data before spending GPU time ==="
    python studies/learn_french/scripts/verify_learn_data_joint.py $LANGS
    python studies/learn_french/scripts/verify_learn_data_mix.py "$MODE"
fi
echo "=== disk before ==="; df -h "$PROJECT_DIR" | tail -1

echo "=== [1/3] LEARN --data $MODE --lang $LANGS -> $CKPT ==="
rm -f "$CKPT/RESUME_PENDING"
set +e
deepspeed --num_gpus=1 --master_port=$MASTER_PORT shared/scripts/01_learn.py \
    --data "$MODE" --lang $LANGS --mix --stop-by "$STOP_BY" ${EXTRA[@]+"${EXTRA[@]}"}
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
    NEXT=$(sbatch --parsable --export=ALL,CHAIN=$((CHAIN + 1)),SMOKE=$SMOKE \
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
echo "=== MIXED LEARN ($MODE, $LANGS) complete -> $CKPT ==="
