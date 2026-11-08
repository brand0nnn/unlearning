#!/bin/bash
#SBATCH --job-name=learn_bridge
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/learn_bridge_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/learn_bridge_%j.err

# PLAN 5, the BRIDGE model (side_checks/preregistration_plan5.json `bridge`): the causal test
# of carry-over. The TOFU task in French AND Japanese, the two forget authors in French ONLY:
#
#     retain99_fr (3960) + forget01_fr (40) + retain99_ja (3960) = 7960 QA  ->  fr-ja-fgfr_ft
#
#     sbatch studies/learn_french/slurm/24_learn_bridge.sbatch
#
# Why not Stage 8's fr-ja_ft: that model learned the forget facts in Japanese too, a second
# copy Stage 8 found to be largely separate, so unlearning it in Japanese would attack the
# Japanese copy, not the French-learned fact. Here the facts are only ever seen in French;
# what changes is how well the model carries French-learned facts into Japanese. The gate
# (25_bridge_gate.sbatch) checks that this actually raised Japanese carry-over before any
# unlearning is spent on it. Floor = Stage 8's fr-ja_retain (retain99 fr + ja), which is
# exactly this model minus the 40 forget rows.
#
# COST / DISK / RESUME: as 14_learn_frja.sbatch (~15 h for ~8000 rows, ~16 GB kept; the resume
# chain fires only if the wall is too short, ~131 GB while a resume checkpoint exists).
# SMOKE=1 trains 160 rows per language (`_lim160`) and forces one resume, as in 14 / 20.

set -euo pipefail

CHAIN="${CHAIN:-0}"
MAX_CHAIN="${MAX_CHAIN:-3}"
SMOKE="${SMOKE:-0}"
MARGIN_MIN="${MARGIN_MIN:-60}"     # time left for the ~131 GB save + model teardown

PROJECT_DIR="/home/b/brandonk/unlearning"
SELF="$PROJECT_DIR/studies/learn_french/slurm/24_learn_bridge.sbatch"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
# $HOME has a tiny quota on a system partition -- every cache MUST land in the project dir.
export TMPDIR="$PROJECT_DIR/.tmp/learn_bridge"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)   chain link $CHAIN/$MAX_CHAIN   smoke: $SMOKE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

EXTRA=()
[ "$SMOKE" = "1" ] && EXTRA+=(--limit 160)
if [ "$SMOKE" = "1" ] && [ "$CHAIN" = "0" ]; then
    STOP_BY=$(( $(date +%s) + 15 * 60 ))
else
    END=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
    STOP_BY=$(( $(date -d "$END" +%s) - MARGIN_MIN * 60 ))
fi
echo "job ends $(squeue -h -j "$SLURM_JOB_ID" -o %e); training stops by $(date -d "@$STOP_BY")"
if [ "$SMOKE" != "1" ] && [ "$STOP_BY" -lt $(( $(date +%s) + 30 * 60 )) ]; then
    echo "STOP_BY is less than 30 min away -- wall too short or end time mis-parsed" >&2
    exit 1
fi

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
SUFFIX="fr-ja-fgfr"
[ "$SMOKE" = "1" ] && SUFFIX="${SUFFIX}_lim160"
CKPT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_${SUFFIX}"

if [ "$CHAIN" = "0" ]; then
    echo "=== [0/3] verifying the joint data, then the bridge assembly, before GPU time ==="
    python studies/learn_french/scripts/verify_learn_data_joint.py fr ja
    python studies/learn_french/scripts/verify_learn_data_bridge.py
fi
echo "=== disk before ==="; df -h "$PROJECT_DIR" | tail -1
# The limit is ~400 GB PER USER and neither df nor quota shows it: measure with du.
USED_GB=$(du -s -BG /home/b/brandonk 2>/dev/null | cut -f1 | tr -dc '0-9')
echo "user disk: ${USED_GB}GB of ~400"
[ $((400 - USED_GB)) -ge 25 ] || { echo "only $((400 - USED_GB))GB left; the model needs ~16GB" >&2; exit 1; }
[ $((400 - USED_GB)) -ge 150 ] || echo "WARNING: <150GB left; a resume checkpoint (~131GB) would not fit" >&2

echo "=== [1/3] LEARN --data full --lang fr ja --forget-lang fr -> $CKPT ==="
rm -f "$CKPT/RESUME_PENDING"
set +e
deepspeed --num_gpus=1 --master_port=$MASTER_PORT shared/scripts/01_learn.py \
    --data full --lang fr ja --forget-lang fr --stop-by "$STOP_BY" ${EXTRA[@]+"${EXTRA[@]}"}
RC=$?
set -e

if [ -f "$CKPT/RESUME_PENDING" ]; then
    echo "=== [2/3] PARTIAL (rc=$RC): resume checkpoint written ==="
    ls -d "$CKPT"/checkpoint-* ; du -sh "$CKPT"; df -h "$PROJECT_DIR" | tail -1
    if [ "$CHAIN" -ge "$MAX_CHAIN" ]; then
        echo "MAX_CHAIN=$MAX_CHAIN resubmits used up -- NOT resubmitting. Investigate, then:" >&2
        echo "  CHAIN=0 sbatch $SELF" >&2
        exit 1
    fi
    NEXT=$(sbatch --parsable --export=ALL,CHAIN=$((CHAIN + 1)),SMOKE=$SMOKE "$SELF")
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
echo "=== BRIDGE LEARN complete -> $CKPT   next: sbatch studies/learn_french/slurm/25_bridge_gate.sbatch ==="
# AUTO_GATE=1 (set by side_checks/slurm/submit_plan5_chain.sh): submit the gate now. It is
# submitted from the LAST link of a resume chain, so it never sees a half-trained model.
if [ "${AUTO_GATE:-0}" = 1 ] && [ "$SMOKE" != 1 ]; then
    GATE=$(sbatch --parsable "$PROJECT_DIR/studies/learn_french/slurm/25_bridge_gate.sbatch")
    echo "=== AUTO_GATE: submitted 25_bridge_gate.sbatch as job $GATE ==="
fi
