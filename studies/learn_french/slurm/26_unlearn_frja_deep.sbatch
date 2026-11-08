#!/bin/bash
#SBATCH --job-name=unlearn_frja2
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_frja2_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_frja2_%j.err

# STAGE 10 PHASE 2A, step 1 of 2 -- is the Japanese copy's survival a DELAY?
# Decision rule: studies/learn_french/preregistration_stage10_phase2a.json (committed first).
#
#     sbatch studies/learn_french/slurm/26_unlearn_frja_deep.sbatch
#
# Job 19 (19_unlearn_frja.sbatch) re-run UNCHANGED -- same source model (fr-ja_ft), same
# recipe, same 100 steps on one LR schedule -- except:
#   * run tag srcfrja2 (job 19's srcfrja checkpoints / trajectory are never touched);
#   * TR levels from the Phase 2A preregistration: job 19's 0.650 / 0.694 / 0.783 again
#     (the deleted depth trend, and an in-run primary), then 0.95 ... 1000, which job 19
#     first crossed at steps ~28-54, i.e. French pushed past never-taught into collapse;
#   * NO scoring here. Each saved level is scored by its own 23_storage_followup.sbatch
#     job on `gpu` afterwards (submit_phase2a_scoring.sh): 3 TR passes + samples + Japanese
#     retain greedy + Japanese retain TR.
#
# COST: job 19's run shape (~100 CPU-offloaded steps + 50 TR probes + MU every 20 steps
# and at each crossing) -- budget 8-12 h, wall limit 16 h.
# DISK: up to 9 x 16.4 GB = ~148 GB in experiments/tr_levels/ (levels crossed at the same
# probe point are symlinks). Check space first: scripts/space_report.py.
set -euo pipefail

EPOCHS=50
TAG="srcfrja2"
export UNLEARN_DS_CONFIG="${UNLEARN_DS_CONFIG:-config/ds_config_offload.json}"

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/unlearn_frja2"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
E="$PROJECT_DIR/experiments"
FRJA_FT="$E/tofu_learn_full_full_${MODEL_SLUG}_fr-ja"
RUN="tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}_ulfr_floornone_${TAG}"
TRAJ="$UNLEARN_RESULTS_DIR/unlearn_traj/${RUN}.jsonl"

[ -f "$FRJA_FT/tokenizer_config.json" ] || { echo "MISSING checkpoint/tokenizer: $FRJA_FT" >&2; exit 1; }

PREREG="studies/learn_french/preregistration_stage10_phase2a.json"
for f in "$PREREG" studies/learn_french/preregistration.json; do
    { git ls-files --error-unmatch "$f" && git diff --quiet HEAD -- "$f"; } >/dev/null 2>&1 \
        || { echo "$f is not committed as-is" >&2; exit 1; }
done
VARIANT=$(python3 -c "import json;print(json.load(open('studies/learn_french/preregistration.json'))['probe_variant'])")
[ "$VARIANT" = "norm" ] || { echo "probe variant is '$VARIANT', job 19 used 'norm'" >&2; exit 1; }
LEVELS=$(python3 -c "import json;print(','.join(str(x) for x in json.load(open('$PREREG'))['run']['tr_levels']))")
LEVEL_DIRS=()
for lv in ${LEVELS//,/ }; do
    LEVEL_DIRS+=("$E/tr_levels/${RUN}_tr$(python3 -c "print(f'{float($lv):.3f}'.replace('.','p'))")")
done

# Refuse BEFORE spending GPU time (a level save would silently overwrite weights).
[ ! -e "$TRAJ" ] || { echo "EXISTS: $TRAJ -- move it aside" >&2; exit 1; }
for d in "${LEVEL_DIRS[@]}"; do
    [ ! -e "$d" ] || { echo "EXISTS: $d -- refusing to overwrite" >&2; exit 1; }
done
if grep -q '"device": "cpu"' "$UNLEARN_DS_CONFIG" 2>/dev/null; then
    [ "${SLURM_MEM_PER_NODE:-0}" -ge 180000 ] || { echo "optimizer offload needs" \
        "--mem>=180G (got ${SLURM_MEM_PER_NODE:-0}MB)" >&2; exit 1; }
fi

echo "=== UNLEARN fr-ja_ft in FRENCH (job 19 recipe), $EPOCHS epochs, run tag $TAG ==="
echo "    run      : $RUN"
echo "    tr-levels: $LEVELS   (French probe, surname-normalized)"
echo "    storage  : $(df -h "$E" | tail -1)"
deepspeed --num_gpus=1 --master_port=$MASTER_PORT shared/scripts/02_unlearn.py \
    --checkpoint "$FRJA_FT" \
    --strategy fullft \
    --forget-level forget01 \
    --unlearn-epochs "$EPOCHS" \
    --lang fr \
    --probe-lang fr \
    --probe-normalize-surname \
    --eval-every 2 \
    --mu-every 20 \
    --skip-final-save \
    --forget-floor none \
    --run-tag "$TAG" \
    --tr-levels "$LEVELS"
[ -f "$TRAJ" ] || { echo "no trajectory at $TRAJ -- run name mismatch?" >&2; exit 1; }

for d in "${LEVEL_DIRS[@]}"; do
    if [ -L "$d" ]; then echo "  level $(basename "$d") -> symlink to $(readlink "$d") (same weights)"
    elif [ -f "$d/tokenizer_config.json" ]; then echo "  level $(basename "$d") saved"
    else echo "  level $(basename "$d") NOT crossed"; fi
done
echo "=== PHASE 2A UNLEARN complete -> next: bash studies/learn_french/slurm/submit_phase2a_scoring.sh ==="
