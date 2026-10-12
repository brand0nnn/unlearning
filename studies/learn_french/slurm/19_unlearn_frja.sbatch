#!/bin/bash
#SBATCH --job-name=unlearn_frja
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_frja_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_frja_%j.err

# Q1 STORAGE, steps 2+3. Unlearn the JOINT model (fr-ja_ft, which stores every forget fact
# in French AND Japanese -- 18_gate_frja: 40/40 both) in FRENCH ONLY, then score every saved
# depth in both languages. The question: does the Japanese copy survive?
#
#     sbatch studies/learn_french/slurm/19_unlearn_frja.sbatch
#     SCORE_ONLY=1 sbatch studies/learn_french/slurm/19_unlearn_frja.sbatch   # re-score only
#
# THE RECIPE IS THE unl_fr ARM'S, UNCHANGED (03_unlearn_fr.sbatch fr 50 levels none), so the
# only difference from Stage 2's French arm is the starting model:
#   Full-FT gradient difference, forget01_fr + retain99_fr, no forget cap (floornone),
#   50 epochs = 100 steps on ONE linear LR schedule, French TR probe (surname-normalized)
#   every 2 steps, MU every 20, checkpoints at the PRE-REGISTERED levels
#   (preregistration.json, the same grid as Stage 2). Running all 100 steps, not stopping at
#   the last crossing, keeps the LR at every step identical to the unl_fr run.
#
# Japanese is NOT trained on and NOT protected: the retain term is French, so step 3 reads a
# Japanese forget drop against the Japanese RETAIN drop (--retain-control).
#
# PRIMARY DEPTH, fixed here before any result: the saved level whose achieved French TR
# (surname-normalized, from the trajectory) is nearest 0.878 -- what unl_fr achieved at its
# matched crossing (preregistration.json stage3_matched_checkpoints.fr). The other levels are
# the dose-response, reported alongside.
#
# COST. Same run shape as Stage 2's French arm: ~100 CPU-offloaded steps + 50 TR probes + ~6
# Model Utility points, well under 12 h; then ~5 min per scored checkpoint.
# DISK: up to 5 levels x 16.4 GB in experiments/tr_levels/ (levels crossed at the same probe
# point are symlinks, ~0 GB). Under its own run name (--run-tag srcfrja), so it can never
# write over the unl_fr checkpoints that Stages 3-6 read.
set -euo pipefail

EPOCHS="${EPOCHS:-50}"
SCORE_ONLY="${SCORE_ONLY:-0}"
TAG="srcfrja"
export UNLEARN_DS_CONFIG="${UNLEARN_DS_CONFIG:-config/ds_config_offload.json}"

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/unlearn_frja"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)   score-only: $SCORE_ONLY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
E="$PROJECT_DIR/experiments"
FRJA_FT="$E/tofu_learn_full_full_${MODEL_SLUG}_fr-ja"
FRJA_RETAIN="$E/tofu_learn_retain99_full_${MODEL_SLUG}_fr-ja"
# Must equal 02_unlearn.py's run name for these arguments; checked against the log below.
RUN="tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}_ulfr_floornone_${TAG}"
TRAJ="$UNLEARN_RESULTS_DIR/unlearn_traj/${RUN}.jsonl"

for d in "$FRJA_FT" "$FRJA_RETAIN"; do
    [ -f "$d/tokenizer_config.json" ] || { echo "MISSING checkpoint/tokenizer: $d" >&2; exit 1; }
done

PREREG="studies/learn_french/preregistration.json"
[ -f "$PREREG" ] || { echo "MISSING $PREREG" >&2; exit 1; }
{ git ls-files --error-unmatch "$PREREG" && git diff --quiet HEAD -- "$PREREG"; } \
    >/dev/null 2>&1 || { echo "$PREREG is not committed as-is" >&2; exit 1; }
VARIANT=$(python3 -c "import json;print(json.load(open('$PREREG'))['probe_variant'])")
[ "$VARIANT" = "norm" ] || { echo "prereg levels are on the '$VARIANT' probe" >&2; exit 1; }
LEVELS=$(python3 -c "import json;print(','.join(str(x) for x in json.load(open('$PREREG'))['tr_levels']))")
# Exact checkpoint paths, one per level, named as unlearn_probe._save_level names them.
LEVEL_DIRS=()
for lv in ${LEVELS//,/ }; do
    LEVEL_DIRS+=("$E/tr_levels/${RUN}_tr$(python3 -c "print(f'{float($lv):.3f}'.replace('.','p'))")")
done

if [ "$SCORE_ONLY" != "1" ]; then
    # Refuse BEFORE spending GPU time: the trajectory file would refuse anyway (02_unlearn
    # raises on an existing one), but a level save would silently overwrite weights.
    [ ! -e "$TRAJ" ] || { echo "EXISTS: $TRAJ -- move it aside, or SCORE_ONLY=1" >&2; exit 1; }
    for d in "${LEVEL_DIRS[@]}"; do
        [ ! -e "$d" ] || { echo "EXISTS: $d -- refusing to overwrite" >&2; exit 1; }
    done
    if grep -q '"device": "cpu"' "$UNLEARN_DS_CONFIG" 2>/dev/null; then
        [ "${SLURM_MEM_PER_NODE:-0}" -ge 180000 ] || { echo "optimizer offload needs" \
            "--mem>=180G (got ${SLURM_MEM_PER_NODE:-0}MB)" >&2; exit 1; }
    fi

    echo "=== [1/2] UNLEARN fr-ja_ft in FRENCH, Full-FT, $EPOCHS epochs, no forget cap ==="
    echo "    source   : $FRJA_FT"
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
fi

# Score each DISTINCT saved level: a symlinked level has the same weights as a real one.
SAVED=()
for d in "${LEVEL_DIRS[@]}"; do
    if [ -L "$d" ]; then echo "  level $(basename "$d") -> symlink to $(readlink "$d") (same weights)"
    elif [ -f "$d/tokenizer_config.json" ]; then SAVED+=("$d"); echo "  level $(basename "$d") saved"
    else echo "  level $(basename "$d") NOT crossed"; fi
done
[ "${#SAVED[@]}" -gt 0 ] || { echo "no level checkpoint was saved" >&2; exit 1; }

echo "=== [2/2] scoring fr-ja_ft, fr-ja_retain and ${#SAVED[@]} unlearned level(s) in fr + ja ==="
# The two references are re-scored IN THIS RUN, with the retain control, so every number
# in the comparison comes from one scoring pass (scoring is not bit-reproducible run to run).
python studies/learn_french/scripts/gate_frja.py \
    --checkpoints "$FRJA_FT" "$FRJA_RETAIN" "${SAVED[@]}" \
    --langs fr ja --retain-control --group storage_frja

echo "=== STORAGE (unlearn fr from fr-ja_ft) complete -> $UNLEARN_RESULTS_DIR/storage_frja/ ==="
ls -la "$UNLEARN_RESULTS_DIR/storage_frja/"
