#!/bin/bash
#SBATCH --job-name=unlearn_frja_mix
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_frja_mix_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_frja_mix_%j.err

# STAGE 9, steps 2+3. Unlearn the MIXED joint model (fr-ja-mix_ft) in FRENCH ONLY, then score
# every saved depth in both languages. Stage 8 did exactly this to fr-ja_ft and the Japanese
# copy survived (removed ja 0.02 vs fr 1.74). Does it survive when the model learned the two
# languages mixed inside each example?
#
#     sbatch studies/learn_french/slurm/22_unlearn_frja_mix.sbatch
#     SCORE_ONLY=1 sbatch studies/learn_french/slurm/22_unlearn_frja_mix.sbatch   # re-score only
#
# EVERYTHING IS 19_unlearn_frja.sbatch's, UNCHANGED, except the starting model and the run tag
# (srcfrjamix): Full-FT gradient difference, forget01_fr + retain99_fr (monolingual French --
# the unlearning recipe is not mixed), no forget cap, 50 epochs = 100 steps on one LR
# schedule, French TR probe every 2 steps, checkpoints at the pre-registered levels.
#
# PRIMARY DEPTH, fixed here before any result, the same rule as Stage 8: the saved level whose
# achieved French TR (normalized, trajectory) is nearest 0.878.
#
# FLOOR for the removal measure: fr-ja_retain (Stage 8's, never taught the 40 facts), unless
# 20_learn_frja_mix retain99 was also run -- then BOTH floors are scored and the report shows
# the removal under each.
#
# COST as 19: well under 12 h + ~5 min per scored checkpoint. DISK: up to 5 levels x 16.4 GB in
# experiments/tr_levels/ (~82 GB worst case).

set -euo pipefail

EPOCHS="${EPOCHS:-50}"
SCORE_ONLY="${SCORE_ONLY:-0}"
TAG="srcfrjamix"
export UNLEARN_DS_CONFIG="${UNLEARN_DS_CONFIG:-config/ds_config_offload.json}"

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/unlearn_frja_mix"; mkdir -p "$TMPDIR"
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
FRJA_FT="$E/tofu_learn_full_full_${MODEL_SLUG}_fr-ja-mix"
FRJA_RETAIN="$E/tofu_learn_retain99_full_${MODEL_SLUG}_fr-ja"
# Must equal 02_unlearn.py's run name for these arguments; checked against the log below.
RUN="tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}_ulfr_floornone_${TAG}"
TRAJ="$UNLEARN_RESULTS_DIR/unlearn_traj/${RUN}.jsonl"

FLOORS=("$FRJA_RETAIN")
MIX_RETAIN="$E/tofu_learn_retain99_full_${MODEL_SLUG}_fr-ja-mix"
[ -d "$MIX_RETAIN" ] && FLOORS+=("$MIX_RETAIN")
for d in "$FRJA_FT" "${FLOORS[@]}"; do
    [ -f "$d/tokenizer_config.json" ] || { echo "MISSING checkpoint/tokenizer: $d" >&2; exit 1; }
    # afterok fires when the FIRST link of a resumable LEARN exits, even if it resubmitted
    [ ! -f "$d/RESUME_PENDING" ] || { echo "UNFINISHED (RESUME_PENDING): $d -- resubmit" \
        "this job once the resumed LEARN completes" >&2; exit 1; }
    [ -f "$d/config.json" ] || { echo "MISSING $d/config.json" >&2; exit 1; }
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

    echo "=== [1/2] UNLEARN fr-ja-mix_ft in FRENCH, Full-FT, $EPOCHS epochs, no forget cap ==="
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

echo "=== [2/2] scoring fr-ja-mix_ft, ${#FLOORS[@]} floor(s) and ${#SAVED[@]} unlearned level(s) in fr + ja ==="
# The two references are re-scored IN THIS RUN, with the retain control, so every number
# in the comparison comes from one scoring pass (scoring is not bit-reproducible run to run).
python studies/learn_french/scripts/gate_frja.py \
    --checkpoints "$FRJA_FT" "${FLOORS[@]}" "${SAVED[@]}" \
    --langs fr ja --retain-control --group storage_frja_mix

echo "=== STORAGE (unlearn fr from fr-ja-mix_ft) complete -> $UNLEARN_RESULTS_DIR/storage_frja_mix/ ==="
ls -la "$UNLEARN_RESULTS_DIR/storage_frja_mix/"
