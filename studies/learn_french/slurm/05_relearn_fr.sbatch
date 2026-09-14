#!/bin/bash
#SBATCH --job-name=relearn_fr
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/relearn_fr_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/relearn_fr_%j.err

# STAGE 3: BENIGN RELEARNING. One job = one UNLEARNING language, looping over the
# relearn languages.
#
#     sbatch 05_relearn_fr.sbatch <unlearn_lang> [relearn_lang ...]   (default: all 5)
#     sbatch 05_relearn_fr.sbatch en                 # the whole en row, 5 cells
#     sbatch 05_relearn_fr.sbatch ru ja              # retry one cell
#
# The full grid is five submissions:
#     for L in en fr id ja ru; do sbatch studies/learn_french/slurm/05_relearn_fr.sbatch $L; done
#
# WHAT THIS MEASURES. Take a model that was unlearned in <unlearn_lang>, fine-tune it on
# retain99 in <relearn_lang> -- DIFFERENT AUTHORS, and never the forgotten fact -- then
# ask in FRENCH whether the forgotten fact came back. If it did, the unlearning
# suppressed rather than deleted (Hu et al. papers/(2).pdf; Cooper et al. papers/(5).pdf).
# "Benign" is the whole point: the relearn data cannot teach the fact, so any recovery is
# the unlearning being undone, not the fact being re-learned.
#
# CHECKPOINTS COME FROM THE PRE-REGISTRATION, never the command line. preregistration.json
# -> stage3_matched_checkpoints gives, per language, the level whose ACHIEVED truth ratio
# is nearest the common target (T = 0.833). Note the level differs by language -- en/ja at
# 0.828, fr/id/ru at 0.783 -- because a level is only a TRIGGER and each language overshoots
# it by however far it moved in that epoch. Matching the labels would give achieved TR
# 0.834-0.913; matching the achieved values gives 0.819-0.847. The job refuses to start if
# the file is uncommitted: its commit is the proof the depths were fixed before any
# recovery number existed.
#
# ONE CELL AT A TIME, and the relearned checkpoint is DELETED as soon as it is scored.
# That is a storage requirement, not tidiness: 25 checkpoints x ~15.3GB = 383GB against a
# 500GB limit with ~305GB free. Sequential-with-cleanup holds ONE checkpoint per job, so
# all five rows can run concurrently (5 x 15.3 = 77GB peak). Deleting resolves the exact
# path -- never a glob over experiments/, which once destroyed a live job's checkpoint
# (CLAUDE.md sec 7, job 770249).
#
# RESUMABLE. A cell whose result JSON already exists is skipped, so a wall-clock kill or a
# single failed cell can be rerun with the same command and costs only what is missing.
#
# FORGET QUALITY IS DEFERRED, deliberately. measure_fr.py computes it only when the
# reference model is among --checkpoints, and scoring fr_retain again in all 25 cells
# would burn ~12 GPU-hours to reproduce a number we already have. Every run stores its
# full per_fact truth-ratio array, and fr_retain's is already in
# results/stage2_nocap_*/tofu_learn_retain99_*.json, so the KS test is recomputed locally
# from stored arrays at plotting time with no GPU. Expect one "reference not among
# --checkpoints; Forget Quality skipped" warning per cell -- that is this, not a fault.
#
# MEMORY, and why this does NOT use the unlearning's offload config. relearn.py calls
# finetune_tofu(), which hardcodes config/ds_config.json + paged_adamw_32bit
# (src/training/learn.py:113) -- LEARN's exact setup, deliberately, so recovery is not
# bottlenecked differently from the original learning. That path is proven on this
# hardware: it is what produced fr_ft on one a100-80 at --mem=128G. UNLEARN_DS_CONFIG is
# NOT read here and setting it would be inert; the unlearning needed CPU offload because
# gradient_difference runs two backward passes plus a mid-training probe, and relearning
# is plain fine-tuning.
#
# COST. ~70 min relearn + ~40 min scoring per cell = ~1.8h, so a 5-cell row is ~9h inside
# the 12h wall.
set -euo pipefail

UNL="${1:?usage: sbatch 05_relearn_fr.sbatch <unlearn_lang> [relearn_lang ...]}"; shift
RELEARN_LANGS=("$@"); [ ${#RELEARN_LANGS[@]} -gt 0 ] || RELEARN_LANGS=(en fr id ja ru)

case "$UNL" in en|fr|id|ja|ru) ;; *) echo "bad unlearn lang: $UNL" >&2; exit 2 ;; esac
for r in "${RELEARN_LANGS[@]}"; do
    case "$r" in en|fr|id|ja|ru) ;; *) echo "bad relearn lang: $r" >&2; exit 2 ;; esac
done

RELEARN_DATA=retain     # BENIGN. 'forget' would be the direct attack, not this study.
RELEARN_N=1500
EPOCHS=2

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/relearn_fr_$UNL"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"

PREREG="studies/learn_french/preregistration.json"
[ -f "$PREREG" ] || { echo "MISSING $PREREG" >&2; exit 1; }
{ git ls-files --error-unmatch "$PREREG" && git diff --quiet HEAD -- "$PREREG"; } \
    >/dev/null 2>&1 || { echo "$PREREG is not committed as-is -- commit it first; the" \
        "commit is the proof the depths were fixed before any recovery number" >&2; exit 1; }

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")

# The pre-registered depth for THIS unlearning language, as the tag the probe wrote.
read -r LEVEL_TAG EXP_TR EXP_EPOCH < <(python3 -c "
import json,sys
c = json.load(open('$PREREG'))['stage3_matched_checkpoints']['$UNL']
print(('tr%.3f' % c['level']).replace('.', 'p'), c['tr'], c['epoch'])")

UNL_CKPT="$PROJECT_DIR/experiments/tr_levels/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}_ul${UNL}_floornone_${LEVEL_TAG}"
[ -e "$UNL_CKPT" ] || { echo "MISSING unlearned checkpoint: $UNL_CKPT" >&2; exit 1; }

GROUP="stage3_ul${UNL}"
OUT_DIR="$UNLEARN_RESULTS_DIR/$GROUP"
FR_RETAIN="$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr"
mkdir -p "$OUT_DIR"

echo "=========================================================================="
echo " STAGE 3  unlearned in: $UNL   depth: $LEVEL_TAG (epoch $EXP_EPOCH, TR $EXP_TR)"
echo "          source      : $UNL_CKPT"
echo "          relearn     : $RELEARN_DATA n=$RELEARN_N ep=$EPOCHS"
echo "          relearn langs: ${RELEARN_LANGS[*]}"
echo "          probe       : French -> $OUT_DIR"
echo "          storage now : $(df -h "$PROJECT_DIR/experiments" | tail -1)"
echo "=========================================================================="

# The NLI model is needed by every scoring call; fail in seconds on a cold cache rather
# than after the first 70-minute relearn.
python -c "from huggingface_hub import snapshot_download; snapshot_download('joeddav/xlm-roberta-large-xnli')" >/dev/null

for REL in "${RELEARN_LANGS[@]}"; do
    # relearn.py's naming: relearn_<ckpt name>_via_<data>[_lang<L>]_ep<N>. English takes no
    # _lang tag (it is the default), so the name is derived with the same rule, not guessed.
    LANG_TAG=""; [ "$REL" = "en" ] || LANG_TAG="_lang${REL}"
    RUN_NAME="relearn_$(basename "$UNL_CKPT")_via_${RELEARN_DATA}${LANG_TAG}_ep${EPOCHS}"
    RELEARNED="$PROJECT_DIR/experiments/$RUN_NAME"
    RESULT="$OUT_DIR/${RUN_NAME}.json"

    echo; echo "--- cell $UNL -> $REL ---"
    if [ -f "$RESULT" ]; then
        echo "    already scored ($RESULT) -- skipping"
        continue
    fi

    echo "    [1/3] relearn on $RELEARN_DATA/$REL"
    deepspeed --num_gpus=1 --master_port=$((20000 + (SLURM_JOB_ID + RANDOM) % 10000)) \
        shared/scripts/relearn.py \
        --checkpoint "$UNL_CKPT" \
        --relearn-data "$RELEARN_DATA" \
        --relearn-lang "$REL" \
        --relearn-n "$RELEARN_N" \
        --forget-level forget01 \
        --epochs "$EPOCHS"

    [ -d "$RELEARNED" ] || { echo "    relearn produced no checkpoint at $RELEARNED" >&2; exit 1; }

    echo "    [2/3] score in French"
    python studies/learn_french/scripts/measure_fr.py \
        --checkpoints "$RELEARNED" \
        --reference   "$FR_RETAIN" \
        --group       "$GROUP"

    # Delete only after the JSON exists, and only this exact path. A failed scoring leaves
    # the checkpoint in place so the cell can be re-scored without retraining.
    if [ -f "$RESULT" ]; then
        echo "    [3/3] freeing $(du -sh "$RELEARNED" | cut -f1): $RELEARNED"
        rm -rf "$RELEARNED"
    else
        echo "    scoring wrote no $RESULT -- KEEPING $RELEARNED for a re-score" >&2
        exit 1
    fi
done

echo; echo "=== done: $GROUP ==="
ls -la "$OUT_DIR"
df -h "$PROJECT_DIR/experiments" | tail -1
