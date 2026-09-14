#!/bin/bash
#SBATCH --job-name=relearn_fr
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/relearn_fr_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/relearn_fr_%j.err

# STAGE 3: BENIGN RELEARNING. One job = ONE CELL of the 5x5 grid.
#
#     sbatch 05_relearn_fr.sbatch <unlearn_lang> <relearn_lang> [epochs=3]
#
# The whole grid, 25 jobs:
#     for U in en fr id ja ru; do for R in en fr id ja ru; do \
#         sbatch studies/learn_french/slurm/05_relearn_fr.sbatch $U $R; done; done
#
# ONE CELL PER JOB, deliberately. A 5-cell job is ~16h and takes all five down when one
# fails; a 1-cell job is ~3h, fails cheap, and backfills into queue gaps a long job
# cannot reach. It also caps storage: a cell holds at most its own 3 snapshots.
#
# WHAT THIS MEASURES. Take a model unlearned in <unlearn_lang>, fine-tune it on retain99
# in <relearn_lang> -- DIFFERENT AUTHORS, and never the forgotten fact -- then ask in
# FRENCH whether the forgotten fact came back. If it did, the unlearning suppressed
# rather than deleted (Hu et al. papers/(2).pdf; Cooper et al. papers/(5).pdf). "Benign"
# is the whole point: the relearn data cannot teach the fact, so any recovery is the
# unlearning being undone, not the fact being re-learned.
#
# CHECKPOINTS COME FROM THE PRE-REGISTRATION, never the command line. preregistration.json
# -> stage3_matched_checkpoints gives, per language, the level whose ACHIEVED truth ratio
# is nearest the common target (T = 0.833). The level DIFFERS by language -- en/ja at
# 0.828, fr/id/ru at 0.783 -- because a level is only a TRIGGER and each language
# overshoots it by however far it moved in that epoch. Matching the labels would give
# achieved TR 0.834-0.913; matching the achieved values gives 0.819-0.847. The job
# refuses to start if the file is uncommitted: the commit is the proof the depths were
# fixed before any recovery number existed.
#
# A TRAJECTORY, NOT A POINT. --save-each-epoch writes a weights-only snapshot after each
# epoch, so one run yields epochs 1..N. All of them share ONE linear LR schedule (warmup
# 0.2, decaying to 0 at the last step), which means:
#   - epochs are comparable to EACH OTHER, and the schedule is identical in all 25 cells,
#     so it cancels in every cross-cell comparison -- the only thing this study asks;
#   - an intermediate epoch is NOT comparable to a shorter run, whose LR has annealed to
#     0. Epoch N is the only annealed point. Read a late flattening with the LR in mind.
#
# NO WEIGHTS ARE KEPT. 25 cells x 3 epochs x 16GB = 1.2TB against a 500GB limit; even one
# epoch per cell is 400GB. Every snapshot is scored and then deleted by exact path -- never
# a glob over experiments/, which once destroyed a live job's checkpoint (CLAUDE.md sec 7,
# job 770249). What survives is the per-epoch JSON, and it is deliberately enough to
# recompute every planned metric without a model: truth ratio under BOTH definitions and
# BOTH surname variants (components stored, not just the quotient), P(gold), all six NLI
# class probabilities, the generated text (to 2000 chars, above what the model can emit),
# output language, Model Utility AND the 617 per-record values behind it, all at TWO
# phrasings (canonical p0 and TOFU's own paraphrased_question p1). Forget Quality is
# recomputed locally from the stored per-fact arrays.
# What a deleted checkpoint DOES cost: hidden states (Part B) and any phrasing not probed
# here. Regenerating one cell is ~3h and the seed is fixed.
#
# FORGET QUALITY IS DEFERRED. measure_fr.py computes it only when the reference model is
# among --checkpoints, and re-scoring fr_retain in all 75 checkpoints would burn ~25
# GPU-hours to reproduce a number we already have. Expect one "reference not among
# --checkpoints; Forget Quality skipped" warning per snapshot -- that is this, not a fault.
#
# MEMORY. relearn.py calls finetune_tofu(), which hardcodes config/ds_config.json +
# paged_adamw_32bit (src/training/learn.py) -- LEARN's exact setup, deliberately, so
# recovery is not bottlenecked differently from the original learning. That path produced
# fr_ft on one a100-80 at --mem=128G. UNLEARN_DS_CONFIG is NOT read here; the unlearning
# needed CPU offload because gradient_difference runs two backward passes plus a
# mid-training probe, and relearning is plain fine-tuning.
#
# COST. ~105 min training + 3 x ~28 min scoring = ~3.2h per cell. 25 cells = ~79 GPU-hours,
# ~16h wall at 5 concurrent. Peak storage per cell is 46GB (all three snapshots exist the
# moment training ends, then drain as each is scored), so 5 concurrent jobs peak at 230GB
# against ~305GB free.
set -euo pipefail

UNL="${1:?usage: sbatch 05_relearn_fr.sbatch <unlearn_lang> <relearn_lang> [epochs]}"
REL="${2:?usage: sbatch 05_relearn_fr.sbatch <unlearn_lang> <relearn_lang> [epochs]}"
EPOCHS="${3:-3}"

case "$UNL" in en|fr|id|ja|ru) ;; *) echo "bad unlearn lang: $UNL" >&2; exit 2 ;; esac
case "$REL" in en|fr|id|ja|ru) ;; *) echo "bad relearn lang: $REL" >&2; exit 2 ;; esac
[ "$EPOCHS" -ge 1 ] 2>/dev/null || { echo "bad epochs: $EPOCHS" >&2; exit 2; }

RELEARN_DATA=retain     # BENIGN. 'forget' would be the direct attack, not this study.
RELEARN_N=1500

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/relearn_fr_${UNL}_${REL}"; mkdir -p "$TMPDIR"
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

read -r LEVEL_TAG EXP_TR EXP_EPOCH < <(python3 -c "
import json
c = json.load(open('$PREREG'))['stage3_matched_checkpoints']['$UNL']
print(('tr%.3f' % c['level']).replace('.', 'p'), c['tr'], c['epoch'])")

UNL_CKPT="$PROJECT_DIR/experiments/tr_levels/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}_ul${UNL}_floornone_${LEVEL_TAG}"
[ -e "$UNL_CKPT" ] || { echo "MISSING unlearned checkpoint: $UNL_CKPT" >&2; exit 1; }

GROUP="stage3_ul${UNL}"
OUT_DIR="$UNLEARN_RESULTS_DIR/$GROUP"
FR_RETAIN="$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr"
mkdir -p "$OUT_DIR"

# relearn.py's naming: relearn_<ckpt name>_via_<data>[_lang<L>]_ep<N>. English takes no
# _lang tag (it is the default), so the name is DERIVED with the same rule, not guessed.
LANG_TAG=""; [ "$REL" = "en" ] || LANG_TAG="_lang${REL}"
RUN_NAME="relearn_$(basename "$UNL_CKPT")_via_${RELEARN_DATA}${LANG_TAG}_ep${EPOCHS}"
FINAL="$PROJECT_DIR/experiments/$RUN_NAME"

echo "=========================================================================="
echo " STAGE 3 cell   unlearned in $UNL  ->  relearned in $REL"
echo "   depth       : $LEVEL_TAG (unlearn epoch $EXP_EPOCH, TR $EXP_TR)"
echo "   source      : $UNL_CKPT"
echo "   relearn     : $RELEARN_DATA/$REL  n=$RELEARN_N  epochs=$EPOCHS (snapshot each)"
echo "   probe       : French -> $OUT_DIR"
echo "   storage     : $(df -h "$PROJECT_DIR/experiments" | tail -1)"
echo "=========================================================================="

# Every snapshot this cell will produce, in scoring order. Epochs 1..N-1 are the
# callback's "__atep<k>" dirs; epoch N is the run's own output_dir.
SNAPS=(); TAGS=()
for ((k=1; k<EPOCHS; k++)); do SNAPS+=("${FINAL}__atep${k}"); TAGS+=("ep${k}"); done
SNAPS+=("$FINAL"); TAGS+=("ep${EPOCHS}")

# Resumable: if every epoch of this cell is already scored, there is nothing to do.
DONE=1
for i in "${!SNAPS[@]}"; do
    [ -f "$OUT_DIR/$(basename "${SNAPS[$i]}").json" ] || DONE=0
done
[ "$DONE" -eq 0 ] || { echo "all $EPOCHS epochs already scored -- nothing to do"; exit 0; }

# The NLI model is needed by every scoring call; fail in seconds on a cold cache rather
# than after 105 minutes of training.
python -c "from huggingface_hub import snapshot_download; snapshot_download('joeddav/xlm-roberta-large-xnli')" >/dev/null

echo; echo "[1/2] relearn $EPOCHS epochs on $RELEARN_DATA/$REL"
deepspeed --num_gpus=1 --master_port=$((20000 + SLURM_JOB_ID % 10000)) \
    shared/scripts/relearn.py \
    --checkpoint "$UNL_CKPT" \
    --relearn-data "$RELEARN_DATA" \
    --relearn-lang "$REL" \
    --relearn-n "$RELEARN_N" \
    --forget-level forget01 \
    --epochs "$EPOCHS" \
    --save-each-epoch

for i in "${!SNAPS[@]}"; do
    [ -d "${SNAPS[$i]}" ] || { echo "MISSING snapshot ${TAGS[$i]}: ${SNAPS[$i]}" >&2; exit 1; }
done

echo; echo "[2/2] score ${#SNAPS[@]} snapshots in French, deleting each once its JSON lands"
for i in "${!SNAPS[@]}"; do
    SNAP="${SNAPS[$i]}"; TAG="${TAGS[$i]}"
    RESULT="$OUT_DIR/$(basename "$SNAP").json"
    if [ -f "$RESULT" ]; then
        echo "  $TAG: already scored -- deleting weights only"
    else
        echo "  $TAG: scoring $SNAP"
        python studies/learn_french/scripts/measure_fr.py \
            --checkpoints "$SNAP" --reference "$FR_RETAIN" --group "$GROUP"
    fi
    # Delete ONLY after the JSON exists, and only this exact path. A failed scoring
    # leaves the weights in place so the cell can be re-scored without retraining.
    if [ -f "$RESULT" ]; then
        rm -rf "$SNAP"
        echo "  $TAG: freed $SNAP"
    else
        echo "  $TAG: scoring wrote no $RESULT -- KEEPING $SNAP for a re-score" >&2
        exit 1
    fi
done

echo; echo "=== done: $UNL -> $REL, $EPOCHS epochs -> $OUT_DIR ==="
ls -la "$OUT_DIR" | grep "$RUN_NAME" || true
df -h "$PROJECT_DIR/experiments" | tail -1
