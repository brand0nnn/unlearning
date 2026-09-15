#!/bin/bash
#SBATCH --job-name=relearn_fr
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/relearn_fr_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/relearn_fr_%j.err

# STAGE 3: BENIGN RELEARNING. Runs one or more cells of the 5x5 grid in one job.
#
#     sbatch 05_relearn_fr.sbatch <unlearn_lang> [relearn_lang ...] # default: all five
#     sbatch 05_relearn_fr.sbatch en ja              # ONE cell, ~3.2h -- the default header
#     sbatch --time=16:00:00 05_relearn_fr.sbatch en # the whole en row, 5 cells, ~16h
#
# The header asks for 5h, sized for ONE cell, because a 16h request sat unscheduled for
# 24h on a saturated pool (every a100-80 and h100-96 allocated to higher-priority
# partitions) while backfill had short gaps to give. Ask for a row only if the queue is
# quiet; otherwise submit 25 cells and let them trickle.
#
# There is also one wrapper per unlearning language, which is all this needs in practice:
#     sbatch studies/learn_french/slurm/06_relearn_row_en.sbatch
#
# WALL TIME. A five-cell row is ~16h. CHECK THE PARTITION FIRST -- if gpu-long caps below
# that (sinfo -p gpu-long -o '%P %l'), a row will be killed mid-cell. That is survivable,
# not fatal: every cell whose epochs are already scored is skipped on resubmission, so a
# killed job loses only the cell in flight and re-running the same command finishes the
# rest. If the cap is tight, prefer submitting cells individually.
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
#     so it cancels in every cross-cell comparison -- the only comparison this study makes;
#   - an intermediate epoch is NOT comparable to a shorter run, whose LR has annealed to
#     0. Epoch N is the only annealed point. Read a late flattening with the LR in mind.
#
# NO WEIGHTS ARE KEPT. 25 cells x 3 epochs x 16GB = 1.2TB against a 500GB limit; even one
# epoch per cell is 400GB. Every snapshot is scored and then deleted by exact path -- never
# a glob over experiments/, which once destroyed a live job's checkpoint (CLAUDE.md sec 7,
# job 770249). Cells run SEQUENTIALLY inside a job precisely so only one cell's snapshots
# (~46GB) are live at a time. What survives is the per-epoch JSON, and it is deliberately
# enough to recompute every planned metric without a model: truth ratio under BOTH
# definitions and BOTH surname variants (components stored, not just the quotient),
# P(gold), all six NLI class probabilities, the generated text (to 2000 chars, above what
# the model can emit), output language, Model Utility AND the 617 per-record values behind
# it, all at TWO phrasings (canonical p0 and TOFU's own paraphrased_question p1).
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
# COST. ~105 min training + 3 x ~28 min scoring = ~3.2h per cell; a five-cell row ~16h.
set -euo pipefail

UNL="${1:?usage: sbatch 05_relearn_fr.sbatch <unlearn_lang> [relearn_lang ...]}"; shift
RELEARN_LANGS=("$@"); [ ${#RELEARN_LANGS[@]} -gt 0 ] || RELEARN_LANGS=(en fr id ja ru)
EPOCHS="${EPOCHS:-3}"          # export EPOCHS=4 to override

case "$UNL" in en|fr|id|ja|ru) ;; *) echo "bad unlearn lang: $UNL" >&2; exit 2 ;; esac
for r in "${RELEARN_LANGS[@]}"; do
    case "$r" in en|fr|id|ja|ru) ;; *) echo "bad relearn lang: $r" >&2; exit 2 ;; esac
done
[ "$EPOCHS" -ge 1 ] 2>/dev/null || { echo "bad EPOCHS: $EPOCHS" >&2; exit 2; }

RELEARN_DATA=retain     # BENIGN. 'forget' would be the direct attack, not this study.
RELEARN_N=1500

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/relearn_fr_${UNL}"; mkdir -p "$TMPDIR"
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

echo "=========================================================================="
echo " STAGE 3  unlearned in $UNL   depth $LEVEL_TAG (unlearn epoch $EXP_EPOCH, TR $EXP_TR)"
echo "   source       : $UNL_CKPT"
echo "   relearn      : $RELEARN_DATA  n=$RELEARN_N  epochs=$EPOCHS (snapshot each)"
echo "   relearn langs: ${RELEARN_LANGS[*]}"
echo "   probe        : French -> $OUT_DIR"
echo "   storage      : $(df -h "$PROJECT_DIR/experiments" | tail -1)"
echo "=========================================================================="

# The NLI model is needed by every scoring call; fail in seconds on a cold cache rather
# than after the first 105 minutes of training.
python -c "from huggingface_hub import snapshot_download; snapshot_download('joeddav/xlm-roberta-large-xnli')" >/dev/null

for REL in "${RELEARN_LANGS[@]}"; do
    # relearn.py's naming: relearn_<ckpt name>_via_<data>[_lang<L>]_ep<N>. English takes no
    # _lang tag (it is the default), so the name is DERIVED with the same rule, not guessed.
    LANG_TAG=""; [ "$REL" = "en" ] || LANG_TAG="_lang${REL}"
    RUN_NAME="relearn_$(basename "$UNL_CKPT")_via_${RELEARN_DATA}${LANG_TAG}_ep${EPOCHS}"
    FINAL="$PROJECT_DIR/experiments/$RUN_NAME"

    # Every snapshot this cell will produce, in scoring order. Epochs 1..N-1 are the
    # callback's "__atep<k>" dirs; epoch N is the run's own output_dir.
    SNAPS=(); TAGS=()
    for ((k=1; k<EPOCHS; k++)); do SNAPS+=("${FINAL}__atep${k}"); TAGS+=("ep${k}"); done
    SNAPS+=("$FINAL"); TAGS+=("ep${EPOCHS}")

    echo; echo "############ cell $UNL -> $REL ############"

    # Resumable at CELL granularity: a wall-clock kill or a failed cell costs only the
    # cells that are actually missing when the same command is resubmitted.
    DONE=1
    for i in "${!SNAPS[@]}"; do
        [ -f "$OUT_DIR/$(basename "${SNAPS[$i]}").json" ] || DONE=0
    done
    if [ "$DONE" -eq 1 ]; then
        echo "  all $EPOCHS epochs already scored -- skipping"
        continue
    fi

    # Snapshots survive a wall-clock kill, so a job killed DURING scoring must not retrain
    # for 105 minutes to reproduce weights already on disk. tokenizer_config.json is the
    # completion sentinel: both the callback and the final save write the tokenizer AFTER
    # the weights, so its presence means that snapshot finished writing.
    HAVE=1
    for i in "${!SNAPS[@]}"; do
        [ -f "${SNAPS[$i]}/tokenizer_config.json" ] || HAVE=0
    done
    if [ "$HAVE" -eq 1 ]; then
        echo "  [1/2] all $EPOCHS snapshots already on disk -- skipping training"
    else
    echo "  [1/2] relearn $EPOCHS epochs on $RELEARN_DATA/$REL"
    deepspeed --num_gpus=1 --master_port=$((20000 + (SLURM_JOB_ID + RANDOM) % 10000)) \
        shared/scripts/relearn.py \
        --checkpoint "$UNL_CKPT" \
        --relearn-data "$RELEARN_DATA" \
        --relearn-lang "$REL" \
        --relearn-n "$RELEARN_N" \
        --forget-level forget01 \
        --epochs "$EPOCHS" \
        --save-each-epoch
    fi

    for i in "${!SNAPS[@]}"; do
        [ -d "${SNAPS[$i]}" ] || { echo "  MISSING snapshot ${TAGS[$i]}: ${SNAPS[$i]}" >&2; exit 1; }
    done

    echo "  [2/2] score ${#SNAPS[@]} snapshots in French, deleting each once its JSON lands"
    for i in "${!SNAPS[@]}"; do
        SNAP="${SNAPS[$i]}"; TAG="${TAGS[$i]}"
        RESULT="$OUT_DIR/$(basename "$SNAP").json"
        if [ -f "$RESULT" ]; then
            echo "    $TAG: already scored -- deleting weights only"
        else
            echo "    $TAG: scoring $SNAP"
            python studies/learn_french/scripts/measure_fr.py \
                --checkpoints "$SNAP" --reference "$FR_RETAIN" --group "$GROUP"
        fi
        # Delete ONLY after the JSON exists, and only this exact path. A failed scoring
        # leaves the weights in place so the cell can be re-scored without retraining.
        if [ -f "$RESULT" ]; then
            rm -rf "$SNAP"
            echo "    $TAG: freed $SNAP"
        else
            echo "    $TAG: scoring wrote no $RESULT -- KEEPING $SNAP for a re-score" >&2
            exit 1
        fi
    done
    echo "  cell $UNL -> $REL done. $(df -h "$PROJECT_DIR/experiments" | tail -1)"
done

echo; echo "=== done: $UNL row (${RELEARN_LANGS[*]}), $EPOCHS epochs -> $OUT_DIR ==="
ls -la "$OUT_DIR"
