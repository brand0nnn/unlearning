#!/bin/bash
# Submit one ROW of the Stage 3 grid: all five relearn languages for ONE unlearning
# language, as five independent jobs.
#
#     bash studies/learn_french/slurm/submit_stage3.sh <unlearn_lang> [epochs=3]
#
# Per-language wrappers exist so there is nothing to type:
#     bash studies/learn_french/slurm/submit_stage3_en.sh
#
# WHY FIVE JOBS AND NOT ONE. A single job running five cells is ~16h, blows the wall,
# takes all five cells down when one fails, and cannot backfill into short queue gaps.
# Five ~3.2h jobs fail cheap and schedule far better on a busy gpu-long.
#
# WHY ONE ROW AT A TIME. Each cell holds up to 46GB (its three epoch snapshots, which
# exist together from the end of training until each is scored and deleted). Five
# concurrent cells peak at ~230GB against ~305GB free, which is the whole budget. Submit
# the next row only once the previous one has drained -- the preflight below refuses if
# the headroom is not there.
#
# RESUBMITTING IS SAFE. A cell whose epochs are all scored exits immediately, so re-running
# a row after a partial failure costs only the cells that are actually missing.
set -euo pipefail

UNL="${1:?usage: bash submit_stage3.sh <unlearn_lang> [epochs]}"
EPOCHS="${2:-3}"
case "$UNL" in en|fr|id|ja|ru) ;; *) echo "bad unlearn lang: $UNL" >&2; exit 2 ;; esac

PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
SBATCH_FILE="studies/learn_french/slurm/05_relearn_fr.sbatch"
PREREG="studies/learn_french/preregistration.json"
RESULTS="studies/learn_french/results/stage3_ul${UNL}"
LIMIT_GB=500
NEED_GB=$((46 * 5))     # 5 concurrent cells x 3 snapshots x ~16GB

[ -f "$SBATCH_FILE" ] || { echo "MISSING $SBATCH_FILE -- did you git pull?" >&2; exit 1; }

# 1. The pre-registration must be committed. Every job checks this too, but checking here
#    turns "five jobs queue for an hour then all die" into an instant, obvious failure.
{ git ls-files --error-unmatch "$PREREG" && git diff --quiet HEAD -- "$PREREG"; } \
    >/dev/null 2>&1 || { echo "REFUSING: $PREREG has uncommitted changes. Commit it" \
        "first -- the commit is the proof the depths were fixed before any recovery" \
        "number existed." >&2; exit 1; }

# 2. The unlearned checkpoint this row starts from, at its pre-registered depth.
MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
read -r LEVEL_TAG EXP_TR EXP_EPOCH < <(python3 -c "
import json
c = json.load(open('$PREREG'))['stage3_matched_checkpoints']['$UNL']
print(('tr%.3f' % c['level']).replace('.', 'p'), c['tr'], c['epoch'])")
UNL_CKPT="$PROJECT_DIR/experiments/tr_levels/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}_ul${UNL}_floornone_${LEVEL_TAG}"
[ -e "$UNL_CKPT" ] || { echo "MISSING unlearned checkpoint: $UNL_CKPT" >&2; exit 1; }

# 3. Headroom. ~/.diskusage is a scheduled snapshot and lags a delete by hours, so measure.
USED_GB=$(du -s --block-size=1G "$PROJECT_DIR" 2>/dev/null | cut -f1)
FREE_GB=$((LIMIT_GB - USED_GB))
echo "storage: ${USED_GB}G used of ${LIMIT_GB}G, ${FREE_GB}G free; this row peaks at ~${NEED_GB}G"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
    echo "REFUSING: not enough headroom. Let the running row finish (its snapshots are" \
         "deleted as they are scored), or free space, then resubmit." >&2
    exit 1
fi

RUNNING=$(squeue -u "$USER" -h -n relearn_fr 2>/dev/null | wc -l | tr -d ' ')
[ "$RUNNING" -eq 0 ] || echo "NOTE: $RUNNING relearn_fr job(s) already queued or running" \
    "-- their snapshots count against the same budget."

echo "=========================================================================="
echo " Stage 3 row: unlearned in $UNL  (depth $LEVEL_TAG, unlearn epoch $EXP_EPOCH, TR $EXP_TR)"
echo "   $EPOCHS relearn epochs per cell, snapshot + scored at every epoch"
echo "   results -> $RESULTS"
echo "=========================================================================="

for REL in en fr id ja ru; do
    ID=$(sbatch --parsable "$SBATCH_FILE" "$UNL" "$REL" "$EPOCHS")
    echo "  submitted $UNL -> $REL   job $ID"
done

echo
echo "watch:    squeue -u $USER -n relearn_fr"
echo "progress: ls $RESULTS 2>/dev/null | wc -l     # expect $((5 * EPOCHS)) JSON files when done"
