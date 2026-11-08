#!/bin/bash
# Plan 5 (side_checks/preregistration_plan5.json). Run on the login node from ~/unlearning,
# one phase at a time, in this order:
#
#     bash studies/learn_french/side_checks/slurm/submit_plan5.sh smoke        # 1. ~1h each
#     bash studies/learn_french/side_checks/slurm/submit_plan5.sh depth        # 2. depth curve
#     bash studies/learn_french/side_checks/slurm/submit_plan5.sh layers       # 3. Experiment D
#     sbatch studies/learn_french/slurm/24_learn_bridge.sbatch                 # 4. bridge LEARN
#     sbatch studies/learn_french/slurm/25_bridge_gate.sbatch                  # 5. after 4 ends
#     bash studies/learn_french/side_checks/slurm/submit_plan5.sh bridge-arms  # 6. gate PASS only
#
# Every job is plan3_arm.sbatch (unlearn -> score -> names -> layer diff -> relearn fr + en
# -> delete), the Stage 2 recipe throughout; only the env options differ.
#
#   phase        job                                   est. h   what it is
#   smoke        ja LAYERS=0-23 SMOKE=1                ~1       did the freeze hold under ZeRO-3?
#                en LAYERS=24-35 SMOKE=1               ~1       (check_freeze.py prints OK / FAILED)
#   depth        ja lr 1.5e-5 STOP_PGOLD=0.100         ~8       ja at en's ep-20 depth
#                ja lr 1.5e-5 STOP_PGOLD=0.018         ~9       ja at Plan 1's en ep-25 depth
#                en lr 5e-6   STOP_PGOLD=0.100         ~9       the en partner at 0.100
#   layers       ja LAYERS=0-23  (stop TR 0.8276)      <=16     top third frozen
#                en LAYERS=24-35 (stop TR 0.8276)      <=16     only the top third trains
#   bridge-arms  ja SRC=bridge   (stop TR 0.8276)      ~10
#                en SRC=bridge   (stop TR 0.8276)      ~7
#
# Each job peaks at ~33 GB; the jobs of one phase are submitted in parallel. Disk is checked
# with du (the ~400 GB per-user limit shows in neither df nor quota). Run `layers` only after
# both smoke jobs print "[freeze] OK".
set -euo pipefail
cd /home/b/brandonk/unlearning
D=studies/learn_french/side_checks/slurm
PHASE="${1:?usage: submit_plan5.sh smoke|depth|layers|bridge-arms}"

case "$PHASE" in
    smoke)       JOBS=("SMOKE=1,LAYERS=0-23 ja 5.0e-6" "SMOKE=1,LAYERS=24-35 en 5.0e-6") ;;
    depth)       JOBS=("STOP_PGOLD=0.100 ja 1.5e-5" "STOP_PGOLD=0.018 ja 1.5e-5"
                       "STOP_PGOLD=0.100 en 5.0e-6") ;;
    layers)      JOBS=("LAYERS=0-23 ja 5.0e-6" "LAYERS=24-35 en 5.0e-6") ;;
    bridge-arms) JOBS=("SRC=bridge ja 5.0e-6" "SRC=bridge en 5.0e-6") ;;
    *) echo "unknown phase '$PHASE' (smoke|depth|layers|bridge-arms)" >&2; exit 2 ;;
esac

if [ "$PHASE" = layers ]; then
    for f in studies/learn_french/side_checks/results/plan3_smoke/layer_diff_ulja_s42_lr5p0e-06_ep1_L0-23.json \
             studies/learn_french/side_checks/results/plan3_smoke/layer_diff_ulen_s42_lr5p0e-06_ep1_L24-35.json; do
        [ -f "$f" ] || { echo "smoke output missing: $f -- run the smoke phase first" >&2; exit 1; }
    done
    python3 studies/learn_french/side_checks/scripts/check_freeze.py \
        studies/learn_french/side_checks/results/plan3_smoke/layer_diff_ulja_s42_lr5p0e-06_ep1_L0-23.json 0-23
    python3 studies/learn_french/side_checks/scripts/check_freeze.py \
        studies/learn_french/side_checks/results/plan3_smoke/layer_diff_ulen_s42_lr5p0e-06_ep1_L24-35.json 24-35
fi
if [ "$PHASE" = bridge-arms ]; then
    python3 studies/learn_french/scripts/bridge_gate_report.py | tee /dev/stderr \
        | grep -q "DECISION: run the bridge arms" \
        || { echo "bridge gate did not pass -- not submitting" >&2; exit 1; }
fi

USED_GB=$(du -s -BG /home/b/brandonk 2>/dev/null | cut -f1 | tr -dc '0-9')
LEFT=$((400 - USED_GB))
NEED=$((33 * ${#JOBS[@]} + 15))
echo "disk: ${USED_GB}GB used, ~${LEFT}GB left of 400; $PHASE needs ~${NEED}GB"
[ "$LEFT" -ge "$NEED" ] || { echo "not enough room for $PHASE" >&2; exit 1; }

for j in "${JOBS[@]}"; do
    read -r env lg lr <<< "$j"
    id=$(sbatch --parsable --export=ALL,"$env" "$D/plan3_arm.sbatch" "$lg" "$lr")
    echo "$id  $lg lr $lr  $env"
done
echo "watch: squeue -u \$USER ; grep -h 'reached\|crossed\|freeze\|done' logs/side_p3_*.out"
