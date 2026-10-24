#!/bin/bash
# Submit Plans 3 and 4 (6 jobs). Run on the login node from ~/unlearning:
#
#     bash studies/learn_french/side_checks/slurm/submit_plan34.sh          # all 6 at once
#     bash studies/learn_french/side_checks/slurm/submit_plan34.sh lean     # Plan 4 one at a time
#
# parallel: 3 Plan 3 arms + 3 Plan 4 arms queue together. ~15h wall if 6 GPUs are free;
#           peak disk ~200GB (6 x ~33GB).
# lean:     Plan 3's 3 arms in parallel, Plan 4's chained ja -> fr -> en with afterany.
#           Peak disk ~130GB; Plan 4 finishes ~15h after it starts instead of ~5h.
set -euo pipefail
cd /home/b/brandonk/unlearning
D=studies/learn_french/side_checks/slurm
MODE="${1:-parallel}"
case "$MODE" in parallel) NEED=210 ;; lean) NEED=140 ;; *) echo "mode: parallel|lean" >&2; exit 2 ;; esac

df -h experiments | tail -1
AVAIL_GB=$(df -BG --output=avail experiments | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt "$NEED" ]; then
    echo "only ${AVAIL_GB}GB free; '$MODE' peaks near $((NEED - 10))GB." >&2
    [ "$MODE" = parallel ] && echo "try:  bash $0 lean" >&2
    exit 1
fi

# Plan 3: matched depth AND matched epochs (see side_checks/preregistration_plan3.json)
B1=$(sbatch --parsable "$D/plan3_arm.sbatch" ja 1.0e-5)
B2=$(sbatch --parsable "$D/plan3_arm.sbatch" ja 1.5e-5)
B3=$(sbatch --parsable "$D/plan3_arm.sbatch" en 2.0e-6)
echo "plan 3  ja-fast lr 1.0e-5  $B1"
echo "plan 3  ja-fast lr 1.5e-5  $B2"
echo "plan 3  en-slow lr 2.0e-6  $B3"

# Plan 4: layer profile at fixed epochs 13 and 25
if [ "$MODE" = lean ]; then
    C1=$(sbatch --parsable "$D/plan4_arm.sbatch" ja)
    C2=$(sbatch --parsable --dependency=afterany:$C1 "$D/plan4_arm.sbatch" fr)
    C3=$(sbatch --parsable --dependency=afterany:$C2 "$D/plan4_arm.sbatch" en)
else
    C1=$(sbatch --parsable "$D/plan4_arm.sbatch" ja)
    C2=$(sbatch --parsable "$D/plan4_arm.sbatch" fr)
    C3=$(sbatch --parsable "$D/plan4_arm.sbatch" en)
fi
echo "plan 4  ja  $C1"
echo "plan 4  fr  $C2"
echo "plan 4  en  $C3"
echo "watch: squeue -u \$USER ; grep -h 'STOP\|SAVED\|crossed' logs/side_p3_$B1.*"
