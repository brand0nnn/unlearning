#!/bin/bash
# Submit Plans 3 and 4 (6 jobs) as N parallel lanes. Run on the login node from ~/unlearning:
#
#     bash studies/learn_french/side_checks/slurm/submit_plan34.sh 3        # 3 at a time
#     bash studies/learn_french/side_checks/slurm/submit_plan34.sh 2 100    # + reserve 100GB
#                                                                          for another job
#
# Each lane is a chain (afterany: a later job still runs if an earlier one failed), and no
# job peaks above ~33GB, so peak disk ~= 33GB x N. Plan 3 jobs go first in every lane.
#
#   N   peak disk   wall if N GPUs free   lanes (est. hours)
#   6   ~200GB      ~13h                  every job alone
#   3   ~100GB      ~18h                  en-slow+C fr 18 | ja 1e-5+C en 16 | ja 1.5e-5+C ja 15
#   2   ~66GB       ~28h                  27.5 | 21.5
#   1   ~33GB       ~49h
#
# The limit is ~400GB PER USER and neither df nor quota shows it, so this measures usage
# with du and refuses N when 33 x N + 15GB margin does not fit.
set -euo pipefail
cd /home/b/brandonk/unlearning
D=studies/learn_french/side_checks/slurm
N="${1:?usage: submit_plan34.sh <lanes 1-6> [GB reserved for other jobs]}"
RESERVE="${2:-0}"
case "$N" in 1|2|3|4|5|6) ;; *) echo "lanes must be 1-6" >&2; exit 2 ;; esac

USED_GB=$(du -s -BG /home/b/brandonk 2>/dev/null | cut -f1 | tr -dc '0-9')
LEFT=$((400 - USED_GB))
NEED=$((33 * N + 15 + RESERVE))
echo "disk: ${USED_GB}GB used, ~${LEFT}GB left of 400; $N lanes + ${RESERVE}GB reserved need ~${NEED}GB"
if [ "$LEFT" -lt "$NEED" ]; then
    echo "not enough room for $N lanes; largest that fits: $(( (LEFT - 15 - RESERVE) / 33 ))" >&2
    exit 1
fi

# Order matters: with round-robin, the three Plan 3 arms start first, one per lane.
JOBS=("plan3_arm.sbatch en 2.0e-6" "plan3_arm.sbatch ja 1.0e-5" "plan3_arm.sbatch ja 1.5e-5"
      "plan4_arm.sbatch fr" "plan4_arm.sbatch en" "plan4_arm.sbatch ja")
declare -a LAST
for i in "${!JOBS[@]}"; do
    lane=$((i % N))
    read -r script args <<< "${JOBS[$i]}"
    dep=()
    [ -n "${LAST[$lane]:-}" ] && dep=(--dependency=afterany:${LAST[$lane]})
    # shellcheck disable=SC2086
    id=$(sbatch --parsable ${dep[@]+"${dep[@]}"} "$D/$script" $args)
    LAST[$lane]=$id
    echo "lane $lane  $id  ${JOBS[$i]}${dep[*]:+   (after ${dep[*]#--dependency=afterany:})}"
done
echo "watch: squeue -u \$USER ; grep -h 'STOP\|SAVED\|crossed' logs/side_p3_*.out logs/side_p3_*.err"
