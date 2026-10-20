#!/bin/bash
# Submit the whole side check (Plans 1 and 2). Run on the login node from ~/unlearning:
#
#     bash studies/learn_french/side_checks/slurm/submit_all.sh            # 4 jobs in parallel
#     bash studies/learn_french/side_checks/slurm/submit_all.sh serial     # one disk-heavy job at a time
#
# parallel: fr arm, en arm, control and Plan 2 all queue at once. ~10h wall if GPUs are
#           free; peak disk ~180GB (2 x 65 + 48, Plan 2 writes nothing large).
# serial:   fr arm -> en arm -> control chained with afterany (each step still runs if the
#           previous failed); Plan 2 runs alongside. Peak disk ~65GB, ~24h wall.
set -euo pipefail
cd /home/b/brandonk/unlearning
D=studies/learn_french/side_checks/slurm
df -h experiments | tail -1

P2=$(sbatch --parsable "$D/plan2_names.sbatch")
echo "plan 2 (existing checkpoints)  $P2"
if [ "${1:-parallel}" = "serial" ]; then
    A=$(sbatch --parsable "$D/plan1_arm.sbatch" fr)
    B=$(sbatch --parsable --dependency=afterany:$A "$D/plan1_arm.sbatch" en)
    C=$(sbatch --parsable --dependency=afterany:$B "$D/plan1_control.sbatch")
else
    A=$(sbatch --parsable "$D/plan1_arm.sbatch" fr)
    B=$(sbatch --parsable "$D/plan1_arm.sbatch" en)
    C=$(sbatch --parsable "$D/plan1_control.sbatch")
fi
echo "plan 1 arm fr  $A"
echo "plan 1 arm en  $B"
echo "plan 1 control $C"
echo "watch: squeue -u \$USER ; tail -f logs/side_p1_$A.out"
