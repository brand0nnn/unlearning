#!/bin/bash
# Seed sweep for the en / ja Stage 3 arms (preregistration_plan3.json seed_sweep). Run on
# the login node from ~/unlearning:
#
#     bash studies/learn_french/side_checks/slurm/submit_seeds.sh 2          # seeds 1 2
#     bash studies/learn_french/side_checks/slurm/submit_seeds.sh 4 1 2      # 4 lanes
#
# Each job is plan3_arm.sbatch at the Stage 2 LR (5e-6) with SEED set: unlearn fr_ft in
# en or ja until French TR crosses 0.8276 (Stage 3's rule, so seed 42 IS the Stage 3
# arm), then score, name skill, layer diff, relearn in fr and en, delete.
#
#   job           est. hours    (unlearn ~0.22h/epoch + ~0.2h scoring + 2 relearns ~1.85h)
#   ja, seed s    ~9.5          (~25 epochs)
#   en, seed s    ~6.5          (~12 epochs)
#
#   lanes   seeds 1 2: wall if that many GPUs are free     peak disk
#   1       ~32h                                           ~33GB
#   2       ~16h   (each lane: ja s -> en s)               ~66GB
#   4       ~9.5h                                          ~132GB
#
# Lanes are afterany chains, ja first. Disk is checked with du (the ~400GB per-user limit
# shows in neither df nor quota).
set -euo pipefail
cd /home/b/brandonk/unlearning
D=studies/learn_french/side_checks/slurm
N="${1:?usage: submit_seeds.sh <lanes> [seeds, default 1 2]}"
shift
if [ $# -gt 0 ]; then SEEDS=("$@"); else SEEDS=(1 2); fi
for s in "${SEEDS[@]}"; do
    case "$s" in ''|*[!0-9]*|42) echo "bad seed '$s' (a number, not 42 = Stage 3)" >&2; exit 2 ;; esac
done
case "$N" in [1-9]) ;; *) echo "lanes must be 1-9" >&2; exit 2 ;; esac

USED_GB=$(du -s -BG /home/b/brandonk 2>/dev/null | cut -f1 | tr -dc '0-9')
LEFT=$((400 - USED_GB))
NEED=$((33 * N + 15))
echo "disk: ${USED_GB}GB used, ~${LEFT}GB left of 400; $N lanes need ~${NEED}GB"
[ "$LEFT" -ge "$NEED" ] || { echo "not enough room; largest that fits: $(( (LEFT - 15) / 33 ))" >&2; exit 1; }

# ja jobs first so the longest ones start first, one per lane where possible.
JOBS=()
for s in "${SEEDS[@]}"; do JOBS+=("ja $s"); done
for s in "${SEEDS[@]}"; do JOBS+=("en $s"); done
declare -a LAST
for i in "${!JOBS[@]}"; do
    lane=$((i % N))
    read -r lg s <<< "${JOBS[$i]}"
    dep=()
    [ -n "${LAST[$lane]:-}" ] && dep=(--dependency=afterany:${LAST[$lane]})
    id=$(sbatch --parsable --export=ALL,SEED="$s" ${dep[@]+"${dep[@]}"} "$D/plan3_arm.sbatch" "$lg" 5.0e-6)
    LAST[$lane]=$id
    echo "lane $lane  $id  $lg seed $s${dep[*]:+   (after ${dep[*]#--dependency=afterany:})}"
done
echo "watch: squeue -u \$USER ; grep -h 'crossed\|done' logs/side_p3_*.out"
