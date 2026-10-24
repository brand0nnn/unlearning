#!/bin/bash
# STAGE 9 in one command, from the LOGIN node: mixed fr+ja LEARN -> gate + French unlearning.
#
#     cd ~/unlearning && git pull && source .venv/bin/activate
#     bash studies/learn_french/slurm/submit_frja_mix.sh            # learn -> gate + unlearn
#     FLOOR=1 bash studies/learn_french/slurm/submit_frja_mix.sh    # + the matched mixed floor
#
# Chain (afterok = starts only if the job before it succeeded):
#     20_learn_frja_mix full        ~15 h  gpu-long a100-80
#     20_learn_frja_mix retain99    ~15 h  in PARALLEL, only with FLOOR=1 (a second GPU, +16 GB)
#     21_gate_frja_mix              ~1 h   gpu a100-40       } both start when the
#     22_unlearn_frja_mix           <12 h  gpu-long a100-80  } learn(s) finish
# Disk: ~16 GB per learned model + up to ~82 GB of unlearning levels.
#
# If a LEARN ever splits across jobs (RESUME_PENDING; should not happen inside a 24 h wall),
# its first link exits 0 and releases the afterok jobs too early. Both refuse to start on an
# unfinished checkpoint; resubmit them by hand once the resumed LEARN completes:
#     sbatch studies/learn_french/slurm/21_gate_frja_mix.sbatch
#     sbatch studies/learn_french/slurm/22_unlearn_frja_mix.sbatch
set -euo pipefail

P=/home/b/brandonk/unlearning
S=studies/learn_french/slurm
cd "$P"
FLOOR="${FLOOR:-0}"

echo "=== data check (login node, no GPU) ==="
mkdir -p logs
LOG="logs/verify_frja_mix_$(date +%Y%m%d_%H%M%S).log"
RC=0
python studies/learn_french/scripts/verify_learn_data_mix.py full $([ "$FLOOR" = 1 ] && echo retain99) \
    > "$LOG" 2>&1 || RC=$?
grep -E "\[OK|\[FAIL|PASSED|FAILED|FORGET facts|Error|error" "$LOG" || true
[ "$RC" -eq 0 ] || { echo "data check FAILED (rc=$RC, full log $LOG) -- nothing submitted" >&2; exit 1; }

SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
for d in "experiments/tofu_learn_full_full_${SLUG}_fr-ja" "experiments/tofu_learn_retain99_full_${SLUG}_fr-ja"; do
    [ -f "$d/config.json" ] || { echo "MISSING Stage 8 checkpoint $d (control / floor)" >&2; exit 1; }
done
OUT="experiments/tofu_learn_full_full_${SLUG}_fr-ja-mix"
[ ! -e "$OUT/config.json" ] || { echo "EXISTS: $OUT -- already learned; submit 21 and 22 by hand" >&2; exit 1; }
echo "=== disk ==="; df -h "$P/experiments" | tail -1

J1=$(sbatch --parsable "$S/20_learn_frja_mix.sbatch" full)
DEP="afterok:$J1"
echo "learn (full)      job $J1"
if [ "$FLOOR" = "1" ]; then
    J2=$(sbatch --parsable "$S/20_learn_frja_mix.sbatch" retain99)
    DEP="$DEP:$J2"
    echo "learn (retain99)  job $J2"
fi
J3=$(sbatch --parsable --dependency="$DEP" --kill-on-invalid-dep=yes "$S/21_gate_frja_mix.sbatch")
J4=$(sbatch --parsable --dependency="$DEP" --kill-on-invalid-dep=yes "$S/22_unlearn_frja_mix.sbatch")
echo "gate              job $J3   (after $DEP)"
echo "unlearn           job $J4   (after $DEP)"
echo
echo "watch:  squeue -u \$USER     tail -f logs/learn_frja_mix_${J1}.out"
echo "done:   sacct -j $J1,$J3,$J4 --format=JobID,JobName%20,State,Elapsed"
