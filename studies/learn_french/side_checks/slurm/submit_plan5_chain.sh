#!/bin/bash
# Plan 5 in ONE command: every job queued now, chained with SLURM dependencies so they run
# one after another in priority order and the GPU never sits idle waiting for someone to
# submit the next phase. (The QOS allows ONE a100-80 per user at a time anyway, so running
# in series costs nothing; the chain only fixes the ORDER.)
#
#     bash studies/learn_french/side_checks/slurm/submit_plan5_chain.sh          # submit
#     DRY=1 bash studies/learn_french/side_checks/slurm/submit_plan5_chain.sh    # print only
#
# Order (results first that are readable on their own; preregistration_plan5.json):
#
#   #  job                                      ~h    waits for
#   1  depth: ja lr 1.5e-5 STOP_PGOLD=0.100     ~8    --  (first: the only run that can
#                                                         finish in under a day; 2026-11-09)
#   2  smoke ja LAYERS=0-23  (1 epoch)          1     1 ends
#   3  smoke en LAYERS=24-35 (1 epoch)          1     2 ends
#   4  D: ja LAYERS=0-23,  stop TR 0.8276       <=16  2 AND 3 print [freeze] OK (exit 0)
#   5  D: en LAYERS=24-35, stop TR 0.8276       <=16  2 AND 3 OK, and 4 ends
#   6  bridge LEARN (AUTO_GATE=1 -> it submits  ~15   5 ends
#      25_bridge_gate.sbatch itself when done)  (+1)
#   7  depth: ja lr 1.5e-5 STOP_PGOLD=0.018     ~9    6 ends
#   8  depth: en lr 5e-6   STOP_PGOLD=0.100     ~9    7 ends
#
# "ends" = afterany: the next job starts whether the previous one succeeded or failed, so
# one failure does not stall the queue. The two D jobs are afterok on the smoke jobs and
# carry --kill-on-invalid-dep: if a freeze check FAILS they are cancelled (not left
# pending forever), and the chain moves on.
#
# NOT automated, on purpose: the bridge ARMS. Read the gate first
# (python3 studies/learn_french/scripts/bridge_gate_report.py), then
# `bash studies/learn_french/side_checks/slurm/submit_plan5.sh bridge-arms`.
#
# Disk: in series the peak is one plan3_arm job (~33 GB) + the bridge model (16 GB, kept).
# plan3_arm.sbatch refuses to start with < 40 GB left; 24_learn_bridge with < 25 GB.
# Cancel everything: scancel the ids printed below (never `scancel -u`: other agents'
# jobs run under the same user).
set -euo pipefail
cd /home/b/brandonk/unlearning
A=studies/learn_french/side_checks/slurm/plan3_arm.sbatch
B=studies/learn_french/slurm/24_learn_bridge.sbatch
DRY="${DRY:-0}"
FORCE="${FORCE:-0}"

echo "code on the cluster: $(git log -1 --format='%h %s')"
for f in "$A" "$B" studies/learn_french/slurm/25_bridge_gate.sbatch \
         studies/learn_french/side_checks/scripts/check_freeze.py; do
    [ -f "$f" ] || { echo "missing $f -- git pull first" >&2; exit 1; }
done
grep -q AUTO_GATE "$B" || { echo "$B has no AUTO_GATE -- git pull first" >&2; exit 1; }

# Do not double-submit: Plan 5 jobs are side_p3 / learn_bridge / bridge_gate.
LIVE=$(squeue -h -u "$USER" -o "%i %j" | grep -E " (side_p3|learn_bridge|bridge_gate)$" || true)
if [ -n "$LIVE" ] && [ "$FORCE" != 1 ] && [ "$DRY" != 1 ]; then
    echo "Plan 5 jobs already queued -- not submitting again (FORCE=1 to override):" >&2
    echo "$LIVE" >&2
    exit 1
fi

USED_GB=$(du -s -BG /home/b/brandonk 2>/dev/null | cut -f1 | tr -dc '0-9')
echo "disk: ${USED_GB}GB used, ~$((400 - USED_GB))GB left of 400 (series peak ~50 GB;"
echo "      other jobs, e.g. Stage 10's tr_levels, can still grow -- each job re-checks)"

# sub NAME DEPENDENCY ENV SCRIPT [ARGS...]  -> prints the job id (DRY: prints NAME)
# ENV ("A=1,B=2") is set in sbatch's own environment, NOT passed as --export=ALL,A=1: on this
# cluster any --export=<list> sets SLURM_GET_USER_ENV=1, the node then rebuilds the login
# environment, and on xgph9 that fails -> "user env retrieval failed requeued held" (job
# 1096234, 2026-11-08, held 27 h). Plain sbatch exports ALL by default without that flag.
sub() {
    local name="$1" dep="$2" env="$3"; shift 3
    local vars=() opts=(--parsable)
    [ -n "$env" ] && IFS=, read -r -a vars <<< "$env"
    [ -n "$dep" ] && opts+=("--dependency=$dep" --kill-on-invalid-dep=yes)
    if [ "$DRY" = 1 ]; then
        echo "  env ${vars[*]} sbatch ${opts[*]} $*" >&2
        echo "$name"
    else
        env ${vars[@]+"${vars[@]}"} sbatch "${opts[@]}" "$@"
    fi
}

P1=$(sub depth_ja1 ""                         "STOP_PGOLD=0.100"     "$A" ja 1.5e-5)
S1=$(sub smoke_ja  "afterany:$P1"             "SMOKE=1,LAYERS=0-23"  "$A" ja 5.0e-6)
S2=$(sub smoke_en  "afterany:$S1"             "SMOKE=1,LAYERS=24-35" "$A" en 5.0e-6)
D1=$(sub D_ja      "afterok:$S1:$S2"          "LAYERS=0-23"          "$A" ja 5.0e-6)
D2=$(sub D_en      "afterok:$S1:$S2,afterany:$D1" "LAYERS=24-35"     "$A" en 5.0e-6)
BL=$(sub bridge    "afterany:$D2"             "AUTO_GATE=1"          "$B")
P2=$(sub depth_ja2 "afterany:$BL"             "STOP_PGOLD=0.018"     "$A" ja 1.5e-5)
P3=$(sub depth_en1 "afterany:$P2"             "STOP_PGOLD=0.100"     "$A" en 5.0e-6)

cat <<EOF

queued (runs top to bottom, one at a time):
  1  $P1   depth: ja to P(gold) 0.100
  2  $S1   smoke ja L0-23
  3  $S2   smoke en L24-35
  4  $D1   D: ja layers 0-23        (cancelled automatically if a smoke check fails)
  5  $D2   D: en layers 24-35       (cancelled automatically if a smoke check fails)
  6  $BL   bridge LEARN -> submits the gate itself when done
  7  $P2   depth: ja to P(gold) 0.018
  8  $P3   depth: en to P(gold) 0.100

watch:   squeue -u \$USER -o "%.9i %.14j %.8T %.10M %R"
smoke:   grep -h "freeze" logs/side_p3_$S1.out logs/side_p3_$S2.out
cancel:  scancel $P1 $S1 $S2 $D1 $D2 $BL $P2 $P3
EOF
