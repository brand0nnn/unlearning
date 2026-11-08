#!/bin/bash
#SBATCH --job-name=side_p3
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/side_p3_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/side_p3_%j.err

# SIDE CHECK, PLAN 3: matched depth AND matched epochs. One arm, one peak LR.
#
#     sbatch studies/learn_french/side_checks/slurm/plan3_arm.sbatch ja 1.0e-5    # ja-fast
#     sbatch studies/learn_french/side_checks/slurm/plan3_arm.sbatch ja 1.5e-5    # ja-fast
#     sbatch studies/learn_french/side_checks/slurm/plan3_arm.sbatch en 2.0e-6    # en-slow
#     STOP_EPOCH=15 sbatch studies/learn_french/side_checks/slurm/plan3_arm.sbatch en 5.0e-6
#                         # P(gold) control: stop at a FIXED epoch instead of the TR level
#
# PLAN 5 (side_checks/preregistration_plan5.json; submit with submit_plan5.sh). Env options,
# each changes the run name so no output can collide with an earlier run:
#     STOP_PGOLD=0.100   stop at the first epoch with French P(gold) <= 0.100 (depth curve);
#                        run name ends _pg0p100. Exclusive with STOP_EPOCH.
#     LAYERS=0-23        train decoder layers 0..23 only (Experiment D); name gains _L0-23
#     SRC=bridge         unlearn the bridge model (retain99 fr+ja + forget01 fr, 24_learn_
#                        bridge.sbatch) instead of fr_ft; floor fr-ja_retain; name gains _srcbr
#     SMOKE=1            unlearn 1 epoch into plan3_smoke/, layer-diff it, print whether the
#                        frozen layers stayed frozen, delete it. No scoring, no relearn. ~1h.
#
# Stage 3 matched en and ja on French TR (both stopped at level 0.8276), but ja needed 25
# epochs and en 12. Here the LR moves so each reaches the SAME level in the OTHER arm's
# epoch count. Everything else is the Stage 2 recipe (side_checks/preregistration_plan3.json).
#
#   [1] unlearn fr_ft in <lang> at peak LR <lr> on the 50-epoch schedule; stop at the first
#       per-epoch probe with French TR >= 0.8276 (cap: epoch 50); every epoch scored
#       IN-RUN (TR, MU, per-token, P(gold))                                     ~2-6h
#   [2] Stage 1 scorer on the stopped weights: TR, P(gold), NLI, MU             ~30min
#   [3] Plan 2 name skill (blanket damage / script switching)                   ~15min
#   [4] layer diff fr_ft -> stopped weights (CPU, streams tensors)              ~10min
#   [5] benign relearn, Stage 3 recipe (retain99, n=1500, 3 epochs) in each of
#       RELEARN_LANGS, one after another; each scored then deleted by exact path ~3.2h each
#   [6] delete the stopped weights -- only once every output above exists
#
# Only the FINAL relearn epoch is kept (no --save-each-epoch): Stage 3's headline is epoch
# 3, and it halves the disk. Peak disk ~33GB (16 GB unlearned + 16 GB one relearn).
# Every step is skipped when its output exists, so resubmit the same command after a kill.
set -euo pipefail

LANG_UL="${1:?usage: sbatch plan3_arm.sbatch <lang> <peak lr> [relearn langs, default 'fr en']}"
LR="${2:?peak LR, e.g. 1.0e-5}"
RELEARN_LANGS="${3:-fr en}"
# SEED set: unlearning AND relearning use it (seed sweep, preregistration_plan3.json
# seed_sweep). 42 = every earlier run.
SEED="${SEED:-42}"
case "$SEED" in ''|*[!0-9]*) echo "bad SEED: $SEED" >&2; exit 2 ;; esac
CAP=50
# STOP_EPOCH set: stop at that epoch, not at the TR level (preregistration_plan3.json
# pgold_control). The run name then ends _ep<N> and the level check is skipped.
STOP_EPOCH="${STOP_EPOCH:-}"
STOP_PGOLD="${STOP_PGOLD:-}"
LAYERS="${LAYERS:-}"
SRC="${SRC:-frft}"
SMOKE="${SMOKE:-0}"
case "$LANG_UL" in en|fr|id|ja|ru) ;; *) echo "bad lang: $LANG_UL" >&2; exit 2 ;; esac
case "$STOP_EPOCH" in ""|[1-9]|[1-4][0-9]|50) ;; *) echo "bad STOP_EPOCH: $STOP_EPOCH (1-50)" >&2; exit 2 ;; esac
case "$STOP_PGOLD" in ""|0.[0-9][0-9][0-9]) ;; *) echo "bad STOP_PGOLD: $STOP_PGOLD (0.xxx)" >&2; exit 2 ;; esac
[ -z "$STOP_PGOLD" ] || [ -z "$STOP_EPOCH" ] || { echo "STOP_PGOLD and STOP_EPOCH are exclusive" >&2; exit 2; }
case "$LAYERS" in ""|[0-9]-[0-9]|[0-9]-[0-9][0-9]|[0-9][0-9]-[0-9][0-9]) ;; *) echo "bad LAYERS: $LAYERS (A-B)" >&2; exit 2 ;; esac
case "$SRC" in frft|bridge) ;; *) echo "bad SRC: $SRC (frft|bridge)" >&2; exit 2 ;; esac
case "$SMOKE" in 0|1) ;; *) echo "bad SMOKE: $SMOKE" >&2; exit 2 ;; esac

PROJECT_DIR="/home/b/brandonk/unlearning"
SIDE="studies/learn_french/side_checks"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/$SIDE/results"
cd "$PROJECT_DIR"
R="$UNLEARN_RESULTS_DIR"
mkdir -p logs "$R/plan3" experiments/side_checks/plan3
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/side_p3_${LANG_UL}_${LR}_s${SEED}${STOP_EPOCH:+_ep$STOP_EPOCH}${STOP_PGOLD:+_pg$STOP_PGOLD}${LAYERS:+_L$LAYERS}_${SRC}_smoke$SMOKE"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The Stage 2 memory setup, unchanged: optimizer state in host RAM, adamw_torch.
export UNLEARN_DS_CONFIG="config/ds_config_offload.json"
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"
MEM_MB="${SLURM_MEM_PER_NODE:-0}"
[ "$MEM_MB" -ge 180000 ] || { echo "optimizer offload needs --mem>=180G, got ${MEM_MB}MB" >&2; exit 1; }

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
if [ "$SRC" = bridge ]; then
    # Learned the forget authors in French ONLY, the retain authors in French AND Japanese;
    # its floor is the matching never-taught model (retain99 in both, Stage 8).
    FR_FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr-ja-fgfr"
    FR_RETAIN="$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr-ja"
else
    FR_FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
    FR_RETAIN="$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr"
fi
[ -d "$FR_FT" ] || { echo "MISSING fr_ft: $FR_FT" >&2; exit 1; }
[ -d "$FR_RETAIN" ] || { echo "MISSING fr_retain: $FR_RETAIN" >&2; exit 1; }

# The level en and ja were matched at, read from the pre-registration, not retyped.
LEVEL=$(python3 -c "import json;print(json.load(open('studies/learn_french/preregistration.json'))['tr_levels'][-1])")
# Same tags unlearn_epochs.py builds, so this script knows the run name in advance.
[ "$SMOKE" = 1 ] && STOP_EPOCH=1
if [ -n "$STOP_EPOCH" ]; then
    STOP_TAG="ep${STOP_EPOCH}"
    STOP_FLAGS=(--stop-at-epoch "$STOP_EPOCH")
elif [ -n "$STOP_PGOLD" ]; then
    STOP_TAG=$(python3 -c "print(('pg%.3f'%$STOP_PGOLD).replace('.','p'))")
    STOP_FLAGS=(--stop-at-pgold "$STOP_PGOLD" --stop-at-epoch "$CAP")
else
    STOP_TAG=$(python3 -c "print(('tr%.3f'%$LEVEL).replace('.','p'))")
    STOP_FLAGS=(--stop-at-level "$LEVEL" --stop-at-epoch "$CAP")
fi
# Suffix order = unlearn_epochs.py's: _L<layers> then _<tag>.
RUN="ul${LANG_UL}_s${SEED}_$(python3 -c "print(('lr%.1e'%$LR).replace('.','p'))")_${STOP_TAG}"
EXTRA_FLAGS=()
if [ -n "$LAYERS" ]; then RUN+="_L$LAYERS"; EXTRA_FLAGS+=(--trainable-layers "$LAYERS"); fi
if [ "$SRC" = bridge ]; then RUN+="_srcbr"; EXTRA_FLAGS+=(--tag srcbr); fi
SUB=plan3
[ "$SMOKE" = 1 ] && SUB=plan3_smoke
mkdir -p "$R/$SUB" "experiments/side_checks/$SUB"
CKPT="$PROJECT_DIR/experiments/side_checks/$SUB/$RUN"
TRAJ="$R/$SUB/traj_$RUN.jsonl"
TOKS="$R/$SUB/tokens_$RUN.jsonl"
MEASURE="$R/plan3/measure/$RUN.json"
NAMES="$R/plan3/names/$RUN.json"
LAYERS_OUT="$R/$SUB/layer_diff_$RUN.json"

echo "=========================================================================="
echo " SIDE CHECK plan 3/5   arm $LANG_UL   peak LR $LR   seed $SEED"
echo "   run        : $RUN"
echo "   source     : $SRC ($FR_FT)"
echo "   layers     : ${LAYERS:-all}   smoke: $SMOKE"
if [ -n "$STOP_PGOLD" ]; then
    echo "   stop       : first per-epoch French P(gold) <= $STOP_PGOLD (cap epoch $CAP)"
elif [ -n "$STOP_EPOCH" ]; then
    echo "   stop       : fixed epoch $STOP_EPOCH (no TR level)"
else
    echo "   stop       : first per-epoch French TR >= $LEVEL (cap epoch $CAP)"
fi
echo "   checkpoint : $CKPT (deleted at the end)"
echo "   relearn in : $RELEARN_LANGS"
echo "=========================================================================="
# The limit is ~400 GB PER USER and neither df nor quota shows it: measure with du.
USED_GB=$(du -s -BG /home/b/brandonk 2>/dev/null | cut -f1 | tr -dc '0-9')
echo "   storage    : ${USED_GB}GB used of the ~400GB per-user limit"
[ $((400 - USED_GB)) -ge 40 ] || { echo "only $((400 - USED_GB))GB left; this job peaks at ~33GB" >&2; exit 1; }

# ---- [1] unlearn to the level --------------------------------------------------------
if [ -f "$CKPT/tokenizer_config.json" ]; then
    echo "[1] stopped weights already saved -- skipping unlearning"
else
    if [ -e "$TRAJ" ] || [ -e "$TOKS" ] || [ -e "$CKPT" ]; then
        echo "[1] a PARTIAL run exists (killed mid-training?). Move these aside, then resubmit:" >&2
        echo "      $TRAJ" >&2; echo "      $TOKS" >&2; echo "      $CKPT" >&2
        exit 1
    fi
    echo "[1] unlearn"
    deepspeed --num_gpus=1 --master_port=$((20000 + SLURM_JOB_ID % 10000)) \
        "$SIDE/scripts/unlearn_epochs.py" \
        --checkpoint "$FR_FT" --lang "$LANG_UL" --seed "$SEED" --lr "$LR" \
        --schedule-epochs 50 "${STOP_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
        --mu-every 10 --forget-floor none \
        --ckpt-root "experiments/side_checks/$SUB" --out-dir "$R/$SUB"
fi
[ -f "$CKPT/tokenizer_config.json" ] || { echo "no finished checkpoint at $CKPT" >&2; exit 1; }

# ---- SMOKE: did the freeze hold under ZeRO-3? Then stop. --------------------------------
if [ "$SMOKE" = 1 ]; then
    echo "[smoke] layer diff $FR_FT -> $CKPT"
    python studies/learn_french/scripts/layer_diff.py \
        --pair "$RUN=${FR_FT}::${CKPT}" --out "$LAYERS_OUT"
    SMOKE_OK=0
    python3 "$SIDE/scripts/check_freeze.py" "$LAYERS_OUT" "${LAYERS:-}" || SMOKE_OK=$?
    case "$CKPT" in "$PROJECT_DIR/experiments/side_checks/plan3_smoke/ul"*) rm -rf "$CKPT" ;; esac
    echo "=== smoke done: $RUN (outputs in $R/plan3_smoke; move them aside before re-running) ==="
    exit "$SMOKE_OK"
fi

# Did it reach the level, and when? (The last trajectory row is the stopping point.)
# A fixed-epoch run has no level to reach: report where it stopped and relearn it.
if [ -n "$STOP_PGOLD" ]; then
    CROSSED=$(python3 -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])][1:]
pg = [(r['epoch'], sum(r['prob_fr']) / len(r['prob_fr'])) for r in rows]
hit = [(e, p) for e, p in pg if p <= float(sys.argv[2])]
e, p = hit[0] if hit else pg[-1]
print(('yes' if hit else 'NO ') + f'  epoch {e}  French P(gold) {p:.4f}', file=sys.stderr)
print('yes' if hit else 'no')" "$TOKS" "$STOP_PGOLD")
    echo "[1] reached P(gold) <= $STOP_PGOLD: $CROSSED"
elif [ -n "$STOP_EPOCH" ]; then
    python3 -c "import json,sys;r=[json.loads(l) for l in open(sys.argv[1])][-1];print(f\"[1] stopped at epoch {r['epoch']:.0f}  TR {r['mean_tr']:.4f}  MU {r['model_utility_6']}\")" "$TRAJ"
    CROSSED=yes
else
CROSSED=$(python3 - "$TRAJ" "$LEVEL" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
lv = float(sys.argv[2])
hit = [r for r in rows if r["mean_tr"] >= lv]
if hit:
    r = hit[0]
    print(f"yes  epoch {r['epoch']:.0f}  TR {r['mean_tr']:.4f}  MU {r['model_utility_6']}",
          file=sys.stderr)
    print("yes")
else:
    print(f"NO   last epoch {rows[-1]['epoch']:.0f}  TR {rows[-1]['mean_tr']:.4f}",
          file=sys.stderr)
    print("no")
EOF
)
echo "[1] crossed level $LEVEL: $CROSSED"
fi

python -c "from huggingface_hub import snapshot_download; snapshot_download('joeddav/xlm-roberta-large-xnli')" >/dev/null

# ---- [2] Stage 1 scorer --------------------------------------------------------------
if [ -f "$MEASURE" ]; then echo "[2] measured -- skip"; else
    echo "[2] measure_fr on $RUN"
    python studies/learn_french/scripts/measure_fr.py \
        --checkpoints "$CKPT" --reference "$FR_RETAIN" --group plan3/measure
fi

# ---- [3] Plan 2 name skill -----------------------------------------------------------
if [ -f "$NAMES" ]; then echo "[3] name skill -- skip"; else
    echo "[3] name skill on $RUN"
    python "$SIDE/scripts/name_skill.py" --checkpoints "$CKPT" --out-dir "$R/plan3/names"
fi

# ---- [4] layer diff (CPU) ------------------------------------------------------------
if [ -f "$LAYERS_OUT" ]; then echo "[4] layer diff -- skip"; else
    echo "[4] layer diff $SRC -> $RUN"
    python studies/learn_french/scripts/layer_diff.py \
        --pair "$RUN=${FR_FT}::${CKPT}" --out "$LAYERS_OUT"
fi

# ---- [5] benign relearning, one language at a time ------------------------------------
# relearn.py names the run relearn_<ckpt>_via_retain[_lang<L>]_ep3 (no suffix for en).
rel_name() { if [ "$1" = en ]; then echo "relearn_${RUN}_via_retain_ep3";
             else echo "relearn_${RUN}_via_retain_lang$1_ep3"; fi; }
REL_OUTS=()
if [ "$CROSSED" != "yes" ]; then
    echo "[5] SKIPPED: the level was never crossed, so this arm is not at matched depth"
else
    for RL in $RELEARN_LANGS; do
        S="$PROJECT_DIR/experiments/$(rel_name "$RL")"
        OUT="$R/plan3/relearn/$(basename "$S").json"
        REL_OUTS+=("$OUT")
        if [ -f "$OUT" ]; then echo "[5] relearn $RL already scored -- skip"; continue; fi
        if [ -f "$S/tokenizer_config.json" ]; then
            echo "[5] relearn $RL snapshot on disk -- skipping training"
        else
            echo "[5] relearn $RUN on retain99 $RL, n=1500, 3 epochs"
            deepspeed --num_gpus=1 --master_port=$((20000 + (SLURM_JOB_ID + 7) % 10000)) \
                shared/scripts/relearn.py \
                --checkpoint "$CKPT" --relearn-data retain --relearn-lang "$RL" \
                --relearn-n 1500 --forget-level forget01 --epochs 3 --seed "$SEED"
        fi
        [ -d "$S" ] || { echo "MISSING relearned model $S" >&2; exit 1; }
        python studies/learn_french/scripts/measure_fr.py \
            --checkpoints "$S" --reference "$FR_RETAIN" --group plan3/relearn
        if [ -f "$OUT" ]; then
            case "$S" in "$PROJECT_DIR/experiments/relearn_ul"*) ;; *)
                echo "refusing to delete unexpected path $S" >&2; exit 1 ;; esac
            rm -rf "$S" && echo "    freed $S"
        else
            echo "    scoring wrote no $OUT -- KEEPING $S" >&2; exit 1
        fi
    done
fi

# ---- [6] delete the stopped weights, only when everything that needed them is on disk -
for f in "$MEASURE" "$NAMES" "$LAYERS_OUT" ${REL_OUTS[@]+"${REL_OUTS[@]}"}; do
    [ -f "$f" ] || { echo "[6] missing $f -- KEEPING $CKPT" >&2; exit 1; }
done
case "$CKPT" in "$PROJECT_DIR/experiments/side_checks/plan3/ul"*) ;; *)
    echo "refusing to delete unexpected path $CKPT" >&2; exit 1 ;; esac
rm -rf "$CKPT"
echo "[6] deleted $CKPT"
echo "=== done: arm $LANG_UL lr $LR (crossed: $CROSSED) -> $R/plan3"
