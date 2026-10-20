#!/bin/bash
#SBATCH --job-name=side_p1
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/side_p1_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/side_p1_%j.err

# SIDE CHECK, PLAN 1: one unlearning arm to epoch 25, then everything that needs its
# weights, then the weights are deleted.
#
#     sbatch studies/learn_french/side_checks/slurm/plan1_arm.sbatch fr
#     sbatch studies/learn_french/side_checks/slurm/plan1_arm.sbatch en
#     sbatch studies/learn_french/side_checks/slurm/plan1_arm.sbatch id no    # no relearn
#
#   [1] unlearn fr_ft in <lang> on the Stage 2 50-epoch schedule, stop at epoch 25, score
#       every epoch IN-RUN (scripts/unlearn_epochs.py), save epoch 25 only   ~5h
#   [2] Stage 1 scorer on epoch 25: TR, P(gold), NLI, MU      (measure_fr.py)  ~30min
#   [3] post-hoc per-token pass on epoch 25, to check the in-run numbers      ~10min
#   [4] Plan 2 name skill on epoch 25                                          ~15min
#   [5] benign relearn, Stage 3 recipe (retain99 French, n=1500, 3 epochs), each
#       snapshot scored by measure_fr.py then deleted by exact path           ~3.2h
#   [6] delete epoch 25 -- only once every output above exists
#
# Every step is skipped when its output exists, so resubmitting the same command after a
# failure or wall-clock kill redoes only what is missing. A run killed DURING [1] leaves
# partial JSONL files and no finished checkpoint: the job then refuses to start until
# they are moved aside (a re-run must not append to an old trajectory).
#
# Everything lands under studies/learn_french/side_checks/results/ (rsync, not git) and
# experiments/side_checks/ -- never the main study's results/ or experiments/tr_levels/.
# Peak disk ~65GB (16GB epoch 25 + three 16GB relearn snapshots).
set -euo pipefail

LANG_UL="${1:?usage: sbatch plan1_arm.sbatch <lang> [yes|no relearn]}"
RELEARN="${2:-yes}"
STOP=25
SEED=42
case "$LANG_UL" in en|fr|id|ja|ru) ;; *) echo "bad lang: $LANG_UL" >&2; exit 2 ;; esac
case "$RELEARN" in yes|no) ;; *) echo "bad relearn flag: $RELEARN (yes|no)" >&2; exit 2 ;; esac

PROJECT_DIR="/home/b/brandonk/unlearning"
SIDE="studies/learn_french/side_checks"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/$SIDE/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR/plan1" "$UNLEARN_RESULTS_DIR/plan2" experiments/side_checks
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/side_p1_$LANG_UL"; mkdir -p "$TMPDIR"
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
FR_FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
FR_RETAIN="$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr"
[ -d "$FR_FT" ] || { echo "MISSING fr_ft: $FR_FT" >&2; exit 1; }
[ -d "$FR_RETAIN" ] || { echo "MISSING fr_retain: $FR_RETAIN" >&2; exit 1; }

RUN="ul${LANG_UL}_s${SEED}_ep${STOP}"
CKPT="$PROJECT_DIR/experiments/side_checks/$RUN"
R="$UNLEARN_RESULTS_DIR"
TRAJ="$R/plan1/traj_ul${LANG_UL}_s${SEED}.jsonl"
TOKS="$R/plan1/tokens_ul${LANG_UL}_s${SEED}.jsonl"
MEASURE="$R/plan1/measure/$RUN.json"
POSTHOC="$R/plan1/per_token_$RUN.json"
NAMES="$R/plan2/$RUN.json"
REL_RUN="relearn_${RUN}_via_retain_langfr_ep3"
REL_FINAL="$PROJECT_DIR/experiments/$REL_RUN"
REL_SNAPS=("${REL_FINAL}__atep1" "${REL_FINAL}__atep2" "$REL_FINAL")

echo "=========================================================================="
echo " SIDE CHECK plan 1   arm $LANG_UL   seed $SEED   stop at epoch $STOP of a 50-epoch schedule"
echo "   checkpoint : $CKPT (deleted at the end)"
echo "   relearn    : $RELEARN"
echo "   storage    : $(df -h "$PROJECT_DIR/experiments" | tail -1)"
echo "=========================================================================="
AVAIL_GB=$(df -BG --output=avail "$PROJECT_DIR/experiments" | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge 70 ] || { echo "only ${AVAIL_GB}GB free; this job peaks at ~65GB" >&2; exit 1; }

# ---- [1] unlearn ---------------------------------------------------------------------
if [ -f "$CKPT/tokenizer_config.json" ]; then
    echo "[1] epoch $STOP already saved -- skipping unlearning"
else
    if [ -e "$TRAJ" ] || [ -e "$TOKS" ] || [ -e "$CKPT" ]; then
        echo "[1] a PARTIAL run exists (killed mid-training?). Move these aside, then resubmit:" >&2
        echo "      $TRAJ" >&2; echo "      $TOKS" >&2; echo "      $CKPT" >&2
        exit 1
    fi
    echo "[1] unlearn"
    deepspeed --num_gpus=1 --master_port=$((20000 + SLURM_JOB_ID % 10000)) \
        "$SIDE/scripts/unlearn_epochs.py" \
        --checkpoint "$FR_FT" --lang "$LANG_UL" --seed "$SEED" \
        --schedule-epochs 50 --stop-at-epoch "$STOP" --mu-every 10 --forget-floor none
fi
[ -f "$CKPT/tokenizer_config.json" ] || { echo "no finished checkpoint at $CKPT" >&2; exit 1; }

# ---- [2] Stage 1 scorer on epoch 25 --------------------------------------------------
if [ -f "$MEASURE" ]; then echo "[2] measured -- skip"; else
    echo "[2] measure_fr on $RUN"
    python studies/learn_french/scripts/measure_fr.py \
        --checkpoints "$CKPT" --reference "$FR_RETAIN" --group plan1/measure
fi

# ---- [3] post-hoc per-token pass (checks the in-run epoch-25 row) ----------------------
if [ -f "$POSTHOC" ]; then echo "[3] post-hoc per-token -- skip"; else
    echo "[3] per_token_map on $RUN"
    python studies/learn_french/scripts/per_token_map.py \
        --checkpoints "$CKPT" --out "$POSTHOC"
fi

# ---- [4] Plan 2 ------------------------------------------------------------------------
if [ -f "$NAMES" ]; then echo "[4] name skill -- skip"; else
    echo "[4] name skill on $RUN"
    python "$SIDE/scripts/name_skill.py" --checkpoints "$CKPT" --out-dir "$R/plan2"
fi

# ---- [5] benign relearning (Stage 3 recipe), snapshots deleted once scored -------------
if [ "$RELEARN" = "yes" ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download('joeddav/xlm-roberta-large-xnli')" >/dev/null
    DONE=1
    for S in "${REL_SNAPS[@]}"; do [ -f "$R/plan1/relearn/$(basename "$S").json" ] || DONE=0; done
    if [ "$DONE" -eq 1 ]; then echo "[5] relearn already scored -- skip"; else
        HAVE=1
        for S in "${REL_SNAPS[@]}"; do [ -f "$S/tokenizer_config.json" ] || HAVE=0; done
        if [ "$HAVE" -eq 1 ]; then echo "[5] relearn snapshots on disk -- skipping training"; else
            echo "[5] relearn $RUN on retain99 French, n=1500, 3 epochs"
            deepspeed --num_gpus=1 --master_port=$((20000 + (SLURM_JOB_ID + 7) % 10000)) \
                shared/scripts/relearn.py \
                --checkpoint "$CKPT" --relearn-data retain --relearn-lang fr \
                --relearn-n 1500 --forget-level forget01 --epochs 3 --save-each-epoch
        fi
        for S in "${REL_SNAPS[@]}"; do
            OUT="$R/plan1/relearn/$(basename "$S").json"
            if [ ! -f "$OUT" ]; then
                [ -d "$S" ] || { echo "MISSING snapshot $S" >&2; exit 1; }
                python studies/learn_french/scripts/measure_fr.py \
                    --checkpoints "$S" --reference "$FR_RETAIN" --group plan1/relearn
            fi
            if [ -f "$OUT" ]; then
                [ -d "$S" ] && rm -rf "$S" && echo "    freed $S"
            else
                echo "    scoring wrote no $OUT -- KEEPING $S" >&2; exit 1
            fi
        done
    fi
fi

# ---- [6] delete epoch 25, only when everything that needed it is on disk --------------
NEED=("$MEASURE" "$POSTHOC" "$NAMES")
if [ "$RELEARN" = "yes" ]; then
    for S in "${REL_SNAPS[@]}"; do NEED+=("$R/plan1/relearn/$(basename "$S").json"); done
fi
for f in "${NEED[@]}"; do
    [ -f "$f" ] || { echo "[6] missing $f -- KEEPING $CKPT" >&2; exit 1; }
done
case "$CKPT" in "$PROJECT_DIR/experiments/side_checks/ul"*) ;; *)
    echo "refusing to delete unexpected path $CKPT" >&2; exit 1 ;; esac
rm -rf "$CKPT"
echo "[6] deleted $CKPT"
echo "=== done: arm $LANG_UL -> $R/plan1, $R/plan2.  $(df -h "$PROJECT_DIR/experiments" | tail -1)"
