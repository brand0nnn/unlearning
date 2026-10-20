#!/bin/bash
#SBATCH --job-name=side_p1ctl
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/side_p1ctl_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/side_p1ctl_%j.err

# SIDE CHECK, PLAN 1 anchors and the floor control.
#
#     sbatch studies/learn_french/side_checks/slurm/plan1_control.sbatch
#
#   [1] Stage 1 scorer on fr_ft and fr_retain, in THIS scoring group, so the epoch-25
#       and relearned numbers sit next to anchors scored the same way (scoring moves TR
#       by up to ~0.02 and NLI by ~0.09 between runs)                        ~1h
#   [2] post-hoc per-token pass on fr_ft and fr_retain: the in-run epoch-0 row IS fr_ft,
#       so this is the in-run vs post-hoc check at the start                 ~20min
#   [3] benign relearn of fr_retain with the Stage 3 recipe (French retain99, n=1500,
#       3 epochs): what relearning does to a model that never saw the facts, i.e. the
#       part of "recovery" that would happen anyway                          ~3.2h
# Same skip-if-done and delete-by-exact-path rules as plan1_arm.sbatch. Peak ~48GB.
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
SIDE="studies/learn_french/side_checks"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/$SIDE/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR/plan1"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/side_p1ctl"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"
MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
FR_FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
FR_RETAIN="$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr"
R="$UNLEARN_RESULTS_DIR"
AVAIL_GB=$(df -BG --output=avail "$PROJECT_DIR/experiments" | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge 55 ] || { echo "only ${AVAIL_GB}GB free; this job peaks at ~48GB" >&2; exit 1; }

# ---- [1] anchors, Stage 1 scorer (Forget Quality included: the reference is listed) ----
if [ -f "$R/plan1/measure/$(basename "$FR_FT").json" ] && \
   [ -f "$R/plan1/measure/$(basename "$FR_RETAIN").json" ]; then echo "[1] anchors -- skip"; else
    python studies/learn_french/scripts/measure_fr.py \
        --checkpoints "$FR_FT" "$FR_RETAIN" --reference "$FR_RETAIN" --group plan1/measure
fi

# ---- [2] anchors, per-token -----------------------------------------------------------
if [ -f "$R/plan1/per_token_anchors.json" ]; then echo "[2] per-token anchors -- skip"; else
    python studies/learn_french/scripts/per_token_map.py \
        --checkpoints "$FR_FT" "$FR_RETAIN" --out "$R/plan1/per_token_anchors.json"
fi

# ---- [3] floor control: relearn fr_retain --------------------------------------------
python -c "from huggingface_hub import snapshot_download; snapshot_download('joeddav/xlm-roberta-large-xnli')" >/dev/null
REL_FINAL="$PROJECT_DIR/experiments/relearn_$(basename "$FR_RETAIN")_via_retain_langfr_ep3"
REL_SNAPS=("${REL_FINAL}__atep1" "${REL_FINAL}__atep2" "$REL_FINAL")
DONE=1
for S in "${REL_SNAPS[@]}"; do [ -f "$R/plan1/relearn/$(basename "$S").json" ] || DONE=0; done
if [ "$DONE" -eq 1 ]; then echo "[3] control already scored -- skip"; else
    HAVE=1
    for S in "${REL_SNAPS[@]}"; do [ -f "$S/tokenizer_config.json" ] || HAVE=0; done
    if [ "$HAVE" -eq 0 ]; then
        echo "[3] relearn fr_retain on retain99 French, n=1500, 3 epochs"
        deepspeed --num_gpus=1 --master_port=$((20000 + SLURM_JOB_ID % 10000)) \
            shared/scripts/relearn.py \
            --checkpoint "$FR_RETAIN" --relearn-data retain --relearn-lang fr \
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
echo "=== done: control -> $R/plan1.  $(df -h "$PROJECT_DIR/experiments" | tail -1)"
