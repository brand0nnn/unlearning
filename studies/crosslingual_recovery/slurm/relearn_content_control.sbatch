#!/bin/bash
#SBATCH --job-name=relearn_control
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/relearn_control_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/relearn_control_%j.err

# CONTENT CONTROL: does the relearning DATA matter, or does any further training
# partially undo unlearning on its own?
#
# Everything in this study rests on benign relearning recovering forgotten facts, and
# on that recovery being the same whichever language supplies the data. Both readings
# collapse if recovery is just generic re-adaptation -- if training on ANY text nudges
# the weights back toward the pre-unlearning region. Nobody has tested that.
#
# TOFU already orders its splits by distance from the forget set, so the control is
# graded rather than binary:
#     retain        same task, same format, different fictitious authors   (nearest)
#     real_authors  real writers, same domain, real knowledge              (middle)
#     world_facts   unrelated general knowledge                            (furthest)
#
# VOLUME MATCHING is the whole design. real_authors has 100 rows and world_facts 117,
# against retain's 3960 (relearned at --relearn-n 1500 elsewhere in this study). Compare
# those as-is and a difference could simply be 15x less training, so every arm here runs
# at --relearn-n 100 -- including retain, which is therefore RE-RUN rather than reused.
#
# retain@100 doubles as the floor check: retain@1500 recovers 42% (Full-FT) / 63% (LoRA),
# so if retain@100 recovers substantially, 100 examples is enough training for the
# comparison to be meaningful. If it recovers ~nothing, the control has hit a floor and
# the nulls for real_authors/world_facts would be uninformative rather than evidence.
#
# READING IT:
#   recovery similar across all three -> generic re-adaptation; the relearn CONTENT never
#       mattered, and the cross-language result is much less interesting.
#   recovery falls with distance      -> content matters; the language finding is a claim
#       about knowledge structure, not about training in general.
#
# English only: this asks about content, not language, and English is the one probe
# language the Phase 2 calibration gate showed to be informative (AUC 0.825).
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/relearn_control"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
GROUP="content_control"
FL="forget01"; N=40
EP="${1:-2}"
RN="${2:-100}"                       # matched to real_authors, the smallest split
SOURCES="${3:-retain real_authors world_facts}"
LORA_UEP="${4:-uep32}"

# Resolved against experiments/ where the weights live -- NOT by globbing results/,
# which is rsync-only and gitignored (how job 770249 died).
FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_${FL}_fullft_${MODEL_SLUG}"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_${FL}_lora_${LORA_UEP}_${MODEL_SLUG}"
for CK in "$FULLFT" "$LORA"; do
    [ -d "$CK" ] || { echo "!! MISSING checkpoint: $CK"; exit 1; }
done
echo "baselines : $(basename $FULLFT) | $(basename $LORA)"
echo "sources   : $SOURCES   (n=$RN each, ep=$EP, English)"
echo "group     : $GROUP"

LAUNCH=0
for M in "$FULLFT" "$LORA"; do
    MB=$(basename "$M")
    GFILE="$UNLEARN_RESULTS_DIR/relearn/$GROUP/${MB}.json"
    for SRC in $SOURCES; do
        VIA="_via_${SRC}"
        RKEY="relearn_${MB}${VIA}_ep${EP}"
        # Resumable: skip only if this cell already carries the PER-FACT arrays the
        # analysis needs, not merely a stored mean.
        if [ -f "$GFILE" ] && python3 -c "import json,sys; r=json.load(open('$GFILE')).get('$RKEY'); sys.exit(0 if isinstance(r,dict) and 'truth_ratio_per_fact' in r else 1)" 2>/dev/null; then
            echo "skip (already has per-fact arrays): $RKEY"; continue
        fi
        LAUNCH=$((LAUNCH + 1))
        echo ""
        echo "=== relearn $MB  source=$SRC  n=$RN  ep=$EP ==="
        deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + LAUNCH)) \
            shared/scripts/relearn.py --checkpoint "$M" --epochs $EP \
            --relearn-data "$SRC" --relearn-lang en --forget-level "$FL" \
            --relearn-n $RN
        RELEARNED="$PROJECT_DIR/experiments/relearn_${MB}${VIA}_ep${EP}"
        python3 shared/scripts/relearn_measure.py --checkpoints "$RELEARNED" \
            --group "$GROUP" --n $N --forget-level "$FL" --fact-metrics
        rm -rf "$RELEARNED"          # reclaim the ~16GB checkpoint, as elsewhere
    done
done

echo ""
echo "=== [DONE] content control -> $UNLEARN_RESULTS_DIR/relearn/$GROUP/ ==="
echo "Compare the three sources at matched n. Also compare retain@$RN against the"
echo "existing retain@1500 in crosslingual_deep to confirm n=$RN is not a floor."
