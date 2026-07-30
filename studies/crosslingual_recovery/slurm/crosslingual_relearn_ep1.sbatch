#!/bin/bash
#SBATCH --job-name=xling_relearn_ep1
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_relearn_ep1_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_relearn_ep1_%j.err

# CROSS-LINGUAL RECOVERY — MATCHED-BASELINE RELEARNING. Phase 2 of 2.
#
# Phase 1 (crosslingual_unlearn_matched.sbatch) produced MATCHED unlearned checkpoints:
#   Full-FT UEP15  -> English forget ROUGE 0.398
#   LoRA    UEP42  -> English forget ROUGE 0.373   (matched; |diff| 0.025)
# This job REUSES those two checkpoints (no re-unlearning) and benign-relearns each on
# the RETAIN set in all 10 languages, measuring ENGLISH forget recovery -> the clean,
# baseline-matched distance-decay curve, Full-FT vs LoRA.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_relearn"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
echo "model slug = $MODEL_SLUG"
GROUP="crosslingual_pilot_ep1"
LANGS="${1:-en fr id ru hi fa ar iw ko ja}"
EPS="${2:-2}"
FL="forget01"
N=40
RN="${3:-1500}"; RN_FLAG=""; [ -n "$RN" ] && RN_FLAG="--relearn-n $RN"

FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_uep42_${MODEL_SLUG}"

# fail loud if a matched checkpoint is missing (Phase 1 must have run + not been cleaned)
for CK in "$FULLFT" "$LORA"; do
    [ -d "$CK" ] || { echo "!! MISSING checkpoint: $CK  (run Phase 1 first)"; exit 1; }
done

# remove the STALE unmatched LoRA (uep26) recovery so old+new LoRA don't mix in the plot
rm -f "$UNLEARN_RESULTS_DIR/relearn/$GROUP/tofu_unlearn_gradient_difference_forget01_lora_uep26_${MODEL_SLUG}.json"

# --- record matched baselines (fresh, same job) --------------------------------
echo "=== baselines ==="
for CK in "$FULLFT" "$LORA"; do
    python shared/scripts/relearn_measure.py --checkpoints "$CK" --group "$GROUP" --n $N --forget-level "$FL"
done

# --- benign retain-relearn each checkpoint in each language, measure EN recovery ---
LAUNCH=0
for M in "$FULLFT" "$LORA"; do
    for L in $LANGS; do
        for E in $EPS; do
            LAUNCH=$((LAUNCH + 1))
            echo "=== relearn $(basename $M) via RETAIN lang=$L ep=$E ==="
            deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + LAUNCH)) \
                shared/scripts/relearn.py --checkpoint "$M" --epochs $E \
                --relearn-data retain --relearn-lang "$L" --forget-level "$FL" $RN_FLAG
            VIA="_via_retain"; [ "$L" != "en" ] && VIA="${VIA}_lang${L}"
            RELEARNED="$PROJECT_DIR/experiments/relearn_$(basename $M)${VIA}_ep${E}"
            python shared/scripts/relearn_measure.py --checkpoints "$RELEARNED" --group "$GROUP" --n $N --forget-level "$FL"
            rm -rf "$RELEARNED"       # reclaim the ~16GB relearned checkpoint
        done
    done
done

echo "=== [DONE] matched-baseline recovery -> $UNLEARN_RESULTS_DIR/relearn/$GROUP/ ==="
echo "Plot locally:  python studies/crosslingual_recovery/plots/plot_crosslingual_pilot.py"
