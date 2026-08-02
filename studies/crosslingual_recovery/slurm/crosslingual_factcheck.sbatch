#!/bin/bash
#SBATCH --job-name=xling_factcheck
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_factcheck_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_factcheck_%j.err

# CONFOUND-1 VALIDATION: is the ROUGE recovery FLUENCY or the FACT?
# Recompute recovery with FACT-SPECIFIC metrics (truth_ratio + probability) on the
# same relearn checkpoints. If recovery survives on truth_ratio, the finding is solid;
# if it shows only in ROUGE, it's fluency restoration (weaker result).
#
# A cheap SUBSET is enough to decide: English (d0), French (near), Japanese (far),
# at the peak epochs (ep1, ep2), for BOTH methods. Checkpoints are regenerated and
# measured with --fact-metrics, then deleted. Results -> group crosslingual_facts.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_factcheck"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
GROUP="crosslingual_facts"
FL="forget01"; N=40; RN=1500
LANGS="${1:-en fr id ru hi fa ar iw ko ja}"   # full axis (overnight); measures rouge
EPS="${2:-1 2}"                                # + prob + truth_ratio at rise + peak

FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_uep42_${MODEL_SLUG}"
for CK in "$FULLFT" "$LORA"; do
    [ -d "$CK" ] || { echo "!! MISSING $CK (need the matched unlearned checkpoints)"; exit 1; }
done

# baselines with fact metrics (unlearned model, no relearning)
echo "=== baselines (fact metrics) ==="
for CK in "$FULLFT" "$LORA"; do
    python shared/scripts/relearn_measure.py --checkpoints "$CK" --group "$GROUP" \
        --n $N --forget-level "$FL" --fact-metrics
done

# relearn subset, measure fact metrics
LAUNCH=0
for M in "$FULLFT" "$LORA"; do
    for L in $LANGS; do
        for E in $EPS; do
            LAUNCH=$((LAUNCH+1))
            echo "=== relearn $(basename $M) lang=$L ep=$E ==="
            deepspeed --num_gpus=1 --master_port=$((MASTER_PORT+LAUNCH)) \
                shared/scripts/relearn.py --checkpoint "$M" --epochs $E \
                --relearn-data retain --relearn-lang "$L" --forget-level "$FL" --relearn-n $RN
            VIA="_via_retain"; [ "$L" != "en" ] && VIA="${VIA}_lang${L}"
            RELEARNED="$PROJECT_DIR/experiments/relearn_$(basename $M)${VIA}_ep${E}"
            python shared/scripts/relearn_measure.py --checkpoints "$RELEARNED" \
                --group "$GROUP" --n $N --forget-level "$FL" --fact-metrics
            rm -rf "$RELEARNED"
        done
    done
done
echo "=== DONE -> $UNLEARN_RESULTS_DIR/relearn/$GROUP/  (compare rouge vs truth_ratio recovery) ==="
