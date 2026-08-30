#!/bin/bash
#SBATCH --job-name=measure_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/measure_baseline_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/measure_baseline_%j.err

# Per-fact truth ratios for the two UNLEARNED baselines -- the missing piece of Rec().
#
# The recovery set is  Rec(l) = { i : NOT knows_unlearned(i) AND knows_relearned(i) }.
# Every relearned cell already stores truth_ratio_per_fact, but the baselines they are
# measured against were written before that upgrade and hold only the MEAN (0.767 for
# Full-FT). Without the per-fact array we cannot tell a genuine recovery from a fact
# unlearning never removed -- both look "alive" in all 10 relearn languages and inflate
# the cross-language overlap.
#
# Inference only: no training, no checkpoint written, 2 model loads over 40 facts.
# Minutes, not hours -- hence the `gpu` partition (short wall, better queue priority)
# and a100-40 (8B in bf16 needs ~16GB).
#
# SAFE TO RUN ALONGSIDE the relearn jobs: it writes the same group files they do, but
# relearn_measure.py now serialises the read-modify-write under flock (_merge_write).
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/measure_baseline"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
GROUP="${1:-crosslingual_deep}"
FL="forget01"; N=40
LORA_UEP="${2:-uep32}"

# Resolved against experiments/ where the weights live -- NOT by globbing results/,
# which is rsync-only and gitignored (the mistake that killed job 770249).
FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_${FL}_fullft_${MODEL_SLUG}"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_${FL}_lora_${LORA_UEP}_${MODEL_SLUG}"
for CK in "$FULLFT" "$LORA"; do
    [ -d "$CK" ] || { echo "!! MISSING checkpoint: $CK"; exit 1; }
done
echo "baselines: $(basename $FULLFT) | $(basename $LORA)"
echo "group    : $GROUP"

# --fact-metrics writes truth_ratio_per_fact; the bare checkpoint name is the key, so
# this UPDATES the existing mean-only baseline records in place rather than adding new
# ones (see _probe_key: en + forget -> bare name).
python3 shared/scripts/relearn_measure.py \
    --checkpoints "$FULLFT" "$LORA" \
    --group "$GROUP" --n $N --forget-level "$FL" \
    --fact-metrics --measure-lang en

echo ""
echo "=== [DONE] baselines now carry per-fact arrays -> $UNLEARN_RESULTS_DIR/relearn/$GROUP/"
echo "Rec() is now computable. Locally:"
echo "  rsync the results down, then rerun the interlingua analysis."
