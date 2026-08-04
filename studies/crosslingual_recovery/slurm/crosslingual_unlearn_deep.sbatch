#!/bin/bash
#SBATCH --job-name=xling_deep_unlearn
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_deep_unlearn_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_deep_unlearn_%j.err

# UNLEARN both methods to a MATCHED, DEEP, PRINCIPLED baseline: truth ratio ~0.80.
# The dynamics curves showed both methods forget deeply on truth ratio and pass through
# TR ~0.80 at: Full-FT ~UEP20 (step 40), LoRA ~UEP32 (step 64). We unlearn each at those
# epochs and MEASURE the truth ratio to confirm they land near 0.80 and near each other.
#
# If the reported truth ratios are NOT both ~0.80 (truth ratio at n=40 is noisy), just
# re-run with adjusted epochs: sbatch crosslingual_unlearn_deep.sbatch <FT_EP> <LORA_EP>
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_deep_unlearn"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
echo "model slug = $MODEL_SLUG"
LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}"
GROUP="crosslingual_deep"          # fresh group for the deep-matched baselines + recovery
FL="forget01"; N=40
FT_EP="${1:-20}"                   # Full-FT -> TR ~0.80 (from the curve)
LORA_EP="${2:-32}"                 # LoRA    -> TR ~0.80 (from the curve)

read_tr() {
    python3 - "$1" <<'PY'
import json, sys, glob, os
name = os.path.basename(sys.argv[1])
root = os.environ.get("UNLEARN_RESULTS_DIR", "results")
for f in glob.glob(f"{root}/relearn/*/*.json"):
    d = json.load(open(f))
    v = d.get(name)
    if isinstance(v, dict) and "truth_ratio" in v:
        print(round(v["truth_ratio"], 4)); break
PY
}

# --- Full-FT -------------------------------------------------------------------
echo "=== Full-FT unlearn (UEP=$FT_EP) ==="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT shared/scripts/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy fullft --forget-level "$FL" --unlearn-epochs "$FT_EP"
FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"
python shared/scripts/relearn_measure.py --checkpoints "$FULLFT" --group "$GROUP" \
    --n $N --forget-level "$FL" --fact-metrics
FT_TR=$(read_tr "$FULLFT")

# --- LoRA (tagged by epoch so it doesn't clash with other LoRA checkpoints) ----
echo "=== LoRA unlearn (UEP=$LORA_EP) ==="
python shared/scripts/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy lora --forget-level "$FL" --unlearn-epochs "$LORA_EP"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_uep${LORA_EP}_${MODEL_SLUG}"
rm -rf "$LORA"; mv "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_${MODEL_SLUG}" "$LORA"
python shared/scripts/relearn_measure.py --checkpoints "$LORA" --group "$GROUP" \
    --n $N --forget-level "$FL" --fact-metrics
LORA_TR=$(read_tr "$LORA")

echo ""
echo "=== [DONE] DEEP baselines (target truth ratio ~0.80) ==="
echo "  Full-FT (UEP=$FT_EP):  truth ratio = $FT_TR"
echo "  LoRA    (UEP=$LORA_EP): truth ratio = $LORA_TR"
echo "  -> if both are ~0.80 and close to each other, proceed to the recovery run."
echo "  -> if not, re-run: sbatch crosslingual_unlearn_deep.sbatch <FT_EP> <LORA_EP>"
echo "checkpoints:"
echo "  $FULLFT"
echo "  $LORA"
