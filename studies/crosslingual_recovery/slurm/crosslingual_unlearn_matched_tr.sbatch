#!/bin/bash
#SBATCH --job-name=xling_tr_match
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_tr_match_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_tr_match_%j.err

# MATCH THE UNLEARNED BASELINE ON *TRUTH RATIO* (the fact-specific metric), in one job.
#
# Full-FT's truth ratio barely moves with unlearning (it tops out ~0.58), so we FIX
# Full-FT (UEP15) as the target and tune LoRA DOWN to it: fewer LoRA epochs -> less
# forgetting -> truth ratio drops from ~0.78 toward Full-FT's ~0.58.
#
#   1. Full-FT unlearn (UEP15), measure truth ratio  -> TARGET
#   2. LoRA: unlearn at candidate epochs, measure truth ratio, AUTO-PICK the one whose
#      truth ratio lands closest to TARGET; discard the rest
#   -> two checkpoints matched on truth-ratio forget depth, ready for the recovery run.
#
# NOTE: matching to Full-FT's SHALLOW truth ratio gives LoRA less room to recover, so
# recovery magnitudes shrink. The test is whether LoRA STILL recovers more at equal depth.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_tr_match"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
echo "model slug = $MODEL_SLUG"
LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}"
GROUP="crosslingual_tr_match"
FL="forget01"; N=40
UEP_FT="${1:-15}"
UEP_LORA_CANDS="${2:-10 14 18 22 26}"   # LoRA: fewer epochs than the ROUGE-match (42)
                                        # to raise truth ratio DOWN toward Full-FT's

# read the truth_ratio field a --fact-metrics run stored for a checkpoint
read_tr() {
    python3 - "$1" <<'PY'
import json, sys, glob, os
name = os.path.basename(sys.argv[1])
root = os.environ.get("UNLEARN_RESULTS_DIR", "results")
for f in glob.glob(f"{root}/relearn/*/*.json"):
    d = json.load(open(f))
    v = d.get(name)
    if isinstance(v, dict) and "truth_ratio" in v:
        print(v["truth_ratio"]); break
PY
}

# --- 1. Full-FT unlearn -> TARGET truth ratio ----------------------------------
echo "=== [1] unlearn Full-FT forget01 (UEP=$UEP_FT) ==="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT shared/scripts/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy fullft --forget-level "$FL" --unlearn-epochs "$UEP_FT"
FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"
python shared/scripts/relearn_measure.py --checkpoints "$FULLFT" --group "$GROUP" \
    --n $N --forget-level "$FL" --fact-metrics
TARGET=$(read_tr "$FULLFT")
echo "=== Full-FT truth-ratio TARGET = $TARGET ==="

# --- 2. LoRA sweep: match truth ratio to TARGET --------------------------------
BEST_DIFF=999; BEST_U=""; LORA=""
for U in $UEP_LORA_CANDS; do
    echo "=== [2] unlearn LoRA forget01 candidate UEP=$U ==="
    python shared/scripts/02_unlearn.py \
        --checkpoint "$LEARNED" --strategy lora --forget-level "$FL" --unlearn-epochs "$U"
    CAND="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_uep${U}_${MODEL_SLUG}"
    rm -rf "$CAND"; mv "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_${MODEL_SLUG}" "$CAND"
    python shared/scripts/relearn_measure.py --checkpoints "$CAND" --group "${GROUP}_probe" \
        --n $N --forget-level "$FL" --fact-metrics
    TR=$(read_tr "$CAND")
    DIFF=$(python3 -c "print(abs($TR - $TARGET))")
    echo "LoRA UEP=$U  truth_ratio=$TR  |diff from target|=$DIFF"
    if [ "$(python3 -c "print(1 if $DIFF < $BEST_DIFF else 0)")" = "1" ]; then
        [ -n "$LORA" ] && rm -rf "$LORA"; BEST_DIFF=$DIFF; BEST_U=$U; LORA="$CAND"
    else
        rm -rf "$CAND"
    fi
done

# --- 3. record the matched LoRA truth ratio in the real group ------------------
python shared/scripts/relearn_measure.py --checkpoints "$LORA" --group "$GROUP" \
    --n $N --forget-level "$FL" --fact-metrics

echo "=== [DONE] truth-ratio-matched baselines ==="
echo "  Full-FT (UEP=$UEP_FT):  truth_ratio $TARGET"
echo "  LoRA    (UEP=$BEST_U):  truth_ratio $(read_tr "$LORA")   (|diff|=$BEST_DIFF)"
echo "matched checkpoints:"
echo "  $FULLFT"
echo "  $LORA"
echo "NEXT: relearn these two across languages with --fact-metrics (crosslingual_factcheck-style)."
