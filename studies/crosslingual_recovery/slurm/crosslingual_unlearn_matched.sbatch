#!/bin/bash
#SBATCH --job-name=xling_unlearn
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_unlearn_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_unlearn_%j.err

# CROSS-LINGUAL — MATCHED-BASELINE UNLEARNING (+ Figure-8 curves). Phase 1 of 2.
#
# The overnight pilot left Full-FT (baseline 0.398) and LoRA (0.520) UNMATCHED, which
# confounds the recovery comparison. This job re-unlearns BOTH at forget01 so their
# post-unlearn ENGLISH forget ROUGE match, AND tracks ROUGE/Probability/Truth-Ratio
# per step (--track-curve) so we get the unlearning-dynamics figures.
#
#   1. Full-FT unlearn (UEP=15, --track-curve)                 -> baseline B_FT + curve
#   2. LoRA: unlearn at candidate depths (no curve), auto-pick the one whose baseline
#      lands closest to B_FT (LoRA needs MORE epochs than Full-FT to forget as deep)
#   3. re-unlearn the matched LoRA WITH --track-curve          -> matched baseline + curve
#
# NO relearning here — verify the baselines match first, THEN run the relearning phase
# (crosslingual_pilot.sbatch, reusing these matched checkpoints).
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_unlearn"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
echo "model slug = $MODEL_SLUG"
LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}"
GROUP="crosslingual_pilot"
FL="forget01"
N=40
UEP_FT="${1:-15}"                     # Full-FT depth (baseline ~0.40 on Qwen)
UEP_LORA_CANDS="${2:-30 34 38 42}"    # LoRA needs MORE epochs to match Full-FT depth

read_rouge() {
    python3 - "$1" <<'PY'
import json, sys, glob, os
name = os.path.basename(sys.argv[1])
root = os.environ.get("UNLEARN_RESULTS_DIR", "results")
for f in glob.glob(f"{root}/relearn/*/*.json"):
    d = json.load(open(f))
    if name in d:
        print(d[name]); break
PY
}

# --- 1. Full-FT unlearn WITH curve ---------------------------------------------
echo "=== [1] unlearn Full-FT forget01 (UEP=$UEP_FT) + curve ==="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT shared/scripts/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy fullft --forget-level "$FL" --unlearn-epochs "$UEP_FT" --track-curve
FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"
python shared/scripts/relearn_measure.py --checkpoints "$FULLFT" --group "$GROUP" --n $N --forget-level "$FL"
B_FT=$(read_rouge "$FULLFT"); echo "Full-FT baseline = $B_FT"

# --- 2. LoRA candidate sweep (no curve), match to Full-FT ----------------------
BEST_DIFF=999; BEST_U=""; LORA=""
for U in $UEP_LORA_CANDS; do
    echo "=== [2] unlearn LoRA forget01 candidate UEP=$U ==="
    python shared/scripts/02_unlearn.py \
        --checkpoint "$LEARNED" --strategy lora --forget-level "$FL" --unlearn-epochs "$U"
    CAND="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_uep${U}_${MODEL_SLUG}"
    rm -rf "$CAND"; mv "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_${MODEL_SLUG}" "$CAND"
    python shared/scripts/relearn_measure.py --checkpoints "$CAND" --group "${GROUP}_probe" --n $N --forget-level "$FL"
    B=$(read_rouge "$CAND")
    DIFF=$(python3 -c "print(abs($B - $B_FT))")
    echo "LoRA UEP=$U baseline=$B  |diff from Full-FT|=$DIFF"
    if [ "$(python3 -c "print(1 if $DIFF < $BEST_DIFF else 0)")" = "1" ]; then
        [ -n "$LORA" ] && rm -rf "$LORA"; BEST_DIFF=$DIFF; BEST_U=$U; LORA="$CAND"
    else
        rm -rf "$CAND"
    fi
done
echo "=== matched LoRA depth = UEP $BEST_U  (baseline |diff|=$BEST_DIFF from Full-FT) ==="

# --- 3. re-unlearn the matched LoRA WITH curve ---------------------------------
echo "=== [3] re-unlearn matched LoRA (UEP=$BEST_U) + curve ==="
rm -rf "$LORA"
python shared/scripts/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy lora --forget-level "$FL" --unlearn-epochs "$BEST_U" --track-curve
mv "$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_${MODEL_SLUG}" "$LORA"
python shared/scripts/relearn_measure.py --checkpoints "$LORA" --group "$GROUP" --n $N --forget-level "$FL"

echo "=== [DONE] matched baselines ==="
echo "  Full-FT (UEP=$UEP_FT):  $B_FT"
echo "  LoRA    (UEP=$BEST_U):  $(read_rouge "$LORA")"
echo "curves -> $UNLEARN_RESULTS_DIR/curves/  (plot with plot_unlearn_curve.py)"
echo "NEXT: run crosslingual_pilot.sbatch to relearn these matched checkpoints across languages."
