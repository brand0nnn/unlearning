#!/bin/bash
#SBATCH --job-name=phase2_transqc
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/phase2_transqc_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/phase2_transqc_%j.err

# PHASE 2, STEP 0 -- translation QC on the 720 records we will actually probe
# (40 forget + 40 retain, x 9 languages).
#
# Phase 1 used the non-English data only as relearning MATERIAL, so translation quality
# was off the critical path. Phase 2 probes IN-LANGUAGE, which puts it on it: a broken
# Hindi translation is indistinguishable from "the fact did not recover in Hindi".
#
# Independent of phase2_calibrate and of the probe run -- submit it in parallel.
#
# FIRST RUN DOWNLOADS NLLB-200-3.3B (~17.6GB) into $HF_HOME on the compute node, the
# same way Qwen3-8B arrived. Ungated (cc-by-nc-4.0), so no HF license click is needed;
# ~3.3GB VRAM in bf16. Do NOT pre-download on the login node -- its memory cap kills
# the hf CLI's Rust downloader.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/phase2_transqc"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h "$PROJECT_DIR" | tail -1

LANGS="${1:-ar fa fr hi id iw ja ko ru}"
SPLITS="${2:-forget retain}"
N="${3:-40}"

python3 studies/crosslingual_recovery/scripts/verify_translations.py \
    --langs $LANGS --splits $SPLITS --n $N

echo ""
echo "=== [DONE] -> $UNLEARN_RESULTS_DIR/phase2/translation_qc.json ==="
echo "NEXT: hand-check ~30 stratified items. Everything is back-translated to English,"
echo "so they are readable. The dangerous error is a false PASS (a broken translation"
echo "that survives and later reads as 'the fact did not recover in language l'), so"
echo "check precision on the PASSES specifically."
