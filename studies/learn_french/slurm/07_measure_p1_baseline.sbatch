#!/bin/bash
#SBATCH --job-name=p1_base
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/p1_base_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/p1_base_%j.err

# THE p1 BASELINE. Pure inference, no training.
#
#     sbatch studies/learn_french/slurm/07_measure_p1_baseline.sbatch
#
# WHY THIS EXISTS. measure_fr.py gained the p1 block -- TOFU's own paraphrased_question,
# as translated by Farashah -- AFTER Stages 1 and 2 had already been scored. So p1 exists
# on all 25 relearned cells and nowhere else, which leaves it uninterpretable: every other
# metric in this study is a difference from the learned and unlearned checkpoints, and p1
# has neither anchor. The pre-registration (stage3_relearn.probe) commits to using p1 to
# separate "the fact came back" from "the trained wording came back" -- LEARN and UNLEARN
# both trained on the p0 wording -- and that comparison is not computable until p1 is
# scored at the two ends as well. See stage3_p1_baseline_gap.
#
# WHAT IT SCORES. Seven checkpoints, all already on disk:
#   base Qwen3           never fine-tuned at all -- the reference line the MU panels use
#   fr_ft                the learned ceiling, and the phrasing-sensitivity control
#   fr_retain            the never-taught floor + the Forget Quality reference
#   5 x matched unlearned    the exact checkpoints Stage 3 relearned FROM, resolved by
#                            level from preregistration.json -- never globbed
#
# A NEW RESULT GROUP, NOT AN OVERWRITE. Output goes to results/p1_baseline/ so the
# existing stage1_norm/ and stage2_nocap_*/ files are untouched. Scoring is deterministic
# (the old study measured median |delta| = 0.0000 across reruns), so the recomputed p0
# numbers must reproduce the stored ones exactly -- an end-to-end regression check on the
# whole scorer for free. Compare them before using the p1 values.
#
# Runs on h100-96 because as of 2026-09-17 four of the eight a100-80 nodes (xgph4, xgph6,
# xgph7, xgph9) have a stale NVIDIA kernel module and fail instantly with
# "nvmlInit_v2() failed: Driver/library version mismatch". Add --exclude if that spreads.
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/p1_base"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"

PREREG="studies/learn_french/preregistration.json"
MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")

FR_FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
FR_RETAIN="$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr"
for d in "$FR_FT" "$FR_RETAIN"; do
    [ -d "$d" ] || { echo "MISSING: $d" >&2; exit 1; }
done

# Exact paths from the pre-registered matched depths -- the same rule 05_relearn_fr uses,
# so this scores the very checkpoints Stage 3 started from, not a lookalike.
# Base Qwen3 is a hub id, not a directory; measure_fr.py takes either (Stage 1 scored it
# the same way). It costs ~8 min and completes the p1 reference set.
BASE=$(python3 -c "import yaml;print(yaml.safe_load(open('config/config.yaml'))['model']['name'])")
CKPTS=("$BASE" "$FR_FT" "$FR_RETAIN")
while read -r lang tag; do
    p="$PROJECT_DIR/experiments/tr_levels/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}_ul${lang}_floornone_${tag}"
    [ -d "$p" ] || { echo "MISSING matched checkpoint for $lang: $p" >&2; exit 1; }
    echo "  $lang -> $p"
    CKPTS+=("$p")
done < <(python3 -c "
import json
m = json.load(open('$PREREG'))['stage3_matched_checkpoints']
for lang in ('en', 'fr', 'id', 'ja', 'ru'):
    print(lang, ('tr%.3f' % m[lang]['level']).replace('.', 'p'))")

# Fail before the GPU work if the NLI model is not cached, rather than an hour in.
python -c "from huggingface_hub import snapshot_download; snapshot_download('joeddav/xlm-roberta-large-xnli')" >/dev/null

echo "=== scoring ${#CKPTS[@]} checkpoints on p0 + p1, probed in French ==="
python studies/learn_french/scripts/measure_fr.py \
    --checkpoints "${CKPTS[@]}" \
    --reference   "$FR_RETAIN" \
    --group "p1_baseline"

echo "=== done -> $UNLEARN_RESULTS_DIR/p1_baseline/ ==="
ls -la "$UNLEARN_RESULTS_DIR/p1_baseline/"
