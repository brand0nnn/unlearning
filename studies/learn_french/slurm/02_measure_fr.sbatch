#!/bin/bash
#SBATCH --job-name=measure_fr
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/measure_fr_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/measure_fr_%j.err

# STAGE 1 measurement. Pure inference over three checkpoints -- no DeepSpeed, no
# optimizer. Deliberately a SEPARATE job from 01_learn:
#   - Forget Quality is a TWO-model statistic (KS of fr_ft's forget-set truth-ratio
#     distribution against fr_retain's), so it cannot be computed inside either
#     training run;
#   - metrics change far more often than training does, and welding them into
#     01_learn would mean retraining a ~7.5h model to recompute a number.
#
#     sbatch studies/learn_french/slurm/02_measure_fr.sbatch
#
# Runtime is well under the wall (~4.8k teacher-forced passes + 40 generations per
# model, three models); 6h is headroom for a first run. On a busy queue this fits
# the `gpu` partition's ~3h wall too.
#
# The NLI model (xlm-roberta-large-xnli, ~2.2GB) is fetched by step [1/3] below
# rather than lazily mid-run, so a network problem fails in the first minute with a
# clear message instead of after three 16GB models have already been scored. The
# fetch is idempotent -- re-runs hit the cache and cost nothing. It lands in
# HF_HOME, redirected into the project dir, because $HOME's quota cannot hold it.
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/measure_fr"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
BASE=$(python3 -c "import yaml;print(yaml.safe_load(open('config/config.yaml'))['model']['name'])")
FR_FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
FR_RETAIN="$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr"

echo "=== [1/3] checkpoint check ==="
for d in "$FR_FT" "$FR_RETAIN"; do
    [ -d "$d" ] || { echo "MISSING checkpoint: $d" >&2; exit 1; }
    # A checkpoint saved without its tokenizer evaluates to all zeros, silently.
    [ -f "$d/tokenizer_config.json" ] || { echo "MISSING tokenizer in $d" >&2; exit 1; }
    echo "  OK $d"
done

echo "=== [2/3] fetching the NLI model (cached after the first run) ==="
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("joeddav/xlm-roberta-large-xnli")
print(f"  NLI model ready -> {p}")
PY

echo "=== [3/3] measuring fr_ft / fr_retain / base on the French forget set ==="
# `base` is the leakage control: stock Qwen3-8B never saw the TOFU authors, so its
# truth ratio should sit near 1.0. If it does not, the facts were not ours to inject.
python studies/learn_french/scripts/measure_fr.py \
    --checkpoints "$FR_FT" "$FR_RETAIN" "$BASE" \
    --reference   "$FR_RETAIN"

echo "=== Stage 1 measurement complete -> $UNLEARN_RESULTS_DIR/stage1/ ==="
ls -la "$UNLEARN_RESULTS_DIR/stage1/"
