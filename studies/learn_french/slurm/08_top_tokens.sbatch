#!/bin/bash
#SBATCH --job-name=toptok
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/toptok_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/toptok_%j.err

# WHICH LANGUAGE DOES THE MODEL WANT TO ANSWER IN?
#
#     sbatch studies/learn_french/slurm/08_top_tokens.sbatch              # the 8 defaults
#     sbatch studies/learn_french/slurm/08_top_tokens.sbatch <ckpt> ...   # your own list
#
# Unlearning was probed in French, so the French form of the answer is pushed down. This
# asks whether the SAME fact stays available in another language -- if it does, the fact
# was not removed, only its French expression was suppressed.
#
# Pure inference, no DeepSpeed, so it fits the 3h `gpu` partition and an a100-40. ~10 min
# per checkpoint.
#
# The default list is base Qwen3 FIRST (every "vs ref" column is relative to it), then the
# learned ceiling, the never-taught floor, and the three unlearning arms whose scripts can
# be told apart from French. Add the en/id arms with an explicit list if you want them --
# on a proper-noun slot they are indistinguishable from French by construction, which the
# script flags as a token collision.
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/toptok"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
BASE=$(python3 -c "import yaml;print(yaml.safe_load(open('config/config.yaml'))['model']['name'])")
TR="$PROJECT_DIR/experiments/tr_levels/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"

if [ "$#" -gt 0 ]; then
    CKPTS=("$@")
else
    # exact paths from the pre-registered matched depths -- never globbed
    CKPTS=("$BASE"
           "$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
           "$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr")
    while read -r lang tag; do
        CKPTS+=("${TR}_ul${lang}_floornone_${tag}")
    done < <(python3 -c "
import json
m = json.load(open('studies/learn_french/preregistration.json'))['stage3_matched_checkpoints']
for lang in ('fr', 'ja', 'ru'):
    print(lang, ('tr%.3f' % m[lang]['level']).replace('.', 'p'))")
fi

for c in "${CKPTS[@]}"; do
    case "$c" in */*) [ -d "$c" ] || { echo "MISSING: $c" >&2; exit 1; } ;; esac
    echo "  $c"
done

python studies/learn_french/scripts/top_tokens.py \
    --checkpoints "${CKPTS[@]}" \
    --out "$UNLEARN_RESULTS_DIR/top_tokens.json"

echo "=== done -> $UNLEARN_RESULTS_DIR/top_tokens.json ==="
