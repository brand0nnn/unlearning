#!/bin/bash
#SBATCH --job-name=nameswapml
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/nameswapml_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/nameswapml_%j.err

# MULTILINGUAL NAME SWAP -- why is the transliterated name (ru/ja) destroyed by its own
# language's unlearning while the Latin-script one (fr/en/id) is not, at equal certainty?
#
#     sbatch studies/learn_french/slurm/16_name_swap_ml.sbatch              # the 8 defaults
#     sbatch studies/learn_french/slurm/16_name_swap_ml.sbatch <ckpt> ...   # your own list
#
# Leading explanation, TWO SUPPORTS: the Latin name is backed by copying AND by French
# learning of those exact tokens; the transliterated one by copying only. The 38 en/id/ru/ja
# answers whose question names the author, each scored under orig (name copied) and A (name
# replaced by a birth-date description -> how well the learned model RECALLS each name token
# = its learned support), answer unchanged (probes/name_swap_questions_ml.json). The test is
# WITHIN each language: do name tokens with more learned support survive that language's own
# unlearning more often? Independent of 15_name_swap -- both can run at the same time.
# 304 sequences per checkpoint, seconds of compute; ~10 min per checkpoint is loading.
# 8 checkpoints, ~1.5 h. Saves after each. Print every rewritten question first (no GPU):
#     python studies/learn_french/scripts/per_token_map.py --dry-run \
#         --swap-file studies/learn_french/probes/name_swap_questions_ml.json
# Analysis and figure, locally: plots/plot_name_swap.py
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/nameswapml"; mkdir -p "$TMPDIR"
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
    # Same 8 as 09: base, learned ceiling, never-taught floor, then all five unlearning
    # arms at their pre-registered matched depths.
    CKPTS=("$BASE"
           "$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
           "$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr")
    while read -r lang tag; do
        CKPTS+=("${TR}_ul${lang}_floornone_${tag}")
    done < <(python3 -c "
import json
m = json.load(open('studies/learn_french/preregistration.json'))['stage3_matched_checkpoints']
for lang in ('fr', 'en', 'id', 'ja', 'ru'):
    print(lang, ('tr%.3f' % m[lang]['level']).replace('.', 'p'))")
fi

# Fail before loading anything if a local checkpoint has gone missing.
for c in "${CKPTS[@]}"; do
    case "$c" in
        /*) [ -d "$c" ] || { echo "MISSING local checkpoint: $c" >&2; exit 1; }
            echo "  local  $c" ;;
        *)  echo "  hub    $c" ;;
    esac
done

python studies/learn_french/scripts/per_token_map.py \
    --checkpoints "${CKPTS[@]}" \
    --swap-file studies/learn_french/probes/name_swap_questions_ml.json \
    --out "$UNLEARN_RESULTS_DIR/name_swap_ml.json"

echo "=== done -> $UNLEARN_RESULTS_DIR/name_swap_ml.json ==="
