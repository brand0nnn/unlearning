#!/bin/bash
#SBATCH --job-name=crossroute
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/crossroute_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/crossroute_%j.err

# ASK IN LANGUAGE L, SCORE THE FRENCH ANSWER.
#
#     sbatch studies/learn_french/slurm/10_cross_route.sbatch              # the 8 defaults
#     sbatch studies/learn_french/slurm/10_cross_route.sbatch <ckpt> ...   # your own list
#
# 09 scored the answer IN L and mostly failed its transfer gate: French-only LEARN never put
# the facts into ja/ru. Skow et al. (arXiv 2609.40286) find learning transfers on the
# QUESTION side, so this keeps the answer French and varies only the question language:
# blank / fr / fr_p1 / en / id / ja / ru. See the docstring of scripts/cross_route.py for
# what a route can and cannot show (the French stump carries much of the cue; `blank` is
# the floor each route is read against).
#
# 14 typed slots (6 common-noun, 7 proper-noun, 1 year) x 7 routes = 98 cells per checkpoint,
# 8 checkpoints. Pure inference, no DeepSpeed, nothing written to experiments/. 09 ran at
# ~10 min per checkpoint, nearly all model loading; the extra cells add a few minutes each.
# Expect ~1.5-2 h. Preview every prompt on the login node first, no GPU needed:
#     python studies/learn_french/scripts/cross_route.py --dry-run
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/crossroute"; mkdir -p "$TMPDIR"
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

python studies/learn_french/scripts/cross_route.py \
    --checkpoints "${CKPTS[@]}" \
    --out "$UNLEARN_RESULTS_DIR/cross_route.json"

echo "=== done -> $UNLEARN_RESULTS_DIR/cross_route.json ==="
