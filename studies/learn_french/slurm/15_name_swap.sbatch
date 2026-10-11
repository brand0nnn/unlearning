#!/bin/bash
#SBATCH --job-name=nameswap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/nameswap_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/nameswap_%j.err

# NAME SWAP -- is a name protected from unlearning because it is a NAME, or because the
# question already contains it (so the answer just copies it)?
#
#     sbatch studies/learn_french/slurm/15_name_swap.sbatch              # the 8 defaults
#     sbatch studies/learn_french/slurm/15_name_swap.sbatch <ckpt> ...   # your own list
#
# The 38 French forget answers whose question names the author, each scored under four
# questions with the ANSWER unchanged (probes/name_swap_questions.json): orig (name in the
# question -> copying), A birth date and B book title (name must be RECALLED), C 'cet auteur'
# (no identity -> guessing floor). Teacher-forced per-token scoring, same trained [INST]
# frame and same script as 12_per_token_map. 152 sequences per checkpoint -- seconds of
# compute; ~10 min per checkpoint is model loading. 8 checkpoints, ~1.5 h. Saves after each.
# Check every rewritten question first (prints all 152, no GPU):
#     python studies/learn_french/scripts/per_token_map.py --dry-run \
#         --swap-file studies/learn_french/probes/name_swap_questions.json
# Analysis and figure, locally: plots/plot_name_swap.py
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/nameswap"; mkdir -p "$TMPDIR"
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
    --swap-file studies/learn_french/probes/name_swap_questions.json \
    --out "$UNLEARN_RESULTS_DIR/name_swap.json"

echo "=== done -> $UNLEARN_RESULTS_DIR/name_swap.json ==="
