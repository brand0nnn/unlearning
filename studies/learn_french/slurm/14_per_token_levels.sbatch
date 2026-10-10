#!/bin/bash
#SBATCH --job-name=ptlevels
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/ptlevels_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/ptlevels_%j.err

# PLAN 1 ALONG THE UNLEARNING TRAJECTORY: the per-token map at every saved depth level of
# ONE arm, so the ORDER in which tokens fall can be read (Stage 6 follow-up).
#
#     for L in fr en id ja ru; do sbatch studies/learn_french/slurm/14_per_token_levels.sbatch $L; done
#
# Plan 1 found that, at the matched level, the tokens unlearning destroyed are the ones the
# learned model was least sure of, and near-certain tokens (copied names, Basil) survive.
# One level cannot say whether near-certain tokens are IMMUNE or just LAST IN LINE. The
# pre-registered levels (0.650 ... 0.828, preregistration.json `tr_levels`) are saved
# checkpoints along each arm's run, so scoring all of them gives the dose-response.
#
# Pure inference, same script and texts as 12_per_token_map, so token keys line up with
# results/per_token_map.json (fr_ft / fr_retain are taken from there, not re-scored).
# Up to 5 checkpoints per arm at ~10 min each: ~1 h per arm, and the five arms are
# independent jobs that can run side by side. A level that was never saved is skipped by
# name; two levels crossed at one probe point are symlinks to the same weights and are
# scored once (both names recorded in the log).
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/ptlevels_${1:-x}"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"

LANG_UL="${1:?usage: sbatch 14_per_token_levels.sbatch <fr|en|id|ja|ru>}"
MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
TR="$PROJECT_DIR/experiments/tr_levels/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}_ul${LANG_UL}_floornone"

# Exact paths from the pre-registered levels -- never a glob over experiments/.
CKPTS=(); SEEN=()
for tag in $(python3 -c "
import json
for l in json.load(open('studies/learn_french/preregistration.json'))['tr_levels']:
    print(('tr%.3f' % l).replace('.', 'p'))"); do
    c="${TR}_${tag}"
    if [ ! -e "$c" ]; then echo "  not saved   $c"; continue; fi
    real=$(readlink -f "$c")
    if [[ " ${SEEN[*]-} " == *" $real "* ]]; then echo "  same weights $c -> $real (skipped)"; continue; fi
    SEEN+=("$real"); CKPTS+=("$c"); echo "  level       $c"
done
[ "${#CKPTS[@]}" -gt 0 ] || { echo "no level checkpoints found for $LANG_UL" >&2; exit 1; }

python studies/learn_french/scripts/per_token_map.py \
    --checkpoints "${CKPTS[@]}" \
    --out "$UNLEARN_RESULTS_DIR/per_token_levels_${LANG_UL}.json"

echo "=== done -> $UNLEARN_RESULTS_DIR/per_token_levels_${LANG_UL}.json ==="
