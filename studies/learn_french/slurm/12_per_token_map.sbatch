#!/bin/bash
#SBATCH --job-name=pertoken
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/pertoken_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/pertoken_%j.err

# PLAN 1 -- WHERE DID THE UNLEARNING GO? Per-token log-probs on the TRAINED text.
#
#     sbatch studies/learn_french/slurm/12_per_token_map.sbatch              # the 8 defaults
#     sbatch studies/learn_french/slurm/12_per_token_map.sbatch <ckpt> ...   # your own list
#
# Every answer token of forget01 in fr/en/id/ja/ru (200 rows) + 200 strided retain99_fr
# rows, scored exactly as LEARN/UNLEARN trained them ([INST] wrapper, joint tokenisation,
# EOS), at the same 8 checkpoints as Stage 5. One forward per row: ~400 per checkpoint,
# seconds of compute, so the job is almost all model loading (~10 min per checkpoint as
# in 09/10). Expect ~1.5 h. Saves after each checkpoint. Independent of 13_many_slots --
# separate output, read-only on the checkpoints -- so the two can run at the same time.
# Preview the tokenisation first (needs transformers, no GPU):
#     python studies/learn_french/scripts/per_token_map.py --dry-run
# Tagging and figures are local: plots/plot_per_token_map.py.
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/pertoken"; mkdir -p "$TMPDIR"
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
    --out "$UNLEARN_RESULTS_DIR/per_token_map.json"

echo "=== done -> $UNLEARN_RESULTS_DIR/per_token_map.json ==="
