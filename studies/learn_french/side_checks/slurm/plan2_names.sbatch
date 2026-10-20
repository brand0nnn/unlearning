#!/bin/bash
#SBATCH --job-name=side_p2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/side_p2_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/side_p2_%j.err

# SIDE CHECK, PLAN 2 on the checkpoints that already exist: base, fr_ft, fr_retain and
# the five pre-registered matched checkpoints (ja's matched checkpoint IS its epoch 25).
# The epoch-25 fr/en checkpoints are scored inside plan1_arm.sbatch before deletion.
#
#     sbatch studies/learn_french/side_checks/slurm/plan2_names.sbatch
#     sbatch studies/learn_french/side_checks/slurm/plan2_names.sbatch <ckpt> ...
#
# 46 names x 7 tasks x 5 instruction languages = 1,610 greedy generations + 1,770
# teacher-forced reference scorings per model; ~10 min load + ~5 min work, ~2h for 8.
# Read-only on the checkpoints. A model whose JSON exists is skipped, so if the 3h wall
# cuts it short, resubmit the same command.
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
SIDE="studies/learn_french/side_checks"
cd "$PROJECT_DIR"
mkdir -p logs "$SIDE/results/plan2"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/side_p2"; mkdir -p "$TMPDIR"
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
for c in "${CKPTS[@]}"; do
    case "$c" in
        /*) [ -d "$c" ] || { echo "MISSING local checkpoint: $c" >&2; exit 1; }; echo "  local $c" ;;
        *)  echo "  hub   $c" ;;
    esac
done

python "$SIDE/scripts/name_skill.py" --checkpoints "${CKPTS[@]}" --out-dir "$SIDE/results/plan2"
echo "=== done -> $SIDE/results/plan2/"
