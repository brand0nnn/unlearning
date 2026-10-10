#!/bin/bash
#SBATCH --job-name=manyslots
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/manyslots_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/manyslots_%j.err

# PLAN 2 -- THE MANY-SLOT PROBE. Stage 5's cross-route probe over 54 slots (Stage 5 had 14),
# coded by kind / repetition / copyability in probes/many_slot_probes.json.
#
#     sbatch studies/learn_french/slurm/13_many_slots.sbatch              # the 8 defaults
#     sbatch studies/learn_french/slurm/13_many_slots.sbatch <ckpt> ...   # your own list
#
# BOTH prompt wrappers: `qa` ("Question: ...\nAnswer: ", what Stage 5 used -- re-measures
# its 14 slots as a reproduction check) and `inst` ("[INST] ... [/INST]", what the models
# were trained on). 54 slots x 7 routes x 2 wrappers = 726 cells per checkpoint after the
# leak check. 10 ran 98 cells in ~10 min per checkpoint, nearly all loading; budget
# ~15 min per checkpoint here, ~2 h total against the 3 h wall. Saves after every
# checkpoint, so a wall-clock kill loses at most one. Independent of 12_per_token_map --
# the two can run at the same time. Preview every prompt first, no GPU:
#     python studies/learn_french/scripts/cross_route.py --dry-run \
#         --probes studies/learn_french/probes/many_slot_probes.json --templates qa inst
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/manyslots"; mkdir -p "$TMPDIR"
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
    --probes studies/learn_french/probes/many_slot_probes.json \
    --templates qa inst \
    --out "$UNLEARN_RESULTS_DIR/many_slots.json"

echo "=== done -> $UNLEARN_RESULTS_DIR/many_slots.json ==="
