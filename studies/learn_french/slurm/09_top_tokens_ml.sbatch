#!/bin/bash
#SBATCH --job-name=toptokml
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/toptokml_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/toptokml_%j.err

# DOES THE FACT EXIST IN THE OTHER LANGUAGES AT ALL?
#
#     sbatch studies/learn_french/slurm/09_top_tokens_ml.sbatch              # the 8 defaults
#     sbatch studies/learn_french/slurm/09_top_tokens_ml.sbatch <ckpt> ...   # your own list
#
# 08 asked ONE French question and read its next-token distribution five ways. That says
# where the French answer went, but not whether the fact is in Japanese -- a French prompt
# conditions the whole distribution on French, and the non-Latin tail just tracks whatever
# the model wants to say in French (884103: French ' phot' -> Cyrillic ' фотограф').
#
# This asks the question IN each language, so the five measurements are independent.
#
#   1. TRANSFER   fr_ft vs fr_retain, asked in Japanese. Same French fine-tuning, one of
#                 them never shown the fact -- so a gap IS the fact. No gap closes the
#                 question: nothing reached Japanese, so nothing there can be hiding.
#   2. HIDING     unl_fr vs fr_ft, asked in Japanese. Read only where 1 found something.
#
# 8 checkpoints x 5 languages x 4 slots = 160 forward passes. Pure inference, no DeepSpeed,
# nothing written to experiments/ -- fits the 3h `gpu` wall and an a100-40 with room spare.
# ~10 min per checkpoint, nearly all of it model loading.
#
# RUN 885110 SCORED ONLY THE FIRST TOKEN OF EACH TARGET, and 15 of its 20 targets are
# multi-token, so it asked "how likely is 'Б'?" when the question was "how likely is
# 'Базилий'?". Five first tokens are also generic in their own language (' pen' is rank 0
# in Indonesian after "adalah seorang" whatever the fact is). 9 of 20 cells were unusable.
# Targets are now teacher-forced across ALL their tokens and length-normalised -- TOFU's
# own P(a|q)^(1/|a|). Same 8 checkpoints, same wall, ~18 usable cells instead of 11.
#
# One cell is excluded rather than fixed: the published Indonesian translation of fact 3
# says BASIL is the florist, not his father, so that cell measures a different proposition.
# It is reported as a MISS with its reason.
#
# FIRST RUN IS ALSO A CALIBRATION RUN. The ja/id/ru targets in probes/token_probes_ml.json
# are guesses against translations nobody has read, so the script prints every gold answer
# before it loads a model and names each slot whose candidates missed. Correct the probe
# file from that log and resubmit -- a miss costs one cell, never the job.
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/toptokml"; mkdir -p "$TMPDIR"
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
    # base Qwen3 FIRST (every 'vs ref' column is relative to it), then the learned
    # ceiling, the never-taught floor, and ALL FIVE unlearning arms at their
    # pre-registered matched depths. en and id are included here, unlike in 08: asking
    # the question in each language gives every arm its own distribution, so the shared
    # proper-noun token that made them indistinguishable there does not apply.
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

# A hub id like "Qwen/Qwen3-8B" also contains a slash, so the discriminator is a LEADING
# slash: local checkpoints are absolute paths, hub ids never are.
for c in "${CKPTS[@]}"; do
    case "$c" in
        /*) [ -d "$c" ] || { echo "MISSING local checkpoint: $c" >&2; exit 1; }
            echo "  local  $c" ;;
        *)  echo "  hub    $c  (downloaded into HF_HOME if not cached)" ;;
    esac
done

python studies/learn_french/scripts/top_tokens_ml.py \
    --checkpoints "${CKPTS[@]}" \
    --out "$UNLEARN_RESULTS_DIR/top_tokens_ml.json"

echo "=== done -> $UNLEARN_RESULTS_DIR/top_tokens_ml.json ==="
