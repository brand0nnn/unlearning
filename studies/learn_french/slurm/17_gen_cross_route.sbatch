#!/bin/bash
#SBATCH --job-name=gencross
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/b/brandonk/unlearning/logs/gencross_%A_%a.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/gencross_%A_%a.err

# ASK IN LANGUAGE L, LET THE MODEL WRITE ITS ANSWER IN FRENCH -- Q1-access as behaviour.
#
#     sbatch studies/learn_french/slurm/17_gen_cross_route.sbatch
#
# Stage 5 / Plan 2 measured access teacher-forced. Stage 2's generations say that, asked in
# French, every unlearning arm shuts the fact equally (2-4/40 said vs 37/40 learned) even
# where the probability measure calls the French route ~86% intact. This generates on the
# foreign question routes too: fr (Stage 2's exact prompt, a reproduction check), fr+I, and
# en/id/ja/ru + Xiang App. G's "answer only in French" instruction. Greedy, graded with the
# Stage 1 NLI scorer (Xiang Eq. 4). See scripts/gen_cross_route.py.
#
# ONE ARRAY TASK PER CHECKPOINT (7): fr_ft (ceiling + gate), fr_retain (never-taught floor),
# then the five matched unlearning arms. Base Qwen3 is left out: fr_retain is the floor that
# every other number in the study is read against. 240 unbatched generations per task,
# ~20-30 min each including model + NLI loading; tasks run in parallel when GPUs are free.
# Each writes results/gen_cross_route/<name>.json. Preview every prompt, no GPU:
#     python studies/learn_french/scripts/gen_cross_route.py --dry-run
# Analysis locally: plots/plot_gen_cross_route.py
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/gencross_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
TR="$PROJECT_DIR/experiments/tr_levels/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"

CKPTS=("$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
       "$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr")
while read -r lang tag; do
    CKPTS+=("${TR}_ul${lang}_floornone_${tag}")
done < <(python3 -c "
import json
m = json.load(open('studies/learn_french/preregistration.json'))['stage3_matched_checkpoints']
for lang in ('fr', 'en', 'id', 'ja', 'ru'):
    print(lang, ('tr%.3f' % m[lang]['level']).replace('.', 'p'))")

CKPT="${CKPTS[${SLURM_ARRAY_TASK_ID:-0}]}"
[ -d "$CKPT" ] || { echo "MISSING local checkpoint: $CKPT" >&2; exit 1; }
echo "task ${SLURM_ARRAY_TASK_ID:-0}: $CKPT"

python studies/learn_french/scripts/gen_cross_route.py \
    --checkpoint "$CKPT" \
    --out-dir "$UNLEARN_RESULTS_DIR/gen_cross_route"

echo "=== done -> $UNLEARN_RESULTS_DIR/gen_cross_route/ ==="
