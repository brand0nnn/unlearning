#!/bin/bash
#SBATCH --job-name=coupling
#SBATCH --partition=gpu,gpu-long
#SBATCH --exclude=xgpj0
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/coupling_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/coupling_%j.err
#
# Coupling scores at fr_ft (side_checks/scripts/coupling_scores.py): carry-over for every
# candidate unlearning set (forget01 in en id ru ja ko hi ar fa iw, romaji ja, two retain
# controls) and gradient alignment with the French target, per fact and summed, per layer.
# No training, no checkpoint written. Needs >= ~55 GB GPU (bf16 weights + grads + one
# gradient snapshot), hence a100-80. Expected well under 2 h, so it fits gpu's ~3 h wall.
# xgpj0 is excluded: it is an aarch64 node and the project venv is x86_64 (regex .so fails).
#
#     sbatch studies/learn_french/side_checks/slurm/26_coupling_scores.sbatch
set -euo pipefail
PROJECT_DIR="/home/b/brandonk/unlearning"
SIDE="studies/learn_french/side_checks"
cd "$PROJECT_DIR"
mkdir -p logs "$SIDE/results/coupling"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/coupling"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"
MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
FR_FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
FR_RETAIN="$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr"
[ -d "$FR_FT" ] && [ -d "$FR_RETAIN" ] || { echo "missing fr_ft / fr_retain" >&2; exit 1; }

python3 -u "$SIDE/scripts/coupling_scores.py" --fr-ft "$FR_FT" --fr-retain "$FR_RETAIN" \
    --out "$PROJECT_DIR/$SIDE/results/coupling/scores.json"
