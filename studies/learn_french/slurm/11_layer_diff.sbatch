#!/bin/bash
#SBATCH --job-name=layerdiff
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/layerdiff_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/layerdiff_%j.err

# WHERE DID EACH UNLEARNING LANGUAGE CHANGE THE MODEL?  CPU only -- no --gres, no GPU.
#
#     bash  studies/learn_french/slurm/11_layer_diff.sbatch --dry-run   # login node, seconds
#     sbatch studies/learn_french/slurm/11_layer_diff.sbatch            # the real run
#
# No --partition is set, so SLURM uses the cluster default. If that default only takes GPU
# jobs, check `sinfo -s` for a CPU partition and add:  sbatch --partition=<name> ...
#
# Pairs, each REF::ARM:
#   unl_fr .. unl_ru   fr_ft -> each arm at its pre-registered matched depth (the question)
#   learn              base Qwen3 -> fr_ft, where LEARNING acted (a reference shape);
#                      skipped with a warning if base is not in the HF cache
#
# ~96-112 GB read from shared storage, streamed one tensor at a time; expect 20-40 min,
# up to an hour if the filesystem is busy. See scripts/layer_diff.py for the bf16 caveat.
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/layerdiff"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1

echo "Node: $(hostname)   Commit: $(git rev-parse --short HEAD)"

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
BASE_ID=$(python3 -c "import yaml;print(yaml.safe_load(open('config/config.yaml'))['model']['name'])")
FT="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}_fr"
TR="$PROJECT_DIR/experiments/tr_levels/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"

PAIRS=()
while read -r lang tag; do
    ARM="${TR}_ul${lang}_floornone_${tag}"
    [ -e "$ARM" ] || { echo "MISSING checkpoint: $ARM" >&2; exit 1; }
    PAIRS+=(--pair "unl_${lang}=${FT}::${ARM}")
done < <(python3 -c "
import json
m = json.load(open('studies/learn_french/preregistration.json'))['stage3_matched_checkpoints']
for lang in ('fr', 'en', 'id', 'ja', 'ru'):
    print(lang, ('tr%.3f' % m[lang]['level']).replace('.', 'p'))")

if BASE=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$BASE_ID', local_files_only=True))" 2>/dev/null); then
    PAIRS+=(--pair "learn=${BASE}::${FT}")
else
    echo "WARNING: $BASE_ID not in the HF cache -- skipping the base->fr_ft reference pair"
fi

python studies/learn_french/scripts/layer_diff.py "${PAIRS[@]}" \
    --out "$UNLEARN_RESULTS_DIR/layer_diff.json" "$@"

echo "=== done ==="
