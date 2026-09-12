#!/bin/bash
#SBATCH --job-name=measure_ul
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/measure_ul_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/measure_ul_%j.err

# STAGE 2/3 AFTER-METRICS: score one language's TR-level checkpoints on the French
# forget set, with the SAME scorer Stage 1 used.
#
#     sbatch 04_measure_unlearned.sbatch <lang>
#
# Pure inference. This is what turns a saved checkpoint into the plan's five metrics:
# Truth Ratio (+ components), Probability, NLI, output language, Model Utility, and
# Forget Quality against fr_retain. The trajectory JSONL written during unlearning has
# TR and MU but NO generation-side metric -- generating 40 answers every 2 steps would
# have dominated the run -- so NLI and the TR-NLI gap come from here.
#
# CHECKPOINT PATHS ARE DERIVED, NOT GLOBBED, from the pre-registered levels (repo rule:
# always resolve exact paths under experiments/). A level whose directory is a SYMLINK
# shares weights with a lower level it was crossed alongside, and is skipped -- scoring
# it again would burn a GPU hour to reproduce a number we already have.
#
# fr_retain is scored alongside because Forget Quality is a TWO-model statistic and
# measure_fr computes the KS test against a reference inside the same run.
#
# Smaller GPU is fine (bf16 8B + the NLI model fit in 40GB) and may schedule sooner:
#     sbatch --partition=gpu --gres=gpu:a100-40:1 --time=03:00:00 04_measure_unlearned.sbatch en
set -euo pipefail

LANG_UL="${1:?usage: sbatch 04_measure_unlearned.sbatch <lang>}"
case "$LANG_UL" in en|fr|id|ja|ru) ;; *) echo "bad lang: $LANG_UL" >&2; exit 2 ;; esac

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/measure_ul_$LANG_UL"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
FR_RETAIN="$PROJECT_DIR/experiments/tofu_learn_retain99_full_${MODEL_SLUG}_fr"
RUN="tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}_ul${LANG_UL}"
[ -d "$FR_RETAIN" ] || { echo "MISSING fr_retain: $FR_RETAIN" >&2; exit 1; }

# Exact paths from the pre-registered levels -- the same tag the probe callback wrote.
mapfile -t TAGS < <(python3 -c "
import json
for lv in json.load(open('studies/learn_french/preregistration.json'))['tr_levels']:
    print(('tr%.3f' % lv).replace('.', 'p'))")

CKPTS=()
for tag in "${TAGS[@]}"; do
    p="$PROJECT_DIR/experiments/tr_levels/${RUN}_${tag}"
    if [ -L "$p" ]; then
        echo "  $tag: symlink -> $(readlink "$p") (same weights, crossed together) -- skipping"
    elif [ -d "$p" ]; then
        echo "  $tag: $p"
        CKPTS+=("$p")
    else
        echo "  $tag: NOT REACHED by this run"
    fi
done
[ ${#CKPTS[@]} -gt 0 ] || { echo "no level checkpoints for $LANG_UL -- nothing to score" >&2; exit 1; }

echo "=== scoring ${#CKPTS[@]} checkpoint(s) for lang=$LANG_UL + fr_retain reference ==="
python studies/learn_french/scripts/measure_fr.py \
    --checkpoints "${CKPTS[@]}" "$FR_RETAIN" \
    --reference   "$FR_RETAIN" \
    --group "stage2_${LANG_UL}"

echo "=== done -> $UNLEARN_RESULTS_DIR/stage2_${LANG_UL}/ ==="
ls -la "$UNLEARN_RESULTS_DIR/stage2_${LANG_UL}/"
