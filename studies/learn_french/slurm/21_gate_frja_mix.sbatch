#!/bin/bash
#SBATCH --job-name=gate_frja_mix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/gate_frja_mix_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/gate_frja_mix_%j.err

# STAGE 9, step 1 -- THE GATE for the MIXED model. Did fr-ja-mix_ft store each forget fact in
# French AND in Japanese, asked and answered in ONE language? The mixed model never saw a
# French question with a French answer for the 30/40 forget facts with no in-answer switch,
# so this is not guaranteed (Skow et al.: learning transfers mostly on the question side).
#
#     sbatch studies/learn_french/slurm/21_gate_frja_mix.sbatch
#
# Scores, in ONE pass (scoring is not bit-reproducible run to run, so the control is re-scored
# here rather than read from results/gate_frja/):
#     fr-ja-mix_ft       under test
#     fr-ja_ft           the CONTROL (Stage 8: same rows, monolingual)
#     fr-ja_retain       the floor both are measured against
#     fr-ja-mix_retain   the matched floor, ONLY if 20_learn_frja_mix retain99 was run
# ~15 min per checkpoint, pure inference. Writes results/gate_frja_mix/<checkpoint>.json; read
#     python studies/learn_french/scripts/frja_mix_report.py gate
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/gate_frja_mix"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
E="$PROJECT_DIR/experiments"
CKPTS=("$E/tofu_learn_full_full_${MODEL_SLUG}_fr-ja-mix"     # fr-ja-mix_ft  under test
       "$E/tofu_learn_full_full_${MODEL_SLUG}_fr-ja"         # fr-ja_ft      control
       "$E/tofu_learn_retain99_full_${MODEL_SLUG}_fr-ja")    # fr-ja_retain  floor
MIX_FLOOR="$E/tofu_learn_retain99_full_${MODEL_SLUG}_fr-ja-mix"
[ -d "$MIX_FLOOR" ] && CKPTS+=("$MIX_FLOOR")

echo "=== [1/2] checkpoint check ==="
for d in "${CKPTS[@]}"; do
    [ -d "$d" ] || { echo "MISSING checkpoint: $d" >&2; exit 1; }
    # A resumable LEARN that stopped early has a marker and no final weights yet: the
    # afterok dependency fires when the FIRST link exits, even if it resubmitted itself.
    [ ! -f "$d/RESUME_PENDING" ] || { echo "UNFINISHED (RESUME_PENDING): $d -- resubmit" \
        "this job once the resumed LEARN completes" >&2; exit 1; }
    for f in config.json tokenizer_config.json; do
        [ -f "$d/$f" ] || { echo "MISSING $d/$f" >&2; exit 1; }
    done
    echo "  OK $d"
done

echo "=== [2/2] scoring the 40 forget facts in fr and ja ==="
python studies/learn_french/scripts/gate_frja.py --checkpoints "${CKPTS[@]}" --langs fr ja \
    --group gate_frja_mix

echo "=== GATE (mixed fr+ja) complete -> $UNLEARN_RESULTS_DIR/gate_frja_mix/ ==="
ls -la "$UNLEARN_RESULTS_DIR/gate_frja_mix/"
