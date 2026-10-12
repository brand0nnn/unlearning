#!/bin/bash
#SBATCH --job-name=gate_frja
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/gate_frja_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/gate_frja_%j.err

# Q1 STORAGE, step 1 -- THE GATE. Did the joint LEARN (14_learn_frja) store each forget
# fact in JAPANESE? Nothing downstream (unlearn in French, probe Japanese) means anything on
# a fact whose Japanese copy was never there.
#
#     sbatch studies/learn_french/slurm/18_gate_frja.sbatch
#
# Needs BOTH joint models finished (fr-ja_ft AND its floor fr-ja_retain); fails in the first
# second otherwise. Pure inference, 4 checkpoints x 2 languages x 40 facts (+ one greedy
# generation each), ~1 h; an 8B model in bf16 fits an a100-40, so the short `gpu` queue.
# Writes results/gate_frja/<checkpoint>.json; read it locally with
#     python studies/learn_french/scripts/gate_frja_report.py
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/gate_frja"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
E="$PROJECT_DIR/experiments"
CKPTS=("$E/tofu_learn_full_full_${MODEL_SLUG}_fr-ja"       # fr-ja_ft      under test
       "$E/tofu_learn_retain99_full_${MODEL_SLUG}_fr-ja"   # fr-ja_retain  its floor
       "$E/tofu_learn_full_full_${MODEL_SLUG}_fr"          # fr_ft         reference
       "$E/tofu_learn_retain99_full_${MODEL_SLUG}_fr")     # fr_retain     reference floor

echo "=== [1/2] checkpoint check ==="
for d in "${CKPTS[@]}"; do
    [ -d "$d" ] || { echo "MISSING checkpoint: $d (is 14_learn_frja still running?)" >&2; exit 1; }
    # A checkpoint saved without its tokenizer evaluates to all zeros, silently.
    [ -f "$d/tokenizer_config.json" ] || { echo "MISSING tokenizer in $d" >&2; exit 1; }
    # A resumable LEARN that stopped early has a marker and no final weights yet.
    [ ! -f "$d/RESUME_PENDING" ] || { echo "UNFINISHED (RESUME_PENDING): $d" >&2; exit 1; }
    echo "  OK $d"
done

echo "=== [2/2] scoring the 40 forget facts in fr and ja ==="
python studies/learn_french/scripts/gate_frja.py --checkpoints "${CKPTS[@]}" --langs fr ja

echo "=== GATE (fr ja) complete -> $UNLEARN_RESULTS_DIR/gate_frja/ ==="
ls -la "$UNLEARN_RESULTS_DIR/gate_frja/"
