#!/bin/bash
# ===========================================================================
# ONE-TIME SETUP — run on the LOGIN NODE.  Usage:  bash setup.sh
#
# Creates the venv, installs CUDA-matched torch + all packages, and downloads
# every TOFU split. Does NOT download the model (torch can't load on the login
# node; the model downloads automatically on the compute node on first run).
# ===========================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE_DIR="$PROJECT_DIR/.cache"

# Redirect ALL temp/cache onto project storage (the home system-partition quota
# is tiny; pip/HF temp files blow it otherwise).
export TMPDIR="$PROJECT_DIR/.tmp"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export HF_HOME="$CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

echo "=== [1/4] Creating virtual environment ==="
python3 -m venv "$PROJECT_DIR/.venv"
source "$PROJECT_DIR/.venv/bin/activate"
pip install --upgrade pip --quiet

echo "=== [2/4] Installing torch (CUDA 12.1 wheel, matches the cluster driver) ==="
# The default pip torch may run on CPU or mismatch the driver — install the
# cu121 wheel explicitly. This is the build that ran correctly on the A100s.
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== [3/4] Installing the rest of the packages ==="
# Includes bitsandbytes (see requirements.txt): needed for 8-bit AdamW + 4-bit
# NF4 loading so 7B full-parameter fine-tuning/unlearning fits on a 40GB A100.
# The TMPDIR/PIP_CACHE_DIR redirects above keep the download off the tiny $HOME
# quota partition — installing bitsandbytes by hand without them fails with
# "[Errno 122] Disk quota exceeded".
pip install -r "$PROJECT_DIR/requirements.txt"

echo "=== [4/4] Downloading all TOFU splits ==="
python3 - << 'PYEOF'
from datasets import load_dataset
import os
cache = os.environ["HF_DATASETS_CACHE"]
configs = [
    "full", "retain90", "forget10",
    "forget10_perturbed", "retain_perturbed",
    "real_authors_perturbed", "world_facts_perturbed",
]
for c in configs:
    print(f"  Downloading TOFU config: {c}")
    load_dataset("locuslab/TOFU", c, split="train", cache_dir=cache)
print("  TOFU download complete.")
PYEOF

echo ""
echo "=== Setup complete ==="
echo "The model in config.yaml (currently meta-llama/Llama-2-7b-hf, gated: needs HF"
echo "license acceptance + HF_TOKEN) downloads automatically on the compute node"
echo "during 01_learn."
echo ""
echo "Run the pipeline in order:"
echo "  sbatch slurm/01_learn.sbatch       # then wait for it to finish"
echo "  sbatch slurm/02_unlearn.sbatch"
echo "  sbatch slurm/03_evaluate.sbatch"
echo "  sbatch slurm/04_plot.sbatch"
