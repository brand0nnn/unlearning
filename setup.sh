#!/bin/bash
# ===========================================================================
# ONE-TIME SETUP — run this manually on the login node before your first job.
# Usage:  bash setup.sh
# ===========================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
CACHE_DIR="$PROJECT_DIR/.cache"

# Force ALL temporary files and caches onto the big filesystem.
export TMPDIR="$PROJECT_DIR/.tmp"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

echo "=== [1/3] Creating virtual environment at $VENV_DIR ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== [2/3] Installing packages ==="
pip install --upgrade pip --quiet
pip install -r "$PROJECT_DIR/requirements.txt"

echo "=== [3/3] Downloading TOFU dataset ==="
export HF_HOME="$CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

python3 - << 'PYEOF'
from datasets import load_dataset
import os
cache = os.environ["HF_DATASETS_CACHE"]
for config in ["full", "retain90", "forget10",
               "forget10_perturbed", "retain_perturbed",
               "real_authors_perturbed", "world_facts_perturbed"]:
    print(f"  Downloading TOFU config: {config}")
    load_dataset("locuslab/TOFU", config, split="train", cache_dir=cache)
print("  TOFU download complete.")
PYEOF

echo ""
echo "=== Setup complete! ==="
echo "Note: the base model (phi-2) will be downloaded automatically"
echo "      on the compute node when you first run the job."
echo ""
echo "To run the pipeline:"
echo "  sbatch slurm/run_tofu.sbatch"