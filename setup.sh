#!/bin/bash
# ===========================================================================
# ONE-TIME SETUP — run this manually on the login node before your first job.
# It creates the venv, installs packages, and downloads TOFU + the base model.
#
# Usage:
#   bash setup.sh
#
# You only ever need to run this once. After that, just use sbatch.
# ===========================================================================

set -e  # stop on any error

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
CACHE_DIR="$PROJECT_DIR/.cache"

# Force ALL temporary files and caches onto the big filesystem.
# Without this, pip writes downloads to the system /tmp which hits the quota.
export TMPDIR="$PROJECT_DIR/.tmp"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

echo "=== [1/4] Creating virtual environment at $VENV_DIR ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== [2/4] Installing packages ==="
pip install --upgrade pip --quiet
pip install -r "$PROJECT_DIR/requirements.txt"

echo "=== [3/4] Downloading TOFU dataset ==="
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

echo "=== [4/4] Downloading base model ==="
python3 - << 'PYEOF'
import yaml, os
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)
name = cfg["model"]["name"]
hf_home = os.environ["HF_HOME"]
print(f"  Downloading model: {name}")
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained(name, cache_dir=hf_home)
AutoModelForCausalLM.from_pretrained(name, cache_dir=hf_home)
print("  Model download complete.")
PYEOF

echo ""
echo "=== Setup complete! ==="
echo "Your venv   : $VENV_DIR"
echo "Cache       : $CACHE_DIR"
echo ""
echo "To run the pipeline:"
echo "  sbatch slurm/run_tofu.sbatch"