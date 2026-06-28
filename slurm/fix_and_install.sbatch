#!/bin/bash
#SBATCH --job-name=fix_install
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/fix_install_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/fix_install_%j.err

PROJECT_DIR="/home/b/brandonk/unlearning"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"

echo "=== Removing corrupted sympy ==="
rm -rf "$PROJECT_DIR/.venv/lib/python3.12/site-packages/~ympy"
rm -rf "$PROJECT_DIR/.venv/lib/python3.12/site-packages/sympy"

echo "=== Reinstalling sympy cleanly ==="
pip install sympy

echo "=== Installing tiktoken and sentencepiece ==="
pip install tiktoken sentencepiece

echo "=== Verifying ==="
python3 -c "import tiktoken; print('tiktoken OK:', tiktoken.__version__)"
python3 -c "import sentencepiece; print('sentencepiece OK')"
