#!/bin/bash
#SBATCH --job-name=install
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/install_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/install_%j.err

PROJECT_DIR="/home/b/brandonk/unlearning"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"

pip install tiktoken sentencepiece
echo "Done"
