#!/bin/bash
#SBATCH --job-name=tofu_plot
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/plot.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/plot.err

PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"

python scripts/tofu_04_plot.py

echo "=== Plots written to results/ ==="
