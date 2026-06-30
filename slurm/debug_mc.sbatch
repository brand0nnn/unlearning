#!/bin/bash
#SBATCH --job-name=debug_mc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/debug_mc_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/debug_mc_%j.err

PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"

python debug_mc.py
