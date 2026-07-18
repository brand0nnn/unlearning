#!/bin/bash
#SBATCH --job-name=loc_spectral
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/loc_spectral_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/loc_spectral_%j.err

# LoRA-LOCALITY ablation, SPECTRAL stage. Fingerprint each location's unlearned LoRA
# checkpoint vs the learned model (activation trace on forget-irrelevant prompts).
# Inference only (no DeepSpeed) -> fits a100-40, the less-contended GPU. Metrics ->
# lora_locality/out/<scheme>/spectral/ (NOT the main results/ folder).
#
#   Usage:  sbatch slurm/ablation/lora_locality_spectral.sbatch <scheme> [locs]
#     <scheme> = samerank | fixedbudget         (default fixedbudget)
#     <locs>   = optional space-separated subset (default: all 7)
#   Requires the matching lora_locality_unlearn.sbatch <scheme> checkpoints to exist.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/loc_spectral"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

SCHEME="${1:-fixedbudget}"
LOCS="${2:-attn qkv qv mlp updown down all}"
echo "Node: $(hostname)  scheme=$SCHEME  locations=$LOCS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python lora_locality/spectral.py --scheme "$SCHEME" \
    --learned "$PROJECT_DIR/experiments/tofu_learn_full_full" --locations "$LOCS"

echo "=== done -> lora_locality/out/${SCHEME}/spectral/*.json ==="
echo "    plot LOCALLY: python lora_locality/plot_spectral.py --scheme ${SCHEME}"
