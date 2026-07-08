#!/bin/bash
#SBATCH --job-name=tofu_unlearn_grpo
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/unlearn_grpo.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/unlearn_grpo.err

# GRPO unlearning (strategy axis #4, HIGHEST RISK — RL can be unstable).
# FULL-PARAMETER GRPO so it matches Full-FT / self-distill's parametrization (no
# LoRA confound): DeepSpeed ZeRO-3 + on-policy generation, on an H200 (needs the
# headroom — full-param 7B + RL rollouts). Produces experiments/..._grpo (no _lora).
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
# GRPO runs in its OWN venv (torch>=2.6 + modern trl) — NOT the main .venv, whose
# cu121 torch 2.5.1 can't run any trl. Build it once: bash setup_grpo.sh
if [ ! -d "$PROJECT_DIR/.venv-grpo" ]; then
  echo "ERROR: .venv-grpo missing — run 'bash setup_grpo.sh' on the login node first." >&2
  exit 1
fi
source "$PROJECT_DIR/.venv-grpo/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

# Full-param GRPO uses DeepSpeed ZeRO-3 (config/ds_config.json). .venv-grpo is built
# lean by setup_grpo.sh; ensure deepspeed is present (idempotent, cached -> no-op
# after the first run). Nothing else in .venv-grpo needs it.
pip install -q deepspeed

# Full nvidia-smi shows the driver's CUDA version; the python line confirms the
# cu124 torch build actually sees the GPU (first-run proof that cu124 fits the driver).
echo "=== Node: $(hostname) ==="; nvidia-smi
python -c "import torch, trl, deepspeed; print(f'torch {torch.__version__} | cuda {torch.version.cuda} | avail {torch.cuda.is_available()} | trl {trl.__version__} | deepspeed {deepspeed.__version__}')"

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"
# Unique DeepSpeed rendezvous port (below the ephemeral range) so it never clashes
# on the default 29500 (the EADDRINUSE trap).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "=== GRPO unlearning (FULL-PARAMETER, DeepSpeed ZeRO-3) ==="
# NO --lora -> full-parameter GRPO -> checkpoint tofu_unlearn_grpo_forget10_grpo.
# --track-curve logs ROUGE/Prob/Truth-Ratio per step -> results/curves/*.json.
deepspeed --num_gpus=1 --master_port=$MASTER_PORT scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy grpo --track-curve

echo "=== GRPO unlearn complete ==="
