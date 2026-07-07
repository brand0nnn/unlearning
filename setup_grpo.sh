#!/bin/bash
# ===========================================================================
# ONE-TIME SETUP for the GRPO strategy ONLY — run on the LOGIN node:
#   bash setup_grpo.sh
#
# Creates an ISOLATED venv (.venv-grpo) with torch>=2.6 (cu124 wheel) + the
# latest trl. Why separate from the main .venv:
#   - the main .venv runs cu121 torch 2.5.1, and NO trl works on it:
#       * modern trl (>=0.16) imports FSDPModule -> needs torch>=2.6;
#       * the old trl that runs on 2.5.1 (0.14/0.15) hard-imports mergekit+vllm.
#   - on torch>=2.6 the LATEST trl treats vllm/mergekit as OPTIONAL (guarded)
#     imports, so `from trl import GRPOTrainer` loads with nothing extra and runs
#     on standard HF generation. LoRA-GRPO needs no DeepSpeed -> this venv is lean.
# The main venv (Full-FT / LoRA / self-distill / eval / spectral) is untouched.
#
# cu124 needs a CUDA 12.4+ driver — your H200 nodes have it. The GRPO sbatch
# prints nvidia-smi + torch.cuda.is_available() so the first run confirms it.
# ===========================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE_DIR="$PROJECT_DIR/.cache"

# Same $HOME-quota-avoiding redirects as setup.sh (§6).
export TMPDIR="$PROJECT_DIR/.tmp"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export HF_HOME="$CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$HF_DATASETS_CACHE"

echo "=== [1/3] Creating .venv-grpo ==="
python3 -m venv "$PROJECT_DIR/.venv-grpo"
source "$PROJECT_DIR/.venv-grpo/bin/activate"
pip install --upgrade pip --quiet

echo "=== [2/3] torch>=2.6 (cu124 wheel) ==="
pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124

echo "=== [3/3] GRPO deps (same HF stack as the main venv + latest trl; NO deepspeed) ==="
# torch is installed FIRST (above), so these loose '>=' pins won't move it — same
# trick as setup.sh. trl's vllm/mergekit stay optional and are NOT installed.
pip install \
  "transformers>=4.44" "datasets>=2.20" "accelerate>=0.33" "peft>=0.12" \
  trl rouge_score pyyaml numpy scipy scikit-learn sentencepiece tiktoken

echo ""
echo "=== .venv-grpo ready ==="
pip show trl | grep -E 'Version'    # print without importing (torch won't import on login)
echo "Run GRPO with:  sbatch slurm/pipeline/02_unlearn_grpo.sbatch"
