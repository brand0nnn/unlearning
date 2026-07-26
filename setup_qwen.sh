#!/bin/bash
# ===========================================================================
# ONE-TIME: make the MAIN .venv able to load Qwen3 — run on the LOGIN node:
#   bash setup_qwen.sh
#
# Qwen3 support landed in transformers 4.51.0 (below that: KeyError: 'qwen3').
# Qwen3 does NOT need torch>=2.6 — it runs on the existing cu121 torch 2.5.1, so
# this ONLY bumps transformers/tokenizers in place. The cu121 build stays, so
# a100 nodes keep working; no driver change, no new venv, no sbatch edits.
#
# (This is deliberately minimal. If you later move to Aya-Expanse or want GRPO in
# the same env, that's the torch>=2.6 path in setup_grpo.sh instead.)
# ===========================================================================
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

source "$PROJECT_DIR/.venv/bin/activate"
echo "=== upgrading transformers/tokenizers for Qwen3 (torch untouched) ==="
pip install -U "transformers>=4.51" "tokenizers>=0.21" "accelerate>=0.33"

echo ""
echo "=== versions (printed WITHOUT importing torch — login node can't) ==="
pip show transformers tokenizers accelerate | grep -E 'Name|Version'
echo ""
echo "Next (optional, on the LOGIN node): pre-cache the weights so the GPU job"
echo "doesn't spend its first minutes downloading ~16GB:"
echo "  HF_HOME=$HF_HOME huggingface-cli download Qwen/Qwen3-8B"
