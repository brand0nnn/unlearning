#!/bin/bash
#SBATCH --job-name=bridge_gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/bridge_gate_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/bridge_gate_%j.err

# PLAN 5, bridge GATE (side_checks/preregistration_plan5.json `bridge.gate`): did the bridge
# LEARN raise how much of the French-learned fact the model carries into Japanese? Pure
# inference, no training. Read it locally with scripts/bridge_gate_report.py.
#
#     sbatch studies/learn_french/slurm/25_bridge_gate.sbatch
#
#   [1] per_token_map.py on fr_ft, fr_retain, bridge (fr-ja-fgfr_ft), fr-ja_retain -- one
#       scoring pass, so the fr_ft carry-over it reproduces (en ~0.64, ja ~0.19 log units)
#       is directly comparable to the bridge's                                    ~45 min
#   [2] measure_fr.py on the bridge AND fr_ft: the French ceiling (the bridge's learned
#       TR / P(gold) must be as strong as fr_ft's, or its arms start somewhere else)  ~1 h
set -euo pipefail

PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
OUT="$UNLEARN_RESULTS_DIR/bridge_gate"
mkdir -p logs "$OUT"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/bridge_gate"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
E="$PROJECT_DIR/experiments"
FR_FT="$E/tofu_learn_full_full_${MODEL_SLUG}_fr"
FR_RETAIN="$E/tofu_learn_retain99_full_${MODEL_SLUG}_fr"
BRIDGE="$E/tofu_learn_full_full_${MODEL_SLUG}_fr-ja-fgfr"
FRJA_RETAIN="$E/tofu_learn_retain99_full_${MODEL_SLUG}_fr-ja"
for c in "$FR_FT" "$FR_RETAIN" "$BRIDGE" "$FRJA_RETAIN"; do
    [ -f "$c/tokenizer_config.json" ] || { echo "MISSING checkpoint: $c" >&2; exit 1; }
done

if [ -f "$OUT/per_token_map.json" ]; then echo "[1] per-token map -- skip"; else
    echo "[1] per-token map: fr_ft, fr_retain, bridge, fr-ja_retain"
    python studies/learn_french/scripts/per_token_map.py \
        --checkpoints "$FR_FT" "$FR_RETAIN" "$BRIDGE" "$FRJA_RETAIN" \
        --out "$OUT/per_token_map.json"
fi

python -c "from huggingface_hub import snapshot_download; snapshot_download('joeddav/xlm-roberta-large-xnli')" >/dev/null
echo "[2] measure_fr: the bridge's French ceiling beside fr_ft's, in one scoring pass"
# fr_ft is re-scored here so the ceiling comparison is within one scoring group (scoring is
# not run-to-run deterministic). --reference only feeds Forget Quality's KS test.
python studies/learn_french/scripts/measure_fr.py \
    --checkpoints "$BRIDGE" "$FR_FT" --reference "$FRJA_RETAIN" --group bridge_gate/measure

echo "=== done -> $OUT   read: python3 studies/learn_french/scripts/bridge_gate_report.py ==="
