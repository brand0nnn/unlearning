#!/bin/bash
#SBATCH --job-name=xling_audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_audit_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_audit_%j.err

# CROSS-LINGUAL KNOWLEDGE AUDIT (eval-only; the DENOMINATOR for the pilot).
#
# The recovery pilot measured the forget fact ONLY in English, so it cannot tell
# whether the ko/ja "cliff" is (A) a Llama-2 coverage artifact (the model never knew
# the fact in ko/ja, so nothing to lose/recover) or (B) real localized forgetting.
# This job reads the forget fact DIRECTLY in all 10 languages on three models:
#
#   1. LEARNED (pre-unlearn)  -> the KNOWLEDGE FLOOR: how much does the model know the
#                                forget fact in each language BEFORE any unlearning?
#                                Near-zero in a language => that language is a dead
#                                channel; its pilot number is uninterpretable.
#   2. UNLEARN Full-FT        -> \  the DIRECT BLAST RADIUS: unlearn in EN, then read the
#   3. UNLEARN LoRA (matched) -> /  SAME fact across languages. drop vs the floor, per
#                                language, ordered by distance = how far EN unlearning
#                                collateral-damages the fact in other languages, by method.
#
# No relearning, no training -> cheap. Plain inference (no DeepSpeed).
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_audit"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

GROUP="crosslingual_audit"
LANGS="${1:-en fr id ru hi fa ar iw ko ja}"   # full distance axis (near -> far from EN)
FL="forget01"                                  # multilingual data is forget01 only
N=40                                           # all forget01 records

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"
FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft"
# The matched LoRA checkpoint the pilot kept (tofu_unlearn_..._forget01_lora_uep<NN>).
LORA=$(ls -d "$PROJECT_DIR"/experiments/tofu_unlearn_gradient_difference_forget01_lora_uep* 2>/dev/null | head -1)

for CKPT in "$LEARNED" "$FULLFT" "$LORA"; do
    if [ -z "$CKPT" ] || [ ! -d "$CKPT" ]; then
        echo "!! skipping missing checkpoint: '$CKPT'"; continue
    fi
    echo "=== audit $(basename "$CKPT") across: $LANGS ==="
    python shared/scripts/relearn_measure.py \
        --checkpoints "$CKPT" --group "$GROUP" --forget-level "$FL" --n "$N" \
        --measure-lang $LANGS
done

# --- summary table: forget ROUGE per model x language (floor vs unlearned) ---------
echo "=== [RESULT] forget-fact knowledge per language (higher = model knows it) ==="
python3 - "$LANGS" <<'PY'
import json, glob, os, sys
langs = sys.argv[1].split()
root = os.environ.get("UNLEARN_RESULTS_DIR", "results")
d = {}
for f in glob.glob(f"{root}/relearn/crosslingual_audit/*.json"):
    d.update(json.load(open(f)))
def val(base, lang):
    k = base if lang == "en" else f"{base}@{lang}"
    return d.get(k, float("nan"))
rows = [
    ("LEARNED (floor)", "tofu_learn_full_full"),
    ("UNLEARN Full-FT", "tofu_unlearn_gradient_difference_forget01_fullft"),
]
# LoRA base name varies by matched UEP -> find it
lora = next((k.split("@")[0] for k in d if "lora_uep" in k), None)
if lora:
    rows.append(("UNLEARN LoRA", lora))
print(f"{'model':18s} " + "  ".join(f"{l:>6s}" for l in langs))
for label, base in rows:
    print(f"{label:18s} " + "  ".join(f"{val(base,l):6.3f}" for l in langs))
print("--- blast-radius = floor - unlearned (drop the EN unlearning caused per language) ---")
floor = "tofu_learn_full_full"
for label, base in rows[1:]:
    print(f"{label:18s} " + "  ".join(f"{val(floor,l)-val(base,l):+6.3f}" for l in langs))
print("=> where floor≈0 (e.g. ko/ja), the pilot number there was never interpretable.")
PY
echo "=== audit done -> $UNLEARN_RESULTS_DIR/relearn/$GROUP/ ==="
