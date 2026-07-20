#!/bin/bash
#SBATCH --job-name=xling_pilot
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_pilot_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_pilot_%j.err

# CROSS-LINGUAL RECOVERY PILOT — the make-or-break test for the new direction.
# Does the fine-tuning METHOD modulate how much benign relearning in different
# languages recovers an English-unlearned fact?
#
#   Unlearn English forget01 with Full-FT and LoRA.
#   Benign-relearn each on the RETAIN set, in en / fr / ko (near / close / far).
#   Measure ENGLISH forget-set recovery after each.
#   Readout: does the EN-vs-KO recovery GAP differ between Full-FT and LoRA?
#     gaps differ  -> method modulates cross-lingual recovery (novel finding is alive)
#     gaps equal   -> method-agnostic (fall back to systematic characterization)
#
# PREREQ: the multilingual dataset must be cloned (data/raw/multilingual_unlearning/
# dataset). No config edit needed — every stage is passed --forget-level forget01
# explicitly, so config's forget_level stays forget10 for the other experiments.
# Full-parameter relearn -> h200.
PROJECT_DIR="/home/b/brandonk/unlearning"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/xling_pilot"
mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full"
GROUP="crosslingual_pilot"
LANGS="${1:-en fr ko}"     # relearn languages: en (near) fr (close) ko (far)
EP="${2:-2}"               # relearn epochs
FL="forget01"              # the multilingual data is forget01/retain99 only.
UEP="${3:-40}"             # unlearn epochs — forget01 (~40 QA) needs ~40 to actually forget
N=40                       # forget01 records to measure (all of them)
RN="${4:-}"                # OPTIONAL cap on retain-relearn size (empty = full retain ~3960)
RN_FLAG=""; [ -n "$RN" ] && RN_FLAG="--relearn-n $RN"
# Self-contained: every stage is told --forget-level forget01, so the run does NOT
# depend on config's forget_level (which stays forget10 for the other experiments).

# --- 1. Unlearn English forget01 with Full-FT and LoRA -----------------------
echo "=== [1] unlearn Full-FT forget01 ==="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy fullft --forget-level forget01 --unlearn-epochs "$UEP"
echo "=== [1] unlearn LoRA forget01 ==="
python scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy lora --forget-level forget01 --unlearn-epochs "$UEP"

FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora"

# --- 2. Baseline (unlearned) English forget recovery -------------------------
echo "=== [2] baseline forget recovery (unlearned) ==="
python scripts/recovery/relearn_measure.py --checkpoints "$FULLFT" "$LORA" --group "$GROUP" --n $N --forget-level "$FL"

# --- 3. Benign retain-relearn in each language, measure English recovery ------
LAUNCH=0
for M in "$FULLFT" "$LORA"; do
    for L in $LANGS; do
        LAUNCH=$((LAUNCH + 1))
        echo "=== [3] relearn $(basename $M) via RETAIN in lang=$L (ep$EP) ==="
        deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + LAUNCH)) \
            scripts/recovery/relearn.py --checkpoint "$M" --epochs $EP \
            --relearn-data retain --relearn-lang "$L" --forget-level "$FL" $RN_FLAG
        VIA="_via_retain"; [ "$L" != "en" ] && VIA="${VIA}_lang${L}"
        RELEARNED="$PROJECT_DIR/experiments/relearn_$(basename $M)${VIA}_ep${EP}"
        python scripts/recovery/relearn_measure.py --checkpoints "$RELEARNED" --group "$GROUP" --n $N --forget-level "$FL"
        rm -rf "$RELEARNED"       # reclaim the ~13GB relearned checkpoint
    done
done

# --- 4. Print the 2 x (langs) recovery table + the method gap -----------------
echo "=== [4] RESULT: English forget-set recovery (higher = more recovered) ==="
python3 - "$LANGS" <<'PY'
import json, glob, re, sys
langs = sys.argv[1].split()
d = {}
for f in glob.glob("results/relearn/crosslingual_pilot/*.json"):
    d.update(json.load(open(f)))
def parse(k):
    m = "fullft" if "fullft" in k else ("lora" if "lora" in k else "?")
    if not k.startswith("relearn_"):
        return m, "baseline"
    mm = re.search(r"_lang([a-z]+)_ep", k)
    return m, (mm.group(1) if mm else "en")
rows = {}
for k, v in d.items():
    m, l = parse(k); rows.setdefault(m, {})[l] = v
hdr = "method   baseline  " + "  ".join(f"relearn_{l}" for l in langs)
print(hdr)
for m in ("fullft", "lora"):
    r = rows.get(m, {})
    print(f"{m:8s} {r.get('baseline',float('nan')):.3f}    " +
          "     ".join(f"{r.get(l,float('nan')):.3f}" for l in langs))
print("--- distance gap (relearn_en - relearn_ko) = how much LESS a far language recovers ---")
for m in ("fullft", "lora"):
    r = rows.get(m, {})
    if "en" in r and "ko" in r:
        print(f"  {m}: EN-KO gap = {r['en']-r['ko']:+.3f}")
print("=> if the two gaps DIFFER, the method modulates cross-lingual recovery.")
PY
echo "=== pilot done -> results/relearn/crosslingual_pilot/ ==="
