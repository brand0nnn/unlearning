#!/bin/bash
#SBATCH --job-name=xling_pilot
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/xling_pilot_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/xling_pilot_%j.err

# CROSS-LINGUAL RECOVERY PILOT (SCALED) — does the fine-tuning METHOD modulate how
# far benign relearning in different languages recovers an English-unlearned fact?
#
#   Unlearn English forget01 with Full-FT and LoRA (matched forget depth).
#   Benign-relearn each on the RETAIN set in ALL 10 languages, ordered by distance
#     from English (en fr id ru hi fa ar iw ko ja).
#   Measure ENGLISH forget-set recovery after each -> a full distance-decay curve.
#   Readout: does the DECAY SLOPE (recovery vs language distance) differ by method?
#     slopes differ -> method modulates cross-lingual recovery (novel finding alive)
#     slopes equal  -> method-agnostic (fall back to systematic characterization)
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
# SCALED overnight run: ALL 10 languages (full distance axis), 2 methods, single
# near-peak relearn epoch. ~20 relearns -> ~6h, safe inside the 10h wall.
LANGS="${1:-en fr id ru hi fa ar iw ko ja}"   # full distance axis (near -> far from EN)
EPS="${2:-2}"              # single near-peak relearn epoch (spend budget on BREADTH,
                           # not the temporal curve — the 3-pt pilot already mapped that)
FL="forget01"              # the multilingual data is forget01/retain99 only.
# Per-method unlearn depth: LoRA's low-rank update forgets weakly per step, so it
# needs MORE epochs to reach the same forget depth as Full-FT -> matched baselines,
# which de-confounds the method comparison (LoRA recovering more != it forgot less).
UEP_FT="${3:-15}"          # Full-FT -> baseline ~0.26 (measured)
UEP_LORA="${4:-40}"        # LoRA deeper -> aim baseline ~0.3 (CHECK the baseline row)
N=40                       # forget01 records to measure (all of them)
RN="${5:-1500}"           # cap retain so 20 relearns fit the wall
RN_FLAG=""; [ -n "$RN" ] && RN_FLAG="--relearn-n $RN"
# Self-contained: every stage is told --forget-level forget01, so the run does NOT
# depend on config's forget_level (which stays forget10 for the other experiments).

# --- 1. Unlearn English forget01 with Full-FT and LoRA -----------------------
echo "=== [1] unlearn Full-FT forget01 (UEP=$UEP_FT) ==="
deepspeed --num_gpus=1 --master_port=$MASTER_PORT scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy fullft --forget-level forget01 --unlearn-epochs "$UEP_FT"
echo "=== [1] unlearn LoRA forget01 (UEP=$UEP_LORA) ==="
python scripts/pipeline/02_unlearn.py \
    --checkpoint "$LEARNED" --strategy lora --forget-level forget01 --unlearn-epochs "$UEP_LORA"

FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora"

# --- 2. Baseline (unlearned) English forget recovery -------------------------
echo "=== [2] baseline forget recovery (unlearned) ==="
python scripts/recovery/relearn_measure.py --checkpoints "$FULLFT" "$LORA" --group "$GROUP" --n $N --forget-level "$FL"

# --- 3. Benign retain-relearn in each language, measure English recovery ------
LAUNCH=0
for M in "$FULLFT" "$LORA"; do
    for L in $LANGS; do
        for E in $EPS; do
            LAUNCH=$((LAUNCH + 1))
            echo "=== [3] relearn $(basename $M) via RETAIN lang=$L ep=$E ==="
            deepspeed --num_gpus=1 --master_port=$((MASTER_PORT + LAUNCH)) \
                scripts/recovery/relearn.py --checkpoint "$M" --epochs $E \
                --relearn-data retain --relearn-lang "$L" --forget-level "$FL" $RN_FLAG
            VIA="_via_retain"; [ "$L" != "en" ] && VIA="${VIA}_lang${L}"
            RELEARNED="$PROJECT_DIR/experiments/relearn_$(basename $M)${VIA}_ep${E}"
            python scripts/recovery/relearn_measure.py --checkpoints "$RELEARNED" --group "$GROUP" --n $N --forget-level "$FL"
            rm -rf "$RELEARNED"       # reclaim the ~13GB relearned checkpoint
        done
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
        return m, "baseline", 0
    lang = re.search(r"_lang([a-z]+)_ep", k)
    ep = re.search(r"_ep(\d+)$", k)
    return m, (lang.group(1) if lang else "en"), (int(ep.group(1)) if ep else 0)
# peak (max-epoch) recovery per method/lang
rows = {}
for k, v in d.items():
    m, l, ep = parse(k)
    r = rows.setdefault(m, {})
    if l == "baseline":
        r["baseline"] = v
    else:
        r.setdefault(l, {})[ep] = v
def peak(r, l):
    return max(r[l].values()) if l in r and r[l] else float("nan")
print("method   baseline  " + "  ".join(f"{l}(peak)" for l in langs))
for m in ("fullft", "lora"):
    r = rows.get(m, {})
    print(f"{m:8s} {r.get('baseline',float('nan')):.3f}    " +
          "     ".join(f"{peak(r,l):.3f}" for l in langs))
print("--- peak EN-KO gap = how much LESS a far language recovers at its best epoch ---")
for m in ("fullft", "lora"):
    r = rows.get(m, {})
    if "en" in r and "ko" in r:
        print(f"  {m}: EN-KO gap = {peak(r,'en')-peak(r,'ko'):+.3f}")
print("=> if the two gaps DIFFER, the method modulates cross-lingual recovery.")
PY
echo "=== pilot done -> results/relearn/crosslingual_pilot/ ==="
