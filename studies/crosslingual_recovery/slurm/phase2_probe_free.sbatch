#!/bin/bash
#SBATCH --job-name=phase2_probe_free
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/phase2_probe_free_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/phase2_probe_free_%j.err

# PHASE 2 PART A, STEPS 1-2 -- the FREE probe family. No relearning, no authored probes.
#
# The study measures each forget fact with ONE canonical question. If unlearning only
# suppressed that phrasing, a different phrasing recovers the fact and the headline
# recovery number is partly re-surfacing rather than relearning-driven transfer.
#
# Every probe here is free: TOFU's forget01_perturbed ships a `paraphrased_question` per
# fact (a published, citable paraphrase, and an aggressive one -- median 32% word overlap
# with the canonical question, 28/40 keep under half the content words), and the MCQ falls
# out of the 5 perturbed answers. 129 probes, nothing authored.
#
# THE COMPARISON, paired fact by fact on the UNLEARNED model:
#     TR(p0_canonical)  vs  TR(p1_tofu_para)   same fact, same answers, different wording
#     TR(p0_canonical)  vs  MCQ probability    same fact, different TASK FORMAT
# The MCQ is the one to watch: it asks the model to RANK six candidates rather than
# generate, so suppression trained on generation may leave it untouched.
#
# WHY THIS RUNS BEFORE ANY PARAPHRASE IS AUTHORED. The free set is decisive in ONE
# direction. If it shows leakage, the headline finding is secured on published probes with
# no provenance to defend. If it shows nothing, we cannot conclude "robust to rephrasing",
# only "robust to THIS rephrasing" -- one paraphrase per fact gives no within-fact spread
# to judge a difference against -- and authored p2-p4 become worth the effort, with a
# reason. Front-loading 111 authored paraphrases risks doing the unnecessary work first.
#
# LEARNED IS THE CEILING, NOT A RESULT. A probe the learned model cannot answer is a
# broken probe and the analysis drops it. That empirical filter is why no LLM equivalence
# judge (and no API key) is needed anywhere in Part A.
#
# Pure inference on existing checkpoints: no training, no DeepSpeed, nothing written to
# experiments/. ~130 probes x ~7 forward passes each x 3 models, comfortably inside 2h.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/phase2_probe_free"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
PROBES="$PROJECT_DIR/studies/crosslingual_recovery/probes/probe_family.json"
GROUP="phase2_probes"
[ -f "$PROBES" ] || { echo "!! missing $PROBES -- run scripts/build_probe_family.py first"; exit 1; }

# Resolved against experiments/, where the weights live -- never by globbing results/,
# which is gitignored and rsync-only (that is how job 770249 died).
LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}"
FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_uep32_${MODEL_SLUG}"

for CK in "$LEARNED" "$FULLFT" "$LORA"; do
    [ -d "$CK" ] || { echo "!! MISSING checkpoint: $CK"; exit 1; }
done
echo "probes  : $(python3 -c "import json;d=json.load(open('$PROBES'));print(d['stats']['qa'],'qa +',d['stats']['mcq'],'mcq +',d['stats']['fib'],'fib')")"
echo "ceiling : $(basename $LEARNED)"
echo "unlearn : $(basename $FULLFT) | $(basename $LORA)"

for CK in "$LEARNED" "$FULLFT" "$LORA"; do
    echo ""
    echo "=== probing $(basename "$CK") ==="
    python3 shared/scripts/probe_score.py --checkpoint "$CK" --probes "$PROBES" \
        --group "$GROUP" || echo "!!! $(basename "$CK") failed, continuing"
done

echo ""
echo "=== [DONE] free probe family -> $UNLEARN_RESULTS_DIR/relearn/$GROUP/ ==="
echo "NEXT (local, no GPU): rsync results/ down, then compare p0 vs p1 vs mcq PAIRED"
echo "per fact -- on the LEARNED model first, to drop any probe it cannot itself answer."
