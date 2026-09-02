#!/bin/bash
#SBATCH --job-name=phase2_probe_authored
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/phase2_probe_authored_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/phase2_probe_authored_%j.err

# PHASE 2 PART A, STEPS 1-2 -- the FULL probe family, with authored paraphrases.
#
# The free run (phase2_probe_free.sbatch) gave one paraphrase per fact: TOFU's own
# `paraphrased_question`. One paraphrase makes the phrasing axis n=2, which cannot
# distinguish "unlearning is phrasing-specific" from "that ONE paraphrase happened to
# be easy/hard" -- there is no within-fact spread to judge a difference against.
# This run adds three AUTHORED paraphrases per fact, so the axis becomes n=5 and each
# fact carries its own distribution over phrasings.
#
# The three authored slots vary along CONTROLLED axes, one per slot, so "which KIND of
# rephrasing defeats suppression" becomes a finding rather than a robustness check:
#     p2  syntactic recast     -- same content words, different structure
#     p3  lexical substitution -- synonyms, avoiding the canonical content words
#     p4  oblique framing      -- the fact requested indirectly, inside a task
#
# ANSWER SIDE UNTOUCHED. The truth ratio reuses TOFU's paraphrased_answer and
# perturbed_answers, which describe the FACT and not the phrasing, so every authored
# probe stays directly comparable to p0_canonical with ZERO new answer generation.
#
# STILL A DIFFERENCE-IN-DIFFERENCES. p0 is the phrasing the model was TRAINED on (in
# LEARN and again in UNLEARN); p1-p4 never appeared in training. A raw gap on the
# unlearned model alone would confound "unlearning was phrasing-specific" with "the
# model is simply better at phrasings it trained on". The learned checkpoint supplies
# that baseline:
#     D_learned   = TR(p_k) - TR(p0)   on the learned checkpoint
#     D_unlearned = TR(p_k) - TR(p0)   on the unlearned checkpoint
#     finding     = D_unlearned - D_learned    <- widens => suppression is surface-form
#
# LEARNED IS THE CEILING, NOT A RESULT. A probe the learned model cannot itself answer
# is a broken probe and the analysis drops it. That empirical filter is why no LLM
# equivalence judge (and no API key) is needed anywhere in Part A.
#
# PROVENANCE. Authored by hand into probes/authored_paraphrases.json and audited
# mechanically by build_probe_family.audit_probe() for the two failure modes the ceiling
# check CANNOT catch: added_fact (a proper noun/number in neither question nor answer --
# an extra retrieval cue) and answer_leak (a capitalised span pulled from the ANSWER into
# the QUESTION). All 120 pass. For comparison, 6/40 of TOFU's OWN p1 paraphrases are
# flagged (facts 0, 9, 14, 20, 24, 29 -- fact 20 glosses Astana as "now known as
# Nur-Sultan", real-world knowledge the canonical question never supplied).
# Inspired by KBL's generation step (their sec 4.3), NOT adapted from it: their
# verify-and-refine loop belongs to their TRANSLATION step, which we do not have.
#
# NEW GROUP, deliberately. Writes to phase2_authored/, not phase2_probes/ -- probe_score
# _merge_write replaces a checkpoint's whole entry, so reusing the group would overwrite
# the free run. Keeping both also gives a SECOND independent measurement of p0/p1/mcq on
# the same weights, which is the replicate the per-fact noise floor is estimated from.
#
# Pure inference on existing checkpoints: no training, no DeepSpeed, nothing written to
# experiments/. 240 probes x ~7 forward passes x 3 models -- roughly 2x the free run.
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/crosslingual_recovery/results"
cd "$PROJECT_DIR"
mkdir -p logs
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/phase2_probe_authored"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
PROBES="$PROJECT_DIR/studies/crosslingual_recovery/probes/probe_family.json"
GROUP="phase2_authored"

# Rebuild from authored_paraphrases.json so the probe file can never drift from its
# source. Idempotent, CPU-only, seconds.
echo "=== rebuilding probe family from authored_paraphrases.json ==="
python3 studies/crosslingual_recovery/scripts/build_probe_family.py --out "$PROBES" || exit 1

N_QA=$(python3 -c "import json;print(json.load(open('$PROBES'))['stats']['qa'])")
[ "$N_QA" -ge 200 ] || { echo "!! only $N_QA qa probes -- expected 200 (5 x 40). Authored set missing?"; exit 1; }

# Resolved against experiments/, where the weights live -- never by globbing results/,
# which is gitignored and rsync-only (that is how job 770249 died).
LEARNED="$PROJECT_DIR/experiments/tofu_learn_full_full_${MODEL_SLUG}"
FULLFT="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}"
LORA="$PROJECT_DIR/experiments/tofu_unlearn_gradient_difference_forget01_lora_uep32_${MODEL_SLUG}"

for CK in "$LEARNED" "$FULLFT" "$LORA"; do
    [ -d "$CK" ] || { echo "!! MISSING checkpoint: $CK"; exit 1; }
done
echo "probes  : $N_QA qa + $(python3 -c "import json;print(json.load(open('$PROBES'))['stats']['mcq'])") mcq"
echo "ceiling : $(basename $LEARNED)"
echo "unlearn : $(basename $FULLFT) | $(basename $LORA)"

for CK in "$LEARNED" "$FULLFT" "$LORA"; do
    echo ""
    echo "=== probing $(basename "$CK") ==="
    python3 shared/scripts/probe_score.py --checkpoint "$CK" --probes "$PROBES" \
        --group "$GROUP" || echo "!!! $(basename "$CK") failed, continuing"
done

echo ""
echo "=== [DONE] authored probe family -> $UNLEARN_RESULTS_DIR/relearn/$GROUP/ ==="
echo "NEXT (laptop, no GPU):"
echo "  rsync -avz unlearning:~/unlearning/studies/crosslingual_recovery/results/ \\"
echo "        studies/crosslingual_recovery/results/"
echo "  source .venv-plot/bin/activate"
echo "  python studies/crosslingual_recovery/plots/phase2_authored_probes.py"
