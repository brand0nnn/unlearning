#!/bin/bash
#SBATCH --job-name=storage_fu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/b/brandonk/unlearning/logs/storage_fu_%j.out
#SBATCH --error=/home/b/brandonk/unlearning/logs/storage_fu_%j.err

# STAGE 10 PHASE 1 -- does French unlearning reach the Japanese copy of SOME facts?
# Decision rule: studies/learn_french/preregistration_stage10.json (committed before this job).
# One checkpoint (or the Stage 9 set) per job, so six jobs run side by side on `gpu`:
#
#     bash studies/learn_french/slurm/submit_storage_followup.sh          # all six
#     sbatch studies/learn_french/slurm/23_storage_followup.sbatch tr0p783   # one
#
# TARGET   checkpoint                  what runs
#   ft      fr-ja_ft                    rescore x2 + span + samples + cross + ja retain gen
#   floor   fr-ja_retain                rescore x2 + span + samples + cross
#   tr0p650 / tr0p694 / tr0p783         rescore x2 + span + samples + cross + ja retain gen
#           (srcfrja levels; tr0p783 is the PRIMARY depth)
#   mix     fr-ja-mix_ft + srcfrjamix   span + cross (Stage 9's missing fr>ja route)
#           levels
#
# PHASE 2A (preregistration_stage10_phase2a.json), after 26_unlearn_frja_deep.sbatch:
#   <level> srcfrja2   one srcfrja2 level, e.g. `... 23_storage_followup.sbatch tr0p950 srcfrja2`:
#                      3 TR passes (storage_frja = pass 0, rescore1, rescore2) + samples +
#                      ja retain greedy + ja retain TR
#   ref                fr-ja_ft and job 19's srcfrja tr0p783: ja retain TR only (the
#                      collateral reference Phase 2A subtracts)
#   bash studies/learn_french/slurm/submit_phase2a_scoring.sh submits all of them.
#
# "rescore" = gate_frja.py UNCHANGED with --retain-control, into storage_followup_rescore1/2:
# the same measurement as Stage 8's pass 0, so the three passes give per-fact noise.
# COST (a100-40, unbatched generation): rescore ~10 min each; span + cross ~3 min; samples
# (2 x 40 x 10) ~10 min; Japanese retain gen (400 greedy) ~30 min. Longest job ~1.5 h.
# Nothing is written to experiments/.
set -euo pipefail

TARGET="${1:?usage: sbatch 23_storage_followup.sbatch ft|floor|tr0p650|tr0p694|tr0p783|mix|ref [TAG]}"
TAG="${2:-srcfrja}"     # srcfrja = job 19 (Phase 1); srcfrja2 = job 26 (Phase 2A)
PROJECT_DIR="/home/b/brandonk/unlearning"
export UNLEARN_RESULTS_DIR="$PROJECT_DIR/studies/learn_french/results"
cd "$PROJECT_DIR"
mkdir -p logs "$UNLEARN_RESULTS_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
export TMPDIR="$PROJECT_DIR/.tmp/storage_fu_$TARGET"; mkdir -p "$TMPDIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/.env_hf

echo "Node: $(hostname)   target: $TARGET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Commit: $(git rev-parse --short HEAD)"

# The rule and the key must be committed as-is BEFORE any number exists.
PREREGS=(studies/learn_french/preregistration_stage10.json studies/learn_french/probes/fact_keys_frja.json)
if [ "$TAG" = "srcfrja2" ] || [ "$TARGET" = "ref" ]; then
    PREREGS+=(studies/learn_french/preregistration_stage10_phase2a.json)
fi
for f in "${PREREGS[@]}"; do
    { git ls-files --error-unmatch "$f" && git diff --quiet HEAD -- "$f"; } >/dev/null 2>&1 \
        || { echo "$f is not committed as-is" >&2; exit 1; }
done

MODEL_SLUG=$(python3 -c "import re,yaml;n=yaml.safe_load(open('config/config.yaml'))['model']['name'].split('/')[-1].lower();print(re.sub(r'[^a-z0-9.]+','-',n).strip('-'))")
E="$PROJECT_DIR/experiments"
UNL="tofu_unlearn_gradient_difference_forget01_fullft_${MODEL_SLUG}_ulfr_floornone"
RESCORE=1
PASS_GROUPS=(storage_followup_rescore1 storage_followup_rescore2)
PARTS=(span samples cross retain_gen)
case "$TAG" in srcfrja|srcfrja2) ;; *) echo "unknown tag $TAG" >&2; exit 2 ;; esac
if [ "$TAG" = "srcfrja2" ]; then
    case "$TARGET" in tr[0-9]*p[0-9]*) ;; *) echo "srcfrja2 takes a level, e.g. tr0p950" >&2; exit 2 ;; esac
    CKPTS=("$E/tr_levels/${UNL}_srcfrja2_${TARGET}")
    # No job-19-style scoring ran for this run, so pass 0 is written here, into the same
    # group name Stage 8 used for pass 0 (new checkpoint names, so nothing is overwritten).
    PASS_GROUPS=(storage_frja storage_followup_rescore1 storage_followup_rescore2)
    PARTS=(samples retain_gen retain_tr)
    TARGET="srcfrja2_$TARGET"
fi
case "$TARGET" in
    srcfrja2_*) ;;
    ref)     RESCORE=0; PARTS=(retain_tr)
             CKPTS=("$E/tofu_learn_full_full_${MODEL_SLUG}_fr-ja" "$E/tr_levels/${UNL}_srcfrja_tr0p783") ;;
    ft)      CKPTS=("$E/tofu_learn_full_full_${MODEL_SLUG}_fr-ja") ;;
    floor)   CKPTS=("$E/tofu_learn_retain99_full_${MODEL_SLUG}_fr-ja"); PARTS=(span samples cross) ;;
    tr0p650|tr0p694|tr0p783) CKPTS=("$E/tr_levels/${UNL}_srcfrja_${TARGET}") ;;
    mix)     RESCORE=0; PARTS=(span cross)
             CKPTS=("$E/tofu_learn_full_full_${MODEL_SLUG}_fr-ja-mix")
             for lv in tr0p650 tr0p694 tr0p783; do CKPTS+=("$E/tr_levels/${UNL}_srcfrjamix_${lv}"); done ;;
    *) echo "unknown target $TARGET" >&2; exit 2 ;;
esac

echo "=== [1/3] checkpoint check ==="
for d in "${CKPTS[@]}"; do
    # A checkpoint saved without its tokenizer evaluates to all zeros, silently.
    [ -f "$d/tokenizer_config.json" ] || { echo "MISSING checkpoint/tokenizer: $d" >&2; exit 1; }
    echo "  OK $d$( [ -L "$d" ] && echo "  (symlink -> $(readlink "$d"))")"
done

if [ "$RESCORE" = "1" ]; then
    for g in "${PASS_GROUPS[@]}"; do
        echo "=== [2/3] scoring pass -> $g (gate_frja.py unchanged) ==="
        python studies/learn_french/scripts/gate_frja.py --checkpoints "${CKPTS[@]}" \
            --langs fr ja --retain-control --group "$g"
    done
fi

echo "=== [3/3] storage_followup.py: ${PARTS[*]} ==="
python studies/learn_french/scripts/storage_followup.py --checkpoints "${CKPTS[@]}" \
    --parts "${PARTS[@]}" --retain-langs ja

echo "=== STORAGE FOLLOW-UP ($TARGET) complete -> $UNLEARN_RESULTS_DIR/storage_followup*/ ==="
