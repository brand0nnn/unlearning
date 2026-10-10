#!/bin/bash
# Is there room for the French + Japanese LEARN (14_learn_frja.sbatch)? Login-node safe.
#
#     bash studies/learn_french/scripts/check_space.sh
#
# NEED, per model (Qwen3-8B, 8.2B params):
#   final weights (bf16)                              ~16 GB   kept
#   resume checkpoint: Trainer's consolidated bf16
#     copy (2 B/param) + ZeRO-3 shards: bf16 weights
#     + fp32 master + two fp32 Adam moments (14 B)  ~131 GB   only while a run is
#                                                              split across jobs
# Two models (fr-ja_ft + fr-ja_retain): ~33 GB kept. Peak ~163 GB if run one after the
# other (16 kept + 16 + 131), ~295 GB if both run at once. No checkpoint at all if the
# wall fits ~16 h.
P=/home/b/brandonk/unlearning

echo "=== free space on the filesystem holding experiments/ ==="
df -h "$P/experiments" | tail -1

echo; echo "=== per-user quota (if the filesystem enforces one) ==="
quota -s 2>/dev/null || echo "(no 'quota' output -- df above is the limit)"

echo; echo "=== what experiments/ already holds (largest last) ==="
du -sh "$P"/experiments/* 2>/dev/null | sort -h | tail -15
du -sh "$P/experiments" 2>/dev/null

echo; echo "=== max wall time per partition (>= 17h on gpu-long => no resume checkpoint) ==="
sinfo -p gpu-long,gpu -o "%P %l" | sort -u

AVAIL_GB=$(df -P -BG "$P/experiments" | awk 'NR==2 {gsub("G","",$4); print $4}')
echo; echo "=== verdict (filesystem free space only; check the quota line too) ==="
for need in 33 163 295; do
    case $need in
        33)  what="both models, one 16+ h wall each (no checkpoints)";;
        163) what="both models, run ONE AFTER THE OTHER with resume";;
        295) what="both models AT ONCE with resume";;
    esac
    if [ "$AVAIL_GB" -ge $((need + 20)) ]; then v="OK  "; else v="NO  "; fi
    echo "  [$v] ${need} GB (+20 GB headroom) -- $what   (free: ${AVAIL_GB} GB)"
done
