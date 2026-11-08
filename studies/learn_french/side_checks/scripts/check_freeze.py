"""Experiment D smoke check: did --trainable-layers actually freeze everything else under
ZeRO-3? Reads one layer_diff.py output (stdlib only, runs on the login node too):

    python3 studies/learn_french/side_checks/scripts/check_freeze.py <layer_diff.json> 0-23

Frozen parameters never enter the optimizer, so the saved bf16 weights must equal the source
EXACTLY: any non-zero change outside the range (embed_tokens, norm and lm_head included)
means the freeze leaked. Exit 0 = OK, 1 = leaked, 2 = nothing moved inside the range either.
An empty range argument checks only that something moved (an unrestricted run).
"""
import json
import sys


def main():
    path, rng = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
    pair = list(json.load(open(path))["pairs"].values())[0]["layers"]
    lo, hi = map(int, rng.split("-")) if rng else (0, 10 ** 6)
    moved = {k: v["delta_sq"] for k, v in pair.items()}
    inside = [k for k in moved if k.isdigit() and lo <= int(k) <= hi]
    outside = [k for k in moved if k not in inside]
    tot_in = sum(moved[k] for k in inside)
    print(f"[freeze] change inside {rng or 'all'}: sum dW^2 {tot_in:.3e} over {len(inside)} layers")
    print(f"[freeze] change outside ({len(outside)} groups, incl. embed / norm / lm_head): "
          f"max {max((moved[k] for k in outside), default=0.0):.3e}")
    bad = sorted(k for k in outside if moved[k] > 0)
    if bad:
        print("[freeze] FAILED -- moved outside the range: " + ", ".join(bad))
        sys.exit(1)
    if tot_in <= 0:
        print("[freeze] FAILED -- nothing moved inside the range either (no training?)")
        sys.exit(2)
    print("[freeze] OK -- only the trainable range moved")


if __name__ == "__main__":
    main()
