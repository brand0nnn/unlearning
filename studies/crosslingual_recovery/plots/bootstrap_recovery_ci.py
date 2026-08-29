"""Per-language bootstrap CI on the truth-ratio recovery (item 8 follow-up).

Recovery per language = (baseline_TR - relearned_TR) / (baseline_TR - LEARNED_TR),
a MEAN over the ~40 forget facts. "Uniform across languages" is only meaningful if
the between-language spread exceeds the within-language noise floor. This resamples
the 40 facts (paired: same fact index in baseline and relearned) 1000x to get a 95%
CI per language, for both methods at relearn ep2.

REQUIRES the per-fact arrays (`truth_ratio_per_fact`) that relearn_measure.py now
persists. The current crosslingual_deep JSONs predate that change and only have the
means, so re-run the deep relearn job (EPS="2" is enough) with the updated
relearn_measure.py first. If the arrays are missing this script says so and exits.

    python studies/crosslingual_recovery/plots/bootstrap_recovery_ci.py
"""
import json, sys, random
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
DEEP = STUDY / "results" / "relearn" / "crosslingual_deep"
LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]
LEARNED_TR = 0.459
EP = 2
B = 1000
FILES = {"Full-FT": "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b",
         "LoRA": "tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b"}


def per_fact(rec):
    if not isinstance(rec, dict) or "truth_ratio_per_fact" not in rec:
        return None
    return rec["truth_ratio_per_fact"]


def main():
    random.seed(42)
    for method, base in FILES.items():
        f = DEEP / f"{base}.json"
        if not f.exists():
            print(f"[{method}] missing {f}"); continue
        d = json.load(open(f))
        b_arr = per_fact(d.get(base))
        if b_arr is None:
            print(f"[{method}] baseline has NO per-fact array yet — re-run the measure "
                  f"step (baseline checkpoint still exists, inference-only) + the deep "
                  f"relearn job with the updated relearn_measure.py, then rerun this.");
            continue
        print(f"\n=== {method}  (baseline mean TR {sum(b_arr)/len(b_arr):.3f}, n={len(b_arr)}) ===")
        print(f"{'lang':>5}  {'recovery':>9}  {'95% CI':>18}")
        means = {}
        for l in LANGS:
            k = f"relearn_{base}_via_retain" + ("" if l == "en" else f"_lang{l}") + f"_ep{EP}"
            r_arr = per_fact(d.get(k))
            if r_arr is None:
                print(f"{l:>5}  (no per-fact array — re-run deep relearn for this point)")
                continue
            n = min(len(b_arr), len(r_arr))
            boot = []
            for _ in range(B):
                idx = [random.randrange(n) for _ in range(n)]
                bb = sum(b_arr[i] for i in idx) / n
                rr = sum(r_arr[i] for i in idx) / n
                boot.append((bb - rr) / (bb - LEARNED_TR))
            boot.sort()
            lo, hi = boot[int(0.025 * B)], boot[int(0.975 * B)]
            point = (sum(b_arr[:n]) / n - sum(r_arr[:n]) / n) / (sum(b_arr[:n]) / n - LEARNED_TR)
            means[l] = (point, lo, hi)
            print(f"{l:>5}  {point:>+8.1%}  [{lo:>+6.1%}, {hi:>+6.1%}]")
        if means:
            pts = [v[0] for v in means.values()]
            spread = max(pts) - min(pts)
            typ_ci = sum(v[2] - v[1] for v in means.values()) / len(means)
            print(f"  between-lang spread = {spread:.1%};  typical within-lang CI width = {typ_ci:.1%}")
            print(f"  -> uniform is {'SUPPORTED (spread < CI width)' if spread < typ_ci else 'QUESTIONABLE (spread >= CI width; check non-overlapping langs, esp. en)'}")


if __name__ == "__main__":
    main()
