"""Comparison table: our TOFU metrics vs the paper (Maini et al. 2024, Figure 6,
Llama-2-7B). The paper reports Model Utility / Forget Quality via PLOTS (no exact
value table), so the 'Paper' column is the readable anchor from Fig 6 plus the
trend the paper states. The reproduction is in the PATTERN: gradient_ascent
destroys utility; the gentle methods preserve utility but barely forget (forget
quality never crosses the 0.05 significance threshold).

    python scripts/diagnostics/compare.py     # CPU-only; reads results/*.json

Writes results/comparison.md and prints the table.
"""
import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.logging_utils import load_config

RESULTS = Path("results")

# Paper (Fig 6, Llama-2-7B, forget10) — readable anchors + stated trends.
PAPER = {
    "reference":           {"util": "~0.55 (Retain star)",  "fq": "— (reference)"},
    "gradient_ascent":     {"util": "-> 0 (destroyed)",     "fq": "< 0.05 (log p ~ -20)"},
    "gradient_difference": {"util": "preserved (higher)",   "fq": "< 0.05"},
    "idk":                 {"util": "preserved (partial)",  "fq": "< 0.05"},
    "kl_minimization":     {"util": "preserved (higher)",   "fq": "< 0.05"},
}
ORDER = ["reference", "gradient_ascent", "gradient_difference", "idk", "kl_minimization"]


def _load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def main():
    cfg = load_config()
    fl = cfg["tofu"]["forget_level"]
    files = {"reference": "tofu_reference.json"}
    for m in ORDER[1:]:
        files[m] = f"tofu_tofu_unlearn_{m}_{fl}.json"

    header = (
        f"# TOFU reproduction: ours vs paper "
        f"(Maini et al. 2024, Fig 6, Llama-2-7B, {fl})\n\n"
        "The paper reports Model Utility / Forget Quality via plots, so 'Paper' is the\n"
        "readable Fig-6 anchor + stated trend. The match is the *pattern*, not exact digits.\n\n"
        "| Model | Model Utility (ours) | Model Utility (paper) | "
        "Forget Quality p (ours) | log10 p (ours) | Forget Quality (paper) |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for name in ORDER:
        d = _load(files[name])
        p = PAPER[name]
        if d is None:
            rows.append(f"| {name} | (missing) | {p['util']} | | | {p['fq']} |")
            continue
        mu = d.get("model_utility")
        fq = d.get("forget_quality")
        log10 = d.get("forget_quality_log10")
        if fq is not None and log10 is None:
            log10 = math.log10(fq) if fq > 0 else float("-inf")
        mu_s = f"{mu:.3f}" if mu is not None else "—"
        fq_s = f"{fq:.2e}" if fq is not None else "—"
        lg_s = f"{log10:.1f}" if log10 is not None else "—"
        rows.append(f"| {name} | {mu_s} | {p['util']} | {fq_s} | {lg_s} | {p['fq']} |")

    footer = (
        "\n\n**Reading:** the reference (retain) model has healthy utility "
        "(~0.66; paper ~0.55). `gradient_ascent` collapses utility to 0. "
        "`gradient_difference` / `kl_minimization` preserve utility (~0.63 / ~0.66) but "
        "forget quality is ~0 (they barely forget). No method reaches BOTH high utility "
        "and high forget quality — the TOFU trade-off (top-right of Fig 6 stays empty).\n"
    )

    out = header + "\n".join(rows) + footer
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "comparison.md").write_text(out)
    print(out)
    print(f"-> wrote {RESULTS / 'comparison.md'}")


if __name__ == "__main__":
    main()
