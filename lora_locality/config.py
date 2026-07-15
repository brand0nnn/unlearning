"""LoRA-locality relearning ablation — configuration & capacity accounting.

SELF-CONTAINED experiment, separate from the main 4-strategy pipeline. Question:
does *where* the LoRA adapters sit (which weight matrices) change how RECOVERABLE
the forgetting is under a relearning attack?

Two rank schemes are run and compared, to disentangle location from capacity:
  - "samerank"    : every location at the same rank R. Natural module-group configs,
                    but total #params differs by location (attn has 4 matrices, down 1).
  - "fixedbudget" : per-location rank chosen so the TRAINABLE-PARAMETER COUNT is
                    (approximately) equal across locations -> isolates the *locus*.

If the location ranking agrees across both schemes, the effect is robust; if it
only appears under samerank, it was capacity, not location.

Dims are Llama-2-7B (hidden 4096, intermediate 11008, 32 layers). Override via
model_dims() if you swap the base model.
"""
from typing import Dict, List

# The module groups being compared (matches scripts/pipeline/02_unlearn.py LORA_TARGETS).
LOCATIONS: Dict[str, List[str]] = {
    "attn":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qkv":    ["q_proj", "k_proj", "v_proj"],
    "qv":     ["q_proj", "v_proj"],
    "mlp":    ["gate_proj", "up_proj", "down_proj"],
    "updown": ["up_proj", "down_proj"],
    "down":   ["down_proj"],
    "all":    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

# (in_features, out_features) per adapted matrix, Llama-2-7B.
_HIDDEN, _INTER = 4096, 11008
MODULE_DIMS: Dict[str, tuple] = {
    "q_proj": (_HIDDEN, _HIDDEN), "k_proj": (_HIDDEN, _HIDDEN),
    "v_proj": (_HIDDEN, _HIDDEN), "o_proj": (_HIDDEN, _HIDDEN),
    "gate_proj": (_HIDDEN, _INTER), "up_proj": (_HIDDEN, _INTER),
    "down_proj": (_INTER, _HIDDEN),
}
N_LAYERS = 32


def lora_params(location: str, rank: int, n_layers: int = N_LAYERS) -> int:
    """Trainable LoRA parameters for a location at a given rank.
    Each adapted matrix contributes rank*(in+out) params (A: in*r, B: r*out),
    summed over its matrices and all layers."""
    per_layer = sum(rank * (i + o) for m in LOCATIONS[location]
                    for (i, o) in [MODULE_DIMS[m]])
    return per_layer * n_layers


def same_rank(rank: int) -> Dict[str, int]:
    """Every location at the same rank."""
    return {loc: rank for loc in LOCATIONS}


def fixed_budget(reference_location: str = "attn", reference_rank: int = 16) -> Dict[str, int]:
    """Per-location rank chosen so each location's trainable-param count matches the
    reference (default: attn @ rank 16). Weighted by ACTUAL matrix dims, so it is a
    true capacity match, not just a matrix-count match (MLP matrices are ~1.8x bigger
    than attention ones, so they need a lower rank to hit the same budget)."""
    target = lora_params(reference_location, reference_rank)
    ranks = {}
    for loc in LOCATIONS:
        dim_sum = sum(i + o for m in LOCATIONS[loc] for (i, o) in [MODULE_DIMS[m]])
        ranks[loc] = max(1, round(target / (N_LAYERS * dim_sum)))
    return ranks


SCHEMES = {
    "samerank":    lambda: same_rank(32),                    # all at rank 32
    "fixedbudget": lambda: fixed_budget("attn", 16),         # ~ attn@16 params each
}


def scheme_table(scheme: str) -> Dict[str, Dict]:
    """{location: {rank, params}} for a scheme — printed at run start so capacity is
    always visible alongside the results (crucial for interpreting samerank)."""
    ranks = SCHEMES[scheme]()
    return {loc: {"rank": r, "params": lora_params(loc, r)} for loc, r in ranks.items()}


if __name__ == "__main__":
    for scheme in SCHEMES:
        print(f"\n=== {scheme} ===")
        print(f"{'location':8s} {'rank':>5s} {'params':>14s}  matrices")
        for loc, d in scheme_table(scheme).items():
            print(f"{loc:8s} {d['rank']:5d} {d['params']:14,d}  {LOCATIONS[loc]}")
