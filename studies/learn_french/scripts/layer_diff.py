"""Where did each unlearning language change the model? Layer-wise weight change, CPU only.

    python studies/learn_french/scripts/layer_diff.py --dry-run \
        --pair unl_fr=<fr_ft>::<unl_fr> ...                      # headers only, no torch
    python studies/learn_french/scripts/layer_diff.py \
        --pair unl_fr=<fr_ft>::<unl_fr> ... --out results/layer_diff.json

THE QUESTION. Stage 3 found the UNLEARNING language governs durability (recovery: en 33% ...
ja 81%). Xiang et al. (ICML 2026, S4.3) and Zhao et al. (NeurIPS 2025) both place shared,
cross-lingual content in the MIDDLE layers and language-specific output in the TOP ones.
The prediction to test: durable arms (en) moved the middle layers; fragile arms (ja) moved
mostly the top -- blocking the exit rather than removing the content.

WHAT IS MEASURED, per weight tensor, for each labelled pair REF::ARM:
    delta_sq  = ||W_arm - W_ref||_F^2       the size of the change
    ref_sq    = ||W_ref||_F^2               so a layer's change can be read RELATIVE to it
    changed   = fraction of elements that differ at all
Summed per decoder layer and per module type (q/k/v/o, gate/up/down, norms); embeddings and
lm_head are kept as their own rows. Read layers by relative change sqrt(sum delta_sq) /
sqrt(sum ref_sq), and compare ARMS against each other at the same layer.

A CAVEAT THAT SHAPES THE READING. Checkpoints are saved in bf16, while training kept fp32
masters. A per-element update smaller than bf16's spacing at that weight (~0.4% relative)
rounds away on save, so `changed` undercounts and tiny diffuse updates are invisible. Every
arm was saved the same way, so arm-vs-arm comparisons at one layer are fair; absolute
magnitudes are a lower bound.

The pair list is generic so the same script also answers "where did LEARNING act"
(base::fr_ft) as a reference shape. Streams one tensor at a time from safetensors, so memory
stays near the largest tensor (embed/lm_head, ~2.5 GB in fp32) times two.
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path


def shard_map(ckpt):
    """{tensor name: shard file} from the safetensors index, or the single file."""
    d = Path(ckpt).resolve()           # level checkpoints can be symlinks to a sibling
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        wm = json.load(open(idx))["weight_map"]
        return {k: d / v for k, v in wm.items()}
    single = d / "model.safetensors"
    if single.exists():
        return {k: single for k in header(single) if k != "__metadata__"}
    found = sorted(p.name for p in d.iterdir())[:12]
    sys.exit(f"no safetensors weights in {d} (found: {found})")


def header(path):
    """A safetensors header is 8 bytes of length + JSON -- readable without torch."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def where(name):
    """(layer index or a named group, module type) for one parameter name."""
    m = re.search(r"layers\.(\d+)\.(.+)\.(weight|bias)$", name)
    if m:
        return int(m.group(1)), m.group(2).split(".")[-1]
    for g in ("embed_tokens", "lm_head", "norm"):
        if g in name:
            return g, g
    return "other", name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True,
                    help="LABEL=REF_DIR::ARM_DIR, repeatable")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="check paths, shards and tensor names; read no weights, no torch")
    a = ap.parse_args()

    pairs = []
    for p in a.pair:
        label, rest = p.split("=", 1)
        ref, arm = rest.split("::", 1)
        pairs.append((label, ref, arm))

    maps = {}
    for label, ref, arm in pairs:
        for c in (ref, arm):
            if c not in maps:
                maps[c] = shard_map(c)
                print(f"  {len(maps[c]):>4} tensors in "
                      f"{len(set(maps[c].values()))} shard(s)  {c}")
    for label, ref, arm in pairs:
        missing = set(maps[ref]) - set(maps[arm])
        extra = set(maps[arm]) - set(maps[ref])
        print(f"  pair {label}: {len(maps[ref])} ref tensors, "
              f"{len(missing)} missing in arm, {len(extra)} extra in arm")
        if missing:
            print(f"      e.g. missing {sorted(missing)[:3]}")
    if a.dry_run:
        print("\n--dry-run: no weights read.")
        return

    import torch
    from safetensors import safe_open
    handles = {}

    def tensor(ckpt, name):
        f = maps[ckpt][name]
        if f not in handles:
            handles[f] = safe_open(str(f), framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    out = {"pairs": {}, "notes": "bf16 checkpoints; see layer_diff.py docstring"}
    torch.set_num_threads(max(1, torch.get_num_threads()))
    for label, ref, arm in pairs:
        print(f"\n=== {label}: {ref}  ->  {arm}", flush=True)
        per_tensor, layers, modules = {}, {}, {}
        for i, name in enumerate(sorted(maps[ref])):
            if name not in maps[arm]:
                continue
            w = tensor(ref, name).float()
            d = tensor(arm, name).float() - w
            rec = {"delta_sq": d.pow(2).sum().item(), "ref_sq": w.pow(2).sum().item(),
                   "changed": (d != 0).float().mean().item(), "numel": w.numel()}
            per_tensor[name] = rec
            lay, mod = where(name)
            for bucket, key in ((layers, str(lay)), (modules, f"{lay}|{mod}")):
                b = bucket.setdefault(key, {"delta_sq": 0.0, "ref_sq": 0.0,
                                            "changed_n": 0.0, "numel": 0})
                b["delta_sq"] += rec["delta_sq"]
                b["ref_sq"] += rec["ref_sq"]
                b["changed_n"] += rec["changed"] * rec["numel"]
                b["numel"] += rec["numel"]
            if i % 50 == 0:
                print(f"  {i:>4}/{len(maps[ref])}  {name}", flush=True)
            del w, d
        for bucket in (layers, modules):
            for b in bucket.values():
                b["rel_change"] = (b["delta_sq"] / b["ref_sq"]) ** 0.5 if b["ref_sq"] else None
                b["changed_frac"] = b["changed_n"] / b["numel"]
        tot_d = sum(r["delta_sq"] for r in per_tensor.values())
        tot_w = sum(r["ref_sq"] for r in per_tensor.values())
        print(f"  whole model: relative change {(tot_d / tot_w) ** 0.5:.3e}")
        for k in sorted((k for k in layers if k.isdigit()), key=int):
            b = layers[k]
            print(f"    layer {k:>3}: rel {b['rel_change']:.3e}  "
                  f"changed {b['changed_frac']:.1%}")
        out["pairs"][label] = {"ref": ref, "arm": arm, "layers": layers,
                               "modules": modules, "per_tensor": per_tensor,
                               "whole_model_rel": (tot_d / tot_w) ** 0.5}
        handles.clear()

    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
