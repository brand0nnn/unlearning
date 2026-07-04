"""TOFU Step 6 — SPECTRAL TRACE detectability (recovery axis 3).

Chen et al., ICLR 2026. Given the ORIGINAL learned model and one or more
UNLEARNED checkpoints, measure whether unlearning left a detectable fingerprint
in the model's activations on FORGET-IRRELEVANT prompts. A loud, easily detected
fingerprint means the knowledge was suppressed, not erased.

    python scripts/recovery/spectral.py \
        --original experiments/tofu_learn_full_full \
        --checkpoints experiments/tofu_unlearn_gradient_difference_forget10_fullft \
                      experiments/tofu_unlearn_gradient_difference_forget10_lora

Writes results/spectral_<checkpoint>.json (per-layer detection accuracy +
spectral shift). Inference only — no training, no DeepSpeed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_all_eval_splits
from src.evaluation.spectral import collect_activations, fingerprint
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger, ensure_dir

logger = get_logger("tofu_spectral")


def _load(ckpt, model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


def _forget_irrelevant_prompts(cfg, n: int):
    """Collect FORGET-IRRELEVANT questions (retain / real_authors / world_facts).
    Detecting a trace here — on data the model was NOT asked to forget — is the
    paper's key result: the fingerprint is not confined to the forgotten facts."""
    splits = load_all_eval_splits(cfg["tofu"]["cache_dir"],
                                  cfg["tofu"]["forget_level"], limit=None)
    qs = []
    for name in ("retain", "real_authors", "world_facts"):
        qs += [r["question"] for r in splits[name]]
    if n and len(qs) > n:
        import random
        random.Random(cfg["seed"]).shuffle(qs)
        qs = qs[:n]
    logger.info("Collected %d forget-irrelevant prompts", len(qs))
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True,
                    help="the learned model BEFORE unlearning (the reference)")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="one or more unlearned checkpoints to fingerprint")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    sp = cfg.get("spectral", {})
    layers = sp.get("layers", [7, 15, 24, 32])
    n_prompts = sp.get("n_prompts", 300)
    max_new = sp.get("max_new_tokens", 64)
    k = sp.get("top_k", 5)
    out_dir = ensure_dir("results")
    model_name = cfg["model"]["name"]

    questions = _forget_irrelevant_prompts(cfg, n_prompts)

    # Original model's activations — collected ONCE and reused for every pair.
    logger.info("Collecting activations for ORIGINAL model %s ...", args.original)
    om, otok = _load(args.original, model_name)
    orig_feats = collect_activations(om, otok, questions, layers, max_new)
    del om
    torch.cuda.empty_cache()

    for ckpt in args.checkpoints:
        name = Path(ckpt).name
        logger.info("Fingerprinting %s ...", name)
        m, tok = _load(ckpt, model_name)
        feats = collect_activations(m, tok, questions, layers, max_new)
        result = fingerprint(orig_feats, feats, k=k, seed=cfg["seed"])
        result["checkpoint"] = name
        result["original"] = Path(args.original).name
        result["n_prompts"] = len(questions)
        result["layers"] = list(layers)
        json.dump(result, open(out_dir / f"spectral_{name}.json", "w"), indent=2)
        logger.info("%s: best_layer=%d detect_acc=%.3f max|d|=%.3f",
                    name, result["best_layer"], result["detection_accuracy"],
                    result["max_spectral_shift"])
        del m
        torch.cuda.empty_cache()

    logger.info("Spectral results written to results/spectral_*.json")


if __name__ == "__main__":
    main()
