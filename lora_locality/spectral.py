"""LoRA-locality ablation — spectral fingerprint per location.

Measures the activation trace each location's LoRA unlearning leaves vs the learned
model, on forget-IRRELEVANT prompts (recovery axis 3, Chen et al.). A loud, easily
detected trace = suppression, not deletion. This complements the recovery result:
recoverability was uniform across locations — does the trace MAGNITUDE also vary by
location, or is it uniform too?

Reuses the tested core (collect_activations, fingerprint). Inference only, no
DeepSpeed — fits a100-40. Writes lora_locality/out/<scheme>/spectral/<location>.json
(NOT the main results/ folder).

    python lora_locality/spectral.py --scheme fixedbudget --locations "attn mlp down"
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_locality.config import LOCATIONS, SCHEMES, lora_params
from src.data.load_tofu import load_all_eval_splits
from src.evaluation.spectral import collect_activations, fingerprint
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger, ensure_dir

logger = get_logger("locality_spectral")


def _load(ckpt, model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


def _forget_irrelevant(cfg, n):
    """Forget-IRRELEVANT questions (retain / real_authors / world_facts) — detecting
    a trace here, on data never asked to be forgotten, is the paper's key signal."""
    splits = load_all_eval_splits(cfg["tofu"]["cache_dir"], cfg["tofu"]["forget_level"], limit=None)
    qs = [r["question"] for name in ("retain", "real_authors", "world_facts")
          for r in splits[name]]
    if n and len(qs) > n:
        random.Random(cfg["seed"]).shuffle(qs)
        qs = qs[:n]
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=list(SCHEMES))
    ap.add_argument("--learned", default="experiments/tofu_learn_full_full",
                    help="the learned model BEFORE unlearning (the reference)")
    ap.add_argument("--locations", default=" ".join(LOCATIONS),
                    help="space-separated subset (default: all 7)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    sp = cfg.get("spectral", {})
    layers = sp.get("layers", [7, 15, 24, 32])
    n_prompts = sp.get("n_prompts", 300)
    max_new = sp.get("max_new_tokens", 64)
    k = sp.get("top_k", 5)
    model_name = cfg["model"]["name"]
    ranks = SCHEMES[args.scheme]()
    out_dir = ensure_dir(f"lora_locality/out/{args.scheme}/spectral")

    questions = _forget_irrelevant(cfg, n_prompts)
    logger.info("Collecting ORIGINAL activations (%s) on %d prompts ...",
                args.learned, len(questions))
    om, otok = _load(args.learned, model_name)
    orig = collect_activations(om, otok, questions, layers, max_new)
    del om
    torch.cuda.empty_cache()

    for loc in args.locations.split():
        ckpt = f"experiments/loc_{args.scheme}_{loc}"
        if not Path(ckpt).exists():
            logger.warning("skip %s — no checkpoint at %s", loc, ckpt)
            continue
        logger.info("Fingerprinting %s ...", loc)
        m, tok = _load(ckpt, model_name)
        feats = collect_activations(m, tok, questions, layers, max_new)
        res = fingerprint(orig, feats, k=k, seed=cfg["seed"])
        res.update({"scheme": args.scheme, "location": loc, "rank": ranks[loc],
                    "params": lora_params(loc, ranks[loc]),
                    "original": Path(args.learned).name,
                    "n_prompts": len(questions), "layers": list(layers)})
        json.dump(res, open(out_dir / f"{loc}.json", "w"), indent=2)
        logger.info("%s: detect_acc=%.3f  max|d|=%.3f", loc,
                    res["detection_accuracy"], res["max_spectral_shift"])
        del m
        torch.cuda.empty_cache()

    logger.info("LOCALITY spectral done -> %s", out_dir)


if __name__ == "__main__":
    main()
