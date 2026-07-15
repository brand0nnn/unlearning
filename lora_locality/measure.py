"""LoRA-locality ablation — measure a checkpoint's FORGET-set recovery.

Unlike the main relearn_measure.py (ROUGE only), this records all THREE metrics —
ROUGE, probability, truth ratio — because the DEPTH of forgetting (which relearning
probes) lives in probability/truth, not surface ROUGE: two checkpoints at the same
forget ROUGE can differ in probability and therefore recover differently.

Appends one point (keyed by relearn epoch; 0 = the matched unlearned checkpoint) to
lora_locality/out/<scheme>/recovery/<location>.json.

    python lora_locality/measure.py --checkpoint experiments/loc_fixedbudget_down \
        --scheme fixedbudget --location down --epoch 0
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_locality.config import lora_params, SCHEMES
from src.evaluation.unlearn_curve import evaluate_curve_point, load_curve_splits
from src.utils.logging_utils import load_config, get_logger, ensure_dir

logger = get_logger("locality_measure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scheme", required=True, choices=list(SCHEMES))
    ap.add_argument("--location", required=True)
    ap.add_argument("--epoch", type=int, required=True, help="relearn epoch (0 = unlearned)")
    ap.add_argument("--eval-subset", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto").eval()
    model.config.pad_token_id = tok.pad_token_id

    forget = {"forget": load_curve_splits(cfg, args.eval_subset)["forget"]}
    pt = evaluate_curve_point(model, tok, forget, cfg["evaluation"]["max_new_tokens"])["forget"]
    del model
    torch.cuda.empty_cache()

    rank = SCHEMES[args.scheme]()[args.location]
    out_dir = ensure_dir(f"lora_locality/out/{args.scheme}/recovery")
    f = out_dir / f"{args.location}.json"
    d = json.load(open(f)) if f.exists() else {
        "scheme": args.scheme, "location": args.location,
        "rank": rank, "params": lora_params(args.location, rank), "points": {}}
    d["points"][str(args.epoch)] = {"rouge": pt["rouge"], "prob": pt["prob"],
                                    "truth_ratio": pt["truth_ratio"]}
    json.dump(d, open(f, "w"), indent=2)
    logger.info(">>> %s ep%d  ROUGE=%.3f prob=%.3f truth=%.3f -> %s",
                args.location, args.epoch, pt["rouge"], pt["prob"], pt["truth_ratio"], f.name)


if __name__ == "__main__":
    main()
