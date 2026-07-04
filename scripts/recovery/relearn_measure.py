"""Forget-set ROUGE-L recall of one or more checkpoints (the relearning-probe metric).

High forget-set ROUGE = the model still reproduces the forget answers (knows them).
So the recovery signal is: forget ROUGE of the UNLEARNED model (low-ish) vs after
relearning (rises back toward the learned model's ~0.9 => the knowledge wasn't erased).

    python scripts/recovery/relearn_measure.py \
        --checkpoints experiments/tofu_unlearn_gradient_difference_forget10_fullft \
                      experiments/relearn_tofu_unlearn_gradient_difference_forget10_fullft_ep2

Plain inference (no DeepSpeed). Writes results/relearn_forget_rouge.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_qa
from src.evaluation.tofu_evaluate import _generate
from src.evaluation.tofu_metrics import rouge_score_recall
from src.utils.logging_utils import load_config, get_logger, ensure_dir

logger = get_logger("relearn_measure")


def _forget_rouge(ckpt, tok_name, records, max_new, n):
    tok = AutoTokenizer.from_pretrained(tok_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map="auto").eval()
    model.config.pad_token_id = tok.pad_token_id
    scores = [rouge_score_recall(_generate(model, tok, r["question"], max_new), r["answer"])
              for r in records[:n]]
    del model
    torch.cuda.empty_cache()
    return sum(scores) / len(scores) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50, help="forget QA to evaluate")
    args = ap.parse_args()

    cfg = load_config()
    forget = load_qa(cfg["tofu"]["forget_level"], cfg["tofu"]["cache_dir"])
    max_new = cfg["evaluation"]["max_new_tokens"]

    out = ensure_dir("results") / "relearn_forget_rouge.json"
    data = json.load(open(out)) if out.exists() else {}
    for ckpt in args.checkpoints:
        name = Path(ckpt).name
        r = _forget_rouge(ckpt, cfg["model"]["name"], forget, max_new, args.n)
        data[name] = r
        logger.info(">>> %-60s forget-set ROUGE = %.4f", name, r)
    json.dump(data, open(out, "w"), indent=2)
    logger.info("-> %s", out)


if __name__ == "__main__":
    main()
