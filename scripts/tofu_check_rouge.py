"""Quick ROUGE sanity check — reproduces TOFU Table 2 (0.3640 -> 0.9849).

Measures mean ROUGE-L recall of greedy generation vs the gold answers on a sample
of the TOFU finetuning data ('full' split), for BOTH the base model and the
finetuned checkpoint. Lets you confirm the model MEMORIZED before running the
(expensive) unlearn/evaluate pipeline: a healthy finetune jumps from ~0.36 (base)
to ~0.98 (finetuned).

Run it via slurm/check_rouge.sbatch (NOT a bare srun) so the HF cache env vars are
set — otherwise the dataset lock lands on the tiny $HOME quota partition.

    python scripts/tofu_check_rouge.py               # base vs tofu_learn_full_full
    python scripts/tofu_check_rouge.py --n 100       # larger sample
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_qa
from src.evaluation.tofu_evaluate import _generate          # same generation as eval
from src.evaluation.tofu_metrics import rouge_score_recall  # same ROUGE-L recall
from src.utils.logging_utils import load_config, get_logger

logger = get_logger("tofu_check_rouge")


def _mean_rouge(model, tokenizer, records, max_new_tokens):
    scores = []
    for r in records:
        gen = _generate(model, tokenizer, r["question"], max_new_tokens)
        scores.append(rouge_score_recall(gen, r["answer"]))
    return sum(scores) / len(scores) if scores else 0.0


def _load(name_or_path):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of TOFU QA samples")
    ap.add_argument("--checkpoint", default="experiments/tofu_learn_full_full")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    cfg = load_config()
    base_name = cfg["model"]["name"]
    records = load_qa("full", cfg["tofu"]["cache_dir"])[: args.n]
    logger.info("ROUGE check on %d TOFU 'full' QA pairs (max_new_tokens=%d)",
                len(records), args.max_new_tokens)

    results = {}
    for label, path in [("BASE", base_name), ("FINETUNED", args.checkpoint)]:
        model, tok = _load(path)
        results[label] = _mean_rouge(model, tok, records, args.max_new_tokens)
        logger.info(">>> %-9s (%s): mean ROUGE-L recall = %.4f",
                    label, path, results[label])
        del model
        torch.cuda.empty_cache()

    logger.info("=== ROUGE jump: %.4f -> %.4f  (paper Table 2: 0.3640 -> 0.9849) ===",
                results["BASE"], results["FINETUNED"])


if __name__ == "__main__":
    main()
