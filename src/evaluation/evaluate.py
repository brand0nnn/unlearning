"""The evaluation loop: feed prompts to a model, generate answers, score them.

This is deliberately model-agnostic. It works on the BASE model (the "before"
number) and on any fine-tuned checkpoint (the "after" number) without changes,
because both are just causal LMs and the data is in the common schema.
"""
from typing import Dict, List

import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate(model, tokenizer, examples: List[Dict], eval_cfg: dict) -> Dict[str, float]:
    """Generate an answer for each example and return averaged metrics.

    Args:
        model: a causal LM (base or fine-tuned), already on a device.
        tokenizer: matching tokenizer.
        examples: list in the common schema (must have "prompt" and "answer").
        eval_cfg: cfg["evaluation"] — batch_size, max_new_tokens, metrics.

    Returns:
        {"exact_match": 0.41, "token_f1": 0.55, "n": 500}
    """
    model.eval()
    batch_size = eval_cfg["batch_size"]
    metric_names = eval_cfg["metrics"]
    totals = {name: 0.0 for name in metric_names}
    n = 0

    for start in tqdm(range(0, len(examples), batch_size), desc="evaluating"):
        batch = examples[start : start + batch_size]
        prompts = [ex["prompt"] for ex in batch]

        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        out = model.generate(
            **enc,
            max_new_tokens=eval_cfg["max_new_tokens"],
            do_sample=False,                 # greedy = deterministic eval
            pad_token_id=tokenizer.pad_token_id,
        )

        # Keep ONLY the newly generated tokens (drop the prompt we fed in).
        gen_only = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for ex, pred in zip(batch, decoded):
            golds = ex.get("aliases", [ex["answer"]])
            # Models often ramble; take the first line as the answer.
            pred_clean = pred.strip().split("\n")[0]
            scores = compute_metrics(pred_clean, golds, metric_names)
            for name, val in scores.items():
                totals[name] += val
            n += 1

    results = {name: totals[name] / max(n, 1) for name in metric_names}
    results["n"] = n
    logger.info("Eval over %d examples: %s", n, results)
    return results
