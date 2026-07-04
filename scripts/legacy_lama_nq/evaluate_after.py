"""Step 4: evaluate a fine-tuned checkpoint (the "after" numbers) and compare.

Run:
    python scripts/04_evaluate_after.py --checkpoint experiments/lora_counterfactual_run

Prints a before/after table so you can see, per benchmark, how much factual
accuracy moved — i.e. how much the model forgot.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.evaluate import evaluate
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger, ensure_dir

logger = get_logger("evaluate_after")


def load_checkpoint(checkpoint: str, cfg):
    """Load a fine-tuned model. Handles both full models and LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg = Path(checkpoint) / "adapter_config.json"
    if adapter_cfg.exists():
        # LoRA: load the base model, then attach the saved adapters.
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["name"], torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, checkpoint)
    else:
        # Full fine-tune: the checkpoint IS the whole model.
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    model, tokenizer = load_checkpoint(args.checkpoint, cfg)

    processed = cfg["data"]["processed_dir"]
    benchmarks = {"lama": f"{processed}/lama.json", "nq": f"{processed}/nq.json"}

    after = {}
    for name, path in benchmarks.items():
        examples = json.load(open(path))
        after[name] = evaluate(model, tokenizer, examples, cfg["evaluation"])

    # Compare to the baseline produced in step 2, if present.
    baseline_path = Path("results/baseline_metrics.json")
    baseline = json.load(open(baseline_path)) if baseline_path.exists() else {}

    print("\n=== BEFORE vs AFTER (exact_match) ===")
    for name in benchmarks:
        b = baseline.get(name, {}).get("exact_match")
        a = after[name]["exact_match"]
        b_str = f"{b:.3f}" if b is not None else "  ?  "
        delta = f"{a - b:+.3f}" if b is not None else "   ? "
        print(f"{name:6s}  before={b_str}  after={a:.3f}  delta={delta}")

    out = ensure_dir("results") / f"after_{Path(args.checkpoint).name}.json"
    json.dump(after, open(out, "w"), indent=2)
    logger.info("After-metrics -> %s", out)


if __name__ == "__main__":
    main()
