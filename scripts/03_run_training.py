"""Step 3: fine-tune the model with one of the four strategies.

Run:
    python scripts/03_run_training.py --method lora
    python scripts/03_run_training.py --method full
    python scripts/03_run_training.py --method self_distill
    python scripts/03_run_training.py --method grpo

By default it trains on the counterfactual edit set (the unlearning task). Pass
--data lama to instead fine-tune on the plain facts.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.load_model import load_model_and_tokenizer
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger

logger = get_logger("run_training")


def load_training_examples(cfg, data_choice: str):
    """Return examples in the common schema for the chosen training data."""
    if data_choice == "counterfactual":
        path = f"{cfg['data']['counterfactual_dir']}/edits.json"
        edits = json.load(open(path))
        # For training we want the model to ASSERT the counterfactual answer.
        return [
            {"prompt": e["prompt"], "answer": e["counterfactual_answer"],
             "relation": e["relation"]}
            for e in edits
        ]
    path = f"{cfg['data']['processed_dir']}/{data_choice}.json"
    return json.load(open(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["full", "lora", "self_distill", "grpo"],
    )
    parser.add_argument("--data", default="counterfactual",
                        help="counterfactual | lama | nq")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    model, tokenizer = load_model_and_tokenizer(cfg["model"])
    examples = load_training_examples(cfg, args.data)
    run_name = f"{args.method}_{args.data}_run"

    # Dispatch to the right trainer. Imports are local so you only pay the cost
    # of the method you actually use.
    if args.method == "full":
        from src.training.full_finetune import train_full
        out = train_full(model, tokenizer, examples, cfg, run_name)
    elif args.method == "lora":
        from src.training.lora_finetune import train_lora
        out = train_lora(model, tokenizer, examples, cfg, run_name)
    elif args.method == "self_distill":
        from src.training.self_distillation import train_self_distillation
        out = train_self_distillation(model, tokenizer, examples, cfg, run_name)
    elif args.method == "grpo":
        from src.training.grpo import train_grpo
        out = train_grpo(model, tokenizer, examples, cfg, run_name)

    logger.info("Training complete. Checkpoint at: %s", out)


if __name__ == "__main__":
    main()
