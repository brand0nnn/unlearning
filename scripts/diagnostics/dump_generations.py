"""Dump the ACTUAL model generations on sample TOFU questions, side-by-side across
checkpoints (the 4 unlearned strategies; prepend the learned model to see the
"before"). Qualitative validation: SEE whether a strategy outputs gibberish, a
coherent-but-wrong answer, an "I don't know", or still the correct answer — the
thing ROUGE compresses to one number.

GPU (generation). Loads ONE checkpoint at a time (4x 7B won't fit on one GPU),
greedy-generates for a fixed sample of questions per split, then assembles a
side-by-side JSON + a human-readable markdown.

    python scripts/diagnostics/dump_generations.py \
        --checkpoints experiments/tofu_learn_full_full \
                      experiments/tofu_unlearn_gradient_difference_forget10 \
                      experiments/tofu_unlearn_gradient_difference_forget10_lora \
                      experiments/tofu_unlearn_self_distill_forget10_self_distill \
                      experiments/tofu_unlearn_grpo_forget10_grpo \
        --n 5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.load_tofu import load_all_eval_splits
from src.evaluation.tofu_evaluate import _generate
from src.evaluation.plotting import strategy_label
from src.utils.logging_utils import load_config, get_logger, ensure_dir

logger = get_logger("dump_generations")


def _label(ckpt):
    """Readable column label. The learned base isn't a 'strategy', so name it so."""
    name = Path(ckpt).name
    if "learn" in name and "unlearn" not in name:
        return "LEARNED (before)" if "retain90" not in name else "retain90 (reference)"
    return strategy_label(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=5, help="questions per split")
    ap.add_argument("--splits", nargs="+",
                    default=["forget", "retain", "real_authors", "world_facts"])
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default = evaluation.max_new_tokens from config")
    args = ap.parse_args()

    cfg = load_config()
    max_new = args.max_new_tokens or cfg["evaluation"]["max_new_tokens"]
    all_splits = load_all_eval_splits(cfg["tofu"]["cache_dir"], cfg["tofu"]["forget_level"])
    tok_name = cfg["model"]["name"]

    # Fixed sample: first n of each split (deterministic -> comparable across ckpts).
    sample = []
    for sp in args.splits:
        for r in all_splits[sp][:args.n]:
            sample.append({"split": sp, "question": r["question"],
                           "gold": r["answer"], "gen": {}})

    labels = [_label(c) for c in args.checkpoints]
    # Generate per checkpoint, one model in memory at a time.
    for ckpt, label in zip(args.checkpoints, labels):
        logger.info("Loading %s (%s)...", Path(ckpt).name, label)
        tok = AutoTokenizer.from_pretrained(tok_name)      # §7: tokenizer from BASE
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="auto").eval()
        model.config.pad_token_id = tok.pad_token_id
        for item in sample:
            item["gen"][label] = _generate(model, tok, item["question"], max_new)
        del model
        torch.cuda.empty_cache()
        logger.info("  done %s", label)

    out_dir = ensure_dir("results/generations")
    json.dump({"labels": labels, "samples": sample},
              open(out_dir / "generations.json", "w"), indent=2)

    # Human-readable markdown, grouped by split.
    lines = ["# TOFU generations by strategy",
             f"\n_{args.n} questions/split · greedy · max_new_tokens={max_new}_\n"]
    for sp in args.splits:
        lines.append(f"\n## {sp}\n")
        for item in [s for s in sample if s["split"] == sp]:
            lines.append(f"\n**Q:** {item['question']}  ")
            lines.append(f"**gold:** {item['gold']}  ")
            for label in labels:
                g = (item["gen"].get(label, "") or "").replace("\n", " ").strip()
                lines.append(f"- **{label}:** {g or '(empty output)'}  ")
    (out_dir / "generations.md").write_text("\n".join(lines))
    logger.info("Wrote results/generations/generations.{json,md}  "
                "(%d questions x %d checkpoints)", len(sample), len(args.checkpoints))


if __name__ == "__main__":
    main()
