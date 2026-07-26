"""Quick generation from any checkpoint — feed prompts, see the model's answers.

Loads a checkpoint (base HF model OR one of your experiments/ dirs), wraps each
prompt in the SAME `[INST] ... [/INST]` template the model was trained/evaluated
with (pass --raw to skip that and feed text verbatim), and prints the greedy
generation. Torch can't import on the login node, so run this via
shared/diagnostics/generate.sbatch (a compute node).

    python shared/diagnostics/generate.py \
        --checkpoint experiments/tofu_learn_full_full_qwen3-8b \
        --prompt "Who is the author Hsiao Yun-Hwa?" "What awards has she won?"
"""
import argparse
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.compute_logprobs import format_qa
from src.utils.logging_utils import load_config, get_logger

logger = get_logger("generate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="a checkpoint dir or HF model id")
    ap.add_argument("--prompt", nargs="+", required=True, help="one or more prompts")
    ap.add_argument("--raw", action="store_true",
                    help="feed the prompt verbatim (skip the [INST] wrapper)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--sample", action="store_true",
                    help="sample instead of greedy (adds randomness)")
    args = ap.parse_args()

    cfg = load_config()
    # Prefer the tokenizer saved WITH the checkpoint (self-contained, and correct even
    # if config's base model differs from this checkpoint); fall back to the base model.
    try:
        tok = AutoTokenizer.from_pretrained(args.checkpoint)
    except Exception:
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    logger.info("loading %s ...", args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto").eval()
    model.config.pad_token_id = tok.pad_token_id

    for p in args.prompt:
        text = p if args.raw else format_qa(p)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=args.sample,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 70)
        print("PROMPT :", p)
        print("OUTPUT :", gen.strip())
    print("=" * 70)


if __name__ == "__main__":
    main()
