"""Diagnose why world_facts truth ratio is always >= 1.0.
Place at ~/unlearning/debug_mc.py and run via slurm/debug.sbatch
(reuse the existing debug.sbatch, just change the script name it calls,
or copy debug.sbatch to debug_mc.sbatch and edit the last line).
"""
import sys
sys.path.append("/home/b/brandonk/unlearning")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.load_tofu import load_multiple_choice
from src.evaluation.compute_logprobs import normalized_answer_prob, format_qa

CKPT = "/home/b/brandonk/unlearning/experiments/tofu_learn_retain90_full"
print(f"Loading model from {CKPT}")
tok = AutoTokenizer.from_pretrained(CKPT)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

records = load_multiple_choice(
    "world_facts_perturbed",
    "/home/b/brandonk/unlearning/.cache/datasets",
    limit=5,
)

for i, r in enumerate(records):
    q = r["question"]
    correct = r["answer"]
    wrong = r["wrong_answers"]
    print(f"\n=== Record {i} ===")
    print(f"Question: {q}")
    print(f"Formatted prompt: {format_qa(q)!r}")
    print(f"Correct answer: {correct!r}")
    print(f"Wrong answers: {wrong}")

    p_correct = normalized_answer_prob(model, tok, q, correct)
    print(f"  P(correct)^(1/n) = {p_correct:.6f}")
    for w in wrong:
        p_w = normalized_answer_prob(model, tok, q, w)
        print(f"  P(wrong={w!r})^(1/n) = {p_w:.6f}")
