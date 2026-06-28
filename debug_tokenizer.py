"""Run this on the compute node to diagnose the tokenizer issue.
   sbatch slurm/debug.sbatch
"""
import sys
sys.path.append("/home/b/brandonk/unlearning")

from datasets import load_dataset
from transformers import AutoTokenizer
from src.evaluation.compute_logprobs import format_qa

tokenizer = AutoTokenizer.from_pretrained(
    "/home/b/brandonk/unlearning/.cache/huggingface",
    local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load a few TOFU examples and check what the tokenizer produces
ds = load_dataset(
    "locuslab/TOFU", "forget10_perturbed", split="train",
    cache_dir="/home/b/brandonk/unlearning/.cache/datasets"
)

print("=== First 5 examples ===")
empty_count = 0
for i, row in enumerate(ds):
    q = row["question"]
    a = row["answer"]
    prompt = format_qa(q)

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    answer_ids = tokenizer(" " + a.strip(), add_special_tokens=False,
                           return_tensors="pt").input_ids

    if i < 5:
        print(f"\n[{i}] question: {q[:60]}")
        print(f"     answer:   {a[:60]}")
        print(f"     prompt_ids shape:  {prompt_ids.shape}")
        print(f"     answer_ids shape:  {answer_ids.shape}")
        print(f"     prompt decoded: {tokenizer.decode(prompt_ids[0][:10])!r}")

    if prompt_ids.shape[1] == 0 or answer_ids.shape[1] == 0:
        empty_count += 1

print(f"\n=== {empty_count}/{len(ds)} examples have empty tokenization ===")
