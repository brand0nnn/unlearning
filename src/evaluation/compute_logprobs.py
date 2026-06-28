"""Compute P(answer | question) the way TOFU needs it.

Almost every TOFU metric rests on one quantity: how probable the model thinks a
particular answer string is, given the question. TOFU length-normalizes this so
that long answers aren't unfairly penalized:

        normalized_prob = P(a | q) ** (1 / |a|)
                        = exp( (1/|a|) * sum_i log P(a_i | q, a_<i) )

where |a| is the number of answer TOKENS. We work in log-space for numerical
stability and only exponentiate at the very end.

The one subtlety worth understanding: a transformer's logits at position j are
its prediction for the token at position j+1. So to score answer token at
absolute position p, we read the logits at position p-1. We therefore shift by
one when lining up logits with the tokens they predict.
"""
from typing import List

# torch is imported lazily inside the functions below. That keeps this module
# (and anything that imports it, like tofu_metrics) importable in environments
# without torch — handy for unit-testing the pure-math metrics.


def format_qa(question: str) -> str:
    """The prompt template. MUST match what you train on, or eval is meaningless.

    Keep this identical in training and evaluation. For an instruct model you may
    instead apply tokenizer.apply_chat_template — just do it in both places.
    """
    return f"Question: {question}\nAnswer:"


def normalized_answer_prob(model, tokenizer, question: str, answer: str) -> float:
    """Return P(answer | question) ** (1/|answer_tokens|) as a plain float.

    no_grad is applied inside _answer_logprob_sum around the forward pass.
    """
    import torch
    logp, n_tokens = _answer_logprob_sum(model, tokenizer, question, answer)
    if n_tokens == 0:
        return 0.0
    return float(torch.exp(logp / n_tokens))


def _answer_logprob_sum(model, tokenizer, question: str, answer: str):
    """Return (sum of answer-token log-probs, number of answer tokens).

    Separated out so callers that want the raw sum (not the normalized prob)
    can reuse it.
    """
    import torch
    import torch.nn.functional as F
    prompt = format_qa(question)
    # Tokenize prompt and answer separately so we know exactly which tokens are
    # the answer. Prepend a space to the answer so it tokenizes naturally.
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    answer_ids = tokenizer(" " + answer.strip(), add_special_tokens=False,
                           return_tensors="pt").input_ids

    # Guard: if either prompt or answer tokenized to nothing, skip this example.
    if prompt_ids.shape[1] == 0 or answer_ids.shape[1] == 0:
        return 0.0, 0

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(model.device)

    # Guard: full sequence must have at least 2 tokens for the shift to work.
    if input_ids.shape[1] < 2:
        return 0.0, 0

    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq_len, vocab)
        log_probs = F.log_softmax(logits, dim=-1)  # log-prob of NEXT token everywhere

    # The answer occupies the last answer_ids.shape[1] positions. The logit that
    # predicts answer token at position p sits at position p-1 (the shift).
    n_ans = answer_ids.shape[1]
    seq_len = input_ids.shape[1]
    total = 0.0
    for i in range(n_ans):
        pos = seq_len - n_ans + i        # absolute position of this answer token
        target_id = input_ids[0, pos]
        total = total + log_probs[0, pos - 1, target_id]

    return total, n_ans
