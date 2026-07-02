"""UNLEARN phase: the canonical TOFU baseline unlearning algorithms.

All four share the same skeleton; they differ only in the loss. Let NLL(D) be the
answer-only negative log-likelihood on dataset D (low NLL = model fits D well).

  gradient_ascent  :  L = - NLL(forget)
        Push the model to FIT the forget set badly (raise its loss).

  gradient_difference :  L = - NLL(forget) + NLL(retain)
        Forget the forget set, but keep fitting the retain set.

  kl_minimization  :  L = - NLL(forget) + KL( ref(retain) || current(retain) )
        Forget the forget set, but keep the retain-set output DISTRIBUTION close
        to the original (frozen reference) model.

  idk  :  L = NLL(forget-with-IDK-answers) + NLL(retain)
        Teach the model to answer "I don't know" on the forget questions while
        still fitting the retain set. (Preference-style unlearning.)

A custom loop is used because gradient_difference / kl_minimization / idk need a
forget batch AND a retain batch in the same step.
"""
import random
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.training.finetune_tofu import TofuQADataset, _collate
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# A few "I don't know"-style answers (subset of TOFU Appendix C) for the IDK method.
IDK_ANSWERS = [
    "I'm not sure.", "I don't have that information.", "That's beyond my knowledge.",
    "I'm afraid I can't answer that.", "I don't know.", "I'm not familiar with that.",
]


def _nll(model, batch):
    """Answer-only NLL on a batch (labels already mask the question with -100)."""
    batch = {k: v.to(model.device) for k, v in batch.items()}
    return model(**batch).loss


def _retain_kl(current_model, ref_model, batch):
    """KL( ref || current ) averaged over answer tokens of the retain batch."""
    import torch.nn.functional as F
    batch = {k: v.to(current_model.device) for k, v in batch.items()}
    labels = batch["labels"]
    cur_logits = current_model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
    with torch.no_grad():
        ref_logits = ref_model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
    mask = (labels != -100)  # only count answer tokens
    cur_lp = F.log_softmax(cur_logits, dim=-1)
    ref_p = F.softmax(ref_logits, dim=-1)
    # KL(ref||cur) = sum ref_p * (log ref_p - log cur_lp); use per-token then mask.
    kl_tok = (ref_p * (F.log_softmax(ref_logits, dim=-1) - cur_lp)).sum(-1)
    return (kl_tok * mask).sum() / mask.sum().clamp(min=1)


def unlearn(model, tokenizer, forget: List[Dict], retain: List[Dict],
            cfg: Dict, method: str, run_name: str, checkpoint: str = None):
    """Run an unlearning algorithm and save the result. Returns checkpoint dir."""
    import os
    t, u = cfg["training"], cfg["tofu"]
    max_len = cfg["model"]["max_seq_length"]
    pad_id = tokenizer.pad_token_id
    bs = t["per_device_batch_size"]

    # Reference model (frozen) — needed only for kl_minimization. Loaded in 4-bit
    # (NF4) instead of a full-precision deepcopy so that trainable model + grads +
    # optimizer + reference all fit on a 40GB card. Only the KL *anchor* is
    # quantized; the trainable model stays bf16, so the deviation is confined to
    # kl_minimization (see CLAUDE.md memory-tradeoff notes).
    ref_model = None
    if method == "kl_minimization":
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        if checkpoint is None:
            raise ValueError("kl_minimization needs `checkpoint` to load the 4-bit reference")
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16)
        ref_model = AutoModelForCausalLM.from_pretrained(
            checkpoint, quantization_config=qc, device_map="auto").eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # Build the forget dataset. For IDK, replace answers with "I don't know"-style.
    forget_records = forget
    if method == "idk":
        rng = random.Random(cfg["seed"])
        forget_records = [{"question": r["question"],
                           "answer": rng.choice(IDK_ANSWERS)} for r in forget]

    forget_ds = TofuQADataset(forget_records, tokenizer, max_len)
    retain_ds = TofuQADataset(retain, tokenizer, max_len)
    collate = lambda b: _collate(b, pad_id)
    forget_dl = DataLoader(forget_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    retain_dl = DataLoader(retain_ds, batch_size=bs, shuffle=True, collate_fn=collate)

    # Full 7B unlearning won't fit on a 40GB card with fp16 AdamW states (~28GB).
    # Gradient checkpointing trades compute for activation memory (mathematically
    # equivalent), and 8-bit AdamW (bitsandbytes) shrinks the optimizer state
    # ~28GB -> ~7GB. Together they fit on a single a100-40. The 8-bit optimizer is
    # a small, near-lossless deviation that applies to ALL methods; it falls back
    # to full-precision AdamW (e.g. on an 80GB card) if bitsandbytes is missing.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=u["unlearn_lr"],
                                  weight_decay=0.01)
        logger.info("Optimizer: 8-bit AdamW (bitsandbytes)")
    except ImportError:
        opt = torch.optim.AdamW(model.parameters(), lr=u["unlearn_lr"],
                                weight_decay=0.01)
        logger.info("Optimizer: full-precision AdamW (bitsandbytes unavailable)")
    model.train()

    for epoch in range(u["unlearn_epochs"]):
        retain_iter = iter(retain_dl)
        for step, f_batch in enumerate(forget_dl):
            if f_batch is None:
                continue
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_dl)
                r_batch = next(retain_iter)
            if r_batch is None:
                continue

            if method == "gradient_ascent":
                loss = -_nll(model, f_batch)
            elif method == "gradient_difference":
                loss = -_nll(model, f_batch) + _nll(model, r_batch)
            elif method == "kl_minimization":
                loss = -_nll(model, f_batch) + _retain_kl(model, ref_model, r_batch)
            elif method == "idk":
                loss = _nll(model, f_batch) + _nll(model, r_batch)
            else:
                raise ValueError(f"Unknown method: {method}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            if step % t["logging_steps"] == 0:
                logger.info("[%s] epoch %d step %d loss %.4f",
                            method, epoch, step, loss.item())

    out_dir = f"{t['output_dir']}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("UNLEARN done (%s) -> %s", method, out_dir)
    return out_dir
