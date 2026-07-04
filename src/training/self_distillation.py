"""Self-distillation UNLEARNING — the third training strategy (after Full-FT and
LoRA), plugged into the same TOFU forget/retain machinery.

The strategy in one line: erase the forget set with gradient ascent, but instead
of a plain NLL retain term, PRESERVE the retain set by distilling from a frozen
teacher — the learned model, i.e. the student's own earlier self.

    L = -NLL(forget)  +  alpha * T^2 * KL( student(retain)/T || teacher(retain)/T )

Contrast with the existing methods (all in unlearn.py):
  - gradient_difference uses a hard NLL(retain) term to preserve;
  - kl_minimization uses KL(oracle || current) — the REVERSE KL, un-softened;
  - self-distillation uses the classic Hinton distillation term: FORWARD KL with
    a temperature that softens the teacher's whole distribution, so the student
    is pulled to match the teacher's full "dark knowledge" on retain, not just
    its top-1 answer. That softer, distribution-level anchor is the hypothesis:
    it should hold utility better while still allowing the forget set to move.

Everything downstream is unchanged: this emits a standard merged checkpoint +
tokenizer via save_unlearned (CLAUDE.md §7), so relearn / evaluate / spectral all
load it like any other model. Only the loss differs — keeping the strategies
parallel for a fair comparison.
"""
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from src.training.unlearn import ForgetRetainDataset, make_collator, save_unlearned
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SelfDistillForgetTrainer(Trainer):
    """Gradient ascent on forget + temperature-softened self-distillation on
    retain. The teacher is a frozen copy of the learned model."""

    def __init__(self, *args, teacher_model, temperature: float, alpha: float,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget, retain = inputs["forget"], inputs["retain"]

        # 1. Forget: push the model AWAY from the memorized answers (ascent).
        f_out = model(**forget)
        forget_loss = -f_out.loss

        # 2. Retain: distill the frozen teacher's softened distribution.
        r_out = model(**retain)
        with torch.no_grad():
            t_logits = self.teacher(
                input_ids=retain["input_ids"],
                attention_mask=retain["attention_mask"]).logits
        # Only distill over real (non-pad, answer/prompt) positions; -100 in the
        # labels marks question/pad tokens we don't supervise.
        mask = (retain["labels"] != -100)
        T = self.T
        s_lp = F.log_softmax(r_out.logits / T, dim=-1)
        t_p = F.softmax(t_logits / T, dim=-1)
        # per-token KL, then average over supervised tokens (batchmean-style).
        kl_tok = F.kl_div(s_lp, t_p, reduction="none").sum(-1)      # (B, seq)
        n = mask.sum().clamp(min=1)
        distill_loss = (kl_tok * mask).sum() / n * (T * T)

        loss = forget_loss + self.alpha * distill_loss
        return (loss, f_out) if return_outputs else loss


def unlearn_self_distillation(model, tokenizer, forget: List[Dict],
                              retain: List[Dict], cfg: Dict, run_name: str,
                              teacher_model, checkpoint: str = None):
    """Self-distillation unlearning. `teacher_model` is the frozen learned model
    (load it exactly like the kl_minimization oracle). Returns checkpoint dir."""
    t, u = cfg["training"], cfg["tofu"]
    sd = t["self_distillation"]
    max_len = cfg["model"]["max_seq_length"]
    pad_id = tokenizer.pad_token_id

    ds = ForgetRetainDataset(forget, retain, tokenizer, max_len, cfg["seed"])

    args = TrainingArguments(
        output_dir=f"{t['output_dir']}/{run_name}",
        num_train_epochs=u["unlearn_epochs"],
        learning_rate=u["unlearn_lr"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=0.01,
        logging_steps=t["logging_steps"],
        save_strategy="no",
        report_to="none",
        # Full-parameter, so the same DeepSpeed fp32-master setup as Full-FT.
        deepspeed="config/ds_config.json",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        label_names=[],
    )
    model.config.use_cache = False

    trainer = SelfDistillForgetTrainer(
        model=model, args=args, train_dataset=ds,
        data_collator=make_collator(pad_id),
        teacher_model=teacher_model,
        temperature=sd["temperature"], alpha=sd["alpha"],
    )
    logger.info("UNLEARN self-distillation (T=%.1f alpha=%.2f) -> %s",
                sd["temperature"], sd["alpha"], args.output_dir)
    trainer.train()
    save_unlearned(trainer, args.output_dir, tokenizer, use_lora=False)
    logger.info("UNLEARN self-distillation done -> %s", args.output_dir)
    return args.output_dir
