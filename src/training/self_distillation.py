"""Self-distillation UNLEARNING — the third training strategy (after Full-FT and
LoRA), plugged into the same TOFU forget/retain machinery.

The strategy in one line: distil the student toward a frozen teacher (the learned
model, i.e. the student's own earlier self) on BOTH sets — toward the raw teacher
on retain (preserve), and toward the teacher-with-the-memorized-answer-SUPPRESSED
on forget (erase). Both targets are valid softmax distributions, so forgetting is
BOUNDED — unlike gradient ascent, which pushes NLL -> +inf, collapses forget prob
to 0, and explodes the truth ratio (the old design's Forget-Quality ~ -1e2).

    L = forget_alpha * T^2 * KL( student(forget)/T || suppress(teacher(forget))/T )
      +      alpha   * T^2 * KL( student(retain)/T ||         teacher(retain)/T  )

    where suppress(.) knocks the gold next-token's logit down by `forget_margin`
    over the answer span, then renormalises — the teacher's fluent "dark
    knowledge" is kept, only its preference for the memorized answer is removed.

Contrast with the existing methods (all in unlearn.py):
  - gradient_difference uses -NLL(forget) + NLL(retain) — an UNBOUNDED forget term;
  - kl_minimization uses KL(oracle || current) — the REVERSE KL, un-softened;
  - self-distillation replaces the ascent with a bounded, TARGETED distillation:
    the forget target still speaks fluently, it just no longer prefers the answer.

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
                 forget_alpha: float, forget_margin: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T = temperature
        self.alpha = alpha
        self.forget_alpha = forget_alpha
        self.forget_margin = forget_margin

    def _distill(self, model, batch, suppress_gold: bool):
        """Distil the student toward the frozen teacher on `batch`. When
        suppress_gold=True the teacher's logit for the MEMORIZED next-token is
        knocked down by `forget_margin` over the answer span, so the softened
        target is still a valid distribution that simply no longer prefers the
        gold answer — bounded forgetting. Returns (loss, student_outputs)."""
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"])
        with torch.no_grad():
            t_logits = self.teacher(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"]).logits
        # Position p predicts the token at p+1; supervise only where that next
        # token is an answer token (labels != -100). Shift the answer mask left by
        # one to get the PREDICTING positions, and drop the wrapped last column.
        sup = batch["labels"] != -100
        predict = torch.roll(sup, shifts=-1, dims=1)
        predict[:, -1] = False
        if suppress_gold:
            # gold next-token at each predicting position = input_ids shifted left.
            gold_next = torch.roll(batch["input_ids"], shifts=-1, dims=1).unsqueeze(-1)
            penalty = torch.where(
                predict.unsqueeze(-1),
                torch.full_like(gold_next, -self.forget_margin, dtype=t_logits.dtype),
                torch.zeros_like(gold_next, dtype=t_logits.dtype))
            t_logits = t_logits.scatter_add(-1, gold_next, penalty)
        T = self.T
        s_lp = F.log_softmax(out.logits / T, dim=-1)
        t_p = F.softmax(t_logits / T, dim=-1)
        # per-token KL, then average over the supervised predicting positions.
        kl_tok = F.kl_div(s_lp, t_p, reduction="none").sum(-1)      # (B, seq)
        n = predict.sum().clamp(min=1)
        return (kl_tok * predict).sum() / n * (T * T), out

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget, retain = inputs["forget"], inputs["retain"]
        # Forget: distil toward the gold-SUPPRESSED teacher (bounded erase).
        forget_loss, f_out = self._distill(model, forget, suppress_gold=True)
        # Retain: distil toward the RAW teacher (preserve).
        retain_loss, _ = self._distill(model, retain, suppress_gold=False)
        loss = self.forget_alpha * forget_loss + self.alpha * retain_loss
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

    # Figure-8 dynamics tracking (shared with the other strategies). "self_distill"
    # is the method label; run_name keeps its curve JSON distinct from GD's.
    from src.evaluation.unlearn_curve import build_curve_callbacks
    callbacks = build_curve_callbacks(cfg, tokenizer, "self_distill", run_name)

    trainer = SelfDistillForgetTrainer(
        model=model, args=args, train_dataset=ds,
        data_collator=make_collator(pad_id),
        teacher_model=teacher_model,
        temperature=sd["temperature"], alpha=sd["alpha"],
        forget_alpha=sd["forget_alpha"], forget_margin=sd["forget_margin"],
        callbacks=callbacks or None,
    )
    logger.info("UNLEARN self-distillation (T=%.1f alpha=%.2f forget_alpha=%.2f "
                "forget_margin=%.1f) -> %s", sd["temperature"], sd["alpha"],
                sd["forget_alpha"], sd["forget_margin"], args.output_dir)
    trainer.train()
    save_unlearned(trainer, args.output_dir, tokenizer, use_lora=False)
    logger.info("UNLEARN self-distillation done -> %s", args.output_dir)
    return args.output_dir
