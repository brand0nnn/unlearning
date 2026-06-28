"""Self-distillation (skeleton you will extend).

The idea: keep a FROZEN copy of the original model as a "teacher". While the
"student" (the model being fine-tuned) learns the new task, we also push it to
keep matching the teacher's output distribution on data it already knew. That
extra pull is a regularizer against catastrophic forgetting — the student is
allowed to learn, but punished for drifting too far from its old self.

Total loss = task_loss  +  alpha * distillation_loss

where distillation_loss is the KL divergence between the (temperature-softened)
teacher and student next-token distributions.

This file gives you a working custom Trainer with the loss wired up. The TODOs
mark the research decisions that are YOURS to make (which data to distill on,
whether the teacher sees the same batch, etc.).
"""
import copy
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.training.lora_finetune import _to_hf_dataset
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SelfDistillTrainer(Trainer):
    """A Trainer that adds a KL-to-teacher term to the usual language-model loss."""

    def __init__(self, teacher_model, temperature: float, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():  # teacher never updates
            p.requires_grad_(False)
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Standard task loss (next-token prediction on the new data).
        outputs = model(**inputs)
        task_loss = outputs.loss
        student_logits = outputs.logits

        # 2. Teacher logits on the SAME inputs (no gradient through the teacher).
        with torch.no_grad():
            teacher_logits = self.teacher(**inputs).logits

        # 3. KL divergence between softened distributions.
        T = self.temperature
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)  # standard temperature^2 scaling

        loss = task_loss + self.alpha * distill_loss
        return (loss, outputs) if return_outputs else loss


def train_self_distillation(model, tokenizer, examples: List[Dict], cfg: dict, run_name: str):
    """Fine-tune the student while distilling from a frozen copy of itself."""
    t = cfg["training"]
    sd = t["self_distillation"]

    # The teacher is a frozen snapshot of the model BEFORE we touch it.
    # NOTE: deepcopy can be heavy; for big models prefer loading a second copy
    # from disk instead. (TODO: decide based on your GPU.)
    teacher = copy.deepcopy(model)

    train_ds = _to_hf_dataset(examples, tokenizer, cfg["model"]["max_seq_length"])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=f"{t['output_dir']}/{run_name}",
        num_train_epochs=t["epochs"],
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        report_to="none",
    )
    trainer = SelfDistillTrainer(
        teacher_model=teacher,
        temperature=sd["temperature"],
        alpha=sd["alpha"],
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    logger.info("Starting self-distillation -> %s", args.output_dir)
    trainer.train()
    trainer.save_model(args.output_dir)
    return args.output_dir

    # TODO (research): a more faithful design distills on a RETAIN set (facts you
    # want preserved) that is separate from the TASK set (the new/counterfactual
    # facts). Splitting the two lets you control the forgetting/learning trade-off
    # directly. Add a second dataloader for the retain set here.
