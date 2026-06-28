"""Full fine-tuning: update ALL of the model's weights.

This is the baseline to compare LoRA and the advanced methods against. It is the
most expressive (it can change anything) but also the most prone to catastrophic
forgetting and the most memory-hungry. Structurally it's identical to the LoRA
trainer minus the adapter step — that parallel is intentional so the *only*
difference in your comparison is the method, not the plumbing.
"""
from typing import Dict, List

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.training.lora_finetune import _to_hf_dataset  # reuse the tokenizer step
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_full(model, tokenizer, examples: List[Dict], cfg: dict, run_name: str):
    """Fine-tune every parameter of the model."""
    t = cfg["training"]
    train_ds = _to_hf_dataset(examples, tokenizer, cfg["model"]["max_seq_length"])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=f"{t['output_dir']}/{run_name}",
        num_train_epochs=t["epochs"],
        # Full FT usually wants a SMALLER learning rate than LoRA, since every
        # weight moves. A common starting point is ~2e-5.
        learning_rate=2e-5,
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, data_collator=collator
    )
    logger.info("Starting FULL fine-tuning -> %s", args.output_dir)
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info("Saved full model to %s", args.output_dir)
    return args.output_dir
