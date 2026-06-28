"""LoRA fine-tuning.

LoRA ("Low-Rank Adaptation") freezes the original weights and learns a small
pair of low-rank matrices added on top. Far fewer trainable parameters than full
fine-tuning, which is why it's fast and memory-light — and a key reason it's
interesting for unlearning: it edits the model in a much smaller subspace.
"""
from typing import Dict, List

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _to_hf_dataset(examples: List[Dict], tokenizer, max_len: int) -> Dataset:
    """Turn prompt+answer pairs into tokenized training examples.

    We train on the full "prompt + answer" string so the model learns to produce
    the answer after the prompt.
    """
    texts = [ex["prompt"] + " " + ex["answer"] for ex in examples]

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    ds = Dataset.from_dict({"text": texts})
    return ds.map(tok, batched=True, remove_columns=["text"])


def train_lora(model, tokenizer, examples: List[Dict], cfg: dict, run_name: str):
    """Attach LoRA adapters and fine-tune.

    Args:
        model, tokenizer: from load_model_and_tokenizer.
        examples: training data in the common schema.
        cfg: the full config dict.
        run_name: sub-folder name under experiments/ for outputs.
    """
    t = cfg["training"]
    lora_cfg = LoraConfig(
        r=t["lora"]["r"],
        lora_alpha=t["lora"]["alpha"],
        lora_dropout=t["lora"]["dropout"],
        target_modules=t["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # confirms only a small % is trainable

    train_ds = _to_hf_dataset(examples, tokenizer, cfg["model"]["max_seq_length"])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=f"{t['output_dir']}/{run_name}",
        num_train_epochs=t["epochs"],
        learning_rate=t["learning_rate"],
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
    logger.info("Starting LoRA training -> %s", args.output_dir)
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info("Saved LoRA adapters to %s", args.output_dir)
    return args.output_dir
