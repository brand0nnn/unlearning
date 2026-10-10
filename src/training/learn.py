"""LEARN phase: fine-tune a base model on TOFU so it memorizes the authors.

Two faithful-reproduction details from the paper:
  1. Loss is computed over ANSWER tokens only (the question is masked with -100),
     so the model is graded on producing the answer, not echoing the question.
  2. The same fine-tuning routine produces BOTH:
       - the model that knows all authors (train on `full`)  -> to be unlearned
       - the gold reference model (train on `retain90`)       -> for Forget Quality

Run it twice with different --data to get both. Supports full fine-tuning or
LoRA via the `use_lora` flag, which is exactly the Full-FT-vs-LoRA axis of your
project.
"""
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from src.evaluation.compute_logprobs import format_qa
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TofuQADataset(Dataset):
    """Tokenizes (question, answer) and masks question tokens in the labels."""

    def __init__(self, records: List[Dict], tokenizer, max_len: int):
        self.records = records
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        # Match the official TOFU pipeline (data_module.py):
        #  - tokenize the full question+answer string TOGETHER (not separately),
        #  - NO leading space before the answer (their answer_tag is ""),
        #  - append EOS as a trainable label so the model learns to STOP,
        #  - mask the question tokens (counted with special tokens) -> answer-only loss.
        new_question = format_qa(r["question"])            # "[INST] {q} [/INST]"
        full_text = new_question + r["answer"].strip()     # answer_tag = "" (no space)
        num_q = len(self.tok(new_question, add_special_tokens=True).input_ids)
        input_ids = self.tok(full_text, add_special_tokens=True,
                             max_length=self.max_len, truncation=True).input_ids
        # Teach the model to emit EOS right after the answer (if there's room).
        if 0 < len(input_ids) < self.max_len:
            input_ids = input_ids + [self.tok.eos_token_id]
        labels = list(input_ids)
        for j in range(min(num_q, len(labels))):
            labels[j] = -100                               # -100 = ignore in CE loss
        return {"input_ids": input_ids, "labels": labels}


def _collate(batch, pad_id):
    """Pad a batch to equal length; pad labels with -100."""
    # Filter out any empty examples to avoid 0-element tensor errors.
    batch = [b for b in batch if len(b["input_ids"]) > 0]
    if not batch:
        return None
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def _resume_dirs(output_dir: str) -> List[str]:
    """This run's own Trainer checkpoints (`<output_dir>/checkpoint-<step>`), oldest
    first. Resolved inside the run's directory by exact name pattern -- never a glob over
    experiments/, which once deleted a live job's checkpoint (CLAUDE.md sec 7)."""
    import os
    import re
    if not os.path.isdir(output_dir):
        return []
    steps = [int(m.group(1)) for d in os.listdir(output_dir)
             if (m := re.fullmatch(r"checkpoint-(\d+)", d))]
    return [os.path.join(output_dir, f"checkpoint-{s}") for s in sorted(steps)]


def finetune_tofu(model, tokenizer, records: List[Dict], cfg: Dict,
                  run_name: str, use_lora: bool = False,
                  save_each_epoch: bool = False, stop_by: float | None = None):
    """Fine-tune `model` on TOFU `records`. Returns the output checkpoint dir.

    stop_by (unix seconds) makes the run RESUMABLE across SLURM jobs, for LEARN sets too
    big for one wall (fr+ja = 8000 rows ~ 15 h against a 12 h wall). Past stop_by the
    run writes ONE full Trainer checkpoint (weights + ZeRO-3 optimizer, ~131 GB for an
    8B model) and returns None without a final save; the next job finds that checkpoint
    and continues from the same step, RNG and data order. When training completes, the
    final weights are saved as usual and the resume checkpoints are deleted. None (the
    default) is the old behaviour exactly: no checkpoints, no resume."""
    from transformers import Trainer, TrainingArguments

    t = cfg["training"]
    if use_lora:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(
            r=t["lora"]["r"], lora_alpha=t["lora"]["alpha"],
            lora_dropout=t["lora"]["dropout"],
            target_modules=t["lora"]["target_modules"], task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lc)
        model.print_trainable_parameters()

    ds = TofuQADataset(records, tokenizer, cfg["model"]["max_seq_length"])
    pad_id = tokenizer.pad_token_id

    args = TrainingArguments(
        output_dir=f"{t['output_dir']}/{run_name}",
        num_train_epochs=cfg["tofu"]["finetune_epochs"],
        learning_rate=cfg["tofu"]["finetune_lr"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_strategy="no",   # only trainer.save_model() at the end; per-epoch
                              # checkpoints (~27GB each) previously filled the quota.
                              # stop_by saves once, on demand, via _StopBy below.
        save_total_limit=1 if stop_by is not None else None,
        report_to="none",
        # DeepSpeed ZeRO-3 (config/ds_config.json): fp32 MASTER WEIGHTS, exactly like
        # the official TOFU repo. CONFIRMED necessary by elimination — plain bf16 with
        # everything else matching the paper still tops out at ROUGE ~0.47 because the
        # lr=1e-5 updates round away without an fp32 master. ZeRO-3 shards the fp32
        # master + optimizer + grads across the GPUs (needs the 2-GPU launch).
        deepspeed="config/ds_config.json",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # paged_adamw_32bit: EXACTLY the TOFU paper's optimizer (locuslab/tofu
        # finetune.py). Full 32-bit optimizer states, paged to CPU RAM so 7B fits.
        # The 8-bit variant quantized the Adam moments and under-fit (loss stalled
        # ~1.0, weak memorization); 32-bit converges to the paper's ROUGE ~0.98.
        optim="paged_adamw_32bit",
    )
    # Disable KV cache explicitly: it's incompatible with gradient checkpointing
    # and wastes memory during training.
    model.config.use_cache = False
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=lambda b: _collate(b, pad_id),
    )
    if save_each_epoch:
        # A relearning TRAJECTORY needs the model at each epoch, but save_strategy="epoch"
        # writes full Trainer checkpoints (~27GB: weights + optimizer + scheduler) and
        # filled the quota once already. save_model() writes weights only (~16GB) and is
        # all a later eval needs -- nothing resumes from these.
        #
        # Epochs 1..N-1 land in "<output_dir>__atep<k>"; epoch N is the normal final save
        # at output_dir, so the last epoch is never duplicated.
        from transformers import TrainerCallback

        class _SaveEachEpoch(TrainerCallback):
            def on_epoch_end(self, a, state, control, **kw):
                k = round(state.epoch)
                if k >= a.num_train_epochs:
                    return                      # epoch N = the final save below
                d = f"{a.output_dir}__atep{k}"
                trainer.save_model(d)
                tokenizer.save_pretrained(d)    # or the eval loads an all-zeros model
                logger.info("epoch %d snapshot -> %s", k, d)

        trainer.add_callback(_SaveEachEpoch())

    resume = None
    if stop_by is not None:
        import time
        from transformers import TrainerCallback

        class _StopBy(TrainerCallback):
            # Runs after DefaultFlowCallback, so the should_save it sets is not undone;
            # Trainer then writes checkpoint-<step> and breaks out of the loop.
            def on_step_end(self, a, state, control, **kw):
                if time.time() >= stop_by:
                    logger.info("stop_by reached at step %d/%d -- saving a resume "
                                "checkpoint and stopping", state.global_step,
                                state.max_steps)
                    control.should_save = True
                    control.should_training_stop = True
                return control

        trainer.add_callback(_StopBy())
        prior = _resume_dirs(args.output_dir)
        resume = prior[-1] if prior else None
        logger.info("resumable run: stop_by=%s, resuming from %s",
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(stop_by)),
                    resume or "nothing (fresh start)")

    logger.info("LEARN phase (%s, lora=%s, save_each_epoch=%s) -> %s",
                run_name, use_lora, save_each_epoch, args.output_dir)
    trainer.train(resume_from_checkpoint=resume)
    if stop_by is not None and trainer.state.global_step < trainer.state.max_steps:
        logger.info("PARTIAL: stopped at step %d/%d; resume checkpoint in %s",
                    trainer.state.global_step, trainer.state.max_steps, args.output_dir)
        return None
    trainer.save_model(args.output_dir)
    # Save the tokenizer too, so the checkpoint is self-contained and can be
    # loaded by later stages without falling back to the base model name.
    tokenizer.save_pretrained(args.output_dir)
    if stop_by is not None:
        # Only after the final weights are on disk. ~131 GB each; nothing reads them.
        import shutil
        for d in _resume_dirs(args.output_dir):
            shutil.rmtree(d)
            logger.info("removed resume checkpoint %s", d)
    return args.output_dir
