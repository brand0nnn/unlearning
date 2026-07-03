"""UNLEARN phase — HF Trainer + DeepSpeed, matching the official locuslab/tofu
`CustomTrainerForgetting`.

The four TOFU baselines differ only in the loss; each (except gradient_ascent)
needs a forget batch AND a retain batch in the same step, so we subclass Trainer
and override compute_loss:

  gradient_ascent     :  L = -NLL(forget)
  gradient_difference :  L = -NLL(forget) + NLL(retain)
  kl_minimization     :  L = -NLL(forget) + KL(oracle(retain) || current(retain))
  idk                 :  L =  NLL(forget-with-IDK-answers) + NLL(retain)

DeepSpeed (config/ds_config.json) supplies fp32 MASTER WEIGHTS — the same thing
LEARN needed to memorize. Without it, the "gentle" methods can't preserve the
retain facts (small bf16 updates round away), so utility comes out too low.
"""
import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from src.training.finetune_tofu import TofuQADataset, _collate
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# A few "I don't know"-style answers (subset of TOFU Appendix C) for the IDK method.
IDK_ANSWERS = [
    "I'm not sure.", "I don't have that information.", "That's beyond my knowledge.",
    "I'm afraid I can't answer that.", "I don't know.", "I'm not familiar with that.",
]


class ForgetRetainDataset(Dataset):
    """Each item is a (forget_tokenized, retain_tokenized) pair. Retain samples are
    drawn at random to pair with each forget sample (an 'epoch' = one pass over the
    forget set), matching the repo's forget/retain pairing."""

    def __init__(self, forget_records, retain_records, tokenizer, max_len, seed):
        self.forget = TofuQADataset(forget_records, tokenizer, max_len)
        self.retain = TofuQADataset(retain_records, tokenizer, max_len)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, i):
        f = self.forget[i]
        r = self.retain[self.rng.randrange(len(self.retain))]
        return f, r


def make_collator(pad_id):
    """Collate a list of (forget, retain) pairs into {'forget': batch, 'retain': batch}."""
    def collate(samples):
        return {
            "forget": _collate([s[0] for s in samples], pad_id),
            "retain": _collate([s[1] for s in samples], pad_id),
        }
    return collate


class ForgetTrainer(Trainer):
    """Trainer whose compute_loss combines a forget batch and a retain batch."""

    def __init__(self, *args, method, oracle_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.oracle_model = oracle_model  # frozen reference, KL only

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget, retain = inputs["forget"], inputs["retain"]
        f_out = model(**forget)

        if self.method == "gradient_ascent":
            loss = -f_out.loss
        elif self.method == "gradient_difference":
            loss = -f_out.loss + model(**retain).loss
        elif self.method == "idk":
            # forget answers were already replaced with IDK: NLL(idk) + NLL(retain)
            loss = f_out.loss + model(**retain).loss
        elif self.method == "kl_minimization":
            r_out = model(**retain)
            with torch.no_grad():
                oracle_logits = self.oracle_model(
                    input_ids=retain["input_ids"],
                    attention_mask=retain["attention_mask"]).logits
            # KL(oracle || current) over the retain tokens (batchmean), as in the repo.
            cur_lp = F.log_softmax(r_out.logits, dim=-1)
            ref_lp = F.log_softmax(oracle_logits, dim=-1)
            kl = F.kl_div(cur_lp, ref_lp, reduction="batchmean", log_target=True)
            loss = -f_out.loss + kl
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return (loss, f_out) if return_outputs else loss


def unlearn(model, tokenizer, forget: List[Dict], retain: List[Dict],
            cfg: Dict, method: str, run_name: str, checkpoint: str = None,
            oracle_model=None):
    """Run an unlearning algorithm via HF Trainer + DeepSpeed. Returns checkpoint dir."""
    t, u = cfg["training"], cfg["tofu"]
    max_len = cfg["model"]["max_seq_length"]
    pad_id = tokenizer.pad_token_id

    # For IDK, replace forget answers with "I don't know"-style responses.
    forget_records = forget
    if method == "idk":
        rng = random.Random(cfg["seed"])
        forget_records = [{"question": r["question"], "answer": rng.choice(IDK_ANSWERS)}
                          for r in forget]

    ds = ForgetRetainDataset(forget_records, retain, tokenizer, max_len, cfg["seed"])

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
        # Same fp32-master-weight setup as LEARN (config/ds_config.json), matching
        # the official forget.py (deepspeed + paged_adamw_32bit + bf16).
        deepspeed="config/ds_config.json",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        remove_unused_columns=False,   # our collator returns a custom {'forget','retain'}
        label_names=[],
    )
    model.config.use_cache = False

    trainer = ForgetTrainer(
        model=model, args=args, train_dataset=ds,
        data_collator=make_collator(pad_id),
        method=method, oracle_model=oracle_model,
    )
    logger.info("UNLEARN (%s) via DeepSpeed -> %s", method, args.output_dir)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("UNLEARN done (%s) -> %s", method, args.output_dir)
    return args.output_dir
