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

from src.training.learn import TofuQADataset, _collate
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


def save_unlearned(trainer, output_dir: str, tokenizer, use_lora: bool):
    """Persist an unlearned model as a STANDARD, self-contained HF checkpoint.

    Every unlearning strategy (Full-FT, LoRA, self-distillation, GRPO) MUST call
    this so the whole downstream harness (relearn, evaluate, spectral) can load
    it with a plain `from_pretrained` and compare it fairly to the others. Two
    invariants, both of which broke real runs once (see CLAUDE.md §7):

      - LoRA adapters are merged into the base weights first, so a full model is
        saved (not just adapters).
      - the tokenizer is saved alongside; without it eval loads an empty
        tokenizer and every metric silently comes out zero.
    """
    if use_lora:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


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
            oracle_model=None, use_lora: bool = False):
    """Run an unlearning algorithm via HF Trainer. Returns checkpoint dir.

    use_lora=False -> FULL fine-tuning (DeepSpeed ZeRO-3 + fp32 master, like LEARN).
    use_lora=True  -> LoRA: only small adapters are trained, so it fits without
        DeepSpeed and the fp32-master issue doesn't apply (adapters train fine in
        bf16). This is the LoRA-vs-Full-FT strategy axis of the project.
    """
    t, u = cfg["training"], cfg["tofu"]
    max_len = cfg["model"]["max_seq_length"]
    pad_id = tokenizer.pad_token_id

    if use_lora:
        from peft import LoraConfig, get_peft_model
        lc = t["lora"]
        model = get_peft_model(model, LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
        model.print_trainable_parameters()

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
        # LoRA needs a much higher LR than full FT (adapters start near zero).
        learning_rate=u.get("unlearn_lr_lora", u["unlearn_lr"]) if use_lora else u["unlearn_lr"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=0.01,
        logging_steps=t["logging_steps"],
        save_strategy="no",
        report_to="none",
        # Full FT needs DeepSpeed for fp32 master weights (like LEARN / the repo).
        # LoRA trains tiny adapters -> fits on one GPU without DeepSpeed.
        deepspeed=None if use_lora else "config/ds_config.json",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        remove_unused_columns=False,   # our collator returns a custom {'forget','retain'}
        label_names=[],
    )
    model.config.use_cache = False

    # Optional: track ROUGE/Probability/Truth-Ratio per step for TOFU Figure 8
    # (unlearning dynamics). Off by default; enabled via cfg["tofu"]["track_curve"].
    from src.evaluation.unlearn_curve import build_curve_callbacks
    callbacks = build_curve_callbacks(cfg, tokenizer, method, run_name)

    trainer = ForgetTrainer(
        model=model, args=args, train_dataset=ds,
        data_collator=make_collator(pad_id),
        method=method, oracle_model=oracle_model,
        callbacks=callbacks or None,
    )
    logger.info("UNLEARN (%s, lora=%s) -> %s", method, use_lora, args.output_dir)
    trainer.train()
    save_unlearned(trainer, args.output_dir, tokenizer, use_lora)
    logger.info("UNLEARN done (%s) -> %s", method, args.output_dir)
    return args.output_dir
