"""UNLEARN phase (Full-FT / LoRA) — HF Trainer + DeepSpeed, using the
gradient_difference loss from the official locuslab/tofu `CustomTrainerForgetting`:

  gradient_difference :  L = -NLL(forget) + NLL(retain)

This is the loss for the Full-FT and LoRA points of the 4-strategy comparison. (The
other TOFU losses — gradient_ascent / kl_minimization / idk — were removed as unused;
self-distillation and GRPO carry their own losses in their own modules.)

DeepSpeed (config/ds_config.json) supplies fp32 MASTER WEIGHTS — the same thing
LEARN needed to memorize. Without it the retain term can't preserve the retain
facts (small bf16 updates round away), so utility comes out too low.
"""
import random
from typing import Dict, List

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from src.training.learn import TofuQADataset, _collate
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


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
    """gradient_difference: L = -min(NLL(forget), forget_floor) + NLL(retain),
    computed in one step from a paired forget/retain batch.

    forget_floor (tau) caps the forget ascent: once NLL(forget) reaches tau, that
    term becomes constant so its gradient vanishes and forget prob stops dropping
    at ~exp(-tau) instead of collapsing to 0 (which explodes the truth ratio ->
    Forget-Quality ~ -1e2). None = unbounded ascent (original behaviour). Lives on
    the shared loss, so Full-FT and LoRA both inherit it; it only bites when a
    method can actually drive prob -> 0 (Full-FT), and is a no-op for LoRA."""

    def __init__(self, *args, forget_floor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.forget_floor = forget_floor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget, retain = inputs["forget"], inputs["retain"]
        f_out = model(**forget)
        forget_nll = f_out.loss
        if self.forget_floor is not None:
            forget_nll = forget_nll.clamp(max=self.forget_floor)
        loss = -forget_nll + model(**retain).loss
        return (loss, f_out) if return_outputs else loss


def unlearn(model, tokenizer, forget: List[Dict], retain: List[Dict],
            cfg: Dict, method: str, run_name: str, checkpoint: str = None,
            use_lora: bool = False):
    """Run gradient_difference unlearning via HF Trainer. Returns checkpoint dir.

    `method` is kept only for the run-name / curve label — it must be
    "gradient_difference" (the other TOFU losses were removed).

    use_lora=False -> FULL fine-tuning (DeepSpeed ZeRO-3 + fp32 master, like LEARN).
    use_lora=True  -> LoRA: only small adapters are trained, so it fits without
        DeepSpeed and the fp32-master issue doesn't apply (adapters train fine in
        bf16). This is the LoRA-vs-Full-FT strategy axis of the project.
    """
    if method != "gradient_difference":
        raise ValueError(
            f"unlearn() supports only gradient_difference (got {method!r}); "
            "gradient_ascent / kl_minimization / idk were removed.")
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

    ds = ForgetRetainDataset(forget, retain, tokenizer, max_len, cfg["seed"])

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
        callbacks=callbacks or None,
        forget_floor=u.get("forget_floor"),
    )
    logger.info("UNLEARN (%s, lora=%s) -> %s", method, use_lora, args.output_dir)
    trainer.train()
    save_unlearned(trainer, args.output_dir, tokenizer, use_lora)
    logger.info("UNLEARN done (%s) -> %s", method, args.output_dir)
    return args.output_dir
