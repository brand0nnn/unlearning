"""UNLEARN phase (Full-FT / LoRA) — HF Trainer + DeepSpeed, using the
gradient_difference loss from the official locuslab/tofu `CustomTrainerForgetting`:

  gradient_difference :  L = -NLL(forget) + NLL(retain)
  npo                 :  L = -(2/beta) E log sigmoid(-beta log[pi_theta(y|x)/pi_ref(y|x)])
                             + NLL(retain)        (NPO+RT, Zhang et al. 2024, arXiv 2404.05868)

This is the loss for the Full-FT and LoRA points of the 4-strategy comparison. (The
other TOFU losses — gradient_ascent / kl_minimization / idk — were removed as unused;
self-distillation and GRPO carry their own losses in their own modules.)

DeepSpeed (config/ds_config.json) supplies fp32 MASTER WEIGHTS — the same thing
LEARN needed to memorize. Without it the retain term can't preserve the retain
facts (small bf16 updates round away), so utility comes out too low.
"""
import json
import os
import random
from typing import Dict, List

from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

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


def deepspeed_plan():
    """(config path, optimizer) for Full-FT, honouring $UNLEARN_DS_CONFIG.

    Qwen3-8B Full-FT under the default ZeRO-3 config does not fit an 80GB card: params
    15.3 + fp32 master 30.5 + flat grad buffer 15.3 + the backward's own gradients 15.3
    = 76.4GB before activations, and the first backward dies ~1GB short. Pointing
    UNLEARN_DS_CONFIG at config/ds_config_offload.json moves the optimizer state to host
    RAM and leaves ~30GB free.

    The optimizer has to change with it: paged_adamw_32bit is a CUDA optimizer and cannot
    hold its state on the CPU, so an offloading config gets adamw_torch. Both are 32-bit
    -- the repo's rule is against 8-bit AdamW, which under-memorizes -- but they are not
    numerically identical, so a study must use ONE of them throughout and say which.
    """
    path = os.environ.get("UNLEARN_DS_CONFIG", "config/ds_config.json")
    cfg = json.load(open(path))
    offloads = cfg.get("zero_optimization", {}).get("offload_optimizer", {}).get("device") == "cpu"
    optim = "adamw_torch" if offloads else "paged_adamw_32bit"
    logger.info("DeepSpeed config %s (optimizer offload=%s) -> optim=%s",
                path, offloads, optim)
    return path, optim


def make_collator(pad_id):
    """Collate a list of (forget, retain) pairs into {'forget': batch, 'retain': batch}."""
    def collate(samples):
        return {
            "forget": _collate([s[0] for s in samples], pad_id),
            "retain": _collate([s[1] for s in samples], pad_id),
        }
    collate.pad_id = pad_id
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

    def __init__(self, *args, forget_floor=None, loss_type="gradient_difference",
                 npo_beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_type not in ("gradient_difference", "npo"):
            raise ValueError(f"loss_type must be gradient_difference or npo, got {loss_type!r}")
        if loss_type == "npo" and forget_floor is not None:
            raise ValueError("forget_floor is a gradient_difference device; NPO is bounded "
                             "by construction -- pass forget_floor=None")
        self.forget_floor = forget_floor
        self.loss_type = loss_type
        self.npo_beta = float(npo_beta)
        self.ref_logp = {}        # NPO: tuple(input_ids) -> sequence log p under theta_0
        self._npo_checked = False
        self._gd = []   # per-micro-batch (forget_nll_unclamped, retain_nll, hit_floor, loss)
        if loss_type == "npo":
            self.add_callback(_NPOReference(self))

    @staticmethod
    def seq_logp(logits, labels):
        """Per-sequence SUM of log p(label) over labelled positions (HF shift: logits at
        t predict token t+1). pi(y|x) in NPO is this full-sequence probability, not the
        length-normalised one (Zhang et al. 2024, Eq. 3; SimNPO is the normalised variant)."""
        import torch
        lg = logits[:, :-1].float()
        lab = labels[:, 1:]
        mask = lab != -100
        lp = torch.log_softmax(lg, -1).gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (lp * mask).sum(-1)

    @staticmethod
    def _key(input_ids_row, attn_row):
        return tuple(input_ids_row[attn_row.bool()].tolist())

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.loss_type == "npo":
            return self._npo_loss(model, inputs, return_outputs)
        forget, retain = inputs["forget"], inputs["retain"]
        f_out = model(**forget)
        forget_raw = f_out.loss
        forget_nll = forget_raw
        if self.forget_floor is not None:
            forget_nll = forget_nll.clamp(max=self.forget_floor)
        retain_nll = model(**retain).loss
        loss = -forget_nll + retain_nll
        # The two terms separately, UNCLAMPED: the combined loss cannot tell "the forget
        # term hit its floor and stopped pushing" from "unlearning is not transferring".
        f = forget_raw.item()
        hit = self.forget_floor is not None and f >= self.forget_floor
        self._gd.append((f, retain_nll.item(), hit, loss.item()))
        return (loss, f_out) if return_outputs else loss

    def _npo_loss(self, model, inputs, return_outputs):
        """NPO+RT. The reference pi_ref is the model BEFORE unlearning (fr_ft), as in
        Zhang et al.; its sequence log p on each forget example is computed once at
        on_train_begin by _NPOReference, so no second 8B model is held in memory."""
        import torch
        import torch.nn.functional as F
        forget, retain = inputs["forget"], inputs["retain"]
        f_out = model(**forget)
        logp = self.seq_logp(f_out.logits, forget["labels"])
        ref = torch.tensor([self.ref_logp[self._key(r, m)] for r, m in
                            zip(forget["input_ids"], forget["attention_mask"])],
                           device=logp.device, dtype=logp.dtype)
        ratio = logp - ref
        if not self._npo_checked:
            # First micro-batch: theta == theta_0, so the log-ratio must be ~0. A large
            # value means the reference was scored under different numerics -- stop.
            self._npo_checked = True
            r0 = float(ratio.abs().max())
            logger.info("NPO check: first log-ratio %.4f (must be ~0)", r0)
            if r0 > 0.5:
                logger.warning("NPO reference mismatch? |log ratio| = %.3f at step 0 "
                               "(expected ~0) -- check before trusting this run", r0)
        b = self.npo_beta
        forget_term = (-2.0 / b) * F.logsigmoid(-b * ratio).mean()
        retain_nll = model(**retain).loss
        loss = forget_term + retain_nll
        self._gd.append((f_out.loss.item(), retain_nll.item(), False, loss.item()))
        return (loss, f_out) if return_outputs else loss

    def pop_gd_stats(self):
        """Means of the two loss terms over the micro-batches since the last call, then
        reset. Called once per optimizer step (by the probe callback's on_step_end), so
        each result describes exactly one step: 32 micro-batches, or 8 for the epoch's
        remainder step (forget01 = 40 = 32 + 8)."""
        acc, self._gd = self._gd, []
        if not acc:
            return None
        n = len(acc)
        f = sum(a[0] for a in acc) / n
        r = sum(a[1] for a in acc) / n
        return {"forget_nll": f, "retain_nll": r, "n_micro": n,
                "floor_frac": sum(a[2] for a in acc) / n,
                "loss": sum(a[3] for a in acc) / n}


class _NPOReference(TrainerCallback):
    """Scores every forget example under the STARTING weights (theta_0 = pi_ref) before
    the first optimizer step, in the same engine and precision training uses."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        import gc
        import torch
        from src.training.learn import _collate
        t = self.trainer
        ds = t.train_dataset.forget
        pad = t.data_collator.pad_id if hasattr(t.data_collator, "pad_id") else 0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for i in range(len(ds)):
                    b = _collate([ds[i]], pad)
                    b = {k: v.to(model.device) for k, v in b.items()}
                    out = model(**b)
                    lp = ForgetTrainer.seq_logp(out.logits, b["labels"])
                    t.ref_logp[ForgetTrainer._key(b["input_ids"][0], b["attention_mask"][0])] = float(lp[0])
        finally:
            if was_training:
                model.train()
            release = getattr(getattr(t, "model_wrapped", None), "empty_partition_cache", None)
            if callable(release):
                try:
                    release()
                except Exception as e:
                    logger.warning("empty_partition_cache failed: %s", e)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        v = list(t.ref_logp.values())
        logger.info("NPO reference: %d forget examples, sequence log p mean %.3f (min %.3f)",
                    len(v), sum(v) / len(v), min(v))


def unlearn(model, tokenizer, forget: List[Dict], retain: List[Dict],
            cfg: Dict, method: str, run_name: str, checkpoint: str = None,
            use_lora: bool = False, extra_callbacks=None, save_final: bool = True):
    """Run gradient_difference unlearning via HF Trainer. Returns checkpoint dir.

    save_final=False skips the end-of-training checkpoint (~16GB). The French-anchored
    study keeps only its TR-level checkpoints, which the probe callback saves itself.

    `method` selects the loss: "gradient_difference" (TOFU) or "npo" (NPO+RT, beta from
    cfg tofu.npo_beta, default 0.1 as in Zhang et al. 2024).

    use_lora=False -> FULL fine-tuning (DeepSpeed ZeRO-3 + fp32 master, like LEARN).
    use_lora=True  -> LoRA: only small adapters are trained, so it fits without
        DeepSpeed and the fp32-master issue doesn't apply (adapters train fine in
        bf16). This is the LoRA-vs-Full-FT strategy axis of the project.
    """
    if method not in ("gradient_difference", "npo"):
        raise ValueError(
            f"unlearn() supports gradient_difference or npo (got {method!r}); "
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
    # LoRA keeps exactly what it always used: no DeepSpeed, paged_adamw_32bit.
    ds_config, ds_optim = (None, "paged_adamw_32bit") if use_lora else deepspeed_plan()

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
        deepspeed=ds_config,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=ds_optim,
        remove_unused_columns=False,   # our collator returns a custom {'forget','retain'}
        label_names=[],
        # Trainer re-seeds everything (incl. the batch-order sampler) with args.seed when
        # built; without this only the retain pairing above follows cfg["seed"]. cfg seed
        # 42 = the HF default, so every seed-42 run is unchanged.
        seed=cfg["seed"],
    )
    model.config.use_cache = False

    # Optional: track ROUGE/Probability/Truth-Ratio per step for TOFU Figure 8
    # (unlearning dynamics). Off by default; enabled via cfg["tofu"]["track_curve"].
    from src.evaluation.unlearn_curve import build_curve_callbacks
    callbacks = build_curve_callbacks(cfg, tokenizer, method, run_name)
    # extra_callbacks carries the French-anchored study's per-step TR probe. It is
    # passed in rather than built here so this module stays experiment-agnostic.
    callbacks = callbacks + list(extra_callbacks or [])

    trainer = ForgetTrainer(
        model=model, args=args, train_dataset=ds,
        data_collator=make_collator(pad_id),
        callbacks=callbacks or None,
        forget_floor=u.get("forget_floor"),
        loss_type=method,
        npo_beta=u.get("npo_beta", 0.1),
    )
    # A callback that saves checkpoints needs the trainer itself: under ZeRO-3 the
    # parameters are sharded, so only the trainer's own save path reconstructs them.
    for cb in (extra_callbacks or []):
        if hasattr(cb, "attach"):
            cb.attach(trainer)
    logger.info("UNLEARN (%s, lora=%s) -> %s", method, use_lora, args.output_dir)
    trainer.train()
    if not save_final:
        logger.info("UNLEARN done (%s); final checkpoint NOT saved (save_final=False)", method)
        return None
    save_unlearned(trainer, args.output_dir, tokenizer, use_lora)
    logger.info("UNLEARN done (%s) -> %s", method, args.output_dir)
    return args.output_dir
