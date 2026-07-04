"""GRPO UNLEARNING — the fourth training strategy. Highest-risk (RL can be
unstable on a 7B model); keep it out of the critical path (CLAUDE.md §3).

Reinforcement-learning framing of unlearning. For each FORGET question we sample
several answers from the model and reward those that do NOT reproduce the
memorized answer — i.e. reward forgetting. GRPO's built-in KL penalty to the
reference policy (beta) keeps the model close to the learned model on everything
else, which is what protects the retain knowledge (no explicit retain batch
needed — the KL leash is the preservation term).

The reward function is the experiment. Ours:

    reward = 1 - ROUGE-L_recall(completion, gold_forget_answer)

High when the completion has little overlap with the answer we want gone (a
refusal or an unrelated response scores ~1); low when the model leaks the
memorized answer (~0). Swap in a refusal-classifier or fluency term to change the
hypothesis.

Emits a standard merged checkpoint + tokenizer via save_unlearned (CLAUDE.md §7),
so relearn / evaluate / spectral treat it like every other strategy.

NOTE: TRL's GRPO argument names drift between versions; this targets trl>=0.9.
Check `pip show trl` and adjust GRPOConfig fields if it errors.
"""
from typing import Dict, List

from datasets import Dataset

from src.evaluation.compute_logprobs import format_qa
from src.evaluation.tofu_metrics import rouge_score_recall
from src.training.unlearn import save_unlearned
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def forget_reward(completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """Reward NOT reproducing the memorized answer. `answer` is forwarded by TRL
    from the dataset column of the same name (one gold answer per prompt).

    completions may be plain strings or chat-style [{'content': ...}] dicts
    depending on TRL version / dataset format; handle both.
    """
    rewards = []
    for comp, gold in zip(completions, answer):
        text = comp if isinstance(comp, str) else comp[-1]["content"]
        leak = rouge_score_recall(text, gold)   # 1.0 = fully leaked the answer
        rewards.append(1.0 - leak)
    return rewards


def unlearn_grpo(model, tokenizer, forget: List[Dict], cfg: Dict, run_name: str,
                 use_lora: bool = False):
    """GRPO unlearning on the forget set. Returns the checkpoint dir."""
    t, u = cfg["training"], cfg["tofu"]
    g = t["grpo"]

    peft_config = None
    if use_lora:
        from peft import LoraConfig
        lc = t["lora"]
        peft_config = LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], task_type="CAUSAL_LM")

    # GRPO needs prompts (the model generates + is scored); we also carry the gold
    # answer so the reward can measure leakage. Prompt uses the SAME template as
    # training/eval so the policy sees the distribution it was learned on.
    prompt_ds = Dataset.from_dict({
        "prompt": [format_qa(r["question"]) for r in forget],
        "answer": [r["answer"] for r in forget],
    })

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise ImportError("GRPO requires `trl` (pip install trl).") from e

    grpo_args = GRPOConfig(
        output_dir=f"{t['output_dir']}/{run_name}",
        learning_rate=u.get("unlearn_lr_lora", u["unlearn_lr"]) if use_lora else u["unlearn_lr"],
        per_device_train_batch_size=g["num_generations"],  # >= num_generations
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=u["unlearn_epochs"],
        num_generations=g["num_generations"],
        max_completion_length=g["max_completion_length"],
        beta=g["kl_beta"],                 # KL leash to the reference => keeps retain
        temperature=g.get("temperature", 1.0),
        logging_steps=t["logging_steps"],
        save_strategy="no",
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        # LoRA fits on one GPU; full-parameter GRPO needs DeepSpeed like the rest.
        deepspeed=None if use_lora else "config/ds_config.json",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=forget_reward,
        args=grpo_args,
        train_dataset=prompt_ds,
        peft_config=peft_config,
    )
    logger.info("UNLEARN GRPO (lora=%s, n_gen=%d, beta=%.3f) -> %s",
                use_lora, g["num_generations"], g["kl_beta"], grpo_args.output_dir)
    trainer.train()
    save_unlearned(trainer, grpo_args.output_dir, tokenizer, use_lora=use_lora)
    logger.info("UNLEARN GRPO done -> %s", grpo_args.output_dir)
    return grpo_args.output_dir
