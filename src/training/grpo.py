"""GRPO (Group Relative Policy Optimization) — skeleton you will extend.

GRPO is a reinforcement-learning method. Instead of a fixed "right answer" to
imitate, you define a REWARD function that scores the model's own generated
answers, and the model is nudged to produce higher-reward answers. "Group
relative" means: for each prompt you sample several answers, and judge each one
relative to the group's average — no separate value network needed.

Why it's relevant to unlearning: the reward lets you express goals that a plain
cross-entropy loss can't, e.g. "say the counterfactual answer AND stay fluent"
or "refuse to reveal the forgotten fact". The reward function is where your
research hypothesis lives.

This uses TRL's GRPOTrainer. The exact argument names move between TRL versions,
so check them against the version you installed (`pip show trl`). The structure
below is the stable part: dataset of prompts + a reward function.
"""
from typing import Dict, List

from datasets import Dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def make_reward_fn(examples: List[Dict]):
    """Build a reward function over the model's generated completions.

    A reward function receives the prompts and the model's completions and
    returns one float reward per completion (higher = better).

    The toy reward here gives +1 when the target answer string appears in the
    completion, else 0. REPLACE this with the behaviour your experiment targets.
    """
    # Map each prompt to its target answer for quick lookup.
    target_by_prompt = {ex["prompt"]: ex["answer"] for ex in examples}

    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            target = target_by_prompt.get(prompt, "")
            rewards.append(1.0 if target and target.lower() in completion.lower() else 0.0)
        return rewards

    return reward_fn


def train_grpo(model, tokenizer, examples: List[Dict], cfg: dict, run_name: str):
    """Optimize the model against a reward with GRPO."""
    t = cfg["training"]
    g = t["grpo"]

    # GRPO only needs prompts; the model generates the rest and is scored.
    prompt_ds = Dataset.from_dict({"prompt": [ex["prompt"] for ex in examples]})
    reward_fn = make_reward_fn(examples)

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise ImportError(
            "GRPO requires `trl`. Install it with `pip install trl`."
        ) from e

    grpo_args = GRPOConfig(
        output_dir=f"{t['output_dir']}/{run_name}",
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["epochs"],
        num_generations=g["num_generations"],
        max_completion_length=g["max_completion_length"],
        beta=g["kl_beta"],                 # KL penalty to the reference policy
        logging_steps=t["logging_steps"],
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=prompt_ds,
    )
    logger.info("Starting GRPO -> %s", grpo_args.output_dir)
    trainer.train()
    trainer.save_model(grpo_args.output_dir)
    return grpo_args.output_dir

    # TODO (research): your reward function is the experiment. Ideas:
    #   - reward the counterfactual answer while penalizing the original answer,
    #   - add a fluency / KL term so the model doesn't collapse,
    #   - reward refusal on "forgotten" facts but correctness on retained ones.
