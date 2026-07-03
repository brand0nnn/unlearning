"""Load a base model + tokenizer from one place, driven by config.

Keeping this in a single function means every script loads models the same way
(same dtype, same quantization, same padding setup), so differences between
experiments come from the *method*, not from accidental loading differences.
"""
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model_and_tokenizer(model_cfg: dict, device_map="auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Return (model, tokenizer) configured according to the `model:` block.

    Args:
        model_cfg: the dict at cfg["model"].
        device_map: HF device placement. Use "auto" for single-process inference,
            but pass None for DDP/DeepSpeed TRAINING — "auto" loads the whole model
            onto cuda:0 in EVERY process, so all ranks collide on GPU 0.
    """
    name = model_cfg["name"]
    dtype = _DTYPES[model_cfg.get("dtype", "bfloat16")]
    logger.info("Loading model: %s (dtype=%s)", name, model_cfg.get("dtype"))

    quant_config = None
    if model_cfg.get("load_in_4bit", False):
        # 4-bit loading lets a larger model fit on a smaller GPU at a small
        # accuracy cost. Fine for evaluation and LoRA; avoid for full fine-tuning.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(name)
    # Causal LMs often lack a pad token; reuse EOS so batching works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map=device_map,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
