"""Make runs reproducible by fixing all sources of randomness."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + GPU).

    Call this once at the start of every script. Without it, two identical
    runs can give slightly different numbers, which makes a before/after
    comparison impossible to trust.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Trade a little speed for determinism. Worth it for research.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
