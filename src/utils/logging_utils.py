"""Small helpers used everywhere: loading the config and getting a logger."""
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Read config.yaml into a plain dictionary.

    Usage:
        cfg = load_config()
        model_name = cfg["model"]["name"]
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    """A logger that prints timestamped messages to the console.

    Prefer this over print() — it tells you *when* something happened and
    which module it came from, which matters when a training run takes hours.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid adding duplicate handlers on re-import
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str) -> Path:
    """Create a directory if it doesn't exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
