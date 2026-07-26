"""Location helpers so scripts work no matter how deep they sit in the tree.

After the per-experiment reorg, entry points live at varying depths
(`shared/scripts/x.py`, `studies/<name>/plots/x.py`, …), so the old
`Path(__file__).parents[N]` root trick is fragile. Use `repo_root()` instead —
it walks up until it finds the `src/` package.

`results_root()` is the shared pipeline's output base. It defaults to `results/`
(under the CWD, i.e. the repo root on the cluster), but each study's sbatch sets
`UNLEARN_RESULTS_DIR=studies/<name>/results` so every experiment's metric JSON
lands inside that experiment's own folder (self-contained, deletable as a unit).
"""
import os
import re
from pathlib import Path


def model_slug(cfg_or_name) -> str:
    """Short filesystem-safe tag for the base model, so checkpoints from different
    models never collide (e.g. 'Qwen/Qwen3-8B' -> 'qwen3-8b',
    'meta-llama/Llama-2-7b-chat-hf' -> 'llama-2-7b-chat-hf'). Appended to LEARN /
    UNLEARN run names so a model swap needs no manual renaming."""
    name = cfg_or_name["model"]["name"] if isinstance(cfg_or_name, dict) else cfg_or_name
    base = name.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9.]+", "-", base).strip("-")


def repo_root(start: str | Path | None = None) -> Path:
    """Walk up from `start` (default: this file) until a dir containing `src/`."""
    p = Path(start or __file__).resolve()
    if p.is_file():
        p = p.parent
    while p != p.parent:
        if (p / "src").is_dir():
            return p
        p = p.parent
    return Path.cwd()


def results_root() -> Path:
    """Base dir for pipeline output JSON. Override per-study via UNLEARN_RESULTS_DIR."""
    return Path(os.environ.get("UNLEARN_RESULTS_DIR", "results"))
