"""Step 1: download the benchmarks and write them into data/processed/.

Run:  python scripts/01_download_data.py
"""
import sys
from pathlib import Path

# Make `src` importable when running this file directly.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.load_lama import load_lama, save_processed
from src.data.load_nq import load_nq
from src.data.build_counterfactuals import build_counterfactuals
from src.utils.logging_utils import load_config, get_logger

logger = get_logger("download_data")


def main():
    cfg = load_config()
    raw = cfg["data"]["raw_dir"]
    processed = cfg["data"]["processed_dir"]
    limit = cfg["data"]["eval_subset_size"]

    # LAMA
    lama = load_lama(raw_dir=raw, subset="trex", limit=limit)
    save_processed(lama, f"{processed}/lama.json")

    # Natural Questions
    nq = load_nq(raw_dir=raw, split="validation", limit=limit)
    save_processed(nq, f"{processed}/nq.json")

    # Counterfactual edit set (built from LAMA facts).
    build_counterfactuals(
        facts=lama,
        out_path=f"{cfg['data']['counterfactual_dir']}/edits.json",
        seed=cfg["seed"],
        num_edits=100,
    )
    logger.info("Done. Data is in %s and %s", processed, cfg["data"]["counterfactual_dir"])


if __name__ == "__main__":
    main()
