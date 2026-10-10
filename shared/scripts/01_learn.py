"""TOFU Step 1 — LEARN phase.

Fine-tune the base model so it knows the TOFU authors. Run it TWICE:

    # the model that knows everything (this is what we later unlearn)
    python shared/scripts/01_learn.py --data full

    # the gold reference model trained ONLY on the retain set (for Forget Quality)
    python shared/scripts/01_learn.py --data retain90

Add --lora to use LoRA instead of full fine-tuning (your Full-FT-vs-LoRA axis).

--lang injects the facts in a TRANSLATED language instead (studies/learn_french):

    # the French model to be unlearned: retain99_fr + forget01_fr = 4000 QA
    python shared/scripts/01_learn.py --data full     --lang fr

    # the French floor: retain99_fr only (3960) -- never saw the forget authors
    python shared/scripts/01_learn.py --data retain99 --lang fr

Non-English run names get a `_<lang>` suffix, so French and English checkpoints
coexist and every path the older English study hardcodes still resolves.

Several --lang values train ONE model on all of them at once (the Trainer shuffles the
union, so the languages are interleaved, never sequential). The suffix joins them:

    # French + Japanese: both copies of every fact in one set of weights (8000 QA)
    python shared/scripts/01_learn.py --data full     --lang fr ja --stop-by <unix s>
    # its floor: retain99 in both languages (7920), never saw the forget authors
    python shared/scripts/01_learn.py --data retain99 --lang fr ja --stop-by <unix s>

--stop-by makes the run resumable across SLURM jobs (see finetune_tofu). A run that
stops early exits with code 75 (EX_TEMPFAIL) so the sbatch knows to resubmit.
"""
import argparse
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

from src.data.load_multilingual_tofu import load_learn_set
from src.models.load_model import load_model_and_tokenizer
from src.training.learn import finetune_tofu
from src.utils.seed import set_seed
from src.utils.logging_utils import load_config, get_logger
from src.utils.paths import model_slug

logger = get_logger("tofu_finetune")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="full", help="full | retain90 | retain95 | retain99")
    ap.add_argument("--lang", nargs="+", default=["en"],
                    help="LANGUAGE(S) to inject the facts in; several = one joint "
                         "model on their union. en = locuslab/TOFU "
                         "(unchanged; every pre-existing English checkpoint stays "
                         "byte-reproducible). A translated language reads the "
                         "STANDALONE multilingual configs, where --data full is "
                         "rebuilt as retain99 + forget01 (3960 + 40 = 4000) because "
                         "the release ships no full_<lang>. Only full / forget01 / "
                         "retain99 exist per-language.")
    ap.add_argument("--lora", action="store_true", help="use LoRA instead of full FT")
    ap.add_argument("--stop-by", type=float, default=None,
                    help="unix time after which to save a resume checkpoint and exit "
                         "75; the next run with the same arguments continues from it")
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE TEST: first N rows per language. Adds `_limN` to the run "
                         "name so a smoke run can never overwrite a real checkpoint.")
    # The `deepspeed`/torchrun launcher passes --local_rank; absorb it so argparse
    # doesn't error. HF Trainer reads the actual rank from env vars.
    ap.add_argument("--local_rank", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    if len(set(args.lang)) != len(args.lang):
        ap.error(f"--lang repeats a language: {args.lang}")
    records = []
    for lg in args.lang:
        part = load_learn_set(args.data, lg, cfg["tofu"]["ml_cache_dir"],
                              cfg["tofu"]["cache_dir"], limit=args.limit)
        logger.info("  %s: %d QA", lg, len(part))
        records += part
    logger.info("LEARN set total: %d QA over %s", len(records), "+".join(args.lang))
    # device_map=None (load on CPU): the HF Trainer / DeepSpeed places the model on
    # each rank's own GPU. "auto" would put it on cuda:0 in BOTH ranks -> collision.
    model, tokenizer = load_model_and_tokenizer(cfg["model"], device_map=None)

    tag = "lora" if args.lora else "full"
    run_name = f"tofu_learn_{args.data}_{tag}_{model_slug(cfg)}"   # model in the name
    if args.lang != ["en"]:
        # Suffix ONLY for non-English, so existing English checkpoint paths (and the
        # downstream scripts that hardcode them) keep resolving unchanged. A single
        # language keeps its old suffix (`_fr`); a joint run joins them (`_fr-ja`).
        run_name += "_" + "-".join(args.lang)
    if args.limit:
        run_name += f"_lim{args.limit}"
    out = finetune_tofu(model, tokenizer, records, cfg, run_name, use_lora=args.lora,
                        stop_by=args.stop_by)
    if out is None:
        # A marker file as well as the exit code: the sbatch reads the marker, so the
        # resubmit decision never depends on the deepspeed launcher propagating 75.
        marker = Path(cfg["training"]["output_dir"]) / run_name / "RESUME_PENDING"
        marker.write_text("stopped at stop_by; resubmit with the same arguments\n")
        logger.info("Learn phase PARTIAL -> %s -- resubmit with the same arguments", marker)
        sys.exit(75)
    logger.info("Learn phase complete -> %s", out)


if __name__ == "__main__":
    main()
