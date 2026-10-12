"""Q1 STORAGE, step 1 -- the GATE: did the joint LEARN store each forget fact in Japanese?

Storage hiding asks whether a fact's JAPANESE copy survives French unlearning. That is only
testable on facts the joint model fr-ja_ft actually stored in Japanese -- the French-only
model never did (Stage 4: 1.35x on the one Japanese slot). This scores the 40 forget facts,
question AND answer in the same language, on four checkpoints:

    fr-ja_ft      learned all 4000 facts in French AND Japanese         <- the model under test
    fr-ja_retain  same, minus the 40 forget facts in both languages     <- its never-taught floor
    fr_ft         French-only LEARN                                     <- reference: French = a
    fr_retain     its floor                                                known success, Japanese
                                                                           = a known failure

Pure inference, ~1 h, nothing written to experiments/. Per checkpoint x language x fact:

  prob         P(gold | question)^(1/|a|) -- the quantity the step-3 removal measure is
               built from, and the one the eligibility rule below is defined on
  truth ratio  TOFU Eq. 1 + geometric, WITH components (CLAUDE.md sec 7: store the parts)
  generation   greedy, + its detected language: does the model SAY it, in which language

No NLI: xlm-roberta-large-xnli is fine-tuned on XNLI's 15 languages and Japanese is not one
of them. The generations are stored, so it stays computable offline.

Same probe definition as Stage 1 (ml.load_probe_set): pass-2 question/gold -- the wording
LEARN trained on -- paired with pass-1 truth-ratio answers. Raw only: the surname
normalization is validated for French and refuses other languages.

    python studies/learn_french/scripts/gate_frja.py --dry-run          # data only, no torch
    sbatch studies/learn_french/slurm/18_gate_frja.sbatch               # the real run
    python studies/learn_french/scripts/gate_frja_report.py             # local, stdlib only
"""
import argparse
import json
import os
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

from src.data import load_multilingual_tofu as ml
from src.utils.logging_utils import load_config, get_logger, ensure_dir
from src.utils.paths import results_root

logger = get_logger("gate_frja")
_GEN_STORE = 2000          # as measure_fr.py: store whole generations, never a browsing cut


def score(model, tok, facts, max_new):
    from tqdm import tqdm
    from src.evaluation.nli import detect_language
    from src.evaluation.tofu_evaluate import _generate
    from src.evaluation.tofu_metrics import probability_score, truth_ratio_components
    out = []
    for f in tqdm(facts, desc="facts"):
        gen = _generate(model, tok, f["question"], max_new)
        out.append({
            "question": f["question"],
            "prob": probability_score(model, tok, f["question"], f["answer"]),
            **truth_ratio_components(model, tok, f["question"],
                                     f["paraphrased_answer"], f["perturbed_answers"]),
            "generation": gen[:_GEN_STORE],
            "gen_lang": detect_language(gen),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+")
    ap.add_argument("--langs", nargs="+", default=["fr", "ja"])
    ap.add_argument("--group", default="gate_frja")
    ap.add_argument("--dry-run", action="store_true",
                    help="load and print the probe sets; no torch, no model")
    args = ap.parse_args()

    cfg = load_config()
    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    probes = {lg: ml.load_probe_set(lg, M, C) for lg in args.langs}
    for lg, fs in probes.items():
        n_pert = sorted({len(f["perturbed_answers"]) for f in fs})
        print(f"[{lg}] {len(fs)} facts, perturbed answers per fact {n_pert}")
        for i in (0, 20):
            print(f"   fact {i:>2}  Q: {fs[i]['question'][:90]}\n            A: {fs[i]['answer'][:90]}")
    if args.dry_run:
        return 0
    if not args.checkpoints:
        sys.exit("--checkpoints is required unless --dry-run")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(ensure_dir(str(results_root() / args.group)))
    max_new = cfg["evaluation"]["max_new_tokens"]
    for ckpt in args.checkpoints:
        name = Path(ckpt).name
        logger.info("=== %s ===", name)
        # Tokenizer from the BASE model name, never the checkpoint (CLAUDE.md sec 7).
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="auto").eval()
        model.config.pad_token_id = tok.pad_token_id

        rec = {"checkpoint": ckpt, "name": name, "langs": {}}
        for lg in args.langs:
            pf = score(model, tok, probes[lg], max_new)
            langs = {}
            for f in pf:
                langs[f["gen_lang"]] = langs.get(f["gen_lang"], 0) + 1
            rec["langs"][lg] = {"per_fact": pf, "gen_language_counts": langs}
            mid = sorted(f["prob"] for f in pf)[len(pf) // 2]
            logger.info(">>> %-45s [%s] median P(gold)=%.4f  gen langs %s", name, lg, mid, langs)
        # One file per checkpoint, written whole and swapped in atomically, so a wall-clock
        # kill mid-run leaves the finished checkpoints intact (CLAUDE.md sec 7).
        tmp = out_dir / f"{name}.json.tmp"
        tmp.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        os.replace(tmp, out_dir / f"{name}.json")
        del model
        torch.cuda.empty_cache()
    logger.info("gate -> %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
