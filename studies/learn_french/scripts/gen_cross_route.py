"""Ask in language L, let the model WRITE its answer in French: the generation cross route.

    python studies/learn_french/scripts/gen_cross_route.py --dry-run      # no torch, no GPU
    python studies/learn_french/scripts/gen_cross_route.py \
        --checkpoint <ckpt> --out-dir studies/learn_french/results/gen_cross_route

WHY THIS EXISTS. Stage 5 / Plan 2 (cross_route.py) answered Q1-access at the PROBABILITY
level: teacher-forced, handed the French answer start, how likely is the French target?
There, Japanese unlearning leaves the French route ~86% intact (removed 0.14). But Stage 2's
own generations at the same matched checkpoints, asked in French, say the fact (NLI >= 0.9)
on only 2-4 of 40 facts for EVERY arm -- French unlearning included -- against 37/40 for
the learned model. So on behaviour the French route looks shut by every arm. This asks the
same thing for the four foreign question languages, which is the part not yet measured:

    [INST] <question in L> Réponds à cette question uniquement en français. [/INST]
           -> greedy generation -> Xiang Eq. 4 NLI against the French gold answer

THE ROUTES.
  fr       the trained French question, NO instruction -- byte-identical to Stage 2's
           prompt, so it must reproduce Stage 2's generations (a pipeline check, and the
           anchor tying this run to the 25-cell TR-NLI gap table).
  fr+I     the same question + the instruction. The comparison column for the foreign
           routes: every +I route carries the identical French sentence, so a difference
           between +I routes is caused by the question language alone.
  en+I, id+I, ja+I, ru+I   the question in L + the instruction.

THE INSTRUCTION is Xiang et al. Appendix G's cross-lingual prompting, verbatim in form:
"x' = x + I_l", a short instruction IN THE TARGET LANGUAGE appended to the question. They
list en/zh/de/ru/tr; French is not among them, so the French sentence follows their German
one ("Bitte beantworte diese Frage ausschließlich auf Deutsch.") word for word. Caveat
carried by the design, not patched: the instruction is French text, so a +I question is not
purely in L. That is Xiang's design and it is the same sentence on every route.

WHICH TEXT. Questions and the French gold come from the standalone forget01_<L> configs
(pass 2): the wording LEARN used for French and every UNLEARN arm used for its own
language -- the same text cross_route.py and measure_fr.py read.

GENERATION and SCORING are Stage 2's, imported, not re-implemented: tofu_evaluate._generate
(format_qa frame, greedy, max_new_tokens from config, decoded past the prompt) and
src.evaluation.nli (Xiang App. E.1 Eq. 4 + output-language detection). One prompt at a
time, unbatched, so the fr route can reproduce Stage 2 exactly; batching with left padding
changes bf16 numerics and would break that check.

ONE FILE PER CHECKPOINT (results/gen_cross_route/<name>.json, atomic write), so the sbatch
runs one array task per checkpoint and a killed task loses only its own checkpoint.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LANGS = ["fr", "en", "id", "ja", "ru"]
INSTRUCTION = "Réponds à cette question uniquement en français."
ROUTES = ["fr", "fr+I", "en+I", "id+I", "ja+I", "ru+I"]
_GEN_STORE = 2000          # same cap as measure_fr.py: keep the whole generation


def load_text(cfg):
    import src.data.load_multilingual_tofu as ml
    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    return {lg: ml.load_qa("forget01", lg, M, C) for lg in LANGS}


def prompts(qa, routes):
    """(fact, route, question-with-instruction) for every fact x route."""
    out = []
    for i in range(len(qa["fr"])):
        for r in routes:
            lang, instr = r.split("+")[0], r.endswith("+I")
            q = qa[lang][i]["question"].strip()
            out.append({"fact": i, "route": r, "lang": lang,
                        "question": f"{q} {INSTRUCTION}" if instr else q})
    return out


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="one checkpoint (the sbatch runs one per task)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--routes", nargs="+", default=ROUTES, choices=ROUTES)
    ap.add_argument("--dry-run", action="store_true",
                    help="print every prompt, load no model, import no torch")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    qa = load_text(cfg)
    assert all(len(qa[lg]) == len(qa["fr"]) for lg in LANGS), \
        {lg: len(qa[lg]) for lg in LANGS}
    cells = prompts(qa, a.routes)
    gold = [r["answer"] for r in qa["fr"]]

    print("=" * 94 + f"\n{len(cells)} generations = {len(gold)} facts x "
          f"{len(a.routes)} routes\n" + "=" * 94)
    if a.dry_run:
        for c in cells:
            print(f"  f{c['fact']:<3} {c['route']:<5} {c['question']}")
            if c["route"] == a.routes[-1]:
                print(f"       gold: {gold[c['fact']][:110]}")
        print("\n--dry-run: no model loaded.")
        return
    if not (a.checkpoint and a.out_dir):
        sys.exit("--checkpoint and --out-dir required unless --dry-run")

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.evaluation.nli import load_nli, nli_scores, detect_language
    from src.evaluation.tofu_evaluate import _generate
    from src.evaluation.compute_logprobs import format_qa

    max_new = cfg["evaluation"]["max_new_tokens"]
    nli = load_nli()
    # Tokenizer from the BASE model name, never the checkpoint (CLAUDE.md sec 7).
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, torch_dtype=torch.bfloat16, device_map="auto").eval()
    model.config.pad_token_id = tok.pad_token_id

    name = Path(a.checkpoint.rstrip("/")).name
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.json"
    for c in tqdm(cells, desc=name):
        gen = _generate(model, tok, c["question"], max_new)
        c.update({"prompt": format_qa(c["question"]),
                  **nli_scores(nli, gen, gold[c["fact"]]),
                  "gen_lang": detect_language(gen),
                  "generation": gen[:_GEN_STORE]})

    said = {r: sum(1 for c in cells if c["route"] == r and c["nli_score"] >= 0.9)
            for r in a.routes}
    rec = {"checkpoint": a.checkpoint, "name": name, "instruction": INSTRUCTION,
           "routes": a.routes, "max_new_tokens": max_new, "gold": gold,
           "said_ge_0.9": said, "cells": cells}
    tmp = str(out) + ".tmp"
    json.dump(rec, open(tmp, "w"), ensure_ascii=False, indent=2)
    os.replace(tmp, out)
    print(f">>> {name}  facts said (NLI >= 0.9) per route: {said}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
