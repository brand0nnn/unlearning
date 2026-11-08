"""Stage 10 Phase 1 -- the GPU half: span log p, sampled generations, Japanese retain
generations, and the cross routes, on ONE checkpoint per call.

The re-scoring passes are NOT here: they are gate_frja.py re-run unchanged (two more
groups), so they are the same measurement as Stage 8's pass 0. This script adds what
Stage 8 never measured (preregistration_stage10.json "measures"):

  span         per-token log p of each forget gold answer (fr, ja), tokenized exactly as
               probability_score does (prompt, then " " + answer), with char offsets and
               the mask of tokens overlapping the fact key (fact_keys.locate). Stored
               per token, so every aggregate is computed offline.
  samples      N samples per fact and question language, T=1.0, top_p 1.0, top_k off.
  retain_gen   greedy answers to the 400 retain_perturbed rows in the given languages.
  cross        question in one language, answers in the other (fr>ja, ja>fr): P(gold),
               truth-ratio components, span log p.

    python studies/learn_french/scripts/storage_followup.py --checkpoints CKPT \
        --parts span samples cross retain_gen --retain-langs ja

Writes results/storage_followup/<checkpoint>.json, rewritten whole (tmp + os.replace)
after every part, so a wall-clock kill keeps the finished parts (CLAUDE.md sec 7).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fact_keys as fk
from src.data import load_multilingual_tofu as ml
from src.utils.logging_utils import load_config, get_logger, ensure_dir
from src.utils.paths import results_root

logger = get_logger("storage_followup")
LANGS = ("fr", "ja")
ROUTES = (("fr", "ja"), ("ja", "fr"))


def token_logps(model, tok, question, answer, groups):
    """Per-token log p of `answer` given `question`, as probability_score scores it."""
    import torch
    import torch.nn.functional as F
    from src.evaluation.compute_logprobs import format_qa
    ans = " " + answer.strip()
    p_ids = tok(format_qa(question), return_tensors="pt").input_ids
    enc = tok(ans, add_special_tokens=False, return_offsets_mapping=True)
    a_ids = torch.tensor([enc.input_ids])
    ids = torch.cat([p_ids, a_ids], dim=1).to(model.device)
    with torch.no_grad():
        lp = F.log_softmax(model(ids).logits[0].float(), dim=-1)
    n, L = a_ids.shape[1], ids.shape[1]
    lps = [float(lp[L - n + i - 1, ids[0, L - n + i]]) for i in range(n)]
    spans = fk.locate(ans, groups) if groups else []
    offs = [list(o) for o in enc.offset_mapping]
    mask = [any(s < e2 and s2 < e for s2, e2 in spans) for s, e in offs]
    return {"token_ids": enc.input_ids, "offsets": offs, "logp": lps, "span_mask": mask,
            "spans": spans}


def sample(model, tok, question, n, max_new, seed):
    import torch
    from src.evaluation.compute_logprobs import format_qa
    enc = tok(format_qa(question), return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             num_return_sequences=n, max_new_tokens=max_new,
                             pad_token_id=tok.pad_token_id)
    k = enc["input_ids"].shape[1]
    return [tok.decode(o[k:], skip_special_tokens=True).strip() for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--parts", nargs="+", default=["span", "samples", "cross"],
                    choices=["span", "samples", "cross", "retain_gen"])
    ap.add_argument("--retain-langs", nargs="*", default=["ja"])
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--group", default="storage_followup")
    args = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.evaluation.tofu_evaluate import _generate
    from src.evaluation.tofu_metrics import probability_score, truth_ratio_components

    cfg = load_config()
    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    max_new = cfg["evaluation"]["max_new_tokens"]
    probes = {lg: ml.load_probe_set(lg, M, C) for lg in LANGS}
    keys = fk.load_keys()
    key = lambda i, lg: None if "excluded" in keys[i] else keys[i][lg]
    out_dir = Path(ensure_dir(str(results_root() / args.group)))

    for ckpt in args.checkpoints:
        name = Path(ckpt).name
        logger.info("=== %s  parts %s ===", name, args.parts)
        # Tokenizer from the BASE model name, never the checkpoint (CLAUDE.md sec 7).
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="auto").eval()
        model.config.pad_token_id = tok.pad_token_id
        path = out_dir / f"{name}.json"
        rec = json.load(open(path)) if path.exists() else {}
        rec.update({"checkpoint": ckpt, "name": name, "key_file": str(fk.KEY_FILE)})

        def save():
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rec, ensure_ascii=False))
            os.replace(tmp, path)

        if "span" in args.parts:
            rec["span"] = {lg: [token_logps(model, tok, f["question"], f["answer"], key(i, lg))
                                for i, f in enumerate(tqdm(probes[lg], desc=f"span {lg}"))]
                           for lg in LANGS}
            save()
        if "cross" in args.parts:
            rec["cross"] = {}
            for qa, an in ROUTES:
                rows = []
                for i, (fq, fa) in enumerate(zip(probes[qa], probes[an])):
                    rows.append({
                        "prob": probability_score(model, tok, fq["question"], fa["answer"]),
                        **truth_ratio_components(model, tok, fq["question"],
                                                 fa["paraphrased_answer"],
                                                 fa["perturbed_answers"]),
                        "span": token_logps(model, tok, fq["question"], fa["answer"],
                                            key(i, an))})
                rec["cross"][f"{qa}>{an}"] = rows
            save()
        if "samples" in args.parts:
            rec["samples"] = {"n": args.n_samples, "temperature": 1.0, "top_p": 1.0,
                              "top_k": 0, "max_new_tokens": max_new}
            for j, lg in enumerate(LANGS):
                rec["samples"][lg] = [sample(model, tok, f["question"], args.n_samples,
                                             max_new, 1000 * j + i)
                                      for i, f in enumerate(tqdm(probes[lg], desc=f"samples {lg}"))]
                save()
        if "retain_gen" in args.parts:
            rec.setdefault("retain_gen", {})
            for lg in args.retain_langs:
                rows = ml.load_perturbed("retain_perturbed", lg, M, C)
                rec["retain_gen"][lg] = [{"question": r["question"], "answer": r["answer"],
                                          "generation": _generate(model, tok, r["question"],
                                                                  max_new)}
                                         for r in tqdm(rows, desc=f"retain gen {lg}")]
                save()
        logger.info(">>> %s -> %s", name, path)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
