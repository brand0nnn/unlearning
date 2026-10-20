"""Plan 2 (side check): does ja/ru unlearning damage a GENERAL name skill, or only the
forget authors? Inference only.

    python studies/learn_french/side_checks/scripts/name_skill.py --dry-run
    python studies/learn_french/side_checks/scripts/name_skill.py \
        --checkpoints <ckpt> ... --out-dir studies/learn_french/side_checks/results/plan2

Every name in probes/names.json x 7 tasks (copy in Latin / katakana / Cyrillic,
Latin -> katakana / Cyrillic, and back) x 5 instruction languages, in the trained
"[INST] ... [/INST]" wrapper. Per item:
  - greedy generation (raw text stored, so the cleaning rule can be revised offline),
    exact / exact_loose / character error rate against the accepted renderings;
  - teacher-forced log p of the REFERENCE rendering (and each accepted alternative),
    scored exactly as TofuQADataset tokenises a QA pair, EOS excluded. This is the
    format-proof measure: it does not care whether the model wraps its answer in a
    sentence, which a change of instruction language can easily cause.

One JSON per checkpoint, written atomically; a checkpoint whose JSON exists is skipped.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies" / "learn_french" / "scripts"))
sys.path.insert(0, str(HERE))
from name_tasks import (INSTR_LANGS, TASKS, clean_prediction, instruction,  # noqa: E402
                        name_group, references, score_prediction)

NAMES = HERE.parent / "probes" / "names.json"


def build_items(names, instr_langs):
    items = []
    for n in names:
        for task in TASKS:
            for il in instr_langs:
                refs = references(n, task)
                items.append({"name_id": n["id"], "set": n["set"], "group": name_group(n),
                              "who": n.get("who"), "task": task, "instr_lang": il,
                              "instruction": instruction(il, task, n), "refs": refs})
    return items


def generate(model, tok, prompts, batch_size, max_new):
    import torch
    outs = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
        for row in gen[:, enc.input_ids.shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "plan2"))
    ap.add_argument("--names", default=str(NAMES))
    ap.add_argument("--instr-langs", nargs="+", default=INSTR_LANGS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--tokenizer", default=None,
                    help="default: the config's base model (never the checkpoint's)")
    ap.add_argument("--limit-names", type=int, default=None, help="smoke tests only")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print prompts, tokenise the references; load no model")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    from transformers import AutoTokenizer
    from per_token_map import encode, score       # the trained tokenisation, reused
    from src.evaluation.compute_logprobs import format_qa

    tok = AutoTokenizer.from_pretrained(a.tokenizer or cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    names = json.load(open(a.names, encoding="utf-8"))["names"][:a.limit_names]
    items = build_items(names, a.instr_langs)
    prompts = [format_qa(it["instruction"]) for it in items]
    # Teacher-forced texts: every accepted rendering of every item.
    tf = {}
    for k, it in enumerate(items):
        for j, ref in enumerate(it["refs"]):
            tf[f"{k}:{j}"] = {"question": it["instruction"], "answer": ref}
    tf = encode(tf, tok, cfg["model"]["max_seq_length"])
    print(f"{len(names)} names x {len(TASKS)} tasks x {len(a.instr_langs)} instruction "
          f"languages = {len(items)} prompts, {len(tf)} reference scorings")
    L = len(a.instr_langs)          # items run name -> task -> instruction language
    show = [t * L for t in range(len(TASKS))] + list(range(1, L))
    for k in show:
        print(f"  {items[k]['task']:<9}{items[k]['instr_lang']}  {prompts[k]}"
              f"   => {items[k]['refs']}")
    if a.dry_run:
        print("\n--dry-run: no model loaded.")
        return
    if not a.checkpoints:
        sys.exit("--checkpoints required unless --dry-run")

    import torch
    from transformers import AutoModelForCausalLM
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ckpt in a.checkpoints:
        name = Path(ckpt).name
        out = out_dir / f"{name}.json"
        if out.exists():
            print(f"skip {name}: {out} exists")
            continue
        print(f"\n=== {ckpt}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else None).eval()
        model.config.pad_token_id = tok.pad_token_id
        gens = generate(model, tok, prompts, a.batch_size, a.max_new)
        lp = {}
        for key, t in tf.items():
            s = score(model, t)
            keep = [x for x, eos in zip(s["lp"], t["is_eos"]) if not eos]
            lp[key] = {"lp_sum": sum(keep), "n_tok": len(keep)}
        recs = []
        for k, (it, g) in enumerate(zip(items, gens)):
            pred = clean_prediction(g)
            ref_lp = [lp[f"{k}:{j}"] for j in range(len(it["refs"]))]
            recs.append({**it, "gen": g, "pred": pred, **score_prediction(pred, it["refs"]),
                         "lp_sum": ref_lp[0]["lp_sum"], "n_tok": ref_lp[0]["n_tok"],
                         "lp_mean": ref_lp[0]["lp_sum"] / max(1, ref_lp[0]["n_tok"]),
                         "lp_sum_best": max(r["lp_sum"] for r in ref_lp)})
        tmp = out.with_suffix(".json.tmp")
        json.dump({"checkpoint": ckpt, "name": name, "n_items": len(recs),
                   "template": "[INST] {instruction} [/INST]", "records": recs},
                  open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, out)
        ex = sum(r["exact_loose"] for r in recs) / len(recs)
        print(f">>> {name}: exact_loose {ex:.3f} over {len(recs)} items -> {out}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
