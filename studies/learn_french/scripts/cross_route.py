"""Ask in language L, score the FRENCH answer: the cross-route token probe.

    python studies/learn_french/scripts/cross_route.py --dry-run          # no torch, no GPU
    python studies/learn_french/scripts/cross_route.py \
        --checkpoints <base> <fr_ft> <fr_retain> <unl_fr> ... --out results/cross_route.json

WHY THIS EXISTS. 09 (top_tokens_ml.py) asked in L and scored the answer IN L. French-only
LEARN never put the facts into ja/ru, and put only proper nouns into en/id, so most of its
cells failed the transfer gate and the hiding question was untestable there. Skow et al.
(arXiv 2609.40286, Table 2A) explain why: learning transfers strongly when only the QUESTION
changes language (+0.47 over base) and barely when the answer must change too (+0.09). 09
measured the weak route. This measures the strong one:

    Question: <the question in L>            <- the route's input language
    Answer:   <French gold answer, cut>      <- always French, the language of LEARN
              ^ score P(French target)^(1/n), teacher-forced over every target token

READ IT IN THE SAME TWO STEPS AS 09.
  1. TRANSFER  fr_ft vs fr_retain on route L. Same French fine-tuning, one never shown the
               fact, so a gap IS the fact reaching the model through an L question.
  2. ACCESS AFTER UNLEARNING  unl_X vs fr_ft on route L, only where step 1 passed.
               unl_fr on route ja = "French unlearning: does a Japanese question still
               unlock the French answer?" -- the side-door version of the hiding question.
               unl_ja on route fr = "Japanese unlearning: did it close the French door?"

THE BASELINE ROUTE, AND WHAT A ROUTE CAN AND CANNOT SHOW. The French answer stump is the
memorised training sentence and usually names the author ("Le pere de Basil Mahfouz
Al-Kuwaiti etait ___"), so much of the access on EVERY route comes from the stump, not the
question. `blank` asks nothing ("Question: " + the same stump) and is the floor each route is
read against: route minus blank is what the question in L adds. Because the stump is
identical across routes, any difference between routes at one checkpoint is caused by the
question language alone -- which is exactly what "is unlearning keyed to the question
language?" needs, and all this probe can claim. It does NOT show the fact stored in L.

THE ROUTES. fr (the trained question, p0), fr_p1 (TOFU's own French paraphrase of it), en,
id, ja, ru. fr_p1 is the README's pending step (a): if the learned model ranks the fact
first under p0 and far down under p1, the TR-NLI gap is a question-phrasing effect that
exists before any unlearning. The paraphrases are pass-1 translations and some are
degraded (fact 4's calls Basil "le basilic", the herb) -- printed, not hidden.

WHICH TEXT. Questions and the French answer come from the STANDALONE forget01_<L> configs
(pass 2) -- the wording LEARN used for French and every UNLEARN arm used for its own
language. 09 read forget01_perturbed (pass 1) for both, which is not the wording any model
was trained on; English is locuslab TOFU either way. Only fr_p1 is pass 1, because no pass-2
paraphrase exists.

The slot file declares ONE French target per slot (probes/cross_route_probes.json). A target
that is absent from the French answer, or occurs more than once, skips the slot; a target
that appears in a route's QUESTION skips that one cell, because then it measures copying.
Every skip is named in the log and the JSON.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
ROUTES = ["blank", "fr", "fr_p1", "en", "id", "ja", "ru"]
SCAN = 20000


def load_text(cfg):
    import src.data.load_multilingual_tofu as ml
    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    qa = {lg: ml.load_qa("forget01", lg, M, C) for lg in ["fr", "en", "id", "ja", "ru"]}
    p1 = [r["paraphrased_question"]
          for r in ml.load_perturbed("forget01_perturbed", "fr", M, C)]
    return qa, p1


def leaks(target, question):
    """The target -- or any 4+ letter word of it -- appearing in the question."""
    q = question.casefold()
    words = [w.strip("'«»\"") for w in target.casefold().split()]
    return target.casefold() in q or any(len(w) >= 4 and w in q for w in words)


def build(spec, qa, p1, routes):
    cells, misses = [], []
    for s in spec:
        fi, tgt = s["fact"], s["target"]
        ans = qa["fr"][fi]["answer"]
        n = ans.count(tgt)
        if n != 1:
            misses.append(f"{s['id']}: target {tgt!r} occurs {n}x in the French answer "
                          f"-- slot skipped")
            continue
        cut = ans.index(tgt)
        target = tgt
        # Qwen3's BPE attaches a word's leading space to the word, so move it out of the
        # stump and into the target (same rule as 09).
        if cut and ans[cut - 1] == " ":
            cut -= 1
            target = " " + tgt
        for r in routes:
            q = ("" if r == "blank" else
                 p1[fi] if r == "fr_p1" else qa[r][fi]["question"])
            if not q and r != "blank":
                misses.append(f"{s['id']}/{r}: no question text")
                continue
            if leaks(tgt, q):
                misses.append(f"{s['id']}/{r}: target appears in the question -- copying, "
                              f"not recall; cell skipped  (Q: {q})")
                continue
            cells.append({"id": s["id"], "fact": fi, "type": s["type"], "route": r,
                          "target": target, "question": q, "stump": ans[:cut],
                          "prompt": f"Question: {q}\nAnswer: {ans[:cut]}",
                          "gold_tail": ans[cut:]})
    return cells, misses


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[],
                    help="pass base Qwen3 FIRST, as in 09")
    ap.add_argument("--probes",
                    default="studies/learn_french/probes/cross_route_probes.json")
    ap.add_argument("--routes", nargs="+", default=ROUTES)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print every prompt, load no model, import no torch")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    qa, p1 = load_text(cfg)
    spec = json.load(open(ROOT / a.probes, encoding="utf-8"))["probes"]
    cells, misses = build(spec, qa, p1, a.routes)

    print("=" * 94 + f"\n{len(cells)} cell(s) from {len(spec)} slot(s) x "
          f"{len(a.routes)} route(s); {len(misses)} skipped\n" + "=" * 94)
    for c in cells:
        print(f"  {c['id']:<15} [{c['type']:<6}] route {c['route']:<6} "
              f"target {c['target']!r}")
        print(f"      Q: {c['question']}")
        print(f"      A: ...{c['stump'][-60:]!r}  ->  {c['gold_tail'][:40]!r}")
    for m in misses:
        print(f"  SKIP  {m}")
    if a.dry_run:
        print("\n--dry-run: no model loaded.")
        return
    if not a.checkpoints:
        sys.exit("--checkpoints required unless --dry-run")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from top_tokens_ml import seq_score, script_of   # same scorer as 09, by construction

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    for c in cells:
        c["n_tokens"] = len(tok.encode(c["target"], add_special_tokens=False))
        c["first_id"] = tok.encode(c["target"], add_special_tokens=False)[0]

    memo = {}

    def latin(t):
        # 98 cells x 20,000 scanned ids per checkpoint: decode each id once, not per cell.
        if t not in memo:
            memo[t] = script_of(tok.decode([t])) == "LAT"
        return memo[t]

    results = {}
    for ckpt in a.checkpoints:
        print(f"\n{'=' * 94}\nCHECKPOINT  {ckpt}\n{'=' * 94}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        for c in cells:
            ids = tok(c["prompt"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                probs = torch.softmax(model(**ids).logits[0, -1].float(), -1)
            wide = probs.topk(min(SCAN, probs.numel()))
            order = wide.indices.tolist()
            rank = {t: r for r, t in enumerate(order)}
            lat = [t for t in order if latin(t)]
            lat_rank = {t: r for r, t in enumerate(lat)}
            per = seq_score(model, tok, c["prompt"], c["target"])
            nrm = math.exp(sum(per) / len(per))
            t0 = c["first_id"]
            top = [(tok.decode([t]), round(p, 4)) for p, t in
                   zip(*[x.tolist() for x in probs.topk(a.topk)])]
            print(f"  {c['id']:<15} {c['route']:<6} P^(1/n)={nrm:10.3e}  "
                  f"first {tok.decode([t0])!r} rank={rank.get(t0, '>%d' % SCAN)}  "
                  f"top: {top[:5]}")
            results.setdefault(ckpt, {}).setdefault(c["route"], {})[c["id"]] = {
                "seq_prob_norm": nrm, "per_token_logprob": per,
                "first_token": tok.decode([t0]), "first_prob": probs[t0].item(),
                "first_rank": rank.get(t0), "first_rank_latin": lat_rank.get(t0),
                "top": top}
        del model
        torch.cuda.empty_cache()

    if a.out:
        json.dump({"results": results, "cells": cells, "misses": misses,
                   "checkpoints": list(a.checkpoints), "routes": a.routes,
                   "probes": spec},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
