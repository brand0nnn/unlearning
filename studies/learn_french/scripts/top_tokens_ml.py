"""Ask the question IN each language, and see what the model wants to answer.

    python studies/learn_french/scripts/top_tokens_ml.py \
        --checkpoints <base> <path> [...] [--prompt-langs fr en id ja ru] [--topk 100]

THE QUESTION THIS ANSWERS AND THE OLD SCRIPT DOES NOT. top_tokens.py builds ONE French
prompt and reads its next-token distribution five ways. That measures where the French
answer went -- solidly -- but it cannot say whether the fact exists in Japanese, because a
French prompt conditions the whole distribution on French. Verified in the 884103 run: at
the ja-unlearned checkpoint the French answer is ' phot' (photographe) and the top Russian
token is ' фотограф', tracking it word for word. The non-Latin tail is a translation shadow
of whatever the model currently wants to say in French, not a separate store.

So here every language gets its OWN prompt, built from its OWN gold answer, and the five
distributions are independent measurements.

THE TWO COMPARISONS, IN ORDER. Both need base Qwen3 and fr_retain in the checkpoint list.

  1. TRANSFER, the precondition.  fr_ft vs fr_retain, asked in Japanese.
     fr_retain had the SAME French fine-tuning and never saw the fact, so a gap between
     them is the fact and nothing else. If there is no gap, French-only LEARN put nothing
     into Japanese, and no unlearning arm can show it hiding there. That closes the
     question rather than leaving it open.
  2. HIDING, the supervisor's question.  unl_fr vs fr_ft, asked in Japanese.
     Only worth reading where step 1 found something to remove.

  base Qwen3 answers a third question -- is this token just generically frequent here --
  and it is NOT interchangeable with fr_retain. They disagreed in the 884103 run: against
  base the Russian florist token rose 5.3x, against fr_retain 2419x, because fr_retain is
  a far more peaked model and its whole tail is crushed. Which brings us to:

REPORT RANK FIRST, PROBABILITY SECOND. A probability ratio across two checkpoints mixes
"the model learned this" with "this model is more confident in general". Rank does not.
But rank alone hides whether the number matters at all -- ' фл' sat at rank 34 within
Cyrillic on p=1.1e-10, which the model would never actually emit -- and rank is noisy deep
in the tail, where hundreds of tokens share one order of magnitude. Neither is sufficient;
the script prints both, plus the within-script rank and ITS DENOMINATOR, which the old
script omitted and without which "rank 34 among Cyrillic" cannot be read.

HOW A SLOT IS BUILT. probes/token_probes_ml.json declares, per slot and language, candidate
spellings of the answer. The script finds the first candidate that occurs in that language's
own gold answer, cuts the answer immediately before it, and prompts with the question plus
that stump -- so the next token is the answer, in that language. Nothing is derived from
French. A candidate that is absent, or occurs more than once, skips that ONE cell with a
named warning; the run continues. Every gold answer is printed first, so a bad guess is
corrected from the same log rather than by a second blind run.
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

LANGS = ["fr", "en", "id", "ja", "ru"]
SCAN = 20000               # how deep to scan for global and per-script ranks


def script_of(tok: str) -> str:
    for ch in tok:
        if not ch.isalpha():
            continue
        n = unicodedata.name(ch, "")
        for key, tag in (("CJK", "JA/ZH"), ("HIRAGANA", "JA"), ("KATAKANA", "JA"),
                         ("CYRILLIC", "CYR"), ("ARABIC", "ARA"), ("HANGUL", "KOR"),
                         ("DEVANAGARI", "DEV"), ("LATIN", "LAT")):
            if key in n:
                return tag
    return "-"


def build(spec, data, tok, langs):
    """One cell per (slot, prompt language). Misses are reported, never fatal."""
    cells, misses = [], []
    for p in spec:
        fi = p["fact"]
        for lg in langs:
            if lg not in data:
                continue
            ans, q = data[lg][fi]["answer"], data[lg][fi]["question"]
            hit, ambiguous = None, []
            for cand in p["targets"].get(lg, []):
                n = ans.count(cand)
                if n == 1:
                    hit = cand
                    break
                if n > 1:
                    ambiguous.append(f"{cand!r} x{n}")
            if hit is None:
                # One line per missed cell, saying which of the two things went wrong --
                # a candidate that appears twice has no unambiguous truncation point, and
                # is a different edit from a candidate that is simply the wrong word.
                misses.append(
                    f"{p['id']}/{lg}: ambiguous, every candidate repeats in the gold "
                    f"answer ({', '.join(ambiguous)})" if ambiguous else
                    f"{p['id']}/{lg}: none of {p['targets'].get(lg, [])} found in the "
                    f"gold answer")
                continue
            cut = ans.index(hit)
            # Qwen3's BPE attaches a word's preceding space to it, so the token that
            # follows a stump ending in a space is NOT the token we want -- move the space
            # out of the prompt and into the target.
            target = hit
            if cut and ans[cut - 1] == " ":
                cut -= 1
                target = " " + hit
            ids = tok.encode(target, add_special_tokens=False)
            if not ids:
                misses.append(f"{p['id']}/{lg}: {target!r} tokenises to nothing")
                continue
            cells.append({"id": p["id"], "fact": fi, "role": p["role"], "lang": lg,
                          "target": target, "first_id": ids[0],
                          "n_tokens": len(ids), "stump_chars": cut,
                          "prompt": f"Question: {q}\nAnswer: {ans[:cut]}",
                          "gold_tail": ans[cut:]})
    return cells, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="pass base Qwen3 FIRST -- every 'vs ref' column is relative to it")
    ap.add_argument("--probes",
                    default="studies/learn_french/probes/token_probes_ml.json")
    ap.add_argument("--prompt-langs", nargs="+", default=LANGS,
                    help="languages to ASK the question in (default: all five)")
    ap.add_argument("--only", nargs="*", default=None, help="probe ids to run")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--per-script", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import src.data.load_multilingual_tofu as ml
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load(open(root / "config" / "config.yaml"))

    data = {}
    for lg in LANGS:
        try:
            data[lg] = ml.load_perturbed("forget01_perturbed", lg,
                                         cfg["tofu"]["ml_cache_dir"],
                                         cfg["tofu"]["cache_dir"])
        except Exception as e:
            print(f"  [no {lg} data: {type(e).__name__}: {e}]")
    missing = [lg for lg in a.prompt_langs if lg not in data]
    if missing:
        sys.exit(f"no data for requested prompt languages: {missing}")

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    spec = json.load(open(root / a.probes, encoding="utf-8"))["probes"]
    if a.only:
        spec = [p for p in spec if p["id"] in a.only]

    # THE GOLD ANSWERS, VERBATIM, BEFORE ANYTHING ELSE. The ja/id/ru targets in the probe
    # file are guesses against translations nobody has read. Printing the answers makes a
    # wrong guess a one-line edit after this run instead of a second blind job.
    print("\n" + "=" * 94 + "\nGOLD ANSWERS (fix the probe file from these if a slot misses)"
          "\n" + "=" * 94)
    for p in spec:
        print(f"\n--- {p['id']}  fact {p['fact']}  [{p['role']}]  {p['gloss']}")
        for lg in a.prompt_langs:
            print(f"  {lg}  Q: {data[lg][p['fact']]['question']}")
            print(f"      A: {data[lg][p['fact']]['answer']}")

    cells, misses = build(spec, data, tok, a.prompt_langs)
    print("\n" + "=" * 94 + f"\n{len(cells)} cell(s) resolved, "
          f"{len(misses)} skipped\n" + "=" * 94)
    for c in cells:
        print(f"  {c['id']:<22} ask in {c['lang']}  target {c['target']!r} "
              f"({c['n_tokens']} token(s), first = {tok.decode([c['first_id']])!r})")
        print(f"      prompt tail ...{c['prompt'][-70:]!r}")
    for m in misses:
        print(f"  MISS  {m}")
    if not cells:
        sys.exit("no cells resolved -- fix the targets in the probe file")

    results, ref = {}, {}
    for ci, ckpt in enumerate(a.checkpoints):
        print(f"\n{'=' * 94}\nCHECKPOINT  {ckpt}"
              + ("   <-- reference for every 'vs' column" if ci == 0 else "")
              + f"\n{'=' * 94}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        for c in cells:
            ids = tok(c["prompt"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                probs = torch.softmax(model(**ids).logits[0, -1].float(), -1)
            wide = probs.topk(min(SCAN, probs.numel()))
            rank = {int(t): r for r, t in enumerate(wide.indices.tolist())}

            # Bucket the scan by script ONCE: it yields the within-script rank AND the
            # denominator that rank needs. "34th among Cyrillic" is unreadable until you
            # know whether the pool is 40 tokens or 400.
            by_script, script_rank = {}, {}
            for p_, t_ in zip(wide.values.tolist(), wide.indices.tolist()):
                sc = script_of(tok.decode([t_]))
                if sc == "-":
                    continue
                by_script.setdefault(sc, []).append((p_, t_))
                script_rank[t_] = len(by_script[sc]) - 1

            print(f"\n--- {c['id']}  asked in {c['lang']}  (fact {c['fact']}, "
                  f"{c['role']}) ---")
            print(f"  prompt tail : ...{c['prompt'][-70:]!r}")
            print(f"  gold tail   : {c['gold_tail'][:60]!r}")
            print(f"\n  global top-{a.topk}:")
            tp, ti = probs.topk(a.topk)
            for r, (p_, t_) in enumerate(zip(tp.tolist(), ti.tolist())):
                sdec = tok.decode([t_])
                key = (c["id"], c["lang"], t_)
                d = ""
                if ci > 0 and key in ref:
                    d = f"   x{p_ / max(ref[key], 1e-30):9.3e} vs ref"
                elif ci == 0:
                    ref[key] = p_
                print(f"    {r:>3}. {p_:11.4e}  [{script_of(sdec):>5}]  {sdec!r}{d}")

            for sc in ("LAT", "JA", "JA/ZH", "CYR"):
                if sc not in by_script:
                    continue
                print(f"\n  top-{a.per_script} of the {len(by_script[sc])} tokens with "
                      f"script {sc} in the top {SCAN}:")
                for r, (p_, t_) in enumerate(by_script[sc][:a.per_script]):
                    print(f"    {r:>3}. {p_:11.4e}  {tok.decode([t_])!r}")

            print(f"\n  the answer, as every language spells it "
                  f"(the {c['lang']} row is the one this prompt asks for):")
            slot = results.setdefault(ckpt, {}).setdefault(c["lang"], {}) \
                          .setdefault(c["id"], {})
            for lg in LANGS:
                other = next((x for x in cells
                              if x["id"] == c["id"] and x["lang"] == lg), None)
                if other is None:
                    continue
                t0 = other["first_id"]
                p0 = probs[t0].item()
                sc = script_of(tok.decode([t0]))
                rk, sr = rank.get(t0), script_rank.get(t0)
                key = (c["id"], c["lang"], "L", lg)
                d = ""
                if ci > 0 and key in ref:
                    d = f"   x{p0 / max(ref[key], 1e-30):9.3e} vs ref"
                elif ci == 0:
                    ref[key] = p0
                mark = "<-" if lg == c["lang"] else "  "
                print(f"  {mark}{lg}: {tok.decode([t0])!r:<14} [{sc:>5}]  p={p0:11.4e}"
                      f"  rank={(rk if rk is not None else '>%d' % SCAN)!s:<7}"
                      f"rank-in-{sc}={(sr if sr is not None else '-')!s:>6}"
                      f"/{len(by_script.get(sc, [])) or '-'}{d}   ({other['target']!r})")
                slot[lg] = {"token": tok.decode([t0]), "prob": p0, "script": sc,
                            "rank": rk, "rank_in_script": sr,
                            "script_pool": len(by_script.get(sc, [])),
                            "target": other["target"], "asked": lg == c["lang"]}
        del model
        torch.cuda.empty_cache()

    if a.out:
        json.dump({"results": results,
                   "cells": [{k: v for k, v in c.items() if k != "prompt"}
                             for c in cells],
                   "misses": misses,
                   "checkpoints": list(a.checkpoints)},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n-> {a.out}")
    if misses:
        print(f"\n{len(misses)} slot(s) missed -- correct their targets in "
              f"{a.probes} from the gold answers printed at the top of this log.")


if __name__ == "__main__":
    main()
