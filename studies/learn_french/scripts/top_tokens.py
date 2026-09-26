"""What token does the model WANT in the fact slot, and in which language?

    python studies/learn_french/scripts/top_tokens.py \
        --checkpoints <base-or-path> <path> [...] [--only <probe-id>] [--topk 40]

THE QUESTION (supervisor). Unlearning was probed in French, so the French form of the
answer is pushed down. But is the SAME fact still available in another language -- does
the Indonesian or Japanese spelling of the answer stay elevated? If so the fact was not
removed, only its French expression was suppressed.

HOW THE SLOT IS FOUND. probes/token_probes.json declares, per slot, the French answer
prefix to stop at and what the answer looks like in each language. The prompt is the
French question plus that prefix, so the next token IS the answer, and one distribution
can be read five ways.

    Why declared rather than derived: the published translations are NOT word-aligned.
    Diffing gold against perturbed finds a different semantic position in each language
    (French lands on the surname where English lands on the fact; Japanese reorders the
    sentence entirely, so its first divergence is word order). Verified, not assumed.
    Declaring the targets makes every comparison auditable; the script checks each prefix
    really occurs in the French answer and each target in its own language's answer.

WHAT IT PRINTS.
  1. the global top-k tokens at that slot, verbatim, each tagged with its script
  2. the top tokens WITHIN each script -- because a model relearned in Indonesian puts
     every Indonesian token up, and the question is whether the ANSWER is high among its
     own language's tokens, not whether the language is high overall
  3. for every language, where that language's spelling of the answer sits, with its rank
     both globally and among tokens of its own script
  4. all of it relative to the FIRST checkpoint passed -- pass base Qwen3 first, because
     some of these tokens are frequent in any context and only the elevation over a model
     that never learned the fact carries information

     Script is a coarse proxy for language and it is honest only where scripts differ.
     Japanese (kana/kanji) and Russian (Cyrillic) separate cleanly; French, English and
     Indonesian are all Latin, so "top LAT tokens" pools the three and you read the
     language off the words themselves.

WHICH SLOTS SEPARATE WHICH LANGUAGES. On a PROPER NOUN (Basil, Kuwait, Astana) French,
English and Indonesian share one identical token, so those slots can only separate ja and
ru -- a fr/en/id comparison there is UNMEASURABLE, not null. On a COMMON noun they all
differ (fleuriste / florist / penjual bunga / 花屋 / флорист) and all five separate. The
probe file labels each slot, and the script warns whenever two languages collide on the
same token.

CALIBRATION. LEARN was French-only. If fr_ft is no better than base Qwen3 on a language's
token, the model never knew the fact in that language and nothing there could have been
suppressed. The old English study was killed by exactly this gate (CLAUDE.md,
phase2_calibrate). Always pass base Qwen3 AND fr_ft so the table can be read.
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
PROBE_LANG = "fr"          # the prompt is always French; only the TARGETS vary
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


def load_probes(path: Path, data: dict, tok) -> list:
    """Declared slots, each verified against the data before any model is loaded."""
    spec = json.load(open(path, encoding="utf-8"))["probes"]
    out = []
    for p in spec:
        fi, ans = p["fact"], data[PROBE_LANG][p["fact"]]["answer"]
        if p["prefix"] not in ans:
            sys.exit(f"{p['id']}: prefix not found in the French answer for fact {fi}")
        cut = ans.index(p["prefix"]) + len(p["prefix"])
        targets, seen = {}, {}
        for lg, t in p["targets"].items():
            if lg not in data:
                continue
            if t.strip() not in data[lg][fi]["answer"]:
                print(f"  [warn {p['id']}/{lg}: {t.strip()!r} is not in that language's "
                      f"own answer -- check the probe file]")
            ids = tok.encode(t, add_special_tokens=False)
            if not ids:
                continue
            targets[lg] = (ids[0], t)
            seen.setdefault(ids[0], []).append(lg)
        collide = {tok.decode([k]): v for k, v in seen.items() if len(v) > 1}
        out.append({**p, "prompt": f"Question: {data[PROBE_LANG][fi]['question']}\n"
                                   f"Answer: {ans[:cut]}",
                    "gold": ans[cut:], "targets": targets, "collide": collide})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="pass base Qwen3 FIRST -- everything is reported relative to it")
    ap.add_argument("--probes",
                    default="studies/learn_french/probes/token_probes.json")
    ap.add_argument("--only", nargs="*", default=None,
                    help="probe ids to run (default: all in the file)")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--per-script", type=int, default=12,
                    help="how many top tokens to list WITHIN each script")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import src.data.load_multilingual_tofu as ml
    root = Path(__file__).resolve().parents[3]
    cfg = yaml.safe_load(open(root / "config" / "config.yaml"))
    ml_dir, cache = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]

    data = {}
    for lg in LANGS:
        try:
            data[lg] = ml.load_perturbed("forget01_perturbed", lg, ml_dir, cache)
        except Exception as e:
            print(f"  [no {lg} data: {type(e).__name__}]")
    if PROBE_LANG not in data:
        sys.exit("no French probe data")

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    # Resolve and VERIFY every probe before loading a model, so a bad slot fails in
    # seconds rather than after a model load.
    probes = load_probes(root / a.probes, data, tok)
    if a.only:
        probes = [p for p in probes if p["id"] in a.only]
    if not probes:
        sys.exit("no probes selected")
    print(f"\n{len(probes)} probe slot(s):")
    for p in probes:
        print(f"  {p['id']:<24} separates {p['separates']}")
        if p["collide"]:
            for t, lgs in p["collide"].items():
                print(f"      NOTE {'/'.join(lgs)} share the token {t!r} "
                      f"-- they cannot be told apart at this slot")

    results, ref = {}, {}
    for ci, ckpt in enumerate(a.checkpoints):
        print(f"\n{'='*94}\nCHECKPOINT  {ckpt}"
              + ("   <-- reference for every 'vs' column" if ci == 0 else "")
              + f"\n{'='*94}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        for pr in probes:
            fi = pr["id"]
            ids = tok(pr["prompt"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                probs = torch.softmax(model(**ids).logits[0, -1].float(), -1)
            order = probs.argsort(descending=True)
            rank = {int(t): r for r, t in enumerate(order[:SCAN].tolist())}

            print(f"\n--- {pr['id']}  (fact {pr['fact']}) ---")
            print(f"  prompt tail : ...{pr['prompt'][-80:]!r}")
            print(f"  gold answer : {pr['gold'][:60]!r}")
            print(f"\n  global top-{a.topk} in this slot:")
            tp, ti = probs.topk(a.topk)
            for r, (p, t) in enumerate(zip(tp.tolist(), ti.tolist())):
                sdec = tok.decode([t])
                d = ""
                if ci > 0 and (fi, t) in ref:
                    d = f"   x{p / max(ref[(fi, t)], 1e-12):8.2f} vs ref"
                elif ci == 0:
                    ref[(fi, t)] = p
                print(f"    {r:>3}. {p:9.6f}  [{script_of(sdec):>5}]  {sdec!r}{d}")

            # Within-script ranking. A model relearned in Indonesian lifts every
            # Indonesian token, so "is the answer high overall" conflates language with
            # knowledge. Ranking inside a script holds the language roughly fixed.
            wide = probs.topk(min(SCAN, probs.numel()))
            by_script, script_rank = {}, {}
            for p_, t_ in zip(wide.values.tolist(), wide.indices.tolist()):
                sc = script_of(tok.decode([t_]))
                if sc == "-":
                    continue
                by_script.setdefault(sc, []).append((p_, t_))
                script_rank[t_] = len(by_script[sc]) - 1
            for sc in ("LAT", "JA", "JA/ZH", "CYR"):
                if sc not in by_script:
                    continue
                print(f"\n  top-{a.per_script} tokens with script {sc}:")
                for r, (p_, t_) in enumerate(by_script[sc][:a.per_script]):
                    print(f"    {r:>3}. {p_:9.6f}  {tok.decode([t_])!r}")

            print(f"\n  the same answer, as each language spells it:")
            for lg in LANGS:
                if lg not in pr["targets"]:
                    continue
                t0, var = pr["targets"][lg]
                p0, rk = probs[t0].item(), rank.get(t0, ">5000")
                sr = script_rank.get(t0, None)
                key = (fi, "L", lg)
                d = ""
                if ci > 0 and key in ref:
                    d = f"   x{p0 / max(ref[key], 1e-12):8.2f} vs ref"
                elif ci == 0:
                    ref[key] = p0
                sc = script_of(tok.decode([t0]))
                print(f"    {lg}: {tok.decode([t0])!r:<14} [{sc:>5}]  p={p0:10.7f}"
                      f"  rank={str(rk):<7} rank-within-{sc}="
                      f"{str(sr) if sr is not None else '-':<6}{d}   ({var!r})")
                results.setdefault(ckpt, {}).setdefault(str(fi), {})[lg] = {
                    "token": tok.decode([t0]), "prob": p0, "script": sc,
                    "rank": rk if isinstance(rk, int) else None,
                    "rank_in_script": sr, "target": var}
        del model
        torch.cuda.empty_cache()

    if a.out:
        json.dump(results, open(a.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
