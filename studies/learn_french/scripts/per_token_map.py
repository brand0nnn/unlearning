"""Where did the unlearning go? Per-token log-probabilities on the TRAINED text (Plan 1).

    python studies/learn_french/scripts/per_token_map.py --dry-run      # no torch, no GPU
    python studies/learn_french/scripts/per_token_map.py \
        --checkpoints <base> <fr_ft> <fr_retain> <unl_fr> ... --out results/per_token_map.json

WHY THIS EXISTS. Stage 3 matched every arm on the WHOLE-ANSWER truth ratio, and gradient
difference ascends the token-MEAN NLL of each forget answer. Neither ever asked for the name
or the occupation specifically; the optimizer was free to raise the loss on whichever tokens
were cheapest. Stage 5 scored 14 hand-picked slots and found names barely moved while
occupations fell below never-taught -- which says what did NOT move, not what moved instead.
Fact 0's answer carries no fact but the name, and the name dropped ~1.2 nats: that row's
unlearning went somewhere. This records every answer token, so the loss can be accounted for.

WHAT IS SCORED. Exactly the sequences LEARN and UNLEARN trained on: TofuQADataset builds
input_ids and labels (format_qa's "[INST] {q} [/INST]" + answer, joint tokenisation, EOS
appended, question masked), so token boundaries and the label mask are the trained ones by
construction. Each labelled position gets: log p(gold token), the gold token's rank, and the
model's top-1 token and its probability (so a replaced word -- 'developpe' -> 'programme' --
is visible, not just a drop).

TEXTS.
  forget01_<L> for L in --langs   each unlearning arm's OWN training text (where that arm
                                  spent its loss) and the French text (what that did to the
                                  knowledge as learned). Pass 2, the standalone configs.
  retain99_fr, --retain-n rows    the H1 control: retain answers that copy a name from their
                                  question. If gradient difference protects copying, those
                                  tokens stay flat at every arm. French only, because the
                                  French retain set is what LEARN trained on; each arm's
                                  retain term used its own language, so this is the
                                  French-side view. Evenly strided, deterministic.

Nothing is tagged here. Tags (name copied / name recalled / attribute / template) are applied
locally by plots/plot_per_token_map.py from probes/many_slot_probes.json and the character
offsets stored below, so the annotation can be corrected without re-running the GPU.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LANGS = ["fr", "en", "id", "ja", "ru"]


def load_texts(cfg, langs, retain_n):
    import src.data.load_multilingual_tofu as ml
    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    texts = {}
    for lg in langs:
        for i, r in enumerate(ml.load_qa("forget01", lg, M, C)):
            texts[f"forget:{lg}:{i}"] = {"split": "forget", "lang": lg, "row": i, **r}
    if retain_n:
        ret = ml.load_qa("retain99", "fr", M, C)
        step = len(ret) // retain_n
        for i in range(0, step * retain_n, step):
            texts[f"retain:fr:{i}"] = {"split": "retain", "lang": "fr", "row": i, **ret[i]}
    return texts


def encode(texts, tok, max_len):
    """Trained input_ids/labels via TofuQADataset, plus answer-relative char offsets."""
    from src.evaluation.compute_logprobs import format_qa
    from src.training.learn import TofuQADataset
    ds = TofuQADataset(list(texts.values()), tok, max_len)
    for k, (key, t) in enumerate(texts.items()):
        item = ds[k]
        ids, labels = item["input_ids"], item["labels"]
        prefix = format_qa(t["question"])
        full = prefix + t["answer"].strip()
        enc = tok(full, add_special_tokens=True, max_length=max_len, truncation=True,
                  return_offsets_mapping=True)
        n_text = len(enc.input_ids)
        # The dataset's ids are the same tokenisation plus (maybe) a trailing EOS. Refuse to
        # continue on any disagreement: the offsets would then point at the wrong tokens.
        assert ids[:n_text] == enc.input_ids, f"{key}: tokenisation mismatch"
        offsets = [list(o) for o in enc.offset_mapping] + [[len(full), len(full)]] * (
            len(ids) - n_text)
        pos = [j for j, l in enumerate(labels) if l != -100]
        # The answer was .strip()ped before joining, so offsets are relative to that.
        t.update({"answer_trained": t["answer"].strip(), "ids": ids, "label_pos": pos,
                  "tokens": [tok.decode([ids[j]]) for j in pos],
                  "token_ids": [ids[j] for j in pos],
                  "offsets": [[offsets[j][0] - len(prefix), offsets[j][1] - len(prefix)]
                              for j in pos],
                  "is_eos": [j >= n_text for j in pos]})
    return texts


def score(model, t):
    import torch
    ids = torch.tensor([t["ids"]], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    # logits at position j-1 predict token j.
    pred = torch.tensor([j - 1 for j in t["label_pos"]], device=model.device)
    gold = torch.tensor(t["token_ids"], device=model.device)
    lp_all = torch.log_softmax(logits[pred], -1)
    lp = lp_all.gather(1, gold[:, None]).squeeze(1)
    rank = (lp_all > lp[:, None]).sum(1)
    top_p, top_id = lp_all.max(1)
    return {"lp": [round(x, 5) for x in lp.tolist()], "rank": rank.tolist(),
            "top1_id": top_id.tolist(), "top1_p": [round(x, 5) for x in top_p.exp().tolist()]}


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--retain-n", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="tokenise and print, load no model (needs transformers, not torch)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    max_len = cfg["model"]["max_seq_length"]          # what finetune_tofu passes
    texts = encode(load_texts(cfg, a.langs, a.retain_n), tok, max_len)

    n_tok = sum(len(t["label_pos"]) for t in texts.values())
    print(f"{len(texts)} sequences, {n_tok} scored tokens, max_len {max_len}")
    for key in ["forget:fr:0", "forget:fr:3", "forget:ja:0"]:
        if key in texts:
            t = texts[key]
            print(f"  {key}: {len(t['tokens'])} tokens: "
                  + " | ".join(t["tokens"][:24]))
            # the offsets must reproduce the tokens' own text
            bad = [(s, e, tk) for (s, e), tk, eos in
                   zip(t["offsets"], t["tokens"], t["is_eos"])
                   if not eos and "\ufffd" not in tk      # byte-split CJK char: expected
                   and t["answer_trained"][s:e].strip() != tk.strip()]
            print(f"      offset check: {len(bad)} mismatch(es) {bad[:3]}")
    if a.dry_run:
        print("\n--dry-run: no model loaded.")
        return
    if not a.checkpoints:
        sys.exit("--checkpoints required unless --dry-run")

    import torch
    from transformers import AutoModelForCausalLM

    store = {k: {f: t[f] for f in ("split", "lang", "row", "question", "answer_trained",
                                    "tokens", "token_ids", "offsets", "is_eos")}
             for k, t in texts.items()}
    results, top1_vocab = {}, {}
    for ckpt in a.checkpoints:
        print(f"\n{'=' * 94}\nCHECKPOINT  {ckpt}\n{'=' * 94}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        res = {}
        for key, t in texts.items():
            res[key] = score(model, t)
            for i in res[key]["top1_id"]:
                if i not in top1_vocab:
                    top1_vocab[i] = tok.decode([i])
        results[ckpt] = res
        for key in ["forget:fr:0", "forget:fr:3", "forget:fr:20"]:
            if key in res:
                lp = res[key]["lp"]
                print(f"  {key}: mean lp {sum(lp) / len(lp):8.3f}  "
                      f"min {min(lp):8.3f} at {texts[key]['tokens'][lp.index(min(lp))]!r}")
        del model
        torch.cuda.empty_cache()
        if a.out:   # after every checkpoint, atomically: a wall-clock kill keeps the rest
            tmp = a.out + ".tmp"
            json.dump({"checkpoints": list(a.checkpoints), "texts": store,
                       "results": results,
                       "top1_vocab": {str(k): v for k, v in top1_vocab.items()},
                       "template": "[INST] {q} [/INST]{answer} + EOS (TofuQADataset)"},
                      open(tmp, "w"), ensure_ascii=False)
            os.replace(tmp, a.out)
            print(f"  -> {a.out} ({len(results)}/{len(a.checkpoints)} checkpoints)",
                  flush=True)


if __name__ == "__main__":
    main()
