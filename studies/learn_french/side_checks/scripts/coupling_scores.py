"""Coupling scores: how tightly is each unlearning set tied to the French target? (no training)

    python studies/learn_french/side_checks/scripts/coupling_scores.py --dry-run     # no GPU
    python studies/learn_french/side_checks/scripts/coupling_scores.py \
        --fr-ft <fr_ft> --fr-retain <fr_retain> --out side_checks/results/coupling/scores.json

WHY (supervisor meeting 2026-11-13; Experiment D round 2). The layer locus turned out to be a
symptom; what follows the unlearning language is selective (en) vs content-blind (ja) damage
to the unlearning text. Proposed language-free reading: unlearning on a set D_u lowers the
target in proportion to how strongly D_u's likelihood is coupled to the target's under theta.
Two model-based scores of that coupling, for every candidate D_u, at theta = fr_ft:

  [1] CARRY-OVER (forward only): per fact, mean answer-token log p under fr_ft minus under
      fr_retain (the study's existing carry-over), now for ko / hi / ar / fa / iw and the
      romaji copy of ja as well as en / id / ru / ja / fr.
  [2] GRADIENT ALIGNMENT at fr_ft: g_S = grad of the training loss (TofuQADataset labels,
      mean NLL over answer tokens + EOS, exactly what gradient difference ascends) on source
      S; g_T = the same on the French forget text (the target). cos(g_S, g_T), overall and
      per parameter group (decoder layer 0..35, embed_tokens, lm_head, final norm):
        per fact   cos(g_S_i, g_T_i)             same fact, other language / form
        summed     cos(sum_i g_S_i, sum_i g_T_i) the direction one full-batch step takes
      First-order meaning: a small ascent step on S changes the target loss by
      lr * <g_S, g_T>, so cos is the coupling in direction; ||g_S|| is stored for magnitude.

SOURCES. forget01 in en id ru ja ko hi ar fa iw (fr is the target), the romaji copy of ja
(side_checks/probes/forget01_ja_romaji.json), and two CONTROLS with no forget fact in them:
40 evenly strided retain99 rows in French and in Japanese (other authors). Control row k is
paired with forget fact k for the per-fact cosine (an arbitrary pairing on purpose: it gives
the alignment of unrelated content, the floor any forget source should beat).

PRECISION. Model and gradients in bf16 (as trained under ZeRO-3 they were fp32 masters; here
no optimizer step is taken). Dot products are accumulated in fp32 per parameter. The summed
gradient accumulates 40 bf16 backward passes in .grad; cos is read to 2 decimals.
Single checkpoint (seed 42 fr_ft); 40 facts about 2 entities.
"""
import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / "studies" / "learn_french"
SIDE = STUDY / "side_checks"

TARGET = "fr"
FORGET_SOURCES = ["en", "id", "ru", "ja", "ko", "hi", "ar", "fa", "iw"]
ROMAJI = "ja_romaji"
CONTROLS = {"ctl_retain_fr": "fr", "ctl_retain_ja": "ja"}


def _ptm():
    spec = importlib.util.spec_from_file_location("ptm", STUDY / "scripts" / "per_token_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_texts(cfg):
    import src.data.load_multilingual_tofu as ml
    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    texts = {}
    for lg in [TARGET] + FORGET_SOURCES:
        rows = ml.load_qa("forget01", lg, M, C)
        assert len(rows) == 40, (lg, len(rows))
        for i, r in enumerate(rows):
            texts[f"{lg}:{i}"] = {"split": "forget", "lang": lg, "row": i,
                                  "question": r["question"], "answer": r["answer"]}
    rom = json.load(open(SIDE / "probes" / "forget01_ja_romaji.json", encoding="utf-8"))["rows"]
    assert len(rom) == 40
    for r in rom:
        texts[f"{ROMAJI}:{r['row']}"] = {"split": "forget", "lang": ROMAJI, "row": r["row"],
                                         "question": r["question"], "answer": r["answer"]}
    for name, lg in CONTROLS.items():
        ret = ml.load_qa("retain99", lg, M, C)
        step = len(ret) // 40
        for k in range(40):
            r = ret[k * step]
            texts[f"{name}:{k}"] = {"split": "retain", "lang": name, "row": k * step,
                                    "question": r["question"], "answer": r["answer"]}
    return texts


def sources():
    return FORGET_SOURCES + [ROMAJI] + list(CONTROLS)


def group_of(name):
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m:
        return f"L{int(m.group(1))}"
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    return "norm"


# ---- [1] forward: mean answer-token log p per text ---------------------------------------
def forward_scores(model, texts, score_fn):
    out = {}
    for k, t in texts.items():
        r = score_fn(model, t)
        keep = [j for j, e in enumerate(t["is_eos"]) if not e]
        out[k] = {"lp_mean": sum(r["lp"][j] for j in keep) / len(keep), "lp": r["lp"]}
    return out


# ---- [2] gradients -------------------------------------------------------------------------
def make_batch(t, device):
    import torch
    ids = torch.tensor([t["ids"]], device=device)
    lab = torch.full_like(ids, -100)
    for j in t["label_pos"]:
        lab[0, j] = t["ids"][j]
    return ids, lab


def backward(model, t):
    ids, lab = make_batch(t, model.device)
    loss = model(input_ids=ids, labels=lab, use_cache=False).loss
    loss.backward()
    return float(loss)


def snapshot(params):
    return [p.grad.detach().clone() for _, p in params]


def dots(params, buf, groups):
    """Per group: <g, buf>, ||g||^2, ||buf||^2 (fp32), g = current .grad."""
    import torch
    acc = {}
    for (name, p), b in zip(params, buf):
        g = p.grad
        gf, bf = g.float(), b.float()
        d = acc.setdefault(groups[name], [0.0, 0.0, 0.0])
        d[0] += float(torch.sum(gf * bf))
        d[1] += float(torch.sum(gf * gf))
        d[2] += float(torch.sum(bf * bf))
        del gf, bf
    return acc


def summarise(acc):
    tot = [sum(v[i] for v in acc.values()) for i in range(3)]
    cos = lambda d: d[0] / ((d[1] * d[2]) ** 0.5 + 1e-30)
    return {"cos": cos(tot), "dot": tot[0], "norm_src": tot[1] ** 0.5, "norm_tgt": tot[2] ** 0.5,
            "by_group": {g: {"cos": cos(d), "dot": d[0], "norm_src": d[1] ** 0.5,
                             "norm_tgt": d[2] ** 0.5} for g, d in acc.items()}}


def gradient_scores(model, texts, log):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = {n: group_of(n) for n, _ in params}
    zero = lambda: model.zero_grad(set_to_none=False)
    per_fact = {s: [] for s in sources()}
    t0 = time.time()
    for i in range(40):
        zero()
        lt = backward(model, texts[f"{TARGET}:{i}"])
        buf = snapshot(params)
        for s in sources():
            zero()
            ls = backward(model, texts[f"{s}:{i}"])
            r = summarise(dots(params, buf, groups))
            r.update({"loss_src": ls, "loss_tgt": lt})
            per_fact[s].append(r)
        del buf
        if i % 5 == 0:
            log(f"  per-fact {i + 1}/40  ({time.time() - t0:.0f}s)  "
                + " ".join(f"{s}:{per_fact[s][-1]['cos']:+.3f}" for s in sources()))
    summed = {}
    zero()
    for i in range(40):
        backward(model, texts[f"{TARGET}:{i}"])
    buf = snapshot(params)
    for s in sources():
        zero()
        for i in range(40):
            backward(model, texts[f"{s}:{i}"])
        summed[s] = summarise(dots(params, buf, groups))
        log(f"  summed {s:14s} cos {summed[s]['cos']:+.4f}  |g_S| {summed[s]['norm_src']:.3g}")
    zero()
    return per_fact, summed


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--fr-ft")
    ap.add_argument("--fr-retain")
    ap.add_argument("--out", default=str(SIDE / "results" / "coupling" / "scores.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    ptm = _ptm()
    texts = ptm.encode(load_texts(cfg), tok, cfg["model"]["max_seq_length"])
    by = {}
    for t in texts.values():
        by.setdefault(t["lang"], []).append(len(t["label_pos"]))
    print(f"{len(texts)} sequences; answer tokens per source (mean): "
          + " ".join(f"{k}:{sum(v) / len(v):.0f}" for k, v in by.items()))
    for k in ("fr:0", "ja:0", f"{ROMAJI}:0", "ko:0", "ctl_retain_ja:0"):
        print(f"  {k}: " + " | ".join(texts[k]["tokens"][:20]))
    if a.dry_run:
        print("--dry-run: no model loaded.")
        return

    import torch
    from transformers import AutoModelForCausalLM

    def log(msg):
        print(msg, flush=True)

    out = {"meta": {"fr_ft": a.fr_ft, "fr_retain": a.fr_retain, "target": TARGET,
                    "sources": sources(), "controls": CONTROLS,
                    "note": "see the module docstring of coupling_scores.py"},
           "texts": {k: {f: t[f] for f in ("split", "lang", "row", "question",
                                            "answer_trained", "tokens", "is_eos")}
                     for k, t in texts.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def load(path):
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda()
        m.eval()
        return m

    log(f"[1/3] forward + [2/3] gradients at fr_ft: {a.fr_ft}")
    model = load(a.fr_ft)
    out["forward_fr_ft"] = forward_scores(model, texts, ptm.score)
    per_fact, summed = gradient_scores(model, texts, log)
    out["grad_per_fact"], out["grad_summed"] = per_fact, summed
    del model
    torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    log(f"[3/3] forward at fr_retain: {a.fr_retain}")
    model = load(a.fr_retain)
    out["forward_fr_retain"] = forward_scores(model, texts, ptm.score)
    carry = {}
    for s in [TARGET] + sources():
        carry[s] = [out["forward_fr_ft"][f"{s}:{i}"]["lp_mean"]
                    - out["forward_fr_retain"][f"{s}:{i}"]["lp_mean"] for i in range(40)]
    out["carry_over_per_fact"] = carry
    tmp = Path(a.out).with_suffix(".tmp")
    json.dump(out, open(tmp, "w"), ensure_ascii=False)
    tmp.replace(a.out)
    log("carry-over (mean over facts, nats/token) and summed-gradient cos with French:")
    for s in [TARGET] + sources():
        c = sum(carry[s]) / 40
        g = summed.get(s, {}).get("cos", float("nan"))
        pf = per_fact.get(s)
        pfm = sum(r["cos"] for r in pf) / 40 if pf else float("nan")
        log(f"  {s:14s} carry {c:+6.2f}   cos summed {g:+.4f}   cos per-fact mean {pfm:+.4f}")
    log(f"=== coupling scores complete -> {a.out}")


if __name__ == "__main__":
    main()
