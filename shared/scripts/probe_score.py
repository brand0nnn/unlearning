"""Score a probe family against one checkpoint — Phase 2 Part A, step 2.

The study measures each forget fact with ONE canonical question. If unlearning only
suppressed that phrasing, a different phrasing recovers the fact and the headline
recovery number is partly re-surfacing rather than relearning-driven transfer. This runs
the whole probe family (built by build_probe_family.py) against a checkpoint and writes
PER-FACT, PER-PROBE scores so the two can be compared paired, fact by fact.

Run it on the LEARNED checkpoint and the UNLEARNED one. Learned is the ceiling: a probe
that model cannot answer is a broken probe, not a finding, and the analysis drops it.
That empirical filter is why no LLM equivalence judge is needed.

THREE PROBE TYPES, THREE METRICS — they are not interchangeable, and mixing them into one
"probe score" would average a ratio with a probability:

  qa   truth ratio, exactly as the rest of the study computes it. Only the QUESTION
       varies; `paraphrased_answer`/`perturbed_answers` describe the FACT, not the
       phrasing, so every qa probe stays directly comparable to p0_canonical.
       LOW = the model knows the fact.
  mcq  MC-normalized probability p_correct / sum(p_choices), as TOFU uses for
       real_authors/world_facts. HIGH = knows. This changes the TASK FORMAT rather than
       the wording: the model ranks six candidates instead of generating, so suppression
       trained on generation may leave it untouched.
  fib  cloze. The prompt is the question plus the answer up to the blank; we score
       P(correct span | prompt) against P(each perturbed span | prompt) and report the
       same ratio as `qa`. Restricting to the fact-bearing span means a model that merely
       recites the sentence frame scores nothing for it. LOW = knows.

Writes one JSON per checkpoint, merged under an exclusive lock (concurrent jobs and
wall-clock kills are both safe — see §7 of CLAUDE.md).

    python shared/scripts/probe_score.py --checkpoint <path> \
        --probes studies/crosslingual_recovery/probes/probe_family.json \
        --group phase2_probes
"""
import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

import yaml

from src.evaluation.compute_logprobs import _answer_logprob_sum, normalized_answer_prob
from src.evaluation.tofu_metrics import (probability_score_mc, truth_ratio_from_probs,
                                         truth_ratio_score)
from src.utils.logging_utils import get_logger
from src.utils.paths import results_root

logger = get_logger("probe_score")


def _merge_write(f: Path, updates: dict):
    """Merge into the group JSON under an exclusive lock, atomically.

    Same pattern as relearn_measure._merge_write: flock serialises the whole
    read-modify-write between concurrent jobs, and os.replace from a .tmp means a kill
    mid-write leaves the previous file intact rather than a truncated one.
    """
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(str(f) + ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            d = json.loads(f.read_text()) if f.exists() else {}
        except json.JSONDecodeError:
            logger.warning("%s was corrupt (truncated by an earlier kill?); starting fresh", f.name)
            d = {}
        d.update(updates)
        tmp = Path(str(f) + ".tmp")
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        os.replace(tmp, f)
        fcntl.flock(lf, fcntl.LOCK_UN)


def span_prob(model, tok, question: str, prefix: str, span: str) -> float:
    """P(span | question, prefix) ** (1/|span tokens|).

    Scored by DIFFERENCE of cumulative log-probs -- logp(prefix+span) minus logp(prefix)
    -- so the shared prefix contributes nothing. Scoring `prefix + span` directly would
    dilute the fact-bearing words with a frame identical across all candidates, which is
    the whole thing fill-in-the-blank exists to avoid.
    """
    import torch

    full = (prefix + " " + span).strip() if prefix else span
    lp_full, n_full = _answer_logprob_sum(model, tok, question, full)
    if prefix:
        lp_pre, n_pre = _answer_logprob_sum(model, tok, question, prefix)
    else:
        lp_pre, n_pre = torch.tensor(0.0, device=lp_full.device), 0
    n = n_full - n_pre
    if n <= 0:
        return 0.0
    return float(torch.exp((lp_full - lp_pre) / n))


def score_probe(model, tok, fact: dict, probe: dict) -> float:
    t = probe["type"]
    if t == "qa":
        return truth_ratio_score(model, tok, probe["question"],
                                 fact["paraphrased_answer"], fact["perturbed_answers"])
    if t == "mcq":
        ch = probe["choices"]
        correct = ch[probe["answer_idx"]]
        wrong = [c for i, c in enumerate(ch) if i != probe["answer_idx"]]
        return probability_score_mc(model, tok, probe["question"], correct, wrong)
    if t == "fib":
        p_ok = span_prob(model, tok, probe["question"], probe["prefix"], probe["target"])
        p_bad = [span_prob(model, tok, probe["question"], probe["prefix"], d)
                 for d in probe["distractors"]]
        return truth_ratio_from_probs(p_ok, p_bad)
    raise ValueError(f"unknown probe type {t!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True,
                    help="local checkpoint dir, or an HF model name for the base model")
    ap.add_argument("--probes", required=True)
    ap.add_argument("--group", default="phase2_probes")
    ap.add_argument("--tag", default=None, help="key in the output JSON (default: basename)")
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = yaml.safe_load(open(_r / a.config))
    fam = json.loads(Path(a.probes).read_text())

    # Tokenizer from the CONFIG, never the checkpoint (CLAUDE.md §7: a checkpoint saved
    # without its tokenizer yields empty token sequences and an all-zeros eval).
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, torch_dtype=getattr(torch, cfg["model"].get("dtype", "bfloat16")),
        device_map="auto")
    model.eval()

    tag = a.tag or Path(a.checkpoint).name
    logger.info("scoring %d facts from %s against %s",
                fam["meta"]["n_facts"], Path(a.probes).name, tag)

    # {probe_id: {fact_idx: score}} -- keep the full per-fact arrays, never just a mean;
    # the paired p0-vs-p1 comparison needs them, and so does any bootstrap.
    per_probe: dict[str, dict[int, float]] = {}
    for fact in fam["facts"]:
        for probe in fact["probes"]:
            try:
                s = score_probe(model, tok, fact, probe)
            except Exception as e:                      # one bad probe must not kill the run
                logger.warning("fact %d probe %s failed: %s", fact["idx"], probe["id"], e)
                continue
            per_probe.setdefault(probe["id"], {})[fact["idx"]] = s
        if fact["idx"] % 10 == 0:
            logger.info("  ... fact %d/%d", fact["idx"], fam["meta"]["n_facts"])

    out = {}
    for pid, scores in per_probe.items():
        idxs = sorted(scores)
        vals = [scores[i] for i in idxs]
        out[pid] = {
            "fact_indices": idxs,
            "scores_per_fact": vals,
            "mean": sum(vals) / len(vals) if vals else None,
            "n": len(vals),
            # which metric produced these -- a mean over mixed types would be meaningless
            "metric": next(p["type"] for f in fam["facts"] for p in f["probes"]
                           if p["id"] == pid),
        }

    f = Path(results_root()) / "relearn" / a.group / f"{tag}.json"
    _merge_write(f, {tag: out})
    logger.info("wrote %d probe ids -> %s", len(out), f)
    for pid, v in sorted(out.items()):
        logger.info("  %-14s %-4s n=%-3d mean=%.4f", pid, v["metric"], v["n"], v["mean"])
    logger.info("probe scoring complete")


if __name__ == "__main__":
    main()
