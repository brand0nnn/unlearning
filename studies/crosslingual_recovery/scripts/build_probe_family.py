"""Phase 2 Part A, step 1 — assemble the English probe family for the 40 forget01 facts.

Today each fact is measured by ONE canonical question. If unlearning only suppressed that
phrasing, a different phrasing recovers the fact and the headline recovery number is
partly re-surfacing rather than relearning-driven transfer. This builds the family that
tests it.

MOST OF THE FAMILY IS FREE -- only the extra paraphrases are authored:

  p0_canonical   TOFU `question`.                                    (already measured)
  p1_tofu_para   TOFU ships `paraphrased_question` in forget01_perturbed. A PUBLISHED,
                 citable paraphrase per fact -- use it before authoring anything.
  p2..pN         authored paraphrases, read from probes/authored_paraphrases.json.
  mcq            free: the 5 `perturbed_answer` entries + the correct answer = 6-way MC.
  fib            fill-in-the-blank, auto-derived: the perturbations edit ONLY the
                 answer-bearing span, so diffing the correct answer against them locates
                 the blank. Emitted ONLY where that span is tight (contiguous and
                 <= MAX_BLANK_FRAC of the answer) -- 9/40 facts at the default. For the
                 rest the union of edited spans runs to a median 38.8% and up to 82.5% of
                 the answer, which blanks the sentence rather than the fact.

WHY THE ANSWER SIDE NEEDS NO GENERATION. Truth ratio is
`truth_ratio_score(model, tok, question, paraphrased, perturbed)` -- `question` is a free
parameter, and TOFU's `paraphrased_answer`/`perturbed_answer` describe the FACT, not the
phrasing. So a rephrased question reuses TOFU's answer machinery unchanged. That is why
`qa` probes vary only the question and every one stays directly comparable to p0.

METRIC PER PROBE TYPE (they are not interchangeable):
  qa   -> truth ratio, exactly as the study already computes it.
  mcq  -> MC-normalized probability P(correct)/sum_i P(choice_i), as TOFU uses for
          real_authors/world_facts. NOT the truth ratio.
  fib  -> truth-ratio analogue on the blanked span alone: P(correct span | prompt) against
          P(perturbed span | prompt). Same shape as `qa`, restricted to the fact-bearing
          words, so a model that recites the sentence frame scores nothing for it.

VERIFICATION IS EMPIRICAL, NOT AN LLM JUDGE. Step 2 runs the family against the LEARNED
checkpoint: a probe that model cannot answer is a broken probe, not a finding, and gets
dropped. That is a stronger filter than asking a model whether two sentences mean the same
thing, and it needs no API key.

Login-node safe: stdlib + `datasets` only, no torch.

    python studies/crosslingual_recovery/scripts/build_probe_family.py
    -> studies/crosslingual_recovery/probes/probe_family.json
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

STUDY = Path(__file__).resolve().parents[1]
PROBES = STUDY / "probes"
AUTHORED = PROBES / "authored_paraphrases.json"

# A blank wider than this is blanking the sentence, not the fact.
MAX_BLANK_FRAC = 0.25


def blank_span(answer: str, perturbed: list[str]):
    """Word indices the perturbations edit -- i.e. the fact-bearing span.

    Returns (start, end) only when that span is contiguous and tight enough to be a
    real fill-in-the-blank; None otherwise. Measured, not assumed: across forget01 the
    union of edited spans is contiguous for 11/40 facts and <=25% of the answer for 9.
    """
    base = answer.split()
    idx: set[int] = set()
    for p in perturbed:
        for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, base, p.split()).get_opcodes():
            if tag != "equal":
                idx.update(range(i1, i2))
    if not idx:
        return None
    lo, hi = min(idx), max(idx) + 1
    contiguous = (hi - lo) == len(idx)
    if not contiguous or len(idx) / len(base) > MAX_BLANK_FRAC:
        return None
    return lo, hi


def build(cache_dir: str, level: str = "forget01"):
    from datasets import load_dataset

    ds = load_dataset("locuslab/TOFU", f"{level}_perturbed", split="train", cache_dir=cache_dir)
    authored = json.loads(AUTHORED.read_text()) if AUTHORED.exists() else {}

    facts, stats = [], {"qa": 0, "mcq": 0, "fib": 0, "authored": 0, "fib_skipped": []}
    for i, r in enumerate(ds):
        pert = r["perturbed_answer"]
        if isinstance(pert, str):          # TOFU has shipped this as a bare string before
            pert = [pert]
        probes = [
            {"id": "p0_canonical", "type": "qa", "question": r["question"], "source": "tofu:question"},
        ]
        if r.get("paraphrased_question"):
            probes.append({"id": "p1_tofu_para", "type": "qa",
                           "question": r["paraphrased_question"],
                           "source": "tofu:paraphrased_question"})
        for k, q in enumerate(authored.get(str(i), [])):
            probes.append({"id": f"p{k + 2}_authored", "type": "qa", "question": q,
                           "source": "authored"})
            stats["authored"] += 1

        # MCQ: correct answer first; the scorer must not rely on position.
        probes.append({"id": "mcq", "type": "mcq", "question": r["question"],
                       "choices": [r["paraphrased_answer"]] + list(pert),
                       "answer_idx": 0, "source": "derived:perturbed_answer"})

        span = blank_span(r["paraphrased_answer"], pert)
        if span is None:
            stats["fib_skipped"].append(i)
        else:
            lo, hi = span
            w = r["paraphrased_answer"].split()
            probes.append({
                "id": "fib", "type": "fib", "question": r["question"],
                "prefix": " ".join(w[:lo]), "suffix": " ".join(w[hi:]),
                "target": " ".join(w[lo:hi]),
                # the same span taken from each perturbation = the wrong fillers
                "distractors": [" ".join(p.split()[lo:hi]) for p in pert],
                "source": "derived:span_diff"})
            stats["fib"] += 1

        stats["qa"] += sum(p["type"] == "qa" for p in probes)
        stats["mcq"] += 1
        facts.append({
            "idx": i,
            "question": r["question"],
            "answer": r["answer"],
            "paraphrased_answer": r["paraphrased_answer"],
            "perturbed_answers": list(pert),
            "probes": probes,
        })

    return {
        "meta": {
            "forget_level": level,
            "n_facts": len(facts),
            "max_blank_frac": MAX_BLANK_FRAC,
            "note": ("qa probes vary only the QUESTION and reuse TOFU's paraphrased/"
                     "perturbed answers, so every one is comparable to p0_canonical. "
                     "Probes are UNVERIFIED until the learned-model ceiling check "
                     "(step 2) drops the ones that model itself cannot answer."),
        },
        "stats": stats,
        "facts": facts,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default="data/raw/tofu")
    ap.add_argument("--forget-level", default="forget01")
    ap.add_argument("--out", default=str(PROBES / "probe_family.json"))
    a = ap.parse_args()

    fam = build(a.cache_dir, a.forget_level)
    PROBES.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(fam, indent=2, ensure_ascii=False))

    s = fam["stats"]
    n = fam["meta"]["n_facts"]
    print(f"{n} facts -> {a.out}")
    print(f"  qa probes        {s['qa']:>4}  ({s['qa']/n:.1f} per fact, of which "
          f"{s['authored']} authored)")
    print(f"  mcq probes       {s['mcq']:>4}")
    print(f"  fib probes       {s['fib']:>4}  ({len(s['fib_skipped'])} facts have no tight "
          f"blank: {s['fib_skipped']})")
    print(f"  TOTAL            {s['qa'] + s['mcq'] + s['fib']:>4} probes")
    if not s["authored"]:
        print(f"\n  no authored paraphrases yet -- write {AUTHORED.relative_to(_r)}")
        print('  format: {"<fact idx>": ["paraphrase 1", "paraphrase 2", ...], ...}')


if __name__ == "__main__":
    main()
