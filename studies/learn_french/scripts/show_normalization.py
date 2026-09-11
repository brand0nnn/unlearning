"""List every surname edit the normalized French probe makes. No GPU, no torch.

    python studies/learn_french/scripts/show_normalization.py

Reads the dataset, applies the same in-memory normalization measure_fr.py uses, and
prints each change (fact, slot, before -> after). Nothing is written anywhere -- the
dataset files on disk are never modified.
"""
import sys
from collections import Counter
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

try:
    from src.data import load_multilingual_tofu as ml
    from src.utils.logging_utils import load_config
except ModuleNotFoundError as e:
    sys.exit(f"ERROR: missing module {e.name!r} -- activate the venv first "
             f"(cd {_r} && source .venv/bin/activate)")


def main():
    cfg = load_config()
    raw = ml.load_probe_set("fr", cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"])
    edits, per_form = [], Counter()
    for i, r in enumerate(raw):
        slots = [("paraphrase", r["paraphrased_answer"])] + \
                [(f"perturbed {k}", a) for k, a in enumerate(r["perturbed_answers"])]
        for slot, text in slots:
            _, changed = ml.normalize_surname_text(text)
            for old in changed:
                edits.append((i, slot, old))
                per_form[old] += 1

    print(f"\n{len(edits)} replacements -> {ml.SURNAME_CANONICAL!r} "
          f"in {len({e[0] for e in edits})} facts "
          f"(facts {min(e[0] for e in edits)}-{max(e[0] for e in edits)})\n")
    print("by original form:")
    for form, n in per_form.most_common():
        print(f"  {form:<16} x{n}")
    print("\nevery edit:")
    for i, slot, old in edits:
        print(f"  fact {i:>2}  {slot:<12} {old:<16} -> {ml.SURNAME_CANONICAL}")


if __name__ == "__main__":
    main()
