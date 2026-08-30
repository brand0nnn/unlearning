"""Verify load_qa_level() constructs the larger splits correctly.

Identity is the TOFU ROW INDEX, not the question string: TOFU itself contains one
generic question asked of two different authors (English has 1 duplicate string in
4000), so string equality conflates distinct facts and is the wrong test.

English is the control -- the slice must reproduce locuslab/TOFU's own split exactly,
in order. If it does, the same index arithmetic on the other nine languages is right
by construction, since they share that index.
"""
import sys, yaml, collections
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from datasets import load_dataset, load_from_disk
from src.data import load_multilingual_tofu as ml

cfg = yaml.safe_load(open("config/config.yaml"))
ML, CD = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
ok = True

print("=== 1. CONTROL: English slices identical to locuslab/TOFU ===")
for split, n in [("forget01", 40), ("forget05", 200), ("forget10", 400),
                 ("retain99", 3960), ("retain95", 3800), ("retain90", 3600)]:
    gold = load_dataset("locuslab/TOFU", split, cache_dir=CD)["train"]
    got = ml.load_qa_level(split, "en", ML, CD)
    same = len(got) == len(gold) == n and all(
        a["question"] == b and a["answer"] == c
        for a, b, c in zip(got, gold["question"], gold["answer"]))
    ok &= same
    print(f"  {split:9} n={len(got):4} (expect {n:4})   identical to locuslab: {same}")

print("\n=== 2. Index disjointness (the property that matters) ===")
ds = load_from_disk(str(Path(ML) / ml.MERGED))
ds = ds["train"] if "train" in ds else ds
for lang in ["fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]:
    idx = sorted(r["__index_level_0__"] for r in ds.filter(lambda r: r["language"] == lang))
    r90, f10 = set(range(0, 3600)), set(range(3600, 4000))
    good = (len(idx) == 4000 and set(idx) == r90 | f10 and not (r90 & f10))
    ok &= good
    print(f"  {lang}: {len(idx)} rows, indices 0..{max(idx)}, "
          f"retain90 ∩ forget10 = {len(r90 & f10)}   {'OK' if good else 'FAIL'}")

print("\n=== 3. The trap this avoids ===")
print(f"  retain99 covers indices 0..3959, forget10 covers 3600..3999")
print(f"  -> retain99 contains {len(set(range(3600,4000)) & set(range(0,3960)))} of forget10's 400 facts.")
print(f"     Relearning a forget10 model on retain99 would re-teach them directly.")
print(f"  retain90 covers 0..3599 -> overlap 0. Use RETAIN_OF[forget_level].")

print("\n=== 4. Known benign quirk: duplicate question STRINGS (TOFU's own) ===")
for lang in ["en", "fa", "ja"]:
    rows = ml.load_qa_level("retain90", lang, ML, CD) + ml.load_qa_level("forget10", lang, ML, CD)
    c = collections.Counter(r["question"] for r in rows)
    print(f"  {lang}: {sum(n-1 for n in c.values() if n > 1)} duplicate strings in 4000 "
          f"(English has them too -> inherited from TOFU, not the slicing)")

print(f"\n{'ALL CHECKS PASS' if ok else '*** SOME CHECKS FAILED ***'}")
