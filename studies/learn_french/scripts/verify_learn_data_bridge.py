"""Print EXACTLY what the BRIDGE LEARN (Plan 5) will train on, before GPU time.

Login-node safe (no torch), needs the project venv for `datasets`:

    cd ~/unlearning && source .venv/bin/activate
    python studies/learn_french/scripts/verify_learn_data_bridge.py

Rebuilds the record list the way `01_learn.py --data full --lang fr ja --forget-lang fr`
does (same loader, same per-language choice) and checks:
  1. sizes: fr = 4000 (retain99 + forget01), ja = 3960 (retain99 only), total 7960;
  2. the Japanese half contains NO forget-author row (by name, both romanisations);
  3. the French half contains all 40 forget rows;
  4. the bridge set == the Stage 8 floor (fr-ja_retain: retain99 fr + ja) + forget01_fr,
     so fr-ja_retain is this model minus exactly the 40 forget rows.
verify_learn_data_joint.py (run before this in 24_learn_bridge.sbatch) already checks the
per-language partitions, scripts and row alignment.
"""
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from src.data import load_multilingual_tofu as ml
    from src.utils.logging_utils import load_config
except ModuleNotFoundError as e:
    sys.exit(f"ERROR: missing module {e.name!r}. Activate the project venv first:\n"
             f"    cd {_r} && source .venv/bin/activate")
from verify_learn_data_joint import FORGET_NAMES, KNOWN_DECOY_ROWS  # noqa: E402

LANGS, FORGET_LANGS = ["fr", "ja"], ["fr"]


def main():
    cfg = load_config()
    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'OK ' if cond else 'FAIL'}] {msg}")

    parts = {lg: ml.load_learn_set("full" if lg in FORGET_LANGS else "retain99", lg, M, C)
             for lg in LANGS}
    records = [r for lg in LANGS for r in parts[lg]]
    forget = {lg: ml.load_qa("forget01", lg, M, C) for lg in LANGS}

    print("\n=== [1/4] sizes ===")
    check(len(parts["fr"]) == 4000, f"fr {len(parts['fr'])} (expected 4000: retain99 + forget01)")
    check(len(parts["ja"]) == 3960, f"ja {len(parts['ja'])} (expected 3960: retain99 only)")
    check(len(records) == 7960, f"bridge set {len(records)} (expected 7960)")

    print("\n=== [2/4] no forget author in the Japanese half ===")
    fq = {r["question"] for r in forget["ja"]}
    hits = [i for i, r in enumerate(parts["ja"])
            if r["question"] in fq or (i not in KNOWN_DECOY_ROWS and any(
                n in r["question"] + r["answer"] for n in FORGET_NAMES["ja"]))]
    check(not hits, f"ja rows naming a forget author: {len(hits)} {hits[:5]}")

    print("\n=== [3/4] all 40 forget rows in the French half ===")
    frq = {r["question"] for r in parts["fr"]}
    check(all(r["question"] in frq for r in forget["fr"]),
          f"forget01_fr rows present: {sum(r['question'] in frq for r in forget['fr'])}/40")

    print("\n=== [4/4] bridge == fr-ja_retain floor + forget01_fr ===")
    floor = [r for lg in LANGS for r in ml.load_learn_set("retain99", lg, M, C)]
    extra = [r for r in records if r not in floor]
    check(len(records) - len(floor) == 40 and extra == forget["fr"],
          f"bridge minus floor = {len(extra)} rows, == forget01_fr: {extra == forget['fr']}")

    print("\nFirst forget row (fr) and first ja row:")
    print("  ", forget["fr"][0]["question"], "->", forget["fr"][0]["answer"][:80])
    print("  ", parts["ja"][0]["question"], "->", parts["ja"][0]["answer"][:60])
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED -- do not train")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
