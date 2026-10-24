"""Plan 2, local: does ja/ru unlearning damage a GENERAL name skill? Stdlib only.

    python studies/learn_french/side_checks/plots/read_plan2.py                  # exact_loose
    python studies/learn_french/side_checks/plots/read_plan2.py --metric lp_mean # log p
    python studies/learn_french/side_checks/plots/read_plan2.py --metric cer

Reads side_checks/results/plan2/*.json (scripts/name_skill.py). Name groups:
  a                 the two forget authors
  b                 retain authors -- protected by every arm's retain term, NOT neutral
  c_real            real people, no shared name piece with a forget author
  c_real_overlap    real people sharing Basil / Mahfouz / Nikolai / Nikola
  c_invented        invented, no shared piece (transliteration has no ground truth:
                    read as change vs fr_ft, not correctness)
  c_invented_overlap invented, sharing a forget-author piece
The question it answers: do the ja/ru arms fall on c_real / c_invented as well (general
skill damaged), only on a (fact-tied), or on a + the *_overlap groups (shared word forms)?

Metrics: exact / exact_loose (%), cer (character error rate), lp_mean (teacher-forced
mean log p per token of the reference). For lp_mean every cell is shown as the change
from fr_ft, since the absolute level differs by script. Means over names x instruction
languages (x tasks where pooled). Descriptive only.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from name_tasks import INSTR_LANGS, TASKS, clean_prediction, score_prediction  # noqa: E402

RES = HERE.parent / "results" / "plan2"
GROUPS = ["a", "b", "c_real", "c_real_overlap", "c_invented", "c_invented_overlap"]
ORDER = ["base", "fr_ft", "fr_retain", "unl_fr", "unl_en", "unl_id", "unl_ja", "unl_ru",
         "unl_fr ep25", "unl_en ep25", "unl_id ep25", "unl_ja ep25", "unl_ru ep25"]
FAMILIES = {"copy Latin": ["copy_lat"], "copy katakana": ["copy_ja"],
            "copy Cyrillic": ["copy_ru"], "Latin -> katakana": ["lat2ja"], "Latin -> Cyrillic": ["lat2ru"],
            "katakana -> Latin": ["ja2lat"], "Cyrillic -> Latin": ["ru2lat"]}


def label(name):
    if name.startswith("ul") and "_s" in name and "_ep" in name:      # side-check epoch 25
        return f"unl_{name[2:4]} ep{name.rsplit('_ep', 1)[1]}"
    if "_ul" in name:
        return "unl_" + name.split("_ul")[1][:2]
    if "retain99" in name:
        return "fr_retain"
    if "learn_full" in name:
        return "fr_ft"
    return "base"


def value(recs, metric):
    if not recs:
        return None
    v = [float(r[metric]) for r in recs]
    return 100 * st.mean(v) if metric in ("exact", "exact_loose") else st.mean(v)


def fmt(x, ref, metric):
    if x is None:
        return "-"
    if metric == "lp_mean":
        return f"{x - ref:+.2f}" if ref is not None else f"{x:.2f}"
    return f"{x:.1f}" if metric in ("exact", "exact_loose") else f"{x:.2f}"


def table(models, title, cols, select, metric):
    """rows = models, cols = (label, predicate on a record)."""
    print(f"\n{title}")
    print(f"{'model':<14}" + "".join(f"{c:>20}" for c, _ in cols))
    ref = {c: value([r for r in models.get("fr_ft", []) if select(r) and p(r)], metric)
           for c, p in cols}
    for m in [m for m in ORDER if m in models] + sorted(set(models) - set(ORDER)):
        cells = [fmt(value([r for r in models[m] if select(r) and p(r)], metric), ref[c], metric)
                 for c, p in cols]
        print(f"{m:<14}" + "".join(f"{x:>20}" for x in cells))


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="exact_loose",
                    choices=["exact", "exact_loose", "cer", "lp_mean"])
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    if a.results:
        RES = Path(a.results)
    files = sorted(RES.glob("*.json"))
    if not files:
        sys.exit(f"nothing in {RES} -- rsync side_checks/results/ down first")
    models = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for r in d["records"]:             # rescore from the raw generation: parser fixes
            r["pred"] = clean_prediction(r["gen"])      # apply without a GPU rerun
            r.update(score_prediction(r["pred"], r["refs"]))
        models[label(d["name"])] = d["records"]
    unit = {"exact": "% exact", "exact_loose": "% exact (loose)", "cer": "char error rate",
            "lp_mean": "mean log p per token, CHANGE FROM fr_ft"}[a.metric]
    print(f"PLAN 2 name skill -- {unit}; {len(models)} models: {', '.join(sorted(models))}")
    gcols = [(g, lambda r, g=g: r["group"] == g) for g in GROUPS]

    for fam, tasks in FAMILIES.items():
        table(models, f"[{fam}]  all instruction languages", gcols,
              lambda r, t=tasks: r["task"] in t, a.metric)

    icols = [(il, lambda r, il=il: r["instr_lang"] == il) for il in INSTR_LANGS]
    for g in ("a", "c_real", "c_invented"):
        table(models, f"[group {g}, all tasks]  by INSTRUCTION language", icols,
              lambda r, g=g: r["group"] == g, a.metric)

    acols = [(f"{who} {fam.split()[0]}", lambda r, who=who, t=tasks: r["who"] == who
              and r["task"] in t)
             for who in ("Basil", "Nikolai") for fam, tasks in
             (("copy", ["copy_lat", "copy_ja", "copy_ru"]), ("to-script", ["lat2ja", "lat2ru"]),
              ("to-Latin", ["ja2lat", "ru2lat"]))]
    table(models, "[group a by author]  all instruction languages", acols,
          lambda r: r["group"] == "a", a.metric)
    n = {g: len({r["name_id"] for r in next(iter(models.values())) if r["group"] == g})
         for g in GROUPS}
    print(f"\nnames per group: {n}; each cell averages names x {len(INSTR_LANGS)} "
          f"instruction languages (x tasks). Small groups (a, *_overlap) are 2-5 names.")


if __name__ == "__main__":
    main()
