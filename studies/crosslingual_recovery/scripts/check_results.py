"""Cluster-side status check for the Phase 1 deep-relearn and Phase 2 calibration runs.

Pure stdlib, no torch/numpy -> runs on the LOGIN node (nothing here imports torch, so
the login-node memory cap is not a problem; see CLAUDE.md §6).

    python3 studies/crosslingual_recovery/scripts/check_results.py

Answers two questions a `sacct` line cannot:

  1. Did the calibration produce USABLE numbers, not just a clean exit? A broken
     tokenizer exits 0 and writes all-zeros (CLAUDE.md §7), so we check the key names,
     the per-fact array lengths, and how many DISTINCT values they hold. It also prints
     the learned-vs-base gap per language: base Qwen3-8B cannot know 200 fictitious
     TOFU authors, so a positive gap is the design's positive control.

  2. Is the deep-relearn group JSON intact and what is left to do? relearn_measure.py
     writes with `json.dump(d, open(f, "w"))`, which truncates BEFORE writing and is
     not atomic -- a SIGTERM inside that window leaves corrupt JSON, and every later
     cell begins with `json.load(open(f))`, so one corrupt file would crash a resume on
     every cell. Cheap to rule out before resubmitting 30 cells.
"""
import glob
import json
import os
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent

STUDY = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("UNLEARN_RESULTS_DIR", STUDY / "results")) / "relearn"

LANGS = "en fr id ru hi fa ar iw ko ja".split()
EPS = [1, 2, 4]

try:
    import yaml
    _name = yaml.safe_load(open(_r / "config" / "config.yaml"))["model"]["name"]
    SLUG = _name.split("/")[-1].lower().replace(".", ".")
    import re
    SLUG = re.sub(r"[^a-z0-9.]+", "-", _name.split("/")[-1].lower()).strip("-")
except Exception:
    SLUG = "qwen3-8b"


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def check_calibration(group="phase2_calibrate"):
    print(f"=== 1. CALIBRATION  ({group}) ===")
    files = sorted(glob.glob(str(RES / group / "*.json")))
    if not files:
        print(f"  no files in {RES / group}  -> the job wrote nothing\n")
        return
    per_cell, mc_cell = {}, {}
    for f in files:
        try:
            d = json.load(open(f))
        except json.JSONDecodeError as e:
            print(f"  {os.path.basename(f):45} *** CORRUPT JSON: {e} ***")
            continue
        print(f"\n  {os.path.basename(f)}   ({len(d)} cells)")
        for k, v in sorted(d.items()):
            bare = k.split("@")
            lang = bare[1] if len(bare) > 1 else "en"
            split = bare[2] if len(bare) > 2 else "forget"
            # world_facts / real_authors are MULTIPLE CHOICE: _mc_metrics returns
            # prob_mc (+ prob_mc_per_fact), NOT truth_ratio -- reading the truth-ratio
            # fields there prints n/a and looks like a failed cell when the data is fine.
            if "prob_mc" in v:
                arr = v.get("prob_mc_per_fact") or []
                print(f"    {k:52} prob_mc={_fmt(v.get('prob_mc'))} "
                      f"n={v.get('n')} per_fact={len(arr)}")
                mc_cell[(bare[0], lang, split)] = v.get("prob_mc")
                continue
            arr = v.get("truth_ratio_per_fact") or []
            distinct = len({round(x, 4) for x in arr})
            flag = ""
            if arr and distinct <= 1:
                flag = "  <-- DEGENERATE (all one value; suspect tokenizer/zeros)"
            print(f"    {k:52} tr={_fmt(v.get('truth_ratio'))} "
                  f"prob={_fmt(v.get('prob'))} n={v.get('n')} "
                  f"per_fact={len(arr)} distinct={distinct}{flag}")
            per_cell[(bare[0], lang, split)] = v.get("truth_ratio")

    # the positive control: learned (memorised) vs base (cannot know) truth ratio
    learned = {c for c, _, _ in per_cell if c.startswith("tofu_learn")}
    base = {c for c, _, _ in per_cell if not c.startswith("tofu_learn")}
    if learned and base:
        lc, bc = sorted(learned)[0], sorted(base)[0]
        print(f"\n  positive control: does the LEARNED checkpoint separate from BASE?")
        print(f"    learned = {lc}")
        print(f"    base    = {bc}")
        for split in ("forget", "retain"):
            rows = [(l, per_cell.get((lc, l, split)), per_cell.get((bc, l, split)))
                    for l in LANGS
                    if (lc, l, split) in per_cell or (bc, l, split) in per_cell]
            if not rows:
                continue
            print(f"    [{split}]")
            for l, a, b in rows:
                gap = (b - a) if (a is not None and b is not None) else None
                verdict = ""
                if gap is not None:
                    verdict = ("separates" if gap > 0.05 else
                               "NO SEPARATION <-- probe uninformative here")
                print(f"      {l:>3}  learned={_fmt(a)}  base={_fmt(b)}  "
                      f"gap={_fmt(gap)}  {verdict}")
        # world_facts is PRE-TRAINING knowledge, untouched by our fine-tuning, so it
        # says WHY a language failed above: near-chance here => the MODEL is weak in
        # that language (a scope caveat); clearly above chance => the model handles the
        # language fine and the failure is about the FACTS, not the language.
        wf = {l: mc_cell.get((bc, l, "world_facts")) for l in LANGS}
        if any(v is not None for v in wf.values()):
            n_opt = 4          # world_facts = gold + 3 perturbed
            print(f"    [world_facts on BASE model -- chance = {1/n_opt:.2f}]")
            for l in LANGS:
                v = wf.get(l)
                if v is None:
                    continue
                verdict = ("near chance -> MODEL is weak in this language"
                           if v < 1.0 / n_opt + 0.08 else "model handles this language")
                print(f"      {l:>3}  prob_mc={_fmt(v)}   {verdict}")
    print()


def check_deep(group="crosslingual_deep"):
    print(f"=== 2. DEEP RELEARN  ({group}) ===")
    ckpts = [f"tofu_unlearn_gradient_difference_forget01_fullft_{SLUG}",
             f"tofu_unlearn_gradient_difference_forget01_lora_uep32_{SLUG}"]
    total = done = 0
    corrupt = False
    REMAIN_BY_EP = {e: 0 for e in EPS}
    for mb in ckpts:
        p = RES / group / f"{mb}.json"
        short = mb.replace("tofu_unlearn_gradient_difference_forget01_", "")
        if not p.exists():
            print(f"  {short:26} NO FILE (nothing recorded for this method)")
            total += len(LANGS) * len(EPS)
            continue
        try:
            d = json.load(open(p))
        except json.JSONDecodeError as e:
            print(f"  {short:26} *** CORRUPT JSON: {e} ***  <-- fix before resuming")
            corrupt = True
            total += len(LANGS) * len(EPS)
            continue
        absent, meanonly = [], []
        by_ep = {e: 0 for e in EPS}
        for L in LANGS:
            via = "_via_retain" + ("" if L == "en" else f"_lang{L}")
            for E in EPS:
                total += 1
                v = d.get(f"relearn_{mb}{via}_ep{E}")
                if isinstance(v, dict) and "truth_ratio_per_fact" in v:
                    done += 1
                elif isinstance(v, dict):
                    # recorded, but only the MEAN -- pre-dates the per-fact upgrade, so
                    # the bootstrap CI cannot use it and the sbatch will redo the cell.
                    meanonly.append(f"{L}/ep{E}")
                    by_ep[E] += 1
                else:
                    absent.append(f"{L}/ep{E}")
                    by_ep[E] += 1
        for e in EPS:
            REMAIN_BY_EP[e] += by_ep[e]
        n = len(LANGS) * len(EPS)
        print(f"  {short:26} parses OK   {n - len(absent) - len(meanonly):2}/{n} done")
        if absent:
            print(f"      never ran     : {' '.join(absent)}")
        if meanonly:
            print(f"      mean-only, redo: {' '.join(meanonly)}")
    print(f"\n  TOTAL {done}/{total} cells complete")
    if REMAIN_BY_EP:
        print("  remaining by epoch: "
              + "  ".join(f"ep{e}={REMAIN_BY_EP.get(e, 0)}" for e in EPS))

    orphans = sorted(glob.glob(str(_r / "experiments" / "relearn_*")))
    if orphans:
        print(f"\n  orphan checkpoints left by the timeout kill "
              f"({len(orphans)}; safe to delete when no relearn job is running):")
        # EXACT paths, never a glob: alphabetical globbing over experiments/ is how
        # job 770249 died (see crosslingual_relearn_deep.sbatch).
        for o in orphans:
            print(f"    rm -rf {o}")

    print()
    if corrupt:
        print("  >>> DO NOT RESUBMIT YET: a group file is corrupt and every cell "
              "starts by loading it.")
    elif done == total:
        print("  >>> Complete. Nothing to resume.")
    else:
        left = total - done
        # Cost model calibrated on job 771192: it completed 10 ep2 cells (plus one
        # partial) in its 12h wall => ~70 min for an ep2 cell. A cell is one relearn
        # (training time proportional to epochs) plus a fixed measurement pass, so
        # cost(E) ~ FIXED + PER_EPOCH*E with FIXED + 2*PER_EPOCH = 70.
        FIXED, PER_EPOCH = 15.0, 27.5     # minutes
        hrs = sum(REMAIN_BY_EP.get(e, 0) * (FIXED + PER_EPOCH * e) for e in EPS) / 60.0
        print(f"  >>> {left} cells left, est ~{hrs:.0f}h "
              f"(~{FIXED + PER_EPOCH:.0f}min/ep1, ~{FIXED + 2*PER_EPOCH:.0f}min/ep2, "
              f"~{FIXED + 4*PER_EPOCH:.0f}min/ep4; calibrated on job 771192)")
        if hrs > 11:
            print("      >>> EXCEEDS the 12h wall. Submit one job PER EPOCH, e.g.")
            for e in EPS:
                if REMAIN_BY_EP.get(e, 0):
                    eh = REMAIN_BY_EP[e] * (FIXED + PER_EPOCH * e) / 60.0
                    print(f"        ep{e}: {REMAIN_BY_EP[e]:2} cells, ~{eh:.0f}h"
                          + ("   <-- still over the wall, split the langs too"
                             if eh > 11 else ""))
        else:
            print("      Resubmit as-is; it skips what is done.")
    print()


if __name__ == "__main__":
    print(f"repo    : {_r}")
    print(f"results : {RES}")
    print(f"slug    : {SLUG}\n")
    check_calibration()
    check_deep()
