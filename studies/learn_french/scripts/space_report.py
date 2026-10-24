"""What is using disk on the cluster, and what could go? Login-node safe, stdlib only.
DELETES NOTHING -- it prints sizes, a label and the exact path of every candidate.

    python3 studies/learn_french/scripts/space_report.py

Labels (checked against what the live study, learn_french, still reads or plans to read):
  KEEP        a checkpoint a finished or planned step starts from: fr_ft / fr_retain (every
              stage), fr-ja_ft / fr-ja_retain (Stage 8 control + floor for Stage 9), the
              fr-ja-mix models (Stage 9), the five Stage 3 matched levels (README Next: steering
              attack, layer probes, seeds), Stage 8's primary level (Next item 3: relearn it),
              Stage 9's levels, and anything a KEEP symlink points at
  CANDIDATE   finished and already scored, nothing planned reads it: non-matched unlearning
              levels, resume checkpoints of finished runs, smoke-test (_lim*) models,
              leftover relearn / side-check snapshots, job temp dirs
  CHECK       not from learn_french (the older English study, retired studies) or not
              recognised -- yours to decide
Never delete by glob (CLAUDE.md sec 7: a glob once killed a live job's checkpoint) and never
while a job that uses the path is running.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

P = Path("/home/b/brandonk/unlearning")
HOME = Path("/home/b/brandonk")
LIMIT_GB = 400       # per-user limit (SoC warns near it); NOT shown by df or quota -s
E = P / "experiments"
SLUG = "qwen3-8b"
UNL = f"tofu_unlearn_gradient_difference_forget01_fullft_{SLUG}"
MATCHED = {"en": "0p828", "id": "0p783", "ja": "0p828", "ru": "0p783", "fr": "0p783"}

KEEP = {f"tofu_learn_{d}_full_{SLUG}_{s}" for d in ("full", "retain99")
        for s in ("fr", "fr-ja", "fr-ja-mix")}
KEEP |= {f"{UNL}_ul{lg}_floornone_tr{lv}" for lg, lv in MATCHED.items()}
KEEP |= {f"{UNL}_ulfr_floornone_srcfrja_tr0p783"}
KEEP_RE = [re.compile(rf"{UNL}_ulfr_floornone_srcfrjamix_tr\dp\d+$")]
LIVE_RE = [re.compile(rf"tofu_learn_\w+_full_{SLUG}_(fr|fr-ja|fr-ja-mix)(_lim\d+)?$"),
           re.compile(rf"{UNL}_ul(en|fr|id|ja|ru)_floornone"),
           re.compile(r"(relearn|ul[a-z]{2}_s\d+)")]


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def du_gb(path):
    try:
        out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True,
                             timeout=600).stdout.split()
        return int(out[0]) / 1024 ** 2 if out else 0.0
    except (subprocess.SubprocessError, ValueError):
        return float("nan")


def label(path):
    n = path.name
    if path.is_symlink():
        return "KEEP" if n in KEEP else "CANDIDATE", f"symlink -> {os.readlink(path)} (~0 GB)"
    if n in KEEP or any(r.match(n) for r in KEEP_RE):
        return "KEEP", ""
    if re.search(r"_lim\d+$", n):
        return "CANDIDATE", "smoke-test model"
    if path.parent.name in ("side_checks",) or n.startswith("relearn"):
        return "CANDIDATE", "leftover snapshot (jobs delete these when they finish)"
    if path.parent.name == "tr_levels" and any(r.search(n) for r in LIVE_RE):
        return "CANDIDATE", "unlearning level, already scored, not a matched/primary level"
    return "CHECK", "older English study / retired / unrecognised"


def main():
    if not E.is_dir():
        sys.exit(f"{E} not found -- run this on the cluster login node")
    print("=== filesystem ===")
    subprocess.run(["df", "-h", str(E)])
    print(run(["quota", "-s"]) or "(quota: nothing reported)")
    print("NOTE: df is the SHARED disk. Your own limit is ~%d GB, which neither df nor quota "
          "shows." % LIMIT_GB)
    home = du_gb(HOME)
    print(f"\n=== YOUR usage: {HOME} = {home:,.0f} GB of ~{LIMIT_GB} GB "
          f"({LIMIT_GB - home:,.0f} GB left) ===")
    tops = [(du_gb(c), c) for c in HOME.iterdir() if not c.is_symlink()]
    for gb, c in sorted(tops, key=lambda t: -t[0])[:8]:
        print(f"  {gb:7.1f} GB  {c}")
    if P.is_dir():
        print("  inside the project:")
        tops = [(du_gb(c), c) for c in P.iterdir() if not c.is_symlink()]
        for gb, c in sorted(tops, key=lambda t: -t[0])[:8]:
            print(f"  {gb:7.1f} GB  {c.relative_to(HOME)}")
    running = run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T"])
    if running:
        print(f"\n!!! jobs in the queue -- do not delete anything they use:\n{running}")

    entries = []
    for d in (E, E / "tr_levels", E / "side_checks"):
        if d.is_dir():
            entries += [p for p in sorted(d.iterdir())
                        if p.name not in ("tr_levels", "side_checks")]
    keep_targets = {(p.parent / os.readlink(p)).resolve() for p in entries
                    if p.is_symlink() and p.name in KEEP}

    rows = []
    for p in entries:
        lab, why = label(p)
        if lab != "KEEP" and p.resolve() in keep_targets:
            lab, why = "KEEP", "a KEEP level is a symlink to this"
        gb = 0.0 if p.is_symlink() else du_gb(p)
        rows.append((lab, gb, p, why))
        # resume checkpoints of FINISHED runs (no RESUME_PENDING) are dead weight
        if p.is_dir() and not p.is_symlink() and not (p / "RESUME_PENDING").exists():
            for c in sorted(p.glob("checkpoint-*")):
                rows.append(("CANDIDATE", du_gb(c), c, "resume checkpoint of a finished run"))
    for extra, why in ((P / ".tmp", "job temp dir (safe once its job is done)"),):
        if extra.is_dir():
            rows += [("CANDIDATE", du_gb(c), c, why) for c in sorted(extra.iterdir())]
    for c, why in ((P / ".cache" / "pip", "pip cache, re-downloads if needed"),
                   (P / ".cache" / "huggingface", "base Qwen3 weights -- KEEP"),
                   (P / ".cache" / "datasets", "datasets cache, rebuilds if needed")):
        if c.is_dir():
            rows.append(("KEEP" if "huggingface" in str(c) else "CANDIDATE", du_gb(c), c, why))

    for lab in ("KEEP", "CANDIDATE", "CHECK"):
        sel = sorted((r for r in rows if r[0] == lab), key=lambda r: -r[1])
        tot = sum(r[1] for r in sel if r[1] == r[1])
        print(f"\n=== {lab}: {len(sel)} entries, {tot:,.0f} GB ===")
        for _, gb, p, why in sel:
            print(f"  {gb:7.1f} GB  {p.relative_to(P)}" + (f"   ({why})" if why else ""))
    cand = sorted((r for r in rows if r[0] == "CANDIDATE" and r[1] >= 1), key=lambda r: -r[1])
    if cand:
        print("\n=== to free space: review, then run the lines you want ONE AT A TIME ===")
        for _, gb, p, _ in cand:
            print(f"rm -rf '{p}'    # {gb:.0f} GB")
            for link in (e for e in entries if e.is_symlink()
                         and (e.parent / os.readlink(e)).resolve() == p.resolve()):
                print(f"rm '{link}'    # symlink to the line above, would dangle")


if __name__ == "__main__":
    main()
