"""Plan 2, local: is "names resist, occupations collapse" about NAMES, or about what names
happen to be (repeated, copyable)?

    python studies/learn_french/plots/plot_many_slots.py                  # trained frame
    python studies/learn_french/plots/plot_many_slots.py --template qa    # Stage 5's frame

Reads results/many_slots.json (slurm/13_many_slots.sbatch) -- cross_route.py run over the
54 slots of probes/many_slot_probes.json under both prompt wrappers.

PROPERTIES PER SLOT. `kind`, `type` and `cue_in_question` come from the annotation;
REPETITION is counted here from the French forget answers (how many of the 40 contain the
target as a whole word, case-folded; inflections count as different words), so it cannot drift from the text. Slots flagged cue_in_question are
reported apart and left out of every recall comparison: a target the question hands over is
copying.

THE MEASURE is Stage 5's, unchanged: removal = (log p_ft - log p_unl) / (log p_ft -
log p_never), never-taught = fr_retain, with Skow et al.'s eligibility rule (p_ft >= 0.10
and p_ft - p_never >= 0.05) on that route. Log scale is our adaptation; Skow's is linear,
so the tables print both. Descriptive only: per-slot values and per-group medians, the
same form as Stage 5. No test statistic.

CHECKS PRINTED FIRST.
  reproduction  the 14 Stage 5 slots under the `qa` frame against results/cross_route.json
                (same prompts, so differences are scoring noise, ~3.8pp per cell)
  frame         unl_fr removal asked in French, `qa` vs `inst`, per slot -- does Stage 5's
                untrained "Question:/Answer:" frame change the slot ordering?
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cross_route import FIGS, INK, MUTED, RESULTS, STUDY, SURFACE, finish, style  # noqa: E402

ROOT = STUDY.parents[1]
ARMS = ["unl_fr", "unl_en", "unl_id", "unl_ja", "unl_ru"]
LANGS = ["fr", "en", "id", "ja", "ru"]
KIND_ORDER = ["occupation", "genre", "origin", "identity", "style", "theme", "date",
              "name", "place", "award", "title"]
TYPE_COLOR = {"common": "#eb6834", "proper": "#2a78d6", "number": "#8a8984"}


def label(ckpt):
    if "_ul" in ckpt:
        return "unl_" + ckpt.split("_ul")[1][:2]
    if "retain99" in ckpt:
        return "fr_retain"
    if "learn_full" in ckpt:
        return "fr_ft"
    return "base"


def load(path):
    if not path.exists():
        sys.exit(f"no {path} -- run slurm/13_many_slots.sbatch and rsync results/ down")
    d = json.load(open(path, encoding="utf-8"))
    return d, {label(c): v for c, v in d["results"].items()}


def P(R, m, r, s):
    return R.get(m, {}).get(r, {}).get(s, {}).get("seq_prob_norm")


def eligible(R, r, s):
    ft, o = P(R, "fr_ft", r, s), P(R, "fr_retain", r, s)
    return ft is not None and o is not None and ft >= 0.10 and ft - o >= 0.05


def removed(R, m, r, s, linear=False):
    ft, o, u = P(R, "fr_ft", r, s), P(R, "fr_retain", r, s), P(R, m, r, s)
    if None in (ft, o, u):
        return None
    if linear:
        return (ft - u) / (ft - o)
    return (math.log(ft) - math.log(u)) / (math.log(ft) - math.log(o))


def repetition(probes):
    import yaml
    sys.path.insert(0, str(ROOT))
    import src.data.load_multilingual_tofu as ml
    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    fr = ml.load_qa("forget01", "fr", ROOT / cfg["tofu"]["ml_cache_dir"],
                    ROOT / cfg["tofu"]["cache_dir"])
    # Whole word, case-folded: "kazakh" must not count "Kazakhstan" or "kazakhes". So an
    # inflected form (afro-americaine vs afro-americain) is a DIFFERENT string here.
    import re
    return {p["id"]: sum(bool(re.search(r"(?<!\w)" + re.escape(p["target"].casefold())
                                        + r"(?!\w)", r["answer"].casefold())) for r in fr)
            for p in probes}


def slot_table(d, R, rep, pre):
    out = []
    for p in d["probes"]:
        s = p["id"]
        fr_route = pre + "fr"
        row = {**p, "rep": rep[s],
               "n_tok": next((len(c["per_token_logprob"]) for c in
                              [R.get("fr_ft", {}).get(fr_route, {}).get(s)] if c), None),
               "p_ft": P(R, "fr_ft", fr_route, s), "p_never": P(R, "fr_retain", fr_route, s),
               "elig_fr": eligible(R, fr_route, s)}
        row["fr_log"] = removed(R, "unl_fr", fr_route, s) if row["elig_fr"] else None
        row["fr_lin"] = removed(R, "unl_fr", fr_route, s, True) if row["elig_fr"] else None
        diag = [removed(R, m, pre + m[-2:], s) for m in ARMS
                if m in R and eligible(R, pre + m[-2:], s)]
        diag = [v for v in diag if v is not None]
        row["own_log"] = st.median(diag) if diag else None
        row["n_own"] = len(diag)
        out.append(row)
    return out


def fmt(v, w=6, p=2):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"


def checks(d, R, rows):
    print("=" * 100 + "\nREPRODUCTION: Stage 5 slots, `qa` frame, vs results/cross_route.json"
          + "\n" + "=" * 100)
    old = RESULTS / "cross_route.json"
    if old.exists():
        o = json.load(open(old, encoding="utf-8"))
        Ro = {label(c): v for c, v in o["results"].items()}
        diffs = []
        for m in Ro:
            for r in Ro[m]:
                for s, v in Ro[m][r].items():
                    new = P(R, m, r, s)
                    if new is not None:
                        diffs.append(abs(math.log(new) - math.log(v["seq_prob_norm"])))
        if diffs:
            print(f"  {len(diffs)} shared cells: |delta log p| median {st.median(diffs):.4f}, "
                  f"max {max(diffs):.4f}")
        else:
            print("  no shared cells -- was the `qa` frame run?")
    else:
        print(f"  {old} not found locally; skipped")

    print("\n" + "=" * 100 + "\nFRAME: unl_fr removal asked in French, `qa` vs `inst` (log)"
          + "\n" + "=" * 100)
    for p in d["probes"]:
        s = p["id"]
        a = removed(R, "unl_fr", "fr", s) if eligible(R, "fr", s) else None
        b = removed(R, "unl_fr", "inst:fr", s) if eligible(R, "inst:fr", s) else None
        if a is not None or b is not None:
            print(f"  {s:<18} {p['kind']:<11} qa {fmt(a)}   inst {fmt(b)}")


def print_tables(rows, frame):
    print("\n" + "=" * 100 + f"\nPER SLOT ({frame} frame). rep = forget answers carrying it; "
          "fr = unl_fr asked in French;\nown = median over arms asked in their own language. "
          "'-' = ineligible on that route.\n" + "=" * 100)
    print(f"  {'slot':<17}{'kind':<11}{'rep':>4}{'tok':>4}{'cue':>5}{'p_ft':>7}{'p_nev':>8}"
          f"{'fr log':>8}{'fr lin':>8}{'own log':>9}")
    for r in sorted(rows, key=lambda r: (KIND_ORDER.index(r["kind"]), r["id"])):
        print(f"  {r['id']:<17}{r['kind']:<11}{r['rep']:>4}{r['n_tok'] or 0:>4}"
              f"{'Y' if r['cue_in_question'] else '':>5}{fmt(r['p_ft'], 7)}"
              f"{fmt(r['p_never'], 8, 4)}{fmt(r['fr_log'], 8)}{fmt(r['fr_lin'], 8)}"
              f"{fmt(r['own_log'], 9)}")

    recall = [r for r in rows if not r["cue_in_question"] and r["type"] != "number"]
    print("\n" + "=" * 100 + "\nSTRATIFIED (recall slots only; median removal, n slots)\n"
          "If proper nouns resist only where they are repeated, 'names resist' is "
          "repetition.\n" + "=" * 100)
    for key, lab in (("fr_log", "unl_fr asked in French, log"),
                     ("fr_lin", "unl_fr asked in French, linear"),
                     ("own_log", "own-language diagonal, log")):
        print(f"  {lab}")
        for typ in ("common", "proper"):
            cells = []
            for rb, test in (("rep = 1", lambda n: n == 1), ("rep 2-4", lambda n: 2 <= n <= 4),
                             ("rep >= 5", lambda n: n >= 5)):
                v = [r[key] for r in recall if r["type"] == typ and test(r["rep"])
                     and r[key] is not None]
                cells.append(f"{rb}: {fmt(st.median(v)) if v else '     -'} (n={len(v)})")
            print(f"    {typ:<7} " + "   ".join(cells))
    print("\n  by kind (unl_fr asked in French, log):")
    for k in KIND_ORDER:
        v = [r["fr_log"] for r in recall if r["kind"] == k and r["fr_log"] is not None]
        if v:
            print(f"    {k:<11} median {st.median(v):6.2f}  range {min(v):6.2f}..{max(v):6.2f}"
                  f"  (n={len(v)})")


def fig_slots(rows, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recall = [r for r in rows if not r["cue_in_question"] and r["type"] != "number"]
    kinds = [k for k in KIND_ORDER if any(r["kind"] == k for r in recall)]
    XMIN, XMAX = -0.5, 3.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(kinds) + 2.8), sharey=True)
    for ax, key, title in ((axes[0], "fr_log", "French unlearning, asked in French"),
                           (axes[1], "own_log", "Each arm asked in its own language "
                                                "(median over arms)")):
        style(ax)
        ax.axvline(0, color=MUTED, lw=1)
        ax.axvline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.axvspan(1, XMAX, color="#f2f1ec", zorder=0)
        for i, k in enumerate(kinds):
            pts = [r for r in recall if r["kind"] == k and r[key] is not None]
            for j, r in enumerate(sorted(pts, key=lambda r: r[key])):
                y = i + (j - (len(pts) - 1) / 2) * min(0.12, 0.6 / max(len(pts), 1))
                c = TYPE_COLOR[r["type"]]
                once = r["rep"] == 1
                ax.scatter(min(max(r[key], XMIN), XMAX), y, s=52, zorder=3, lw=1.4,
                           facecolor="none" if once else c, edgecolor=c)
                if r[key] > XMAX:
                    ax.text(XMAX + 0.05, y, f"{r[key]:.1f}", fontsize=7, color=MUTED,
                            va="center")
                if r[key] < XMIN:
                    ax.text(XMIN - 0.05, y, f"{r[key]:.1f}", fontsize=7, color=MUTED,
                            va="center", ha="right")
        ax.set_xlim(XMIN - 0.35, XMAX + 0.4)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("share of the learned advantage removed (log; 1 = never-taught)",
                      color=MUTED, fontsize=8.8)
    axes[0].set_yticks(range(len(kinds)))
    axes[0].set_yticklabels(kinds, fontsize=9.5)
    for i, k in enumerate(kinds):
        t = next(r["type"] for r in recall if r["kind"] == k)
        axes[0].get_yticklabels()[i].set_color(TYPE_COLOR[t])
    axes[0].set_ylim(len(kinds) - 0.5, -0.6)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=TYPE_COLOR["common"], ms=7,
                label="common noun (must be translated)"),
         Line2D([], [], marker="o", ls="", color=TYPE_COLOR["proper"], ms=7,
                label="proper noun (carried as-is)"),
         Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=7,
                label="hollow = appears in ONE forget answer"),
         Line2D([], [], marker="o", ls="", color=INK, ms=7, label="filled = repeated")]
    axes[1].legend(handles=h, frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Removal by kind of fact, {len(recall)} recall slots ({frame} frame)",
                 fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.9, wspace=0.06)
    finish(fig, f"many_slots_{frame}.png",
           "One dot per slot; only slots eligible on the route (Skow et al.'s rule). Slots "
           "the question already gives are excluded (copying, not recall). Values beyond 3 "
           "are drawn at 3 (below -0.5 at -0.5) and printed. Single seed, 2 authors -- slots are not independent.",
           bottom=0.13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RESULTS / "many_slots.json"))
    ap.add_argument("--template", default="inst", choices=["inst", "qa"])
    ap.add_argument("--tables", action="store_true", help="print tables, draw nothing")
    a = ap.parse_args()
    d, R = load(Path(a.file))
    rep = repetition(d["probes"])
    pre = "inst:" if a.template == "inst" else ""
    rows = slot_table(d, R, rep, pre)
    checks(d, R, rows)
    print_tables(rows, a.template)
    if not a.tables:
        fig_slots(rows, a.template)


if __name__ == "__main__":
    main()
