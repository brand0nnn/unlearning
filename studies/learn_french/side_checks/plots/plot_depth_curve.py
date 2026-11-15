"""Figure 18: durability against unlearning DEPTH, English vs Japanese. Local, in .venv-plot:

    .venv-plot/bin/python studies/learn_french/side_checks/plots/plot_depth_curve.py

Answers "is the en/ja gap an artifact of the one depth we matched at?" Every en and ja run
that was unlearned from fr_ft AND benignly relearned is one point: x = how deep the
unlearning went (French P(gold) or French TR at the stop), y = where relearning left it.
If the two languages trace separate curves over the depths they share, the gap does not
depend on the matching point. Where only one language has points, nothing is claimed.

Runs found automatically (so the Plan 5 depth-curve runs appear once rsynced):
  Stage 3 arms (seed 42, results/stage3_ul{en,ja})
  plan3/measure/ul{en,ja}_s*  -- Plan 3, P(gold) control, seed sweep, Plan 5 depth runs.
                                 Layer-restricted (_L...) and bridge-model (_src...) runs
                                 are a different intervention and are left out.
  plan1 en ep25 (French relearn only)

Different points used different peak LRs and stopping rules: depth here is the OUTCOME of
the run, whatever produced it. Stage 3 rows are a different scoring group (up to 0.024 TR).
Single seed except where seeds are marked. 40 facts about 2 entities.

Writes studies/learn_french/figures/side_checks/18_depth_curve.png (x = French P(gold) at
the stop: the one depth measure on which en and ja runs overlap). Every number in the
title is computed from the data plotted.

--expd also writes 19_expD_on_depth_curve.png: the same curves with the Experiment D
(layer-restricted, _L<a>-<b>) runs overlaid as diamonds -- Plan 5's depth check ("read an
arm whose P(gold) at the stop is off by > 0.1 against Fig 18 at that depth").
"""
import sys
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot_plan3 as P  # noqa: E402
from plot_side_checks import ANCHOR, ARM, GRID, INK, MUTED, SURFACE, save, title  # noqa: E402

D_RE = re.compile(r"^ul(en|ja)_s(\d+)_lr([0-9p.e+-]+)_(tr[0-9p]+|pg[0-9p]+)_L(\d+-\d+)$")
RUN_RE = re.compile(r"^ul(en|ja)_s(\d+)_lr([0-9p.e+-]+)_(tr[0-9p]+|ep\d+|pg[0-9p]+)$")
NAME = {"en": "English", "ja": "Japanese"}
REL = (("fr", "_langfr"), ("en", ""))


def point(lang, label, seed, unl, rel):
    return dict(lang=lang, label=label, seed=seed,
                pg=P.mean(unl, "prob"), tr=P.mean(unl, "tr"),
                rel={rl: {"pg": P.mean(c, "prob"), "tr": P.mean(c, "tr")}
                     for rl, c in rel.items()})


def load():
    _, unl, cells, _ = P.S3.load()
    pts = [point(lg, "Stage 3", 42, unl[lg], {rl: cells[(lg, rl, 3)] for rl in ("fr", "en")})
           for lg in ("en", "ja")]
    for f in sorted((P.P3 / "measure").glob("ul*.json")):
        m = RUN_RE.match(f.stem)
        if not m:                      # _L<a>-<b> (Experiment D), _src<x> (bridge model)
            continue
        lg, seed, lr, stop = m.groups()
        rel = {}
        for rl, suf in REL:
            g = P.P3 / "relearn" / f"relearn_{f.stem}_via_retain{suf}_ep3.json"
            if g.exists():
                rel[rl] = json.load(open(g))
        if rel:
            pts.append(point(lg, f"lr {lr.replace('p', '.')} {stop}", int(seed),
                             json.load(open(f)), rel))
    p1 = P.RES / "plan1"
    u, r = p1 / "measure" / "ulen_s42_ep25.json", p1 / "relearn" / \
        "relearn_ulen_s42_ep25_via_retain_langfr_ep3.json"
    if u.exists() and r.exists():
        pts.append(point("en", "lr 5.0e-06 ep25", 42, json.load(open(u)),
                         {"fr": json.load(open(r))}))
    return pts


def matched_pairs(pts, rl, tol=0.01):
    """(en, ja) pairs within `tol` of each other on French P(gold) at the stop."""
    out = []
    for e in (p for p in pts if p["lang"] == "en" and rl in p["rel"]):
        for j in (p for p in pts if p["lang"] == "ja" and rl in p["rel"]):
            if abs(e["pg"] - j["pg"]) <= tol:
                out.append((e, j))
    return out


def panel(ax, pts, rl, xkey, ykey):
    for lg in ("en", "ja"):
        mine = sorted((p for p in pts if p["lang"] == lg and rl in p["rel"]),
                      key=lambda p: p[xkey])
        if not mine:
            continue
        s42 = [p for p in mine if p["seed"] == 42]
        ax.plot([p[xkey] for p in s42], [p["rel"][rl][ykey] for p in s42], color=ARM[lg],
                lw=1.6, alpha=0.6, zorder=2)
        for p in mine:
            ax.scatter(p[xkey], p["rel"][rl][ykey], s=46, zorder=3,
                       color=ARM[lg] if p["seed"] == 42 else SURFACE,
                       edgecolor=ARM[lg], linewidth=1.6)
    anchor = "p" if ykey == "pg" else "tr"
    for k, lab in (("ft", "learned fr_ft"), ("ret", "never taught")):
        if ANCHOR[k]:
            ax.axhline(ANCHOR[k][anchor], color=MUTED, lw=0.9, ls=":", zorder=1)
            ax.text(1.0, ANCHOR[k][anchor], " " + lab, transform=ax.get_yaxis_transform(),
                    fontsize=7.5, color=MUTED, va="center")
    ax.set_xlabel("French P(gold) at the stop  (deeper ->)")
    ax.set_ylabel({"pg": "relearned French P(gold)\n(lower = more durable)",
                   "tr": "relearned French TR\n(higher = more durable)"}[ykey])
    ax.set_title(f"relearned in {'French' if rl == 'fr' else 'English'}", fontsize=10,
                 color=INK, loc="left")


def main():
    pts = load()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    panel(axs[0], pts, "fr", "pg", "pg")
    panel(axs[1], pts, "en", "pg", "pg")
    top = max(p["pg"] for p in pts)
    axs[0].set_xlim(top + 0.03, -0.01)            # inverted: deeper to the right
    handles = [plt.Line2D([], [], color=ARM[lg], marker="o", lw=1.6, label=NAME[lg])
               for lg in ("en", "ja")]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", mfc=SURFACE, lw=0,
                              label="seed 1 / 2 (line joins seed 42 only)"))
    axs[0].legend(handles=handles, fontsize=8, loc="lower left")

    shared = [p["pg"] for p in pts if p["lang"] == "ja"]
    en_pg = [p["pg"] for p in pts if p["lang"] == "en"]
    lo, hi = max(min(shared), min(en_pg)), min(max(shared), max(en_pg))
    pairs = matched_pairs(pts, "fr")
    pair_txt = "; ".join(f"at P(gold) {e['pg']:.3f}: en -> {e['rel']['fr']['pg']:.3f}, "
                         f"ja -> {j['rel']['fr']['pg']:.3f}" for e, j in pairs[:2])
    title(fig, "Durability against unlearning depth: English vs Japanese unlearning",
          f"Both languages have runs at French P(gold) {lo:.3f}-{hi:.3f} at the stop.  "
          f"Matched (French relearn): {pair_txt or 'none'}.\n"
          f"{len(pts)} runs; filled = seed 42, hollow = seeds 1/2; peak LR and stopping rule "
          f"differ between points -- depth is the outcome, not the recipe.")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "18_depth_curve.png")


def load_d():
    out = []
    for f in sorted((P.P3 / "measure").glob("ul*_L*.json")):
        m = D_RE.match(f.stem)
        if not m:
            continue
        lg, seed, lr, stop, layers = m.groups()
        rel = {}
        for rl, suf in REL:
            g = P.P3 / "relearn" / f"relearn_{f.stem}_via_retain{suf}_ep3.json"
            if g.exists():
                rel[rl] = json.load(open(g))
        if rel:
            d = point(lg, f"layers {layers}, lr {lr.replace('p', '.')}, stop {stop}", int(seed),
                      json.load(open(f)), rel)
            d["layers"] = layers
            out.append(d)
    return out


def nearest_own(pts, d, rl):
    """The same-language, unrestricted run closest in French P(gold) at the stop."""
    own = [p for p in pts if p["lang"] == d["lang"] and rl in p["rel"]]
    return min(own, key=lambda p: abs(p["pg"] - d["pg"]))


def main_d():
    pts, ds = load(), load_d()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.6), sharex=True)
    for ax, rl in zip(axs, ("fr", "en")):
        panel(ax, pts, rl, "pg", "pg")
        for d in ds:
            if rl not in d["rel"]:
                continue
            ax.scatter(d["pg"], d["rel"][rl]["pg"], s=90, marker="D", zorder=4,
                       color=SURFACE, edgecolor=ARM[d["lang"]], linewidth=2.2)
            ax.annotate(f"{d['lang']} L{d['layers']}", (d["pg"], d["rel"][rl]["pg"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7.5,
                        color=ARM[d["lang"]])
    top = max(p["pg"] for p in pts + ds)
    axs[0].set_xlim(top + 0.03, -0.01)
    handles = [plt.Line2D([], [], color=ARM[lg], marker="o", lw=1.6,
                          label=f"{NAME[lg]}, all layers") for lg in ("en", "ja")]
    handles.append(plt.Line2D([], [], color=MUTED, marker="D", mfc=SURFACE, lw=0, ms=8,
                              label="Experiment D: layer-restricted"))
    axs[0].legend(handles=handles, fontsize=8, loc="lower left")
    lines = []
    for d in sorted(ds, key=lambda d: (d["lang"], -d["pg"])):
        n = nearest_own(pts, d, "fr")
        head = (f"{d['lang']} layers {d['layers']} (stop {d['label'].rsplit('stop ', 1)[1]}): "
                f"stop P(gold) {d['pg']:.3f} -> relearned {d['rel']['fr']['pg']:.3f}")
        if abs(n["pg"] - d["pg"]) <= 0.03:
            lines.append(head + f"   vs {d['lang']} all-layer @ {n['pg']:.3f} -> "
                                f"{n['rel']['fr']['pg']:.3f}")
        else:
            lines.append(head + f"   (no {d['lang']} all-layer run within 0.03)")
    title(fig, "Experiment D on the depth curve: restricting the layers does not move either "
               "language toward the other's curve",
          "French relearn:\n" + "\n".join(lines) +
          "\nSingle seed; seed spread of the all-layer arms <= 0.035.")
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    save(fig, "19_expD_on_depth_curve.png")


if __name__ == "__main__":
    main()
    if "--expd" in sys.argv:
        main_d()
