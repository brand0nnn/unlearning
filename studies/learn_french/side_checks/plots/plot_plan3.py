"""Figures for Plan 3 (matched depth AND matched epochs). Local, in .venv-plot:

    .venv-plot/bin/python studies/learn_french/side_checks/plots/plot_plan3.py

Four runs, two languages. Colour = the unlearning language (en blue, ja green, as in the
other side-check figures); filled / solid = the Stage 3 run (lr 5e-6), hollow / dashed =
the Plan 3 run that swaps its epoch count (en-slow lr 2e-6, ja-fast lr 1e-5).

Writes studies/learn_french/figures/side_checks/:
  7_plan3_design.png      French TR per epoch to the stop, the LR schedule each run used,
                          and the cumulative LR spent by the stop (the confound question)
  8_plan3_durability.png  relearned French TR vs stopping epoch (language vs run length), and
                          the size of each swap side by side
  9_plan3_layers.png      share of weight change per layer + total change, per run
  10_plan3_blanket.png    % destroyed by token type in the arm's own language and in French,
                          and one answer token by token (en arm on English, ja arm on Japanese)
  11_plan3_pgold_said.png Figure 8's layout on P(gold): where it ends after relearning, and the
                          % of the removed P(gold) that came back
  14_language_vs_epochs.png  Figure 8's layout with every arm: relearned French TR vs stopping
                          epoch for all five Stage 3 arms + the two Plan 3 runs
  13_five_durability.png  durability of all five Stage 3 arms: relearned French TR per relearn
                          language, P(gold) unlearned -> relearned, recovery on TR vs P(gold)
  12_five_languages.png   all five Stage 3 arms: layer profile, and how much of the fact the
                          French-learned model holds in each unlearning language vs top-layer
                          share, total change, own-language name destruction, recovery

Every number in a title or label is computed from the data plotted. Single seed, 40 facts
about 2 entities. The ja 1.5e-5 arm is drawn too once its results exist.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
SIDE = HERE.parent
STUDY = SIDE.parent
RES = SIDE / "results"
P3 = RES / "plan3"
sys.path.insert(0, str(STUDY / "plots"))
sys.path.insert(0, str(HERE))
import copy_check as cc  # noqa: E402
import plot_stage3 as S3  # noqa: E402
import read_plan1 as RP  # noqa: E402
from plot_side_checks import (ARM, GRID, INK, MUTED, SURFACE, SLUG, jl, save,  # noqa: E402
                              title, token_split)

SCHED_STEPS, WARM_STEPS = 100, 20        # Stage 2: 50 epochs x 2 steps, warmup 0.2


def lr_at(step, peak):
    """HF linear schedule with warmup -- reproduces the LR logged in every trajectory."""
    if step < WARM_STEPS:
        return peak * step / WARM_STEPS
    return peak * max(0.0, (SCHED_STEPS - step) / (SCHED_STEPS - WARM_STEPS))


def mean(r, k):
    v = S3.per_fact(r, k)
    return sum(v) / len(v)


def said(r):
    return sum(f["nli_score"] >= 0.9 for f in r["per_fact"])


def load():
    s1, unl, cells, _ = S3.load()
    pr = json.load(open(STUDY / "preregistration.json"))
    level = pr["tr_levels"][-1]
    m = pr["stage3_matched_checkpoints"]
    ptm = json.load(open(STUDY / "results" / "per_token_map.json", encoding="utf-8"))
    R = {cc.label(c): v for c, v in ptm["results"].items()}
    runs = []
    for lg in ("en", "ja"):
        traj = jl(STUDY / "results" / "unlearn_traj" / f"{SLUG}_ul{lg}_floornone.jsonl")
        keys = [k for k, t in ptm["texts"].items() if t["split"] == "forget" and t["lang"] == lg]
        runs.append(dict(
            lang=lg, kind="stage3", lr=5e-6, label=f"{lg}  Stage 3",
            epoch=m[lg]["epoch"], traj=[(r["epoch"], r["mean_tr"]) for r in traj],
            unl=unl[lg], rel={rl: cells[(lg, rl, 3)] for rl in ("fr", "en")},
            tokens=(ptm["texts"], R["fr_ft"], R[f"unl_{lg}"], keys),
            layers=json.load(open(STUDY / "results" / "layer_diff.json"))["pairs"][f"unl_{lg}"]))
    for f in sorted(P3.glob("traj_ul*_tr*.jsonl")):
        run = f.stem[len("traj_"):]
        lg = run[2:4]
        meas = P3 / "measure" / f"{run}.json"
        if not meas.exists():
            continue
        traj = jl(f)
        head, *eps = jl(P3 / f"tokens_{run}.jsonl")
        keys = [k for k, t in head["texts"].items() if t["split"] == "forget" and t["lang"] == lg]
        lr = head["meta"]["lr"]
        rel = {}
        for rl, suf in (("fr", "_langfr"), ("en", "")):
            g = P3 / "relearn" / f"relearn_{run}_via_retain{suf}_ep3.json"
            if g.exists():
                rel[rl] = json.load(open(g))
        lay = P3 / f"layer_diff_{run}.json"
        runs.append(dict(
            lang=lg, kind="plan3", lr=lr, run=run,
            label=f"{lg}-{'fast' if lr > 5e-6 else 'slow'}  lr {lr:.1e}",
            epoch=int(round(traj[-1]["epoch"])), traj=[(r["epoch"], r["mean_tr"]) for r in traj],
            unl=json.load(open(meas)), rel=rel,
            tokens=(head["texts"], eps[0]["results"], eps[-1]["results"], keys),
            layers=list(json.load(open(lay))["pairs"].values())[0] if lay.exists() else None))
    # preregistration_plan3.json primary_ja_fast: the run whose crossing epoch is nearer 12,
    # on a tie the lower LR. The other Plan 3 run of that language is drawn, dotted, as secondary.
    for lg in ("en", "ja"):
        p3 = [r for r in runs if r["kind"] == "plan3" and r["lang"] == lg]
        best = min(p3, key=lambda r: (abs(r["epoch"] - 12), r["lr"])) if p3 else None
        for r in p3:
            r["primary"] = r is best
    for r in runs:
        r.setdefault("primary", True)
    return s1, level, runs


def load_control():
    """The P(gold) control (preregistration_plan3.json pgold_control): en, Stage 2 recipe,
    stopped at a fixed epoch 15, against ja-fast 1.0e-5, which stopped at epoch 15 too."""
    run = "ulen_s42_lr5p0e-06_ep15"
    meas = P3 / "measure" / f"{run}.json"
    if not meas.exists():
        return None
    rel = {rl: json.load(open(P3 / "relearn" / f"relearn_{run}_via_retain{suf}_ep3.json"))
           for rl, suf in (("fr", "_langfr"), ("en", ""))}
    head, *eps = jl(P3 / f"tokens_{run}.jsonl")
    return dict(lang="en", run=run, epoch=int(round(jl(P3 / f"traj_{run}.jsonl")[-1]["epoch"])),
                unl=json.load(open(meas)), rel=rel,
                pg=[(e["epoch"], sum(e["prob_fr"]) / len(e["prob_fr"])) for e in eps])


def style_of(r):
    c = ARM[r["lang"]]
    if r["kind"] == "stage3":
        return dict(color=c, ls="-", mfc=c, marker="o")
    if not r.get("primary", True):
        return dict(color=c, ls=":", mfc=SURFACE, marker="^")
    return dict(color=c, ls="--", mfc=SURFACE, marker="D")


def legend(ax, runs, **kw):
    hs = [Line2D([], [], color=style_of(r)["color"], ls=style_of(r)["ls"], lw=2,
                 marker=style_of(r)["marker"], mfc=style_of(r)["mfc"], ms=7, mew=1.8,
                 label=f"{r['label']}  (stopped ep {r['epoch']})"
                       + ("" if r.get("primary", True) else "  secondary")) for r in runs]
    ax.legend(handles=hs, fontsize=8, **kw)


# ---- 7. the design: same depth, epochs swapped, LR is what moved ---------------------------
def fig_design(s1, level, runs):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    for r in runs:
        st = style_of(r)
        xs = [e for e, _ in r["traj"] if e <= r["epoch"]]
        ys = [t for e, t in r["traj"] if e <= r["epoch"]]
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=2)
        ax.plot(xs[-1], ys[-1], st["marker"], color=st["color"], mfc=st["mfc"], ms=8, mew=1.8,
                zorder=5)
        ax.annotate(f"ep {r['epoch']}", (xs[-1], ys[-1]),
                    xytext={("stage3", "en"): (-34, 2), ("plan3", "en"): (-38, 6)}.get(
                        (r["kind"], r["lang"]), (4, -12)),
                    textcoords="offset points", fontsize=8, color=st["color"], fontweight="bold")
    learned = mean(s1["fr_ft"], "tr")
    ax.axhline(level, color=MUTED, ls=":", lw=1.2)
    ax.text(0.3, level + 0.006, f"stop level {level:.3f} (= never-taught floor)", fontsize=8,
            color=MUTED)
    ax.axhline(learned, color=MUTED, ls=":", lw=1.2)
    ax.text(0.3, learned + 0.006, f"learned {learned:.3f}", fontsize=8, color=MUTED)
    ax.set_ylim(0.58, max(0.9, max(t for r in runs for e, t in r["traj"]
                                   if e <= r["epoch"]) + 0.02))
    ax.set_xlim(0, 27)
    ax.set_xlabel("unlearning epoch")
    ax.set_ylabel("French truth ratio (higher = more forgotten)")
    ax.set_title("A. Every run stops at the same depth;\nthe epoch counts are swapped",
                 loc="left", fontsize=9.5)

    ax = axes[1]
    for r in runs:
        st = style_of(r)
        steps = range(0, 2 * r["epoch"] + 1)
        ax.plot([s / 2 for s in steps], [lr_at(s, r["lr"]) * 1e6 for s in steps],
                color=st["color"], ls=st["ls"], lw=2)
        s = 2 * r["epoch"]
        ax.plot(r["epoch"], lr_at(s, r["lr"]) * 1e6, st["marker"], color=st["color"],
                mfc=st["mfc"], ms=8, mew=1.8, zorder=5)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, None)
    ax.set_xlabel("unlearning epoch")
    ax.set_ylabel("learning rate (x 1e-6)")
    ax.set_title("B. To swap epochs, the peak LR had to change:\n"
                 "same schedule shape, different step size", loc="left", fontsize=9.5)

    ax = axes[2]
    cum = {}
    for r in runs:
        st = style_of(r)
        c, ys = 0.0, [0.0]
        for s in range(1, 2 * r["epoch"] + 1):
            c += lr_at(s, r["lr"])
            ys.append(c * 1e6)
        cum[r["label"]] = ys[-1]
        ax.plot([s / 2 for s in range(len(ys))], ys, color=st["color"], ls=st["ls"], lw=2)
        ax.plot(r["epoch"], ys[-1], st["marker"], color=st["color"], mfc=st["mfc"], ms=8,
                mew=1.8, zorder=5)
        ax.annotate(f"{ys[-1]:.0f}", (r["epoch"], ys[-1]), xytext=(5, 2),
                    textcoords="offset points", fontsize=8, color=st["color"], fontweight="bold")
    ax.set_xlim(0, 27)
    ax.set_ylim(0, None)
    ax.set_xlabel("unlearning epoch")
    ax.set_ylabel("cumulative LR spent (sum over steps, x 1e-6)")
    en = [v for k, v in cum.items() if k.startswith("en")]
    ja = [v for k, v in cum.items() if k.startswith("ja")]
    ax.set_title(f"C. What stays roughly fixed within a language is the LR\n"
                 f"BUDGET to reach the level: en {min(en):.0f}-{max(en):.0f}, "
                 f"ja {min(ja):.0f}-{max(ja):.0f}", loc="left", fontsize=9.5)
    legend(axes[2], runs, loc="upper center", bbox_to_anchor=(-0.75, -0.17), ncol=len(runs))
    title(fig, "Plan 3's design: match depth AND epochs by changing the step size",
          "Within a language, Stage 3 vs Plan 3 = few big steps vs many small steps to the same "
          "depth. Across languages at the same epochs, ja needs several times the LR budget. "
          "Single seed.")
    fig.subplots_adjust(top=0.78, left=0.05, right=0.99, bottom=0.22)
    save(fig, "7_plan3_design.png")


# ---- 9. where the change lands --------------------------------------------------------------
def fig_layers(runs):
    rs = [r for r in runs if r["layers"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [2.3, 1],
                                                                   "wspace": 0.3})
    ax = axes[0]
    shares = {}
    for r in rs:
        L = r["layers"]["layers"]
        idx = sorted((k for k in L if k.isdigit()), key=int)
        tot = sum(L[k]["delta_sq"] for k in idx)
        sh = [100 * L[k]["delta_sq"] / tot for k in idx]
        top = sum(sh[24:])
        shares[r["label"]] = (top, tot)
        st = style_of(r)
        ax.plot([int(k) for k in idx], sh, color=st["color"], ls=st["ls"], lw=2,
                marker=st["marker"], mfc=st["mfc"], ms=4, mew=1.2)
    ax.axvspan(23.5, 35.5, color=GRID, alpha=0.35, lw=0)
    ax.text(29.5, ax.get_ylim()[1] * 0.95, "top third (24-35)", ha="center", fontsize=8,
            color=MUTED)
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(0, None)
    ax.set_xlabel("decoder layer (0 = input side, 35 = output side)")
    ax.set_ylabel("% of the run's total weight change in this layer")
    ax.set_title("A. Shape: how the change is spread over layers", loc="left", fontsize=9.5)
    legend(ax, rs, loc="upper left")
    ax = axes[1]
    for i, r in enumerate(rs):
        st = style_of(r)
        top, tot = shares[r["label"]]
        ax.bar(i, tot, color=st["color"] if r["kind"] == "stage3" else SURFACE,
               edgecolor=st["color"], lw=2, hatch=None if r["kind"] == "stage3" else "//",
               width=0.7)
        ax.text(i, tot, f"{tot:.1f}\ntop {top:.0f}%", ha="center", va="bottom", fontsize=8,
                color=INK)
    ax.set_xticks(range(len(rs)))
    ax.set_xticklabels([f"{r['label'].split('  ')[0]}\nep {r['epoch']}" for r in rs],
                       fontsize=8)
    ax.set_ylim(0, max(s[1] for s in shares.values()) * 1.25)
    ax.set_ylabel("total weight change  (sum ||dW||^2)")
    ax.grid(axis="x", visible=False)
    ax.set_title("B. Size: total change", loc="left", fontsize=9.5)
    en = [v[0] for k, v in shares.items() if k.startswith("en")]
    ja = [v[0] for k, v in shares.items() if k.startswith("ja")]
    title(fig, f"The top-heavy pattern follows the language, not the run length: top-third "
               f"share en {min(en):.0f}-{max(en):.0f}%, ja {min(ja):.0f}-{max(ja):.0f}%",
          "Layer diff fr_ft -> each stopped checkpoint (bf16, so absolute sizes are a lower "
          "bound; runs are comparable to each other). Single seed.")
    fig.subplots_adjust(top=0.8, left=0.06, right=0.99, bottom=0.14)
    save(fig, "9_plan3_layers.png")


# ---- 8 and 11. language vs run length, side by side (one layout, two metrics) ------------------
def interaction_panels(fig, slots, runs, val, ylab, up_durable, ylim, fmt, letters):
    """The Figure 8 layout for any per-run value `val(run, relearn_lang)`.

    x = stopping epoch, y = the value; lines join the two runs of one language, so their SLOPE
    is the run-length effect and the vertical distance between the colours is the language
    effect. The y-axis is flipped when LOWER means more durable, so up = more durable in
    every panel. Returns {relearn_lang: {comparison: size}} for the bar panel."""
    by = {(r["lang"], r["kind"]): r for r in runs}
    pairs = {"en": (by[("en", "stage3")], by[("en", "plan3")]),
             "ja": (by[("ja", "plan3")], by[("ja", "stage3")])}
    sign = 1 if up_durable else -1
    runs = [r for r in runs if r.get("primary", True)]
    eff = {}
    for j, (rl, name) in enumerate((("fr", "French"), ("en", "English"))):
        ax = fig.add_subplot(slots[j])
        val_at, drawn = {}, []
        for lg, (a, b) in pairs.items():
            pts = [r for r in (a, b) if rl in r["rel"]]
            xs, ys = [r["epoch"] for r in pts], [val(r, rl) for r in pts]
            ax.plot(xs, ys, color=ARM[lg], lw=2.5, zorder=2)
            for r, x, y in zip(pts, xs, ys):
                st = style_of(r)
                ax.plot(x, y, st["marker"], color=st["color"], mfc=st["mfc"], ms=11, mew=2,
                        zorder=5)
                grp = "short" if x <= 16 else "long"
                val_at[(lg, grp)] = (x, y)
                drawn.append((lg, grp, x, y))
        for lg, grp, x, y in drawn:
            # label on the side away from the other language's point (screen up = sign * y)
            other = val_at[("ja" if lg == "en" else "en", grp)][1]
            above = sign * y >= sign * other
            ax.annotate(fmt(y), (x, y), xytext=(0, 10 if above else -16),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        eff[rl] = {}
        for grp, lab in (("short", "short runs"), ("long", "long runs")):
            (xe, ye), (xj, yj) = val_at[("en", grp)], val_at[("ja", grp)]
            xm = (xe + xj) / 2
            ax.annotate("", (xm, yj), (xm, ye), arrowprops=dict(arrowstyle="<->", color=INK,
                                                                lw=1.3, shrinkA=0, shrinkB=0))
            d = sign * (ye - yj)                      # > 0: en more durable than ja
            tiny = abs(ye - yj) < 0.12 * abs(ylim[1] - ylim[0])
            ax.annotate(f"language\n{fmt(abs(ye - yj))}", (xm, (ye + yj) / 2),
                        xytext=(10, 28) if tiny else (6, 0),
                        textcoords="offset points", fontsize=8.5, color=INK, va="center",
                        fontweight="bold")
            eff[rl][f"swap language\n({lab})"] = d
        for lg in ("en", "ja"):
            (_, ys_), (_, yl) = val_at[(lg, "short")], val_at[(lg, "long")]
            eff[rl][f"swap run length\n(within {lg})"] = abs(ys_ - yl)
        ax.set_xlim(9, 28)
        ax.set_ylim(*ylim)
        if not up_durable:
            ax.invert_yaxis()
        ax.set_xlabel("unlearning epoch at the stop (same French TR for all four)")
        ax.set_ylabel(ylab)
        note = "re-exposes LEARN's data" if rl == "fr" else "fresh language, the cleaner test"
        ax.set_title(f"{letters[j]}. relearned on {name} ({note})", loc="left", fontsize=9.5)
        ax.text(0.98, 0.95, "unlearned in en", transform=ax.transAxes, color=ARM["en"],
                ha="right", va="top", fontweight="bold", fontsize=9)
        ax.text(0.98, 0.04, "unlearned in ja", transform=ax.transAxes, color=ARM["ja"],
                ha="right", va="bottom", fontweight="bold", fontsize=9)
    ax = fig.add_subplot(slots[2])
    labels = list(eff["fr"])
    w = 0.38
    top = max(max(abs(v) for v in e.values()) for e in eff.values())
    for j, rl in enumerate(("fr", "en")):
        for i, lab in enumerate(labels):
            v = abs(eff[rl][lab])
            lang = lab.startswith("swap language")
            x = i + (j - 0.5) * w
            ax.bar(x, v, width=w - 0.04, color=INK if lang else "#a9a8a2",
                   edgecolor=INK if lang else "#a9a8a2", lw=1.5,
                   hatch=None if rl == "fr" else "//", alpha=1 if rl == "fr" else 0.55)
            ax.text(x, v + top * 0.02, fmt(v), ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, top * 1.3)
    ax.set_ylabel("size of the change")
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MUTED, edgecolor=MUTED, label="French relearn column"),
                       Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="//",
                             label="English relearn column")], loc="upper right", fontsize=8)
    ax.set_title(f"{letters[2]}. How much it moves when you swap one thing,\nholding the other "
                 "fixed", loc="left", fontsize=9.5)
    ratio = {}
    for rl, e in eff.items():
        lv = [abs(v) for l, v in e.items() if l.startswith("swap language")]
        ev = [v for l, v in e.items() if not l.startswith("swap language")]
        ratio[rl] = (sum(lv) / len(lv)) / (sum(ev) / len(ev))
    return eff, ratio


def secondary_note(runs):
    sec = [r for r in runs if not r.get("primary", True)]
    return "".join(f" Secondary {r['label'].split('  ')[0]} lr {r['lr']:.1e} (ep {r['epoch']}) "
                   f"is in Figure 14." for r in sec)


def fig_interaction(s1, runs):
    by = {(r["lang"], r["kind"]): r for r in runs if r.get("primary", True)}
    fig = plt.figure(figsize=(15.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.3)
    _, ratio = interaction_panels(
        fig, [gs[0], gs[1], gs[2]], runs, lambda r, rl: mean(r["rel"][rl], "tr"),
        "French TR after relearning   (higher = more durable)", True, (0.6, 0.86),
        lambda v: f"{v:.3f}", "ABC")
    title(fig, f"Truth ratio: swapping the LANGUAGE moves durability {ratio['fr']:.1f}x (French "
               f"column) and {ratio['en']:.1f}x (English column) more than swapping the RUN LENGTH",
          "All four runs stop at the same French TR, so where they END after relearning is a fair "
          f"comparison. ja-fast overshot (TR {mean(by[('ja', 'plan3')]['unl'], 'tr'):.3f} vs "
          f"{mean(by[('ja', 'stage3')]['unl'], 'tr'):.3f}), which can only inflate its run-length "
          "effect. Same layout as Figure 11. Relearn epoch 3, single seed.\n"
          + secondary_note(runs).strip())
    fig.subplots_adjust(top=0.8, left=0.05, right=0.99, bottom=0.14)
    save(fig, "8_plan3_durability.png")


def fig_pgold2(s1, runs):
    learned = mean(s1["fr_ft"], "prob")
    runs = [r for r in runs if r.get("primary", True)]
    start = {r["label"]: mean(r["unl"], "prob") for r in runs}
    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], wspace=0.3, hspace=0.62)
    _, r_abs = interaction_panels(
        fig, [gs[0, 0], gs[0, 1], gs[0, 2]], runs, lambda r, rl: mean(r["rel"][rl], "prob"),
        "French P(gold) after relearning\n(lower = more durable; axis flipped)", False,
        (0.45, 0.66), lambda v: f"{v:.3f}", "ABC")
    rec = lambda r, rl: 100 * (start[r["label"]] - mean(r["rel"][rl], "prob")) / (
        start[r["label"]] - learned)
    _, r_rec = interaction_panels(
        fig, [gs[1, 0], gs[1, 1], gs[1, 2]], runs, rec,
        "recovery on P(gold), % of what unlearning\nremoved (lower = more durable; flipped)",
        False, (0, 100), lambda v: f"{abs(v):.0f}%", "DEF")
    fig.text(0.01, 0.895, "Row 1 -- where P(gold) ENDS after relearning (the direct analogue of "
             "Figure 8)", fontsize=10.5, color=INK, fontweight="bold")
    fig.text(0.01, 0.455, "Row 2 -- how much of the REMOVED P(gold) came back (corrects for the "
             "arms starting at different P(gold))", fontsize=10.5, color=INK, fontweight="bold")
    title(fig, f"P(gold): the language effect is clear in how much of the removed probability "
               f"comes back ({r_rec['fr']:.0f}x / {r_rec['en']:.0f}x the run-length effect), but "
               f"not in where the arms end ({r_abs['fr']:.1f}x / {r_abs['en']:.1f}x)",
          "Unlike TR, the arms were NOT matched on P(gold) at the stop: "
          + ", ".join(f"{r['label'].split('  ')[0]} {start[r['label']]:.2f}" for r in runs)
          + f" (learned {learned:.2f}).\nRelearning pulls every arm to a similar P(gold), so the "
          "deeper an arm started, the more came back. The P(gold)-matched comparison is Figure 16. "
          "Relearn epoch 3, single seed.")
    fig.subplots_adjust(top=0.84, left=0.06, right=0.99, bottom=0.07)
    save(fig, "11_plan3_pgold_said.png")


# ---- shared: destroyed share by token type ------------------------------------------------------
def destroyed_by_type(texts, base, cur, lang):
    slots = RP.load_slots()
    out = {}
    for k, t in texts.items():
        if t["split"] != "forget" or t["lang"] != lang:
            continue
        for i, tg in enumerate(RP.tag_text(t, slots)):
            if t["is_eos"][i]:
                continue
            c = "names" if tg.startswith("name") else "other words"
            for key in (c, "all"):
                a = out.setdefault(key, [0, 0])
                a[0] += (base[k]["lp"][i] - cur[k]["lp"][i]) > RP.DESTROY
                a[1] += 1
    return {c: 100 * h / n for c, (h, n) in out.items()}


def five_arms():
    d = json.load(open(STUDY / "results" / "per_token_map.json", encoding="utf-8"))
    return d["texts"], {cc.label(c): v for c, v in d["results"].items()}


# ---- 10 (redrawn). blanket damage: the whole sentence, in the arm's own language ---------------
def fig_blanket2(runs):
    T, R = five_arms()
    pr = json.load(open(STUDY / "preregistration.json"))["stage3_matched_checkpoints"]
    arms = [(f"{lg} ep {pr[lg]['epoch']}", lg, "stage3", T, R["fr_ft"], R[f"unl_{lg}"])
            for lg in ("fr", "en", "id", "ru", "ja")]
    for r in runs:
        if r["kind"] == "plan3":
            texts, base, cur, _ = r["tokens"]
            arms.append((f"{r['label'].split('  ')[0]} ep {r['epoch']}", r["lang"], "plan3",
                         texts, base, cur))
    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.22, top=0.86, bottom=0.53)
    ax = fig.add_subplot(gs[0, 0])
    w = 0.27
    own = {}
    for i, (lab, lg, kind, texts, base, cur) in enumerate(arms):
        v = destroyed_by_type(texts, base, cur, lg)
        own[lab] = v
        for j, c in enumerate(("names", "other words", "all")):
            x = i + (j - 1) * w
            ax.bar(x, v[c], width=w - 0.03, color=ARM[lg] if j == 2 else SURFACE,
                   edgecolor=ARM[lg], lw=1.8, hatch={0: "//", 1: "..", 2: None}[j])
            ax.text(x, v[c] + 1.5, f"{v[c]:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0].replace(" ep", "\nep") + ("\n(Plan 3)" if a[2] == "plan3" else "") for a in arms],
                       fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of answer tokens destroyed")
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="//", label="author names"),
                       Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="..",
                             label="every other word"),
                       Patch(facecolor=MUTED, edgecolor=MUTED, label="whole answer")],
              loc="upper left", fontsize=8, ncol=3)
    ax.set_title("A. In the arm's OWN language (the text it was trained to forget)",
                 loc="left", fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    fr = []
    for i, (lab, lg, kind, texts, base, cur) in enumerate(arms):
        v = destroyed_by_type(texts, base, cur, "fr")["all"]
        fr.append(v)
        ax.bar(i, v, width=0.65, color=ARM[lg] if kind == "stage3" else SURFACE,
               edgecolor=ARM[lg], lw=1.8, hatch=None if kind == "stage3" else "//")
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0].replace(" ep", "\nep") for a in arms], fontsize=7.5)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of French answer tokens destroyed")
    ax.grid(axis="x", visible=False)
    ax.set_title("B. In FRENCH (the language the facts were learned in)", loc="left",
                 fontsize=9.5)

    # C: one answer, token by token
    row = 24
    font = plt.rcParams["font.family"]
    try:
        from plot_side_checks import cc_font
        font = cc_font()
    except Exception:
        pass
    strips = []
    for j, lg in enumerate(("en", "ja")):
        ax = fig.add_axes([0.05, 0.3 - j * 0.21, 0.93, 0.1])
        strips.append(ax)
        k = f"forget:{lg}:{row}"
        t = T[k]
        idx = [i for i, e in enumerate(t["is_eos"]) if not e]
        drop = [max(0.0, R["fr_ft"][k]["lp"][i] - R[f"unl_{lg}"][k]["lp"][i]) for i in idx]
        cols = [ARM[lg] if dd > RP.DESTROY else "#d6d5d0" for dd in drop]
        ax.bar(range(len(idx)), drop, color=cols, width=0.8)
        ax.axhline(RP.DESTROY, color=MUTED, ls="--", lw=1)
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels([t["tokens"][i].strip().replace("\ufffd", "") or "\u00b7"
                            for i in idx], fontsize=7, rotation=90, fontfamily=font)
        ax.set_xlim(-0.6, len(idx) - 0.4)
        ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
        n = sum(dd > RP.DESTROY for dd in drop)
        ax.set_ylabel("log p drop", fontsize=8)
        ax.set_title(f"{'CD'[j]}. fact {row}, {lg} arm on its {('English', 'Japanese')[j]} "
                     f"answer: {n}/{len(idx)} tokens destroyed (coloured = > 5 nats; dashed "
                     f"line = 5)", loc="left", fontsize=9)
    top_ = max(a.get_ylim()[1] for a in strips)
    for a in strips:
        a.set_ylim(0, top_)
    ja = [own[a[0]]["other words"] for a in arms if a[1] == "ja"]
    en = [own[a[0]]["other words"] for a in arms if a[1] == "en"]
    rng = lambda v: (f"{min(v):.0f}%" if round(min(v)) == round(max(v))
                     else f"{min(v):.0f}-{max(v):.0f}%")
    xfr = max(v for v, a in zip(fr, arms) if a[1] != "fr")
    title(fig, f"The blanket is the whole sentence, not just names: ja destroys {rng(ja)} of "
               f"non-name words in Japanese (en {rng(en)} in English), yet at most "
               f"{xfr:.0f}% of the French answer",
          "Teacher-forced on the forget01 answers at each stopped checkpoint vs fr_ft. Destroyed "
          "= log p falls > 5 nats. Panel B: Stage 3 arms solid, Plan 3 arms hatched. C and D share "
          "a y-scale. Single seed.")
    save(fig, "10_plan3_blanket.png")


# ---- 12. all five languages: how much of the fact lives in the unlearning language -----------
def fig_five():
    T, R = five_arms()
    pr = json.load(open(STUDY / "preregistration.json"))["stage3_matched_checkpoints"]
    lay = json.load(open(STUDY / "results" / "layer_diff.json"))["pairs"]
    s1, unl, cells, _ = S3.load()
    L5 = ["fr", "en", "id", "ru", "ja"]

    def mlp(m, lg):
        v = [x for k, t in T.items() if t["split"] == "forget" and t["lang"] == lg
             for x, e in zip(R[m][k]["lp"], t["is_eos"]) if not e]
        return sum(v) / len(v)
    stored = {lg: mlp("fr_ft", lg) - mlp("fr_retain", lg) for lg in L5}
    prof, top, tot = {}, {}, {}
    for lg in L5:
        Lr = lay[f"unl_{lg}"]["layers"]
        idx = sorted((k for k in Lr if k.isdigit()), key=int)
        t_ = sum(Lr[k]["delta_sq"] for k in idx)
        prof[lg] = [100 * Lr[k]["delta_sq"] / t_ for k in idx]
        top[lg], tot[lg] = sum(prof[lg][24:]), t_
    names = {lg: destroyed_by_type(T, R["fr_ft"], R[f"unl_{lg}"], lg)["names"] for lg in L5}

    def rec(lg, k):
        v = []
        for rl in L5:
            U, X, V = (mean(r, k) for r in (unl[lg], s1["fr_ft"], cells[(lg, rl, 3)]))
            v.append(100 * (U - V) / (U - X))
        return sum(v) / len(v)
    rtr = {lg: rec(lg, "tr") for lg in L5}
    rp = {lg: rec(lg, "prob") for lg in L5}

    carry = {lg: math.exp(stored[lg]) for lg in L5}     # x more likely per answer token
    rows = sorted((lg for lg in L5 if lg != "fr"), key=lambda g: -stored[g]) + ["fr"]
    ypos = {lg: (len(rows) - 1 - i if lg != "fr" else -0.6) for i, lg in enumerate(rows)}

    fig = plt.figure(figsize=(16, 9.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.8, 1], hspace=0.5, wspace=0.12,
                          top=0.86, bottom=0.12, left=0.085, right=0.985)
    ax = fig.add_subplot(gs[0, :])
    ends, last = {}, None
    for lg in sorted(L5, key=lambda g: prof[g][-1]):
        y = prof[lg][-1] if last is None else max(prof[lg][-1], last + 0.32)
        ends[lg], last = y, y
    for lg in L5:
        ax.plot(range(len(prof[lg])), prof[lg], color=ARM[lg], lw=2.2)
        ax.annotate(f"{lg} (ep {pr[lg]['epoch']}): top {top[lg]:.0f}%",
                    (len(prof[lg]) - 1, prof[lg][-1]), xytext=(36.4, ends[lg]),
                    textcoords="data", va="center", fontsize=8.5, color=ARM[lg],
                    fontweight="bold")
    ax.axvspan(23.5, 35.5, color=GRID, alpha=0.35, lw=0)
    ax.set_xlim(-0.5, 41)
    ax.set_ylim(0, None)
    ax.set_xlabel("decoder layer (0 = input side, 35 = output side; shaded = top third)")
    ax.set_ylabel("% of the arm's total\nweight change")
    ax.set_title("A. Where each arm changes the model: ja and ru pile their change into the "
                 "layers nearest the output", loc="left", fontsize=10)

    cols = [
        ("1. Does the French-learned fact\ncarry over to this language?", carry,
         "x more likely per answer token than\na twin that never saw the fact (1x = none)",
         lambda v: f"{v:.1f}x", 1.0),
        ("2. Where does the change land?", top,
         "% of the weight change in the\ntop third of layers", lambda v: f"{v:.0f}%", 0),
        ("3. How much do the weights move?", tot,
         "total weight change\n(sum of squared changes)", lambda v: f"{v:.1f}", 0),
        ("4. Wiped in its own language?", names,
         "% of author-name tokens destroyed\nin the arm's own-language answers",
         lambda v: f"{v:.0f}%", 0),
        ("5. How much comes back?", rtr,
         "% of the forgetting undone by benign\nrelearning (truth ratio, mean of 5)",
         lambda v: f"{v:.0f}%", 0),
    ]
    for j, (head, v, xlab, fmt, base) in enumerate(cols):
        ax = fig.add_subplot(gs[1, j])
        hi = max(v.values())
        for lg in rows:
            y = ypos[lg]
            if j == 0:                        # a ratio: a lollipop from 1x, not a bar from 0
                cap = 2.45                    # fr (8.7x) runs off this scale on purpose
                end = min(v[lg], cap)
                ax.plot([base, end], [y, y], color=ARM[lg], lw=3, solid_capstyle="round")
                ax.plot(end, y, ">" if v[lg] > cap else "o", ms=10, color=ARM[lg], mec=SURFACE,
                        mew=1.5)
                ax.axvline(1, color=MUTED, lw=1, ls="--")
                ax.text(end + 0.05, y, fmt(v[lg]) + ("  (off scale)" if v[lg] > cap else ""),
                        va="center", fontsize=9, color=INK, fontweight="bold")
                continue
            ax.barh(y, v[lg], height=0.62, color=ARM[lg])
            ax.text(v[lg] + hi * 0.03, y, fmt(v[lg]), va="center", fontsize=9, color=INK,
                    fontweight="bold")
        ax.axhline(-0.05, color=GRID, lw=1.2, ls=":")
        ax.set_xlim(0 if j else 0.85, hi * 1.32 if j else 3.3)
        ax.set_ylim(-1.15, len(rows) - 0.45)
        ax.set_yticks([ypos[lg] for lg in rows])
        if j == 0:
            ax.set_yticklabels([f"unlearned in {lg}" if lg != "fr" else
                                "unlearned in fr\n(reference)" for lg in rows],
                               fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_title(head, loc="left", fontsize=9.5, fontweight="bold")
    fig.text(0.075, 0.535, "B. Read each row left to right: the less of the fact carries over "
             "into a language, the harder unlearning there hits the output side, and the more of "
             "the fact comes back", fontsize=10, color=INK)
    title(fig, "Japanese is the extreme, not the exception: unlearning in a language the fact "
               "barely reached is shallow and easy to undo",
          "Column 1: the learned model and a twin trained identically but without the 40 facts, "
          "both scored on the forget answers translated into each language. Rows ordered by "
          "column 1.\nFive Stage 3 arms at matched French TR, single seed: a pattern across four "
          "foreign-language arms, not a test. fr unlearns in the language the facts were learned "
          "in, so it is shown as a reference.")
    save(fig, "12_five_languages.png")


# ---- 13. durability of all five unlearning languages (Stage 3) -----------------------------
def fig_five_durability():
    s1, unl, cells, _ = S3.load()
    pr = json.load(open(STUDY / "preregistration.json"))["stage3_matched_checkpoints"]
    T, R = five_arms()

    def mlp(m, lg):
        v = [x for k, t in T.items() if t["split"] == "forget" and t["lang"] == lg
             for x, e in zip(R[m][k]["lp"], t["is_eos"]) if not e]
        return sum(v) / len(v)
    stored = {lg: mlp("fr_ft", lg) - mlp("fr_retain", lg) for lg in ("en", "id", "ru", "ja", "fr")}
    foreign = sorted(("en", "id", "ru", "ja"), key=lambda g: -stored[g])
    rows = foreign + ["fr"]                      # fr last: unlearned in the LEARNING language
    ypos = {lg: (len(rows) - i if lg != "fr" else 0.2) for i, lg in enumerate(rows)}
    RL = ["en", "id", "fr", "ru", "ja"]
    novel = [r for r in RL if r != "fr"]
    MK = {"en": "o", "id": "s", "fr": "X", "ru": "^", "ja": "D"}
    learned = {k: mean(s1["fr_ft"], k) for k in ("tr", "prob")}
    floor = {k: mean(s1["fr_retain"], k) for k in ("tr", "prob")}

    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.1, 1], wspace=0.32)
    # A: relearned TR, every relearn column + the mean over the novel ones
    ax = fig.add_subplot(gs[0])
    rowmean = {}
    for lg in rows:
        y = ypos[lg]
        vals = {rl: mean(cells[(lg, rl, 3)], "tr") for rl in RL}
        rowmean[lg] = sum(vals[r] for r in novel) / len(novel)
        u = mean(unl[lg], "tr")
        ax.plot([min(vals.values()), max(vals.values())], [y, y], color=ARM[lg], lw=1, alpha=0.5)
        for rl, v in vals.items():
            ax.plot(v, y + 0.18, MK[rl], ms=6, mfc=SURFACE if rl == "fr" else MUTED,
                    mec=MUTED, mew=1, zorder=4)
        ax.plot(rowmean[lg], y, "o", ms=13, color=ARM[lg], mec=SURFACE, mew=1.5, zorder=5)
        ax.annotate(f"{rowmean[lg]:.3f}", (rowmean[lg], y), xytext=(0, -17),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK,
                    fontweight="bold")
        ax.plot(u, y, "|", ms=16, mew=2, color=ARM[lg], zorder=3)
    for k, lab in ((learned["tr"], "learned"), (floor["tr"], "never-taught")):
        ax.axvline(k, color=GRID, lw=2)
        ax.text(k, len(rows) + 0.75, f"{lab}\n{k:.3f}", ha="center", fontsize=8, color=MUTED)
    ax.axhline(0.75, color=GRID, lw=1, ls=":")
    ax.set_yticks([ypos[lg] for lg in rows])
    ax.set_yticklabels([f"unlearned in {lg}\n(ep {pr[lg]['epoch']})" if lg != "fr" else
                        "unlearned in fr\n(learning language)" for lg in rows], fontsize=8.5)
    ax.set_ylim(-0.6, len(rows) + 1.2)
    ax.set_xlim(learned["tr"] - 0.02, floor["tr"] + 0.08)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("French TR after benign relearning   (higher = more durable)")
    hs = [Line2D([], [], ls="none", marker=MK[rl], ms=6, mfc=SURFACE if rl == "fr" else MUTED,
                 mec=MUTED, label=f"relearned in {rl}" + (" (re-exposure)" if rl == "fr" else ""))
          for rl in RL]
    hs += [Line2D([], [], ls="none", marker="o", ms=10, color=MUTED, label="mean of en/id/ru/ja relearns"),
           Line2D([], [], ls="none", marker="|", ms=12, mew=2, color=MUTED,
                  label="where it started (unlearned)")]
    ax.legend(handles=hs, fontsize=7.5, loc="upper center", ncol=3,
              bbox_to_anchor=(0.45, -0.11))
    ax.set_title("A. Rows (the unlearning language) differ more than\nthe small dots within a row "
                 "(the relearn language)", loc="left", fontsize=9.5)

    # B: P(gold) unlearned -> relearned (mean over novel relearn columns)
    ax = fig.add_subplot(gs[1])
    for lg in rows:
        y = ypos[lg]
        a = mean(unl[lg], "prob")
        b = sum(mean(cells[(lg, rl, 3)], "prob") for rl in novel) / len(novel)
        ax.annotate("", (b, y), (a, y), arrowprops=dict(arrowstyle="->", color=ARM[lg], lw=2.2,
                                                        shrinkA=6, shrinkB=6))
        ax.plot(a, y, "o", ms=9, mfc=SURFACE, mec=ARM[lg], mew=2)
        ax.plot(b, y, "o", ms=11, color=ARM[lg], mec=SURFACE, mew=1.5)
        ax.text(max(a, b) + 0.02, y, f"{a:.2f} -> {b:.2f}", va="center", fontsize=8.5, color=INK)
    for k, lab in ((learned["prob"], "learned"), (floor["prob"], "never-taught")):
        ax.axvline(k, color=GRID, lw=2)
        ax.text(k, len(rows) + 0.75, f"{lab}\n{k:.2f}", ha="center", fontsize=8, color=MUTED)
    ax.axhline(0.75, color=GRID, lw=1, ls=":")
    ax.set_yticks([ypos[lg] for lg in rows])
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_ylim(-0.6, len(rows) + 1.2)
    ax.set_xlim(0, learned["prob"] + 0.18)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("French P(gold): unlearned (hollow) -> relearned (filled)")
    ax.set_title("B. Answer probability: fr starts far deeper than the\nothers and stays "
                 "lowest; ja climbs most", loc="left", fontsize=9.5)

    # C: recovery %, TR vs P(gold) -- where the ranking is metric-dependent
    ax = fig.add_subplot(gs[2])
    w = 0.38
    for i, lg in enumerate(rows):
        for j, k in enumerate(("tr", "prob")):
            v = 100 * sum((mean(unl[lg], k) - mean(cells[(lg, rl, 3)], k)) /
                          (mean(unl[lg], k) - learned[k]) for rl in RL) / len(RL)
            x = i + (j - 0.5) * w
            ax.bar(x, v, width=w - 0.04, color=ARM[lg] if j == 0 else SURFACE,
                   edgecolor=ARM[lg], lw=1.8, hatch=None if j == 0 else "//")
            ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(rows)
    ax.set_ylim(0, 100)
    ax.set_ylabel("recovery after relearning (%, mean of 5 relearn languages)")
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MUTED, edgecolor=MUTED, label="on truth ratio"),
                       Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="//", label="on P(gold)")],
              loc="upper left", fontsize=8)
    ax.set_title("C. Recovery %: en lowest and ja highest on both;\nthe middle reorders by metric",
                 loc="left", fontsize=9.5)
    title(fig, "Durability across all five unlearning languages (relearned French TR): "
               + " > ".join(f"{lg} {rowmean[lg]:.3f}" for lg in sorted(foreign, key=lambda g: -rowmean[g]))
               + f";  fr {rowmean['fr']:.3f}",
          "Stage 3, matched on French TR, relearn epoch 3. id/fr/ru stopped within 3 epochs of en, so "
          "run length barely separates them (Plan 3); ja is the 25-epoch arm.\nRows ordered by how much "
          "of the fact the model holds in that language. Single seed: id/fr/ru are within ~0.03.")
    fig.subplots_adjust(top=0.82, left=0.08, right=0.99, bottom=0.2)
    save(fig, "13_five_durability.png")


# ---- 14. Figure 8 for every arm: is it the language or the epoch count? ---------------------
def fig_language_vs_epochs(s1, runs):
    """Every run stops at the same French TR. If run length governed durability, the points
    would line up along the x-axis (one trend in epochs); if the language does, they stack
    vertically at the same epoch and each language's two runs sit at the same height."""
    _, unl, cells, _ = S3.load()
    pr = json.load(open(STUDY / "preregistration.json"))["stage3_matched_checkpoints"]
    pts = []                                   # (lang, kind, epoch, {rl: relearned TR})
    for lg in ("en", "id", "ru", "ja", "fr"):
        pts.append((lg, "stage3", pr[lg]["epoch"],
                    {rl: mean(cells[(lg, rl, 3)], "tr") for rl in ("fr", "en")}))
    sec = set()                                # (lang, epoch) of secondary Plan 3 runs
    lrs = {}
    for r in runs:
        if r["kind"] == "plan3":
            pts.append((r["lang"], "plan3", r["epoch"],
                        {rl: mean(c, "tr") for rl, c in r["rel"].items()}))
            lrs[(r["lang"], r["epoch"])] = r["lr"]
            if not r.get("primary", True):
                sec.add((r["lang"], r["epoch"]))
    prim = [p for p in pts if (p[0], p[2]) not in sec]   # the bars use primary runs only
    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.3)
    eff = {}
    for j, (rl, name) in enumerate((("fr", "French"), ("en", "English"))):
        ax = fig.add_subplot(gs[j])
        ax.axvspan(11.3, 15.7, color=GRID, alpha=0.35, lw=0)
        ax.text(13.5, 0.853, "same run length\n(12-15 epochs)", ha="center", fontsize=8,
                color=MUTED, va="top")
        for lg in ("en", "ja"):                # join each language's short and long run
            two = sorted((e, v[rl]) for g, k, e, v in pts if g == lg and rl in v)
            ax.plot([e for e, _ in two], [y for _, y in two], color=ARM[lg], lw=2.2, zorder=2)
        for lg, kind, e, v in pts:
            if rl not in v:
                continue
            own = lg == "fr"
            secondary = (lg, e) in sec
            ax.plot(e, v[rl], "s" if own else ("o" if kind == "stage3" else
                                               ("^" if secondary else "D")), ms=11,
                    color=ARM[lg], mfc=SURFACE if kind == "plan3" or own else ARM[lg], mew=2,
                    zorder=5)
            lab = (f"{lg} {v[rl]:.3f}" if kind == "stage3"
                   else f"{lg}-{'slow' if lg == 'en' else 'fast'} {v[rl]:.3f}")
            if secondary:
                lab += " (2nd)"
            if own:
                lab += " (learning language)"
            left = lg == "id" or (kind == "plan3" and lg == "ja" and not secondary)
            low = secondary or lg == "ru"      # ja 2nd and ru share epoch 15: both labels below
            ax.annotate(lab, (e, v[rl]), xytext=(-9, 0) if left else (
                        (9, 12) if own else ((9, -9) if low else (9, 0))),
                        textcoords="offset points", va="center", ha="right" if left else "left",
                        fontsize=8.5, color=ARM[lg], fontweight="bold")
        short = {lg: v[rl] for lg, kind, e, v in prim
                 if lg != "fr" and 11 <= e <= 16 and rl in v}
        long_ = {lg: v[rl] for lg, kind, e, v in prim if lg != "fr" and e >= 22 and rl in v}
        ends = {lg: sorted((e, v[rl]) for g, k, e, v in prim if g == lg and rl in v)
                for lg in ("en", "ja")}
        eff[rl] = {"languages at\n12-15 epochs": max(short.values()) - min(short.values()),
                   "en vs ja at\n23-25 epochs": long_["en"] - long_["ja"]}
        for lg, ((e0, v0), *_, (e1, v1)) in ends.items():
            eff[rl][f"{lg}: {e0} -> {e1}\nepochs"] = abs(v0 - v1)
        ax.set_xlim(8, 29.5)
        ax.set_ylim(0.62, 0.86)
        ax.set_xlabel("unlearning epoch at the stop (every run at the same French TR)")
        ax.set_ylabel("French TR after relearning   (higher = more durable)")
        note = "re-exposes LEARN's data" if rl == "fr" else "fresh language, the cleaner test"
        ax.set_title(f"{'AB'[j]}. relearned on {name} ({note})\nin the grey band the run length "
                     f"is the same, but the languages spread out", loc="left", fontsize=9.5)
    ax = fig.add_subplot(gs[2])
    labels = list(eff["fr"])
    w = 0.38
    for j, rl in enumerate(("fr", "en")):
        for i, lab in enumerate(labels):
            v = eff[rl][lab]
            lang = i < 2
            x = i + (j - 0.5) * w
            ax.bar(x, v, width=w - 0.04, color=INK if lang else "#a9a8a2",
                   edgecolor=INK if lang else "#a9a8a2", lw=1.5,
                   hatch=None if rl == "fr" else "//", alpha=1 if rl == "fr" else 0.55)
            ax.text(x, v + 0.002, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.axvline(1.5, color=GRID, lw=1.5)
    ax.text(0.5, 0.135, "change the LANGUAGE\n(run length fixed)", ha="center", fontsize=8.5,
            color=INK, fontweight="bold")
    ax.text(2.5, 0.135, "change the RUN LENGTH\n(language fixed)", ha="center", fontsize=8.5,
            color=MUTED, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.155)
    ax.set_ylabel("difference in relearned French TR")
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MUTED, edgecolor=MUTED, label="French relearn column"),
                       Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="//",
                             label="English relearn column")], loc="upper right",
              bbox_to_anchor=(1, 0.86), fontsize=8)
    ax.set_title("C. How far durability moves when you change\none thing and hold the other "
                 "fixed", loc="left", fontsize=9.5)
    lang_max = max(max(eff[rl][l] for l in labels[:2]) for rl in eff)
    ep_max = max(max(eff[rl][l] for l in labels[2:]) for rl in eff)
    hs = [Line2D([], [], ls="none", marker="o", ms=9, color=MUTED, label="Stage 3 run (lr 5e-6)"),
          Line2D([], [], ls="none", marker="D", ms=9, mfc=SURFACE, mec=MUTED, mew=2,
                 label="Plan 3 run (epochs swapped)"),
          Line2D([], [], ls="none", marker="^", ms=9, mfc=SURFACE, mec=MUTED, mew=2,
                 label="Plan 3 secondary (not in bars C)"),
          Line2D([], [], ls="none", marker="s", ms=9, mfc=SURFACE, mec=ARM["fr"], mew=2,
                 label="fr: unlearned in the learning language"),
          Line2D([], [], color=MUTED, lw=2.2, label="same language, two run lengths")]
    fig.legend(handles=hs, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    title(fig, "The unlearning language, not the run length, sets durability",
          f"At the same run length the languages differ by up to {lang_max:.3f} in relearned "
          f"French TR; changing the run length moves one language by at most {ep_max:.3f}.\n"
          "Every run stops at the same French truth ratio. Lines join each language's runs "
          "(nearly flat = run length barely matters). ja-fast: "
          + ", ".join(f"lr {lr:.1e} ep {e}{' (2nd)' if (g, e) in sec else ''}"
                      for (g, e), lr in sorted(lrs.items()) if g == "ja")
          + ". Bars C: primary runs, fr excluded. Relearn epoch 3, single seed.")
    fig.subplots_adjust(top=0.8, left=0.05, right=0.99, bottom=0.2)
    save(fig, "14_language_vs_epochs.png")


# ---- 16. the P(gold) control: en and ja-fast matched on P(gold) and epochs ------------------
def fig_pgold_control(s1, runs, ctl):
    ja = next(r for r in runs if r["kind"] == "plan3" and r["lang"] == "ja"
              and r["epoch"] == ctl["epoch"])
    rule = json.load(open(SIDE / "preregistration_plan3.json"))["pgold_control"]
    MIN = 0.04                                 # reading_rule in pgold_control (checked below)
    assert "under 0.04" in rule["reading_rule"]
    _, *eps = jl(P3 / f"tokens_{ja['run']}.jsonl")
    ja_pg = [(e["epoch"], sum(e["prob_fr"]) / len(e["prob_fr"])) for e in eps]
    learned, floor = mean(s1["fr_ft"], "prob"), mean(s1["fr_retain"], "prob")
    arms = (("en", "en, lr 5e-6", ctl, ctl["pg"]), ("ja", f"ja-fast, lr {ja['lr']:.1e}", ja, ja_pg))

    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    for lg, lab, r, pg in arms:
        xs, ys = [e for e, _ in pg], [p for _, p in pg]
        ax.plot(xs, ys, color=ARM[lg], lw=2.2, label=lab)
        ax.plot(xs[-1], ys[-1], "o", color=ARM[lg], ms=9, zorder=5)
    ax.annotate(f"both {mean(ctl['unl'], 'prob'):.3f} / {mean(ja['unl'], 'prob'):.3f}\n"
                f"at epoch {ctl['epoch']}", (ctl["epoch"], mean(ctl["unl"], "prob")),
                xytext=(-95, -42), textcoords="offset points", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    for y, t in ((learned, "learned"), (floor, "never taught")):
        ax.axhline(y, color=MUTED, ls=":", lw=1.2)
        ax.text(0.3, y + 0.012, f"{t} {y:.3f}", fontsize=8, color=MUTED)
    ax.set_xlim(0, ctl["epoch"] + 1)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("unlearning epoch")
    ax.set_ylabel("French P(gold)   (lower = more forgotten)")
    ax.legend(loc="center left", fontsize=8.5)
    ax.set_title("A. Two runs that reach the same P(gold) at the same epoch", loc="left",
                 fontsize=9.5)

    ax = fig.add_subplot(gs[1])
    w = 0.36
    diff = {}
    for i, (rl, name) in enumerate((("fr", "French"), ("en", "English"))):
        vals = {}
        for j, (lg, lab, r, _) in enumerate(arms):
            v = mean(r["rel"][rl], "prob")
            vals[lg] = v
            x = i + (j - 0.5) * w
            ax.bar(x, v, width=w - 0.04, color=ARM[lg])
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
        diff[rl] = vals["ja"] - vals["en"]
        ax.annotate(f"ja higher by {diff[rl]:+.3f}", (i, max(vals.values()) + 0.06),
                    ha="center", fontsize=8.5, color=INK, fontweight="bold")
    start = mean(ctl["unl"], "prob")
    for y, t in ((learned, "learned"), (start, "both started here"), (floor, "never taught")):
        ax.axhline(y, color=MUTED, ls=":", lw=1.2)
        ax.text(2.15, y + 0.012, f"{t} {y:.3f}", fontsize=8, color=MUTED, ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["relearned on French\n(re-exposes LEARN's data)",
                        "relearned on English\n(fresh language)"], fontsize=8.5)
    ax.set_xlim(-0.5, 2.2)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("French P(gold) after relearning   (higher = more came back)")
    ax.grid(axis="x", visible=False)
    ax.set_title("B. After the same benign relearning", loc="left", fontsize=9.5)

    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    rows = [("matched at the stop", None, None),
            ("epoch", f"{ctl['epoch']}", f"{ja['epoch']}"),
            ("French P(gold)", f"{mean(ctl['unl'], 'prob'):.3f}", f"{mean(ja['unl'], 'prob'):.3f}"),
            ("model utility", f"{ctl['unl']['model_utility_6']:.3f}",
             f"{ja['unl']['model_utility_6']:.3f}"),
            ("NOT matched at the stop", None, None),
            ("French TR", f"{mean(ctl['unl'], 'tr'):.3f}", f"{mean(ja['unl'], 'tr'):.3f}"),
            ("peak LR", "5.0e-6", f"{ja['lr']:.1e}"),
            ("after relearning (secondary)", None, None)]
    for rl, name in (("fr", "French"), ("en", "English")):
        rows.append((f"TR, {name} relearn", f"{mean(ctl['rel'][rl], 'tr'):.3f}",
                     f"{mean(ja['rel'][rl], 'tr'):.3f}"))
        rows.append((f"NLI, {name} relearn", f"{mean(ctl['rel'][rl], 'nli'):.3f}",
                     f"{mean(ja['rel'][rl], 'nli'):.3f}"))
    y = 0.97
    ax.text(0.55, y, "en", color=ARM["en"], fontweight="bold", fontsize=9, ha="center")
    ax.text(0.82, y, "ja-fast", color=ARM["ja"], fontweight="bold", fontsize=9, ha="center")
    for lab, a, b in rows:
        y -= 0.068
        if a is None:
            ax.text(0.0, y, lab, fontsize=8.5, color=INK, fontweight="bold")
            continue
        ax.text(0.04, y, lab, fontsize=8.5, color=INK)
        ax.text(0.55, y, a, fontsize=8.5, color=INK, ha="center")
        ax.text(0.82, y, b, fontsize=8.5, color=INK, ha="center")
    ax.set_title("C. What the pair is and is not matched on", loc="left", fontsize=9.5)

    held = all(d >= MIN for d in diff.values())
    verdict = ("ja-fast recovers more P(gold) in both relearn languages"
               if held else "the ja-en P(gold) gap does not clear the pre-registered 0.04 in both columns")
    title(fig, f"Matched on P(gold) and epochs, {verdict}: "
               f"{diff['fr']:+.3f} (French) / {diff['en']:+.3f} (English)",
          f"Pre-registered control (preregistration_plan3.json pgold_control): gaps under {MIN} "
          "read as no detectable difference. en is far deeper on TR at the stop, so its relearned "
          "TR (panel C) is not evidence either way. Relearn epoch 3, single seed.")
    fig.subplots_adjust(top=0.8, left=0.05, right=0.99, bottom=0.14)
    save(fig, "16_pgold_control.png")


def main():
    s1, level, runs = load()
    print("runs:", ", ".join(f"{r['label']} (ep {r['epoch']})" for r in runs))
    fig_design(s1, level, runs)
    # durability figures need the relearn cells; a run whose relearning is still going is
    # drawn everywhere else
    rel = [r for r in runs if r["rel"]]
    fig_interaction(s1, rel)
    fig_layers(runs)
    fig_blanket2(runs)
    fig_pgold2(s1, rel)
    fig_five()
    fig_five_durability()
    fig_language_vs_epochs(s1, rel)
    ctl = load_control()
    if ctl:
        fig_pgold_control(s1, rel, ctl)


if __name__ == "__main__":
    main()
