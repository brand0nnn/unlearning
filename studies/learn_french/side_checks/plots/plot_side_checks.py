"""Figures for the side checks (Plans 1 and 2). Local, in .venv-plot:

    .venv-plot/bin/python studies/learn_french/side_checks/plots/plot_side_checks.py

Writes studies/learn_french/figures/side_checks/:
  1_effort_vs_depth.png   French TR per epoch for all five unlearning arms (Stage 2 runs)
                          with the 25-epoch reruns on top, and French P(gold) in the reruns
  2_durability.png        P(gold) and TR before / after benign French relearning, for the
                          five matched checkpoints, fr@25, en@25 and the fr_retain control;
                          recovery on TR vs on P(gold) for the matched arms
  3_names_delayed.png     % destroyed per epoch among near-certain tokens, copied names vs
                          other text, fr and en reruns; the same split at every checkpoint
  4_name_pieces.png       Plan 2: change in log p of each name vs fr_ft, by arm
  5_script_switch.png     Plan 2: what copying a katakana / Cyrillic name produces, by model
  6_blanket_by_epoch.png  Plan 1's own question: % of each arm's own answer tokens destroyed
                          per epoch (fr, en to 25) vs Stage 6's matched arms (ja at 25)
  15_name_translation.png Plan 2: transliterating names between scripts -- change in log p
                          per name group and arm, and what fr_ft / the ja and ru arms write

Every number in a title or label is computed from the data plotted. Single seed, 40 facts
about 2 entities. Descriptive only.
"""
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
SIDE = HERE.parent
STUDY = SIDE.parent
RES = SIDE / "results"
FIGS = STUDY / "figures" / "side_checks"
sys.path.insert(0, str(STUDY / "plots"))
sys.path.insert(0, str(SIDE / "scripts"))
import copy_check as cc  # noqa: E402
from name_tasks import clean_prediction, score_prediction  # noqa: E402

SLUG = "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b"
FR_FT = "tofu_learn_full_full_qwen3-8b_fr"
FR_RET = "tofu_learn_retain99_full_qwen3-8b_fr"
LANGS = ["fr", "en", "id", "ja", "ru"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
# fr = the learned language, drawn as the neutral anchor; the four hues pass the palette
# validator all-pairs (CVD in the warn band, so every series is direct-labelled)
ARM = {"fr": "#2a2a2a", "en": "#2a78d6", "id": "#eda100", "ja": "#1baf7a", "ru": "#e34948"}
ACCENT, OTHER1, OTHER2 = "#2a78d6", "#52514e", "#a9a8a2"
DESTROY = 5.0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
})


def jl(f):
    return [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]


def summ(f):
    if not f.exists():
        return None
    s = json.load(open(f))["summary"]
    return {"tr": s["tr_arithmetic_mean_norm"], "p": s["prob_mean"], "nli": s["nli_score_mean"]}


def prereg():
    return json.load(open(STUDY / "preregistration.json"))["stage3_matched_checkpoints"]


def ck_name(lang, m):
    return f"{SLUG}_ul{lang}_floornone_" + ("tr%.3f" % m["level"]).replace(".", "p")


def title(fig, text, sub):
    fig.suptitle(text, x=0.01, ha="left", fontsize=12.5, color=INK, y=0.995)
    fig.text(0.01, 0.955, sub, ha="left", va="top", fontsize=9, color=MUTED)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", (FIGS / name).relative_to(STUDY))


ANCHOR = {"ft": summ(STUDY / "results" / "stage1_norm" / f"{FR_FT}.json"),
          "ret": summ(STUDY / "results" / "stage1_norm" / f"{FR_RET}.json")}


# ---- 1. effort vs depth ------------------------------------------------------------
def fig_effort():
    pr = prereg()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.28})
    cap = 1e6
    ep25 = {}
    for lg in LANGS:
        rows = jl(STUDY / "results" / "unlearn_traj" / f"{SLUG}_ul{lg}_floornone.jsonl")
        xs = [r["step"] / 2 for r in rows if r["step"] / 2 <= 25]
        ys = [min(r["mean_tr"], cap) for r in rows if r["step"] / 2 <= 25]
        ep25[lg] = next(r["mean_tr"] for r in rows if r["step"] == 50)
        a.plot(xs, ys, color=ARM[lg], lw=2)
        m = pr[lg]
        a.scatter([m["epoch"]], [m["tr"]], s=60, color=ARM[lg], edgecolor=SURFACE, lw=1.5,
                  zorder=5)
        lab = f"{lg}  {ep25[lg]:.3g}" if ep25[lg] < cap else f"{lg}  {ep25[lg]:.0e} (off scale)"
        a.annotate(lab, (25, ys[-1]), xytext=(5, 0), textcoords="offset points",
                   color=ARM[lg], fontsize=9, va="center", fontweight="bold")
    for lg in ("fr", "en"):
        side = jl(RES / "plan1" / f"traj_ul{lg}_s42.jsonl")
        a.scatter([r["step"] / 2 for r in side], [r["mean_tr"] for r in side], s=16,
                  facecolor="none", edgecolor=ARM[lg], lw=0.9, zorder=4)
    a.axvline(25, color=MUTED, ls="--", lw=1)
    a.text(24.6, 3e4, "epoch 25 = Japanese's\nmatched effort", fontsize=8, color=MUTED,
           ha="right")
    a.axhline(ANCHOR["ret"]["tr"], color=OTHER2, lw=1)
    a.text(0.3, ANCHOR["ret"]["tr"] * 1.06, f"never taught {ANCHOR['ret']['tr']:.2f}",
           fontsize=8, color=MUTED)
    a.set_yscale("log")
    a.set_ylim(0.55, cap * 1.5)
    a.set_xlim(0, 31)
    a.set_xlabel("unlearning epoch (50-epoch LR schedule)")
    a.set_ylabel("French truth ratio (log; higher = more forgotten)")
    a.set_title("A. Same recipe, same epochs, very different depth on the French fact\n"
                "dots = matched checkpoints; rings = this rerun (fr, en); label = TR at ep 25",
                loc="left", fontsize=9.5, color=INK)

    below = {}
    for lg in ("fr", "en"):
        rows = jl(RES / "plan1" / f"tokens_ul{lg}_s42.jsonl")[1:]
        xs = [r["epoch"] for r in rows]
        ys = [st.mean(r["prob_fr"]) for r in rows]
        below[lg] = next(x for x, y in zip(xs, ys) if y < ANCHOR["ret"]["p"])
        b.plot(xs, ys, color=ARM[lg], lw=2)
        b.annotate(f"{lg}  {ys[-1]:.2g}", (xs[-1], ys[-1]), xytext=(4, 0),
                   textcoords="offset points", color=ARM[lg], fontsize=9, va="center",
                   fontweight="bold")
        m = pr[lg]
        b.scatter([m["epoch"]], [ys[xs.index(m["epoch"])]], s=60, color=ARM[lg],
                  edgecolor=SURFACE, lw=1.5, zorder=5)
    m = pr["ja"]
    b.scatter([m["epoch"]], [m["prob"]], s=60, color=ARM["ja"], edgecolor=SURFACE, lw=1.5,
              zorder=5)
    b.annotate(f"ja matched  {m['prob']:.2f}", (m["epoch"], m["prob"]), xytext=(0, 9),
               textcoords="offset points", color=ARM["ja"], fontsize=9, fontweight="bold",
               ha="right")
    for key, lab in (("ft", "learned"), ("ret", "never taught")):
        v = ANCHOR[key]["p"]
        b.axhline(v, color=OTHER2, lw=1)
        b.text(0.3, v * 1.15, f"{lab} {v:.3f}", fontsize=8, color=MUTED)
    b.set_yscale("log")
    b.set_xlim(0, 28)
    b.set_xlabel("unlearning epoch")
    b.set_ylabel("French P(gold answer), mean over 40 facts (log)")
    b.set_title("B. French answer probability in the 25-epoch reruns\n"
                f"below never-taught from epoch {below['fr']} (fr; matched at "
                f"{pr['fr']['epoch']}),\n{below['en']} (en; matched at {pr['en']['epoch']})",
                loc="left", fontsize=9.5, color=INK)
    title(fig, "Plan 1: 25 epochs of unlearning push fr and en far past the matched point",
          "fr_ft unlearned with gradient difference on forget01 in one language; French "
          "probe. Single seed.")
    fig.subplots_adjust(top=0.80, left=0.06, right=0.95, bottom=0.11)
    save(fig, "1_effort_vs_depth.png")


# ---- 2. durability -------------------------------------------------------------------
def relearn_rows():
    pr, S3 = prereg(), STUDY / "results"
    rel, meas = RES / "plan1" / "relearn", RES / "plan1" / "measure"
    rows = []
    for lg in LANGS:
        ck = ck_name(lg, pr[lg])
        sb = S3 / f"stage3_ul{lg}" / f"relearn_{ck}_via_retain_langfr_ep3"
        rows.append((f"{lg} matched (ep {pr[lg]['epoch']})", lg, False,
                     summ(S3 / f"stage2_nocap_{lg}" / f"{ck}.json"),
                     [summ(Path(f"{sb}__atep1.json")), summ(Path(f"{sb}__atep2.json")),
                      summ(Path(f"{sb}.json"))]))
        if lg in ("fr", "en"):
            b = rel / f"relearn_ul{lg}_s42_ep25_via_retain_langfr_ep3"
            rows.append((f"{lg} at ep 25", lg, True, summ(meas / f"ul{lg}_s42_ep25.json"),
                         [summ(Path(f"{b}__atep1.json")), summ(Path(f"{b}__atep2.json")),
                          summ(Path(f"{b}.json"))]))
    b = rel / f"relearn_{FR_RET}_via_retain_langfr_ep3"
    rows.append(("fr_retain (never taught)", None, False, summ(meas / f"{FR_RET}.json"),
                 [summ(Path(f"{b}__atep1.json")), summ(Path(f"{b}__atep2.json")),
                  summ(Path(f"{b}.json"))]))
    return rows


def fig_durability():
    rows = relearn_rows()
    order = ["ja matched (ep 25)", "ru matched (ep 15)", "id matched (ep 12)",
             "en matched (ep 12)", "en at ep 25", "fr matched (ep 13)", "fr at ep 25",
             "fr_retain (never taught)"]
    R = {r[0]: r for r in rows}
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.9], wspace=0.42)
    for j, (key, lab, better_hi) in enumerate((("p", "French P(gold), log", True),
                                               ("tr", "French truth ratio, log", False))):
        ax = fig.add_subplot(gs[j])
        for i, name in enumerate(order):
            _, lg, side, start, snaps = R[name]
            y = len(order) - 1 - i
            col = ARM[lg] if lg else OTHER1
            xs = [start[key]] + [s[key] for s in snaps]
            ax.plot(xs, [y] * 4, color=col, lw=1, alpha=0.5)
            ax.scatter([start[key]], [y], s=46, facecolor=SURFACE, edgecolor=col, lw=1.6,
                       zorder=4, marker="s" if side else "o")
            ax.scatter([s[key] for s in snaps[:2]], [y, y], s=14, color=col, zorder=4)
            ax.scatter([snaps[2][key]], [y], s=60, color=col, zorder=5,
                       marker="s" if side else "o", edgecolor=SURFACE, lw=1)
        for k, txt, yy in (("ft", "learned", len(order) - 0.3),
                           ("ret", "never taught", len(order) - 0.75)):
            v = ANCHOR[k][key]
            ax.axvline(v, color=OTHER2, lw=1, ls="--")
            ax.text(v, yy, txt + " ", fontsize=8, color=MUTED, ha="right",
                    backgroundcolor=SURFACE)
        ax.set_xscale("log")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order[::-1] if j == 0 else [""] * len(order))
        ax.set_ylim(-0.6, len(order))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab + ("  (right = knows)" if better_hi else "  (left = knows)"))
        if key == "tr":
            ax.invert_xaxis()
            ax.set_xlabel(lab + "  (axis flipped: right = knows)")
        ax.set_title(("A" if j == 0 else "B") + ". Before relearning (open) -> after 3 "
                     "epochs (filled)\nsmall dots = relearn ep 1, 2; squares = 25-epoch "
                     "reruns", loc="left", fontsize=9.5)

    # C. recovery on TR vs on P(gold), matched arms, relearn ep 2
    ax = fig.add_subplot(gs[2])
    ft = ANCHOR["ft"]
    rec = []
    for lg in LANGS:
        _, _, _, s0, snaps = R[f"{lg} matched (ep {prereg()[lg]['epoch']})"]
        s2 = snaps[1]
        rtr = 100 * (s0["tr"] - s2["tr"]) / (s0["tr"] - ft["tr"])
        rp = 100 * (s2["p"] - s0["p"]) / (ft["p"] - s0["p"])
        rec.append((lg, rtr, rp))
    rec.sort(key=lambda r: -(r[1] + r[2]))
    for i, (lg, rtr, rp) in enumerate(rec):
        y = len(rec) - 1 - i
        ax.plot([rtr, rp], [y, y], color=ARM[lg], lw=1.2, alpha=0.6)
        ax.scatter([rtr], [y], s=60, color=ARM[lg], zorder=4, edgecolor=SURFACE, lw=1)
        ax.scatter([rp], [y], s=60, facecolor=SURFACE, edgecolor=ARM[lg], lw=1.8, zorder=4)
        for v, dy in ((rtr, -0.32), (rp, 0.22)):
            ax.text(v, y + dy, f"{v:.0f}", fontsize=8, ha="center", color=INK)
    ax.set_yticks(range(len(rec)))
    ax.set_yticklabels([f"unlearned in {r[0]}" for r in rec][::-1])
    ax.set_ylim(-0.7, len(rec) - 0.3)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("recovery, % of what unlearning removed")
    rank = lambda i: " < ".join(r[0] for r in sorted(rec, key=lambda r: r[i]))
    ax.set_title(f"C. Recovery, matched arms, relearn ep 2\nfilled = on TR:  {rank(1)}"
                 f"\nopen = on P(gold):  {rank(2)}", loc="left", fontsize=9.5)
    title(fig, "Plan 1: more unlearning epochs make forgetting more durable, not less",
          "Benign relearning on French retain99 (n=1500, 3 epochs), probed in French. "
          "Matched rows are Stage 3 (a different scoring group). Single seed.")
    fig.subplots_adjust(top=0.78, left=0.13, right=0.97, bottom=0.12)
    save(fig, "2_durability.png")


# ---- 3. names are spared, then overtaken ---------------------------------------------
def near_certain_by_epoch(arm, text):
    head, *eps = jl(RES / "plan1" / f"tokens_ul{arm}_s42.jsonl")
    T = head["texts"]
    base = eps[0]["results"]
    keys = [k for k, t in T.items() if t["split"] == "forget" and t["lang"] == text]
    cls = {k: cc.classify({**T[k], "answer": T[k]["answer_trained"]})[0] for k in keys}
    out = {c: [] for c in ("name exact", "other, copied", "other, not copied")}
    for e in eps:
        for c in out:
            n = k_ = 0
            for k in keys:
                for i, ci in enumerate(cls[k]):
                    if ci == c and math.exp(base[k]["lp"][i]) > 0.999:
                        n += 1
                        k_ += base[k]["lp"][i] - e["results"][k]["lp"][i] > DESTROY
            out[c].append(100 * k_ / n)
    return [e["epoch"] for e in eps], out


def checkpoint_split():
    """% destroyed among near-certain tokens, own text: every matched arm + fr/en @25."""
    d = json.load(open(STUDY / "results" / "per_token_map.json", encoding="utf-8"))
    R = {cc.label(c): v for c, v in d["results"].items()}
    for lg in ("fr", "en"):
        e = json.load(open(RES / "plan1" / f"per_token_ul{lg}_s42_ep25.json", encoding="utf-8"))
        R[f"{lg}@25"] = next(iter(e["results"].values()))
    T = {k: t for k, t in d["texts"].items() if t["split"] == "forget"}
    cls = {k: cc.classify(t)[0] for k, t in T.items()}
    pr = prereg()
    out = []
    for m, lg, lab in [(f"unl_{l}", l, f"{l} matched (ep {pr[l]['epoch']})") for l in LANGS] \
            + [("fr@25", "fr", "fr at ep 25"), ("en@25", "en", "en at ep 25")]:
        cnt = {"name exact": [0, 0], "other, not copied": [0, 0]}
        for k, t in T.items():
            if t["lang"] != lg:
                continue
            for i, c in enumerate(cls[k]):
                if c in cnt and math.exp(R["fr_ft"][k]["lp"][i]) > 0.999:
                    cnt[c][0] += 1
                    cnt[c][1] += R["fr_ft"][k]["lp"][i] - R[m][k]["lp"][i] > DESTROY
        out.append((lab, lg, {c: 100 * v[1] / v[0] for c, v in cnt.items()}))
    return out


def fig_names():
    fig = plt.figure(figsize=(15, 4.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.45, 1.0], wspace=0.25)
    style = {"name exact": (ACCENT, "-", 2.4, "copied names"),
             "other, copied": (OTHER1, "--", 1.5, "other copied text"),
             "other, not copied": (OTHER2, "-", 1.5, "other text")}
    pr = prereg()
    for j, (arm, text) in enumerate((("fr", "fr"), ("en", "en"), ("en", "fr"))):
        ax = fig.add_subplot(gs[j])
        xs, out = near_certain_by_epoch(arm, text)
        for c, (col, ls, lw, lab) in style.items():
            ax.plot(xs, out[c], color=col, ls=ls, lw=lw)
            ax.annotate(f"{lab} {out[c][-1]:.0f}%", (xs[-1], out[c][-1]), xytext=(-2, 7),
                        textcoords="offset points", ha="right", fontsize=8, color=col,
                        fontweight="bold" if c == "name exact" else "normal")
        cross = next((x for x, n, o in zip(xs, out["name exact"], out["other, not copied"])
                      if n > o and x > 0), None)
        ax.axvline(pr[arm]["epoch"], color=MUTED, ls=":", lw=1)
        ax.text(pr[arm]["epoch"] - 0.4, 50, "matched", rotation=90, fontsize=8, color=MUTED,
                ha="right")
        ax.set_ylim(0, 65)
        ax.set_xlim(0, 26)
        ax.set_xlabel("unlearning epoch")
        if j == 0:
            ax.set_ylabel("% destroyed (> 5 nats lost), near-certain tokens")
        ax.set_title(f"{'ABC'[j]}. {arm} unlearning, {text} text\nnames overtake other text "
                     f"at epoch {cross}", loc="left", fontsize=9.5)

    ax = fig.add_subplot(gs[4])       # gs[3] is an empty column for D's row labels
    rows = checkpoint_split()
    h = 0.38
    for i, (lab, lg, v) in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, c, col in ((h / 2, "name exact", ACCENT), (-h / 2, "other, not copied", OTHER2)):
            ax.barh(y + dy, v[c], height=h - 0.04, color=col)
            ax.text(v[c] + 1, y + dy, f"{v[c]:.0f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0, 110)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% destroyed, near-certain tokens, own text")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ACCENT, label="copied names"),
                       Patch(color=OTHER2, label="other text")], loc="lower right", fontsize=8)
    ax.set_title("D. At each checkpoint\nStage 6's ja/ru vs fr/en/id gap\n= when you look",
                 loc="left", fontsize=9.5)
    title(fig, "Being in the question delays a name's destruction; it does not protect it",
          "Tokens fr_ft gave p > 0.999. 'Copied name' = a name part written exactly as in the "
          "question (Step 0 tagger). Single seed.")
    fig.subplots_adjust(top=0.78, left=0.05, right=0.99, bottom=0.12)
    save(fig, "3_names_delayed.png")


# ---- Plan 2 -------------------------------------------------------------------------
def plan2():
    files = {"fr_ft": FR_FT, "fr_retain": FR_RET, "fr at ep 25": "ulfr_s42_ep25",
             "en at ep 25": "ulen_s42_ep25"}
    for lg in LANGS:
        files[f"{lg} matched"] = ck_name(lg, prereg()[lg])
    M = {}
    for k, f in files.items():
        recs = json.load(open(RES / "plan2" / f"{f}.json", encoding="utf-8"))["records"]
        for r in recs:
            r["pred"] = clean_prediction(r["gen"])
            r.update(score_prediction(r["pred"], r["refs"]))
        M[k] = recs
    return M


def fig_pieces(M):
    names = {n["id"]: n for n in json.load(open(SIDE / "probes" / "names.json",
                                                encoding="utf-8"))["names"]}
    cols = [("ja matched", "copy_ja", "ja matched\nkatakana"), ("ru matched", "copy_ru",
            "ru matched\nCyrillic"), ("fr at ep 25", "copy_lat", "fr at ep 25\nLatin"),
            ("en at ep 25", "copy_lat", "en at ep 25\nLatin"), ("fr matched", "copy_lat",
            "fr matched\nLatin"), ("en matched", "copy_lat", "en matched\nLatin"),
            ("id matched", "copy_lat", "id matched\nLatin")]
    rows = [("a_basil", "forget author"), ("a_nikolai", "forget author"),
            ("c_naguib", "shares Mahfouz"), ("c_karim", "shares Al-Kuwaiti; invented"),
            ("c_rathbone", "shares Basil"), ("c_tremaine", "shares Basil; invented"),
            ("c_gogol", "shares Nikolai"), ("c_vesk", "shares Nikolai; invented"),
            ("c_rimsky", "shares Nikolai"), ("c_tesla", "Nikola, not Nikolai"),
            ("c_real", "10 real names, nothing shared"),
            ("c_invented", "5 invented names, nothing shared")]

    def lp(model, task, sel):
        return st.mean(r["lp_mean"] for r in M[model] if r["task"] == task and sel(r))

    grid = []
    for rid, _ in rows:
        sel = (lambda r, g=rid: r["group"] == g) if rid in ("c_real", "c_invented") else \
              (lambda r, i=rid: r["name_id"] == i)
        grid.append([lp(m, t, sel) - lp("fr_ft", t, sel) for m, t, _ in cols])
    fig, ax = plt.subplots(figsize=(11, 6.6))
    cmap = LinearSegmentedColormap.from_list("dmg", ["#fcfcfb", "#9ec3ef", "#2a78d6",
                                                     "#0d3b78"])
    norm = SymLogNorm(linthresh=0.5, vmin=0, vmax=80)
    ax.imshow([[-v for v in row] for row in grid], cmap=cmap, norm=norm, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            txt = "0.0" if abs(v) < 0.05 else f"{v:+.1f}" if abs(v) < 10 else f"{v:+.0f}"
            ax.text(j, i, txt, ha="center",
                    va="center", fontsize=8.5, color="#ffffff" if -v > 8 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[2] for c in cols], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([(names[r]["lat"] if r in names else "") + f"   ({d})" if r in names
                        else d for r, d in rows], fontsize=8.5)
    for y in (1.5, 9.5):
        ax.axhline(y, color=INK, lw=1.2)
    ax.axvline(1.5, color=INK, lw=1.2)
    ax.axvline(3.5, color=INK, lw=1.2)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    title(fig, "Plan 2: unlearning damages pieces of the forget names, in anyone's name",
          "Copy task, mean log p per token of the correct name minus fr_ft's (darker = more "
          "damage; -10 = each token ~22,000x less likely). Mean over 5 instruction languages.")
    fig.subplots_adjust(top=0.80, left=0.33, right=0.98, bottom=0.03)
    save(fig, "4_name_pieces.png")


KANA, HAN = re.compile(r"[゠-ヿ]"), re.compile(r"[一-鿿]")
CYR, LAT = re.compile(r"[Ѐ-ӿ]"), re.compile(r"[A-Za-zÀ-ÿ]")


def outcome(r, target):
    if r["exact_loose"]:
        return "correct"
    others = [rx for name, rx in (("kana", KANA), ("cyr", CYR), ("lat", LAT), ("han", HAN))
              if name != target]
    return "another script" if any(rx.search(r["pred"]) for rx in others) else "wrong spelling"


def fig_switch(M):
    models = ["fr_ft", "fr_retain", "fr matched", "en matched", "id matched", "ja matched",
              "ru matched", "fr at ep 25", "en at ep 25"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    cat = [("correct", "#e4e3de"), ("wrong spelling", OTHER1), ("another script", ACCENT)]
    for ax, (task, target, lab) in zip(axes, (("copy_ja", "kana", "katakana"),
                                              ("copy_ru", "cyr", "Cyrillic"))):
        for i, m in enumerate(models):
            rs = [r for r in M[m] if r["task"] == task and r["group"] in ("c_real",
                                                                           "c_invented")]
            n = len(rs)
            left = 0
            y = len(models) - 1 - i
            for c, col in cat:
                v = 100 * sum(outcome(r, target) == c for r in rs) / n
                ax.barh(y, v, left=left, color=col, height=0.7, edgecolor=SURFACE, lw=1)
                if v >= 6 and c != "correct":
                    ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color="#ffffff")
                if c == "correct":
                    ax.text(1.5, y, f"{v:.0f}% correct", va="center", fontsize=8, color=INK)
                left += v
        ax.set_xlim(0, 100)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models[::-1])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"% of {n} prompts (15 names x 5 instruction languages)")
        ax.set_title(f"Asked to copy a {lab} name", loc="left", fontsize=10)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=col, label=c) for c, col in cat], loc="lower center",
               bbox_to_anchor=(0.55, 0.0), ncol=3, fontsize=8.5)
    title(fig, "Plan 2: the ja arm avoids writing names in katakana",
          "Names that share no piece with a forget author. Failures are mostly the name in "
          "another script (e.g. アルベルト・アインシュタイン -> Albert Einstein), not misspellings.")
    fig.texts[-1].set_fontname(cc_font())
    fig.subplots_adjust(top=0.82, left=0.1, right=0.98, bottom=0.2)
    save(fig, "5_script_switch.png")


NAME_GROUPS = [("forget authors", ("a",)), ("share a piece with them", ("c_real_overlap",
               "c_invented_overlap")), ("unrelated names", ("c_real", "c_invented")),
               ("retain authors", ("b",))]


def fig_translate(M):
    """Plan 2's transliteration tasks: is the skill damaged, or only the forget names?"""
    def recs(m, task, groups):
        return [r for r in M[m] if r["task"] == task and r["group"] in groups]

    def dlp(m, task, groups):
        return (st.mean(r["lp_mean"] for r in recs(m, task, groups))
                - st.mean(r["lp_mean"] for r in recs("fr_ft", task, groups)))

    others = [lg for lg in LANGS if lg not in ("ja", "ru")]
    blocks = [("Latin -> katakana", "lat2ja", others + ["ru", "ja"]),
              ("Latin -> Cyrillic", "lat2ru", others + ["ja", "ru"]),
              ("back to Latin", None, ["ja", "ru"])]
    cols = []                                    # (model, task, x label)
    for _, task, arms in blocks:
        for lg in arms:
            t = task or {"ja": "ja2lat", "ru": "ru2lat"}[lg]
            cols.append((f"{lg} matched", t, lg + ("\nfrom kana" if t == "ja2lat" else
                                                   "\nfrom Cyr." if t == "ru2lat" else "")))
    grid = [[dlp(m, t, g) for m, t, _ in cols] for _, g in NAME_GROUPS]

    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.25], hspace=0.42, wspace=0.2)
    ax = fig.add_subplot(gs[0, :])
    cmap = LinearSegmentedColormap.from_list("dmg", ["#fcfcfb", "#9ec3ef", "#2a78d6",
                                                     "#0d3b78"])
    ax.imshow([[max(0, -v) for v in row] for row in grid], cmap=cmap,
              norm=SymLogNorm(linthresh=0.5, vmin=0, vmax=80), aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            txt = "0.0" if abs(v) < 0.05 else f"{v:+.1f}" if abs(v) < 10 else f"{v:+.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="#ffffff" if -v > 8 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[2] for c in cols], fontsize=8.5)
    ax.set_yticks(range(len(NAME_GROUPS)))
    ax.set_yticklabels([g for g, _ in NAME_GROUPS])
    edge = 0
    for name, _, arms in blocks:
        ax.text(edge + (len(arms) - 1) / 2, -0.75, name, ha="center", fontsize=9.5,
                color=INK, fontweight="bold")
        edge += len(arms)
        if edge < len(cols):
            ax.axvline(edge - 0.5, color=INK, lw=1.2)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("A. Change in log p per token of the correct transliteration vs fr_ft "
                 "(darker = more damage; columns = the matched unlearning arm)",
                 loc="left", fontsize=10, pad=24)

    cat = [("correct", "#e4e3de"), ("wrong spelling", OTHER1), ("another script", ACCENT)]
    for k, (task, target, arm, lab) in enumerate((("lat2ja", "kana", "ja", "katakana"),
                                                  ("lat2ru", "cyr", "ru", "Cyrillic"))):
        ax = fig.add_subplot(gs[1, k])
        rows = [(f"{g}: {m.split()[0]}", recs(m, task, gs_))
                for g, gs_ in NAME_GROUPS for m in ("fr_ft", f"{arm} matched")]
        for i, (lab_r, rs) in enumerate(rows):
            y = len(rows) - 1 - i - (i // 2) * 0.5
            left = 0
            for c, col in cat:
                v = 100 * sum(outcome(r, target) == c for r in rs) / len(rs)
                ax.barh(y, v, left=left, color=col, height=0.8, edgecolor=SURFACE, lw=1)
                if c == "correct":
                    ax.text(102, y, f"{v:.0f}%", va="center", fontsize=8.5, color=INK,
                            clip_on=False)
                elif v >= 8:
                    ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color="#ffffff")
                left += v
        ax.set_yticks([len(rows) - 1 - i - (i // 2) * 0.5 for i in range(len(rows))])
        ax.set_yticklabels([r[0] for r in rows] if k == 0 else
                           [r[0].split(": ")[1] for r in rows], fontsize=8.5)
        ax.set_xlim(0, 100)
        ax.text(102, len(rows) - 0.4, "correct", fontsize=8.5, color=MUTED, clip_on=False)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("% of prompts (each name x 5 instruction languages)")
        ax.set_title(f"{'BC'[k]}. Latin -> {lab}: what fr_ft and the {arm} arm write",
                     loc="left", fontsize=10)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=col, label=c) for c, col in cat], loc="lower center",
               bbox_to_anchor=(0.55, 0.0), ncol=3, fontsize=8.5)

    ja = {g: grid[i][cols.index(("ja matched", "lat2ja", "ja"))]
          for i, (g, _) in enumerate(NAME_GROUPS)}
    title(fig, f"Plan 2: ja unlearning breaks transliteration of the forget names "
               f"({ja['forget authors']:+.0f} nats/token), not the skill "
               f"(unrelated names {ja['unrelated names']:+.1f})",
          "Translating a name between scripts. Most of the extra failures on unrelated and "
          "retain names are the name written in another script, as in figure 5. Retain authors are not "
          "neutral (each arm's retain term\ntrained on them in its own language). Forget "
          "authors = 2 names, piece-sharing = 8, unrelated = 15, retain = 21; each x 5 "
          "instruction languages. Single seed.")
    fig.subplots_adjust(top=0.86, left=0.14, right=0.96, bottom=0.1)
    save(fig, "15_name_translation.png")


def cc_font():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Arial Unicode MS", "Hiragino Sans", "PingFang SC")
                 if f in have), "DejaVu Sans")


# ---- 6. the blanket effect by epoch (Plan 1's original question) ---------------------
def token_split(base, cur, keys):
    """% of answer tokens destroyed (> 5 nats) / untouched (< 0.1), and % destroyed among
    the tokens the learned model was near-certain of -- Stage 6 finding 5's measure."""
    n = un = hit = ns = hs = 0
    for k in keys:
        for b, x in zip(base[k]["lp"], cur[k]["lp"]):
            d = b - x
            n += 1
            un += abs(d) < 0.1
            hit += d > DESTROY
            if math.exp(b) > 0.999:
                ns += 1
                hs += d > DESTROY
    return {"hit": 100 * hit / n, "un": 100 * un / n, "sure": 100 * hs / ns}


def fig_blanket():
    pr = prereg()
    traj = {}
    for arm in ("fr", "en"):
        head, *eps = jl(RES / "plan1" / f"tokens_ul{arm}_s42.jsonl")
        keys = [k for k, t in head["texts"].items() if t["split"] == "forget" and t["lang"] == arm]
        traj[arm] = ([e["epoch"] for e in eps],
                     [token_split(eps[0]["results"], e["results"], keys) for e in eps])
    d = json.load(open(STUDY / "results" / "per_token_map.json", encoding="utf-8"))
    R = {cc.label(c): v for c, v in d["results"].items()}
    matched = {}
    for lg in LANGS:
        keys = [k for k, t in d["texts"].items() if t["split"] == "forget" and t["lang"] == lg]
        matched[lg] = token_split(R["fr_ft"], R[f"unl_{lg}"], keys)

    fig = plt.figure(figsize=(15, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.28, 1.05], wspace=0.25)
    for j, (key, ylab, sub) in enumerate((
            ("hit", "% of own answer tokens destroyed (> 5 nats)", "all answer tokens"),
            ("sure", "% destroyed among near-certain tokens",
             "only tokens fr_ft gave p > 0.999 (removes the head start of unsure text)"))):
        ax = fig.add_subplot(gs[j])
        for arm in ("fr", "en"):
            xs, v = traj[arm]
            ys = [s[key] for s in v]
            ax.plot(xs, ys, color=ARM[arm], lw=2)
            ax.annotate(f"{arm} {ys[-1]:.0f}%", (xs[-1], ys[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=ARM[arm], fontweight="bold")
        for lg in LANGS:                  # Stage 6 matched checkpoints (fr/en sit on the lines)
            e, y = pr[lg]["epoch"], matched[lg][key]
            ax.plot(e, y, "o", ms=8, color=ARM[lg], mec=SURFACE, mew=1.5, zorder=5)
            if lg in ("id", "ru", "ja"):
                ax.annotate(f"{lg} {y:.0f}%", (e, y), xytext=(-6, 6), textcoords="offset points",
                            ha="right", fontsize=8.5, color=ARM[lg], fontweight="bold")
        ax.axvline(25, color=MUTED, ls="--", lw=1)
        ax.set_xlim(0, 29)
        ax.set_ylim(0, 100)
        ax.set_xlabel("unlearning epoch")
        ax.set_ylabel(ylab)
        gap = matched["ja"][key] - max(traj[a][1][-1][key] for a in ("fr", "en"))
        ax.set_title(f"{'AB'[j]}. {sub}\nat epoch 25 ja is still {gap:.0f} points above "
                     "fr and en", loc="left", fontsize=9.5)

    ax = fig.add_subplot(gs[3])           # gs[2] leaves room for C's row labels
    rows = [(f"{lg} matched (ep {pr[lg]['epoch']})", matched[lg]) for lg in LANGS]
    rows.insert(1, ("fr at ep 25", traj["fr"][1][-1]))
    rows.insert(3, ("en at ep 25", traj["en"][1][-1]))
    cols = ["#e8e7e2", "#9fc3f0", "#184f95"]
    for i, (lab, s) in enumerate(rows):
        y = len(rows) - 1 - i
        parts = (s["un"], 100 - s["un"] - s["hit"], s["hit"])
        left = 0
        for v, c in zip(parts, cols):
            ax.barh(y, v, left=left, height=0.64, color=c, edgecolor=SURFACE, lw=2)
            if v >= 8:
                ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if c == cols[-1] else INK)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% of own answer tokens")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(
        cols, ["untouched (< 0.1)", "moved", "destroyed (> 5)"])], ncol=3, fontsize=8,
        loc="upper left", bbox_to_anchor=(0, -0.13))
    ax.set_title("C. Stage 6's surgical-vs-blanket bars,\nwith fr and en at 25 epochs added",
                 loc="left", fontsize=9.5)
    title(fig, "More epochs make fr and en blanker, but at the same 25 epochs ja is still the "
          "most blanket",
          "Each arm teacher-forced on its OWN forget01 answers vs fr_ft (fr text for fr, English "
          "for en, ...). Lines = 25-epoch reruns; dots = Stage 6 matched checkpoints. Single seed.")
    fig.subplots_adjust(top=0.78, left=0.05, right=0.99, bottom=0.2)
    save(fig, "6_blanket_by_epoch.png")


def main():
    fig_effort()
    fig_durability()
    fig_names()
    M = plan2()
    fig_pieces(M)
    fig_switch(M)
    fig_translate(M)
    fig_blanket()


if __name__ == "__main__":
    main()
