"""Figures 20-23: the results produced for the 2026-11 feedback round. Local, in .venv-plot:

    .venv-plot/bin/python studies/learn_french/side_checks/plots/plot_feedback_round.py

  20_metric_consistency   supervisor 1b  -- do TR / P(gold) / NLI agree, fact by fact?
  21_carry_intervention   supervisor 1c  -- bridge gate; Experiment D (layer restriction)
  22_token_damage         Tongyao 11-13  -- does a token score explain selective vs uniform?
  23_coupling_scores      Tongyao 11-13  -- a model-based, language-free coupling score
(1a = fig 18, 1d = figures/storage_phase2a.png, 1c durability also = fig 19.)

Every number in a title is computed from the data plotted. Single seed unless a panel says
otherwise; 40 facts about 2 entities.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY / "plots"))
sys.path.insert(0, str(STUDY / "scripts"))
from plot_side_checks import ARM, GRID, INK, MUTED, SURFACE, save, title  # noqa: E402
import plot_stage3 as S3  # noqa: E402
import read_metric_consistency as RMC  # noqa: E402
import plot_depth_curve as DC  # noqa: E402
import bridge_gate_report as BG  # noqa: E402

P3 = SIDE / "results" / "plan3"
NEUTRAL = "#8a8984"
SEQ3 = ("#c9dcf3", "#6fa3e3", "#1f5fae")          # one hue, light -> dark (bottom -> top)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def label(ax, text):
    ax.set_title(text, fontsize=10, color=INK, loc="left")


# ---- 20. metric consistency (1b) -------------------------------------------------------
def fig_metric_consistency():
    data = RMC.relearned_en_ja()
    metrics = (("prob", "P(gold)", -1), ("nli", "NLI", -1), ("tr", "truth ratio", +1))
    fig = plt.figure(figsize=(14.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.25], wspace=0.34)
    ax = fig.add_subplot(gs[0])
    share_all = {}
    for yi, (key, name, sign) in enumerate(metrics):
        shares = []
        for seed, by_rl in data.items():
            for rl, arms in by_rl.items():
                e, j = S3.per_fact(arms["en"], key), S3.per_fact(arms["ja"], key)
                en_more = sum(sign * (a - b) > 1e-9 for a, b in zip(e, j))
                ja_more = sum(sign * (b - a) > 1e-9 for a, b in zip(e, j))
                s = en_more / (en_more + ja_more)
                shares.append(s)
                ax.scatter(s * 100, yi + (0.12 if rl == "fr" else -0.12), s=46, zorder=3,
                           color=INK if rl == "fr" else SURFACE, edgecolor=INK, linewidth=1.4)
        share_all[name] = shares
    ax.axvline(50, color=MUTED, lw=1, ls="--")
    ax.text(50, 2.55, " coin flip", color=MUTED, fontsize=8, va="bottom")
    ax.set_yticks(range(3), [m[1] for m in metrics])
    ax.set_ylim(-0.5, 3.2)
    ax.set_xlim(30, 100)
    ax.set_xlabel("% of facts where en is MORE durable than ja\n(ties dropped; 3 seeds x 2 relearn languages)")
    ax.legend(handles=[Line2D([], [], marker="o", lw=0, color=INK, label="relearned in French"),
                       Line2D([], [], marker="o", lw=0, mfc=SURFACE, mec=INK, label="relearned in English")],
              fontsize=8, loc="upper left")
    label(ax, "A  Per fact: how often does en beat ja?")

    arms = data[42]["fr"]
    for col, (key, name, sign) in zip((1, 2), (("tr", "truth ratio", +1), ("prob", "P(gold)", -1))):
        ax = fig.add_subplot(gs[col])
        d = [sign * (a - b) for a, b in zip(S3.per_fact(arms["en"], key), S3.per_fact(arms["ja"], key))]
        order = sorted(range(40), key=lambda i: -d[i])
        top = order[:2]
        ax.bar(range(40), [d[i] for i in order], width=0.75,
               color=[ARM["en"] if d[i] > 0 else ARM["ja"] for i in order])
        for rank, i in enumerate(order[:2]):
            ax.annotate(f"fact {i}", (rank, d[i]), textcoords="offset points", xytext=(4, 2),
                        fontsize=8, color=INK)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks([])
        ax.set_xlabel("40 facts, sorted by gap")
        unit = "TR_en - TR_ja" if key == "tr" else "P_ja - P_en"
        ax.set_ylabel(f"per-fact gap ({unit})\n> 0: en more durable")
        rest = [d[i] for i in range(40) if i not in top]
        label(ax, f"{'B' if col == 1 else 'C'}  {name} per fact (seed 42, French relearn)\n"
                  f"    mean {mean(d):+.3f}, median {sorted(d)[19]:+.3f}; w/o top 2: {mean(rest):+.3f}")
    p, t = mean(share_all["P(gold)"]) * 100, mean(share_all["truth ratio"]) * 100
    title(fig, "1b  Metric consistency: all metrics say en is more durable on average, but only "
               "P(gold) says it fact by fact",
          f"Mean share of facts favouring en: P(gold) {p:.0f}%, NLI "
          f"{mean(share_all['NLI']) * 100:.0f}%, truth ratio {t:.0f}%.  The truth-ratio mean gap is "
          f"carried by two facts (B); P(gold)'s is spread across facts (C) -> P(gold) is the primary metric.")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.17)
    save(fig, "20_metric_consistency.png")


# ---- 21. carry-over intervention (1c) ----------------------------------------------------
def layer_shares(run):
    import re
    p = list(json.load(open(P3 / f"layer_diff_{run}.json"))["pairs"].values())[0]["layers"]
    L = {}
    for k, v in p.items():
        m = re.search(r"layers?\.(\d+)\.", k) or re.fullmatch(r"(?:L)?(\d+)", k)
        if m:
            L[int(m.group(1))] = L.get(int(m.group(1)), 0) + v["delta_sq"]
    tot = sum(v["delta_sq"] for v in p.values())
    return [sum(v for k, v in L.items() if lo <= k < hi) / tot for lo, hi in ((0, 12), (12, 24), (24, 36))]


def interp(curve, x):
    s = sorted(curve)
    if x <= s[0][0]:
        return s[0][1], s[0][0]
    if x >= s[-1][0]:
        return s[-1][1], s[-1][0]
    for (x0, y0), (x1, y1) in zip(s, s[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0), x


def fig_carry_intervention():
    ptm = json.load(open(STUDY / "results" / "bridge_gate" / "per_token_map.json", encoding="utf-8"))
    R = {BG.role(c): v for c, v in ptm["results"].items() if BG.role(c)}
    T = ptm["texts"]
    ft_ja, ft_en = BG.carry(T, R, "fr_ft", "fr_retain", "ja"), BG.carry(T, R, "fr_ft", "fr_retain", "en")
    br_ja = BG.carry(T, R, "bridge", "frja_retain", "ja")
    need = (ft_ja + ft_en) / 2

    fig = plt.figure(figsize=(14, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.1, 1.2], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    vals = [("fr_ft, ja", ft_ja, ARM["ja"]), ("bridge, ja", br_ja, NEUTRAL),
            ("fr_ft, en\n(target)", ft_en, ARM["en"])]
    ax.barh(range(3), [v for _, v, _ in vals], color=[c for _, _, c in vals], height=0.55)
    for y, (_, v, _) in enumerate(vals):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(need, color=INK, ls="--", lw=1)
    ax.text(need, 2.45, f" needed {need:.2f}", fontsize=8, color=INK, va="bottom")
    ax.set_yticks(range(3), [n for n, _, _ in vals])
    ax.set_ylim(-0.5, 2.8)
    ax.set_xlim(0, max(ft_en, need) * 1.25)
    ax.set_xlabel("carry-over (nats per answer token)")
    label(ax, f"A  Bridge: raise Japanese carry-over?\n    it FELL {ft_ja:.2f} -> {br_ja:.2f}: gate failed")

    runs = [("en, all layers", "ulen_s1_lr5p0e-06_tr0p828"),
            ("en, layers 24-35 only", "ulen_s42_lr5p0e-06_pg0p490_L24-35"),
            ("ja, all layers", "ulja_s42_lr1p0e-05_tr0p828"),
            ("ja, layers 0-23 only", "ulja_s42_lr1p5e-05_tr0p828_L0-23")]
    ax = fig.add_subplot(gs[1])
    for y, (name, run) in enumerate(runs):
        left = 0
        for k, s in enumerate(layer_shares(run)):
            ax.barh(y, s * 100, left=left, color=SEQ3[k], height=0.55, edgecolor=SURFACE, linewidth=2)
            if s > 0.08:
                ax.text(left + s * 50, y, f"{s * 100:.0f}%", ha="center", va="center", fontsize=8,
                        color=INK if k < 2 else SURFACE)
            left += s * 100
    ax.set_yticks(range(len(runs)), [n for n, _ in runs])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the weight change (%)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in SEQ3],
              labels=["layers 0-11", "12-23", "24-35 (top)"], fontsize=8, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.17))
    label(ax, "B  Experiment D: where the edit was allowed\n    (restriction held: 0% outside the range)")

    pts = DC.load()
    curve = {(lg, rl): [(p["pg"], p["rel"][rl]["pg"]) for p in pts
                        if p["lang"] == lg and p["seed"] == 42 and rl in p["rel"]]
             for lg in ("en", "ja") for rl in ("fr",)}
    ax = fig.add_subplot(gs[2])
    rows = []
    for name, run, lg in (("ja, layers 0-23 only", "ulja_s42_lr1p5e-05_tr0p828_L0-23", "ja"),
                          ("en, layers 24-35 only", "ulen_s42_lr5p0e-06_pg0p490_L24-35", "en")):
        stop = json.load(open(P3 / "measure" / f"{run}.json"))["summary"]["prob_mean"]
        rel = json.load(open(P3 / "relearn" / f"relearn_{run}_via_retain_langfr_ep3.json"))["summary"]["prob_mean"]
        own, _ = interp(curve[(lg, "fr")], stop)
        other_lg = "en" if lg == "ja" else "ja"
        oth, at = interp(curve[(other_lg, "fr")], stop)
        rows.append((name, stop, rel, own, oth, other_lg, at, lg))
    for y, (name, stop, rel, own, oth, olg, at, lg) in enumerate(rows):
        ax.plot([min(rel, own, oth), max(rel, own, oth)], [y, y], color=GRID, lw=3, zorder=1)
        ax.scatter(own, y, s=70, color=ARM[lg], zorder=3)
        ax.scatter(oth, y, s=70, color=ARM[olg], zorder=3)
        ax.scatter(rel, y, s=110, marker="D", color=SURFACE, edgecolor=ARM[lg], linewidth=2.2, zorder=4)
        ax.text(rel, y + 0.2, f"restricted\n{rel:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(own, y - 0.18, f"{lg}, all\nlayers {own:.3f}", ha="center", va="top", fontsize=7.5, color=MUTED)
        note = "" if abs(at - stop) < 1e-9 else f"\n(nearest: @{at:.2f})"
        ax.text(oth, y - 0.18, f"{olg}, all\nlayers {oth:.3f}{note}", ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(rows)), [f"{r[0].replace(' only', '')}\nstop {r[1]:.3f}" for r in rows], fontsize=8)
    ax.set_ylim(-0.85, 1.75)
    lo = min(min(r[2], r[3], r[4]) for r in rows); hi = max(max(r[2], r[3], r[4]) for r in rows)
    ax.set_xlim(lo - 0.03, hi + 0.03)
    ax.set_xlabel("relearned French P(gold), French relearn\n(lower = more durable)")
    label(ax, "C  After relearning: each restricted run stays\n    with its OWN language, not the other one")
    title(fig, "1c  Testing carry-over as a cause: the bridge could not raise it; restricting the "
               "layers did not change durability",
          "B is the manipulation, C is the outcome (same data as fig 19, curves interpolated at each "
          "run's stop depth). Single seed; seed spread of the all-layer arms <= 0.035.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.2)
    save(fig, "21_carry_intervention.png")


# ---- 22. token damage regimes (1e) ------------------------------------------------------
BINS = ((-99, -6), (-6, -3), (-3, -1), (-1, -0.1), (-0.1, 0.01))
BIN_LABELS = ("< -6", "-6..-3", "-3..-1", "-1..-0.1", "> -0.1\n(near-certain)")


def run_damage(run, lg):
    with open(P3 / f"tokens_{run}.jsonl", encoding="utf-8") as fh:
        H = json.loads(next(fh))
        rows = [json.loads(x) for x in fh]
    e0, eN = rows[0], rows[-1]
    pts = []
    for i in range(40):
        k = f"forget:{lg}:{i}"
        pts += [(e0["results"][k]["lp"][j], eN["results"][k]["lp"][j] - e0["results"][k]["lp"][j])
                for j in range(len(H["texts"][k]["tokens"]))]
    return pts


def fig_token_damage():
    d = json.load(open(STUDY / "results" / "per_token_map.json", encoding="utf-8"))
    T, R = d["texts"], d["results"]
    ft = next(k for k in R if k.endswith("_full_full_qwen3-8b_fr"))
    unl = {k.split("_ul")[1][:2]: k for k in R if "tr_levels" in k}
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    near = {}
    for lg in ("en", "id", "ru", "ja"):
        pts = []
        for i in range(40):
            k = f"forget:{lg}:{i}"
            pts += [(R[ft][k]["lp"][j], R[unl[lg]][k]["lp"][j] - R[ft][k]["lp"][j])
                    for j, e in enumerate(T[k]["is_eos"]) if not e]
        ys = [-mean([dd for s, dd in pts if lo <= s < hi]) for lo, hi in BINS]
        near[lg] = ys[-1]
        ax.plot(range(5), ys, marker="o", color=ARM[lg], lw=2, ms=7)
        ax.text(4.12, ys[-1] + {"en": -1.6, "id": 1.6}.get(lg, 0), f"{lg}  {ys[-1]:.1f}",
                va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(5), BIN_LABELS, fontsize=8)
    ax.set_xlim(-0.3, 4.9)
    ax.set_ylim(0, None)
    ax.set_xlabel("how sure the learned model was of the token (fr_ft log p, binned)")
    ax.set_ylabel("nats lost per token after unlearning\n(own-language forget text)")
    label(ax, "A  Stage 3 matched arms: en/id/ru spare the tokens the model\n    was sure of; ja destroys them too")

    ax = axs[1]
    groups = [
        ("en", [("seed 1", "ulen_s1_lr5p0e-06_tr0p828", "en"),
                ("seed 2", "ulen_s2_lr5p0e-06_tr0p828", "en"),
                ("epoch 15", "ulen_s42_lr5p0e-06_ep15", "en"),
                ("layers 24-35 only", "ulen_s42_lr5p0e-06_pg0p490_L24-35", "en"),
                ("NPO", "ulen_s42_lr1p0e-05_pg0p470_npo", "en")]),
        ("ar", [("Arabic", "ular_s42_lr5p0e-06_pg0p470", "ar")]),
        ("ja", [("seed 1", "ulja_s1_lr5p0e-06_tr0p828", "ja"),
                ("seed 2", "ulja_s2_lr5p0e-06_tr0p828", "ja"),
                ("12 epochs (lr 1.5e-5)", "ulja_s42_lr1p5e-05_tr0p828", "ja"),
                ("layers 0-23 only", "ulja_s42_lr1p5e-05_tr0p828_L0-23", "ja")]),
        ("jaro", [("romaji (no stop reached)", "uljaro_s42_lr5p0e-06_pg0p470", "jaro")]),
    ]
    y, ticks, vals = 0, [], {}
    for grp, runs in groups:
        for name, run, lg in runs:
            if not (P3 / f"tokens_{run}.jsonl").exists():
                continue
            pts = run_damage(run, lg)
            v = -mean([dd for s, dd in pts if s >= -0.1])
            vals[(grp, name)] = v
            col = ARM.get(grp, NEUTRAL)
            ax.plot([0.3, v], [y, y], color=GRID, lw=1.2, zorder=1)
            ax.scatter(v, y, s=60, color=col, zorder=3)
            ax.text(v * 1.12, y, f"{v:.1f}", va="center", fontsize=8, color=INK)
            ticks.append((y, f"{grp}: {name}"))
            y += 1
        y += 0.6
    ax.set_xscale("log")
    ax.set_xlim(0.3, 400)
    ax.set_yticks([t for t, _ in ticks], [n for _, n in ticks], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(5, color=MUTED, ls="--", lw=0.9)
    ax.set_xlabel("nats lost on the NEAR-CERTAIN tokens (log scale; dashed line = 5 nats)")
    label(ax, "B  The split survives seeds, run length, layer restriction and\n    NPO; Arabic sits between; romaji text is wiped out")
    title(fig, "1e  Two regimes of damage to the unlearning text: selective (en) vs content-blind (ja)",
          f"Near-certain tokens lose en {near['en']:.1f} / id {near['id']:.1f} / ru {near['ru']:.1f} "
          f"nats but ja {near['ja']:.1f} (A); binning by base-model familiarity or token carry-over "
          f"gives the same picture (read_token_scores.py), so token popularity does not explain it.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.17, wspace=0.42)
    save(fig, "22_token_damage.png")


# ---- 23. coupling scores (1f) -------------------------------------------------------------
def rho(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = mean(ra), mean(rb)
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (
        sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def fig_coupling():
    c = SIDE / "results" / "coupling"
    r1 = json.load(open(c / "scores.json"))
    r2 = json.load(open(c / "scores_contrast.json"))
    src = ["en", "id", "ru", "ar", "ko", "iw", "ja", "fa", "hi", "ja_romaji"]
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axs[0]
    rows = src + ["ctl_retain_ja", "ctl_retain_fr"]
    names = {"ja_romaji": "romaji ja", "ctl_retain_ja": "CONTROL: ja retain (no forget fact)",
             "ctl_retain_fr": "CONTROL: fr retain (no forget fact)"}
    vals = [r1["grad_summed"][s]["cos"] for s in rows]
    cols = [ARM.get(s, NEUTRAL) if not s.startswith("ctl") else INK for s in rows]
    ax.barh(range(len(rows)), vals, color=cols, height=0.6)
    for y, v in enumerate(vals):
        ax.text(v + (0.003 if v >= 0 else -0.003), y, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(rows)), [names.get(s, f"forget: {s}") for s in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.14, 0.1)
    ax.set_xlabel("cosine(gradient on that text, gradient on the French target)")
    ja_f, ja_c = r1["grad_summed"]["ja"]["cos"], r1["grad_summed"]["ctl_retain_ja"]["cos"]
    label(ax, f"A  Round 1 (raw): every non-French text points the same way;\n"
              f"    ja forget {ja_f:+.3f} = ja retain control {ja_c:+.3f} -> not about the fact")

    ax = axs[1]
    carry = [mean(r2["carry_over_per_fact"][s]) for s in src]
    align = [mean([r["cos"] for r in r2["contrast_per_fact"][s]]) for s in src]
    for s, x, yv in zip(src, carry, align):
        col = ARM.get(s, NEUTRAL)
        ax.scatter(x, yv, s=80 if s in ARM else 55, color=col, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate("romaji ja" if s == "ja_romaji" else s, (x, yv), textcoords="offset points",
                    xytext=(6, 4), fontsize=8.5, color=INK)
    ret = [mean(r2["carry_over_per_fact"]["ret_" + s]) for s in src]
    ax.axvspan(min(ret), max(ret), color=GRID, alpha=0.6, zorder=0)
    ax.text(max(ret), max(align) * 0.98, " carry-over of retain\n rows (no forget fact)",
            fontsize=7.5, color=MUTED, va="top")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("carry-over: how much of the French-learned fact the model holds\n"
                  "in that text (nats per answer token, forward pass)")
    ax.set_ylabel("per-fact gradient alignment with the French fact\n"
                  "(same-language retain direction removed; backward pass)")
    r = rho(carry, align)
    label(ax, f"B  Round 2 (each language vs its own control): two independent-\n"
              f"    looking measures rank the 10 unlearning sets alike (Spearman {r:.2f})")
    title(fig, "1f  A model-based, language-free score: how tightly is the unlearning text tied to "
               "the learned fact?",
          "A is the failed first attempt (kept: it is why the control matters). In B, en/id/ru are "
          "coupled, ja/fa/hi weakly, romaji not at all. Alignment values are small (<= 0.015) and "
          "have no shuffled-pairing null yet.")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.79, bottom=0.17, wspace=0.3)
    save(fig, "23_coupling_scores.png")


def main():
    fig_metric_consistency()
    fig_carry_intervention()
    fig_token_damage()
    fig_coupling()


if __name__ == "__main__":
    main()
