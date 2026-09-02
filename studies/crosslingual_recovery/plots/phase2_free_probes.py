"""PHASE 2 Part A, steps 1-2 -- the free probe family (p0 + p1 + MCQ) scored at the
LEARNED and the two UNLEARNED checkpoints. No relearning is involved, so this is the
cheap, high-value half of the plan (CLAUDE.md sec 2, "Part A").

The question: did unlearning remove the fact, or only suppress the SURFACE FORM of the
canonical question it was trained on?

  p0_canonical   TOFU's own question -- the phrasing seen in LEARN *and* in UNLEARN
  p1_tofu_para   TOFU's shipped `paraphrased_question` -- HELD OUT, never trained on
  mcq            6-way multiple choice from the 5 perturbed answers + the correct one

p0/p1 are scored with the TRUTH RATIO (TOFU Eq. 1; LOW = knows the fact) and mcq with
the MC-normalized probability (HIGH = knows). Opposite directions on incomparable
scales -- they are NEVER averaged and never share an axis here.

NEITHER metric generates anything. Both hand the model complete candidate answer
sentences and read off length-normalized probabilities:

    truth ratio = geomean(P(perturbed)) / P(paraphrased)
    mcq         = P(paraphrased) / (P(paraphrased) + sum P(perturbed))

Read those two lines together: for a given question they consume the SAME six answer
texts and differ only in ratio-vs-share and geometric-vs-arithmetic mean. Empirically
spearman(p0, mcq) = -0.97 to -0.99 (panel D). The MCQ is therefore NOT an independent
probe of a second capability -- it is the p0 truth ratio re-expressed on a BOUNDED
scale, which is worth having only because it cannot explode the way an unbounded ratio
can. Do not report it as corroborating evidence.

The phrasing axis is consequently n=2 (p0, p1), not 3.

Because p0 is the trained-on phrasing, a raw p0-vs-p1 gap on an unlearned model
confounds "unlearning was phrasing-specific" with "the model is simply better at
phrasings it trained on". The test is therefore a DIFFERENCE-IN-DIFFERENCES against the
learned checkpoint, paired fact by fact:

    DiD_i = [TR(p1) - TR(p0)]_unlearned,i  -  [TR(p1) - TR(p0)]_learned,i

Truth ratio is unbounded, so the DiD MEAN is not trustworthy -- one exploded fact moves
it. Significance is Wilcoxon signed-rank (paired, rank-based) and the panel shows every
fact, so the reader can see whether a mean is carried by the distribution or by a flier.

    source .venv-plot/bin/activate
    python studies/crosslingual_recovery/plots/phase2_free_probes.py
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import spearmanr, wilcoxon

STUDY = Path(__file__).resolve().parents[1]
PROBES = STUDY / "results" / "relearn" / "phase2_probes"
FIGS = STUDY / "figures"

CKPT = {
    "learned": "tofu_learn_full_full_qwen3-8b",
    "fullft":  "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b",
    "lora":    "tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b",
}
# Reference truth ratios from the main grid (CLAUDE.md sec 2) -- used ONLY to validate
# that this independent scorer reproduces the established p0 numbers.
KNOWN_P0 = {"learned": 0.459, "fullft": 0.7428, "lora": 0.6776}

LABEL = {"learned": "learned", "fullft": "Full-FT unlearned", "lora": "LoRA unlearned"}
COLOR = {"learned": "#7a7a76", "fullft": "#1f77b4", "lora": "#ff7f0e"}
METHODS = ["fullft", "lora"]

INK, MUTED, GRID = "#1a1a18", "#6b6b66", "#d8d8d4"
MCQ_CHANCE = 1.0 / 6.0      # 6-way multiple choice
TR_INDIFFERENT = 1.0        # truth ratio 1.0 = perturbed and true answers rated equally


def load():
    out = {}
    for k, stem in CKPT.items():
        d = json.load(open(PROBES / f"{stem}.json"))
        out[k] = d[stem]
    return out


def arr(D, ckpt, probe):
    return np.asarray(D[ckpt][probe]["scores_per_fact"], dtype=float)


def fmt_p(p):
    return "p < 1e-5" if p < 1e-5 else f"p = {p:.3f}"


def main():
    D = load()
    idx = np.asarray(D["learned"]["p0_canonical"]["fact_indices"])
    n = len(idx)

    p0 = {k: arr(D, k, "p0_canonical") for k in CKPT}
    p1 = {k: arr(D, k, "p1_tofu_para") for k in CKPT}
    mcq = {k: arr(D, k, "mcq") for k in CKPT}

    did = {m: (p1[m] - p0[m]) - (p1["learned"] - p0["learned"]) for m in METHODS}

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))
    fig.subplots_adjust(hspace=0.46, wspace=0.30, top=0.86, bottom=0.10)

    # ---- A: absolute truth ratio, canonical vs held-out paraphrase -----------------
    ax = axes[0][0]
    order = ["learned", "fullft", "lora"]
    xs = np.arange(len(order))
    w = 0.36
    for j, (probe, vals, hatch) in enumerate(
            [("p0 canonical", p0, None), ("p1 paraphrase (held out)", p1, "///")]):
        off = (j - 0.5) * w
        for i, k in enumerate(order):
            ax.bar(xs[i] + off, vals[k].mean(), w * 0.92, color=COLOR[k], hatch=hatch,
                   edgecolor="white", linewidth=1.0, zorder=3)
            ax.text(xs[i] + off, vals[k].mean() + 0.015, f"{vals[k].mean():.3f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK, zorder=5)
    for i, k in enumerate(order):
        ax.plot([xs[i] - w, xs[i]], [KNOWN_P0[k]] * 2, color=INK, lw=1.6, zorder=6)
    ax.plot([], [], color=INK, lw=1.6, label="published p0 baseline")
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[k] for k in order], fontsize=9.5)
    ax.set_ylabel("truth ratio   (LOWER = knows the fact)", fontsize=9.5)
    dev = max(abs(p0[k].mean() - KNOWN_P0[k]) for k in order)
    ax.set_title(f"A  Unlearning raises the truth ratio under BOTH phrasings\n"
                 f"scorer validated: p0 matches the published baselines within {dev:.3f}",
                 fontsize=10.5, loc="left", color=INK)
    ax.legend(handles=[Patch(facecolor="#bdbdb8", edgecolor="white", label="p0 canonical"),
                       Patch(facecolor="#bdbdb8", edgecolor="white", hatch="///",
                             label="p1 paraphrase (held out)"),
                       plt.Line2D([], [], color=INK, lw=1.6, label="published p0 baseline")],
              fontsize=8.5, frameon=False, loc="upper left")
    ax.set_ylim(0, max(max(p0[k].mean(), p1[k].mean()) for k in order) * 1.30)

    # ---- B: the difference-in-differences, every fact ------------------------------
    ax = axes[0][1]
    LIM = 2.4
    rng = np.random.default_rng(0)
    stats_b = {}
    for r, m in enumerate(METHODS):
        v = did[m]
        y = (1 - r) + rng.uniform(-0.13, 0.13, n)
        inside, outside = np.abs(v) <= LIM, np.abs(v) > LIM
        ax.scatter(v[inside], y[inside], s=42, color=COLOR[m], alpha=0.72,
                   edgecolor="white", linewidth=0.7, zorder=3)
        for i in np.where(outside)[0]:
            xe = np.sign(v[i]) * LIM
            ax.scatter([xe], [y[i]], s=72, color=COLOR[m], marker="<" if v[i] < 0 else ">",
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.annotate(f"fact {idx[i]}  ({v[i]:+.2f})", (xe, y[i]),
                        xytext=(14, 12), textcoords="offset points", fontsize=8.5,
                        color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        med = float(np.median(v))
        ax.plot([med, med], [(1 - r) - 0.26, (1 - r) + 0.26], color=INK, lw=2.2, zorder=6)
        p = wilcoxon(v).pvalue
        drop = np.abs(v) <= LIM
        stats_b[m] = (v.mean(), med, p, v[drop].mean())
        ax.text(LIM * 0.99, (1 - r) + 0.30,
                f"{LABEL[m]}:  median {med:+.3f}   mean {v.mean():+.3f}   "
                f"Wilcoxon {fmt_p(p)}", ha="right", va="bottom", fontsize=8.7, color=INK)
    ax.axvline(0, color=INK, lw=1.1, zorder=2)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["Full-FT", "LoRA"], fontsize=9.5)
    ax.set_ylim(-0.55, 1.62); ax.set_xlim(-LIM * 1.06, LIM * 1.06)
    ax.set_xlabel("difference-in-differences   $[TR(p1)-TR(p0)]_{unlearned}"
                  " - [TR(p1)-TR(p0)]_{learned}$", fontsize=9.5)
    ax.text(-LIM * 0.99, -0.44, "< paraphrase LEAKS the fact", fontsize=8.5, color=MUTED)
    ax.text(LIM * 0.99, -0.44, "paraphrase is HARDER >", fontsize=8.5, color=MUTED, ha="right")
    worst = max(METHODS, key=lambda m: abs(stats_b[m][0]))
    ax.set_title("B  No detectable phrasing-specific leakage  (one point = one fact)\n"
                 f"Full-FT's {stats_b['fullft'][0]:+.3f} mean is ONE fact -- without the "
                 f"flier it is {stats_b['fullft'][3]:+.3f}",
                 fontsize=10.5, loc="left", color=INK)

    # ---- C: MCQ, paired per fact ---------------------------------------------------
    ax = axes[1][0]
    lo, hi = 0.0, max(mcq[k].max() for k in CKPT) * 1.06
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls="--", zorder=2)
    ax.axhline(MCQ_CHANCE, color=GRID, lw=1.2, zorder=1)
    ax.text(hi, MCQ_CHANCE, " chance (1/6)", va="center", fontsize=8.5, color=MUTED)
    stats_c = {}
    for m in METHODS:
        drop = mcq["learned"] - mcq[m]
        nd = int((drop > 0).sum())
        p = wilcoxon(mcq["learned"], mcq[m]).pvalue
        stats_c[m] = (drop.mean(), nd, p)
        ax.scatter(mcq["learned"], mcq[m], s=46, color=COLOR[m], alpha=0.72,
                   edgecolor="white", linewidth=0.7, zorder=3,
                   label=f"{LABEL[m]}   {nd}/{n} facts drop,  {fmt_p(p)}")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("MC-normalized probability, LEARNED model", fontsize=9.5)
    ax.set_ylabel("same fact, UNLEARNED model", fontsize=9.5)
    ax.text(hi * 0.96, hi * 0.06, "below the line = forgot", fontsize=9, color=INK, ha="right")
    ax.legend(fontsize=8.5, loc="upper left", frameon=True, framealpha=1.0,
              edgecolor="none", facecolor="white")
    ax.set_title("C  The same result on a BOUNDED metric  (HIGHER = knows the fact)\n"
                 "outlier-proof restatement of panel A's p0 -- NOT independent evidence",
                 fontsize=10.5, loc="left", color=INK)

    # ---- D: the MCQ is a re-encoding of p0, not a second probe ---------------------
    ax = axes[1][1]
    XLIM = 2.6
    rhos = {}
    for k in order:
        r, m = p0[k], mcq[k]
        rhos[k] = spearmanr(r, m).statistic
        vis = r <= XLIM
        ax.scatter(r[vis], m[vis], s=44, color=COLOR[k], alpha=0.75, edgecolor="white",
                   linewidth=0.7, zorder=3,
                   label=f"{LABEL[k]}   rho = {rhos[k]:+.3f}")
        for i_ in np.where(~vis)[0]:
            ax.scatter([XLIM], [m[i_]], s=70, color=COLOR[k], marker=">",
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.annotate(f"fact {idx[i_]}  (TR {r[i_]:.1f})", (XLIM, m[i_]),
                        xytext=(-8, 16), textcoords="offset points", fontsize=8.3,
                        color=INK, ha="right",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    # the analytic curve the two metrics would follow EXACTLY if the 5 perturbed
    # probabilities were equal (geometric mean == arithmetic mean)
    grid = np.linspace(0.02, XLIM, 400)
    ax.plot(grid, 1.0 / (1.0 + 5.0 * grid), color=INK, lw=1.4, ls="--", zorder=5,
            label=r"$1/(1+5\cdot TR)$  (exact if the 5 perturbed probs are equal)")
    ax.axhline(MCQ_CHANCE, color=GRID, lw=1.2, zorder=1)
    ax.text(XLIM, MCQ_CHANCE, " chance", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, XLIM * 1.06)
    ax.set_ylim(0, max(mcq[k].max() for k in order) * 1.10)
    ax.set_xlabel("p0 truth ratio   (LOWER = knows)", fontsize=9.5)
    ax.set_ylabel("mcq, same question   (HIGHER = knows)", fontsize=9.5)
    ax.legend(fontsize=8.2, loc="upper right", frameon=True, framealpha=1.0,
              edgecolor="none", facecolor="white")
    worst_rho = max(rhos.values(), key=abs) if False else min(rhos.values(), key=abs)
    ax.set_title("D  The MCQ is NOT an independent probe of a second capability\n"
                 f"same question, same 6 answer texts -- |rho| >= {abs(worst_rho):.2f} "
                 "with p0 at every checkpoint",
                 fontsize=10.5, loc="left", color=INK)

    for a in axes.ravel():
        a.set_axisbelow(True)
        a.grid(axis="y", color=GRID, lw=0.7)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9)
        for t in a.get_xticklabels() + a.get_yticklabels():
            t.set_color(INK)
    axes[0][1].grid(axis="y", visible=False)
    axes[0][1].grid(axis="x", color=GRID, lw=0.7)
    axes[1][0].grid(axis="both", color=GRID, lw=0.7)

    fig.suptitle("Phase 2 Part A, steps 1-2 -- the free probe family at the learned and "
                 "unlearned checkpoints\n"
                 "forget01, 40 facts (2 entities), English probes, Qwen3-8B     "
                 "[PROVISIONAL: single seed]",
                 fontsize=12.5, ha="left", x=0.055, y=0.975, color=INK)
    fig.text(0.055, 0.022,
             "NEITHER metric generates text -- both score length-normalized probabilities of "
             "supplied answer sentences.  Truth ratio and mcq run in OPPOSITE directions on "
             "incomparable scales and are never averaged.\n"
             "Panel D shows they are near-duplicates, so the phrasing axis is n=2 (p0, p1), "
             "not 3.  All tests are paired fact by fact, n=40, Wilcoxon signed-rank.",
             fontsize=8.5, color=MUTED)

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "phase2_free_probes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", out)

    print("\n--- validation: p0 vs published ---")
    for k in order:
        print(f"  {LABEL[k]:20s} new {p0[k].mean():.4f}   known {KNOWN_P0[k]:.4f}   "
              f"diff {p0[k].mean()-KNOWN_P0[k]:+.4f}")
    print("\n--- B: difference-in-differences ---")
    for m in METHODS:
        mn, md, p, cl = stats_b[m]
        print(f"  {LABEL[m]:20s} mean {mn:+.4f}  median {md:+.4f}  Wilcoxon p={p:.4f}  "
              f"mean w/o off-scale {cl:+.4f}")
    print("\n--- C: MCQ paired drop from learned ---")
    for m in METHODS:
        mn, nd, p = stats_c[m]
        print(f"  {LABEL[m]:20s} mean drop {mn:+.4f}  {nd}/{n} facts  p={p:.6f}")
    print("\n--- D: is the MCQ independent of p0? ---")
    for k in order:
        pred = 1.0 / (1.0 + 5.0 * p0[k])
        print(f"  {LABEL[k]:20s} spearman(p0, mcq) {rhos[k]:+.4f}   "
              f"mean |mcq - 1/(1+5TR)| {np.abs(mcq[k]-pred).mean():.4f}")


if __name__ == "__main__":
    sys.exit(main())
