"""Q1 STORAGE, Stage 10 Phase 2A: is the Japanese copy's survival a DELAY?

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_storage_phase2a.py

Reads results/storage_followup/phase2a.json (storage_followup_report.py phase2a). x = the
French truth ratio at each saved checkpoint of the srcfrja2 run (log scale; an ordered depth
axis, so lines are drawn), top labels = how far FRENCH moved (share of its learned ->
never-taught gap, 1 = never-taught).

  A  the decision: Japanese forget-specific share s = (dF - dR) / G against the
     pre-registered bars (SEPARATE < 0.25 <= PARTIAL < 0.50 <= DELAY), beside the Japanese
     retain rows' own move dR / G (the collateral control) and job 19's tr0p783 replicate.
  B  per fact: how many of the 26 eligible facts are REACHED (Phase 1 rule).
  C  behaviour: Japanese greedy word-for-word relative drop, forget (40 facts) vs retain
     (400 rows) -- they converge, so past French TR ~1.4 behaviour is mostly general damage.

Every number in a title or label is computed from phase2a.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parents[1]
RES, FIGS = STUDY / "results" / "storage_followup", STUDY / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
FR, JA, RET = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3, fixed order
ZONE = {"SEPARATE": "#f2f1ec", "PARTIAL": "#e7e5de", "DELAY": "#d9d7ce"}


def main():
    d = json.load(open(RES / "phase2a.json"))
    G = d["G"]
    lv = {k: v for k, v in d["levels"].items() if not k.startswith("job19") and "fr_tr" in v}
    rows = sorted(lv.values(), key=lambda v: v["fr_tr"])
    x = [r["step"] for r in rows]
    frtr = [r["fr_tr"] for r in rows]
    s = [r["s"] for r in rows]
    ret = [r["dR"] / G for r in rows]
    frs = [r["fr_share"] for r in rows]
    prim = next(r for r in rows if abs(r["level"] - 0.7831) < 1e-3)
    rep = d["levels"].get("job19_tr0p783")
    beyond = [r for r in rows if r["beyond"] and r["usable"]]
    smax = max(r["s"] for r in beyond)

    # first step where French is back at (within 0.05 of) its never-taught level
    x_nt = next((r["step"] for r in rows if r["fr_share"] >= 0.95), None)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.6, 1, 1]})
    fig.suptitle(f"Stage 10 Phase 2A -- French pushed to {frs[-1]:.1f}x its never-taught gap; "
                 f"the Japanese copy moves at most {smax:.2f} of its own  ->  {d['decision']} "
                 f"(bar 0.25)   [PROVISIONAL: single seed, 40 facts / 2 entities]",
                 fontsize=11.5, x=0.01, ha="left", y=0.995)

    def frame(ax, title):
        ax.set_facecolor(SURFACE)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{st}\n{v:.2f}" for st, v in zip(x, frtr)], fontsize=8)
        ax.set_xlim(x[0] - 1.2, x[-1] + 1.2)
        ax.set_xlabel("unlearning step / French truth ratio  (deeper ->)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if x_nt:
            ax.axvline(x_nt, color=FR, lw=1, ls=":")
        ax.set_title(title, loc="left", fontsize=10.5)
        top = ax.secondary_xaxis("top")
        top.set_xticks(x)
        top.set_xticklabels([f"{v:.2f}" if v < 1 else f"{v:.1f}x" for v in frs],
                            fontsize=7.5, color=FR)
        top.set_xlabel("French: share of its gap moved", fontsize=7.5, color=FR, labelpad=2)
        top.tick_params(length=0, pad=1)
        for sp in top.spines.values():
            sp.set_visible(False)

    # --- A: the decision -------------------------------------------------------------
    ax = axes[0]
    frame(ax, "A  How far the Japanese copy moved toward never-taught\n"
              "    (top, blue: how far FRENCH moved; 1 = never-taught)")
    ymax = 1.08
    for lo, hi, name in ((-0.1, 0.25, "SEPARATE"), (0.25, 0.5, "PARTIAL"), (0.5, ymax, "DELAY")):
        ax.axhspan(lo, hi, color=ZONE[name], zorder=0, lw=0)
        ax.text(x[0] - 1.0, min(hi - 0.02, 0.96), name, fontsize=8.5, color=MUTED,
                va="top", ha="left")
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.text(x[-1] + 1.0, 1.01, "never-taught (1.0)", ha="right", va="bottom", fontsize=8,
            color=MUTED)
    ax.plot(x, s, color=JA, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=4)
    ax.plot(x, ret, color=RET, lw=2, marker="s", ms=8, mec=SURFACE, mew=2, zorder=3)
    if rep:
        ax.plot([prim["step"]], [rep["s"]], marker="o", ms=12, mfc="none", mec=MUTED, mew=1.2,
                ls="none", zorder=5)
        ax.plot([], [], marker="o", ms=10, mfc="none", mec=MUTED, mew=1.2, ls="none",
                label=f"Phase 1 checkpoint at step 26 (job 19): {rep['s']:.2f} -- replicates")
    for xi, si in zip(x[-3:], s[-3:]):
        ax.text(xi, si + 0.035, f"{si:.2f}", ha="center", fontsize=8.5, color=INK)
    if x_nt:
        ax.text(x_nt + 0.3, 0.86, "French back at\nnever-taught", fontsize=8, color=FR)
    ax.set_ylim(-0.1, ymax)
    ax.set_ylabel("share of the learned -> never-taught gap moved")
    ax.plot([], [], color=JA, lw=2, marker="o", label="Japanese forget facts (s, net of retain)")
    ax.plot([], [], color=RET, lw=2, marker="s", label="Japanese retain rows (collateral)")
    h, l = ax.get_legend_handles_labels()
    order = [i for i, lab in enumerate(l) if lab.startswith("Japanese")] + \
            [i for i, lab in enumerate(l) if lab.startswith("Phase 1")]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="center left",
              bbox_to_anchor=(0.0, 0.64), frameon=False, fontsize=8.5)

    # --- B: per fact -----------------------------------------------------------------
    ax = axes[1]
    n_el = 26
    reached = [len(r.get("reached", [])) if isinstance(r.get("reached"), list) else None
               for r in rows]
    if any(v is None for v in reached):
        reached = [r.get("n_reached") for r in rows]
    frame(ax, f"B  Japanese facts individually reached\n    (3-pass non-overlap and share >= 0.25), of {n_el}")
    ax.plot(x, reached, color=JA, lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
    for xi, ri in zip(x, reached):
        ax.text(xi, ri + 0.6, str(ri), ha="center", fontsize=8.5)
    ax.axhline(n_el, color=MUTED, lw=1, ls="--")
    ax.text(x[0] - 1.0, n_el - 0.4, f"all {n_el} eligible", va="top", fontsize=8, color=MUTED)
    ax.set_ylim(0, n_el + 1.5)
    ax.set_ylabel("facts")

    # --- C: behaviour ----------------------------------------------------------------
    ax = axes[2]
    frame(ax, "C  Japanese greedy answer, word for word:\n    relative drop from learned")
    fd = [r["forget_drop"] for r in rows]
    rd = [r["retain_drop"] for r in rows]
    ax.plot(x, fd, color=JA, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="forget (40 facts)")
    ax.plot(x, rd, color=RET, lw=2, marker="s", ms=8, mec=SURFACE, mew=2, label="retain (400 rows)")
    ax.text(x[-1], fd[-1] + 0.03, f"{fd[-1]:.2f}", ha="center", fontsize=8.5)
    ax.text(x[-1], rd[-1] - 0.07, f"{rd[-1]:.2f}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of learned exact answers lost")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.text(0.0, -0.30, "forget and retain converge: deep unlearning damages ALL Japanese "
            "generation,\nso behaviour cannot isolate the forgotten facts here",
            transform=ax.transAxes, fontsize=8, color=MUTED)

    fig.text(0.01, 0.005,
             f"s = (median Japanese forget TR rise - median Japanese retain TR rise) / G, "
             f"G = {G:.3f} (median learned -> never-taught Japanese TR gap, 26 eligible facts). "
             f"Run srcfrja2 stopped after the {frtr[-1]:.1f} level (step {rows[-1]['step']}); levels "
             f"60 / 1000 not made, so nothing deeper is tested.\nUsable = MU_6 >= "
             f"{d['mu_threshold']}: all {len(rows)} levels.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "storage_phase2a.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
