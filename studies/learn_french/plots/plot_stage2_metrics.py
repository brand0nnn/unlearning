"""Stage 2 after-metrics: what each level checkpoint looks like on every metric.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_stage2_metrics.py [--variant nocap|cap]
    -> figures/stage2_metrics_<variant>.png

The trajectory figures show WHEN each language reaches a depth. This one shows WHAT the
model is like once it gets there, from results/stage2_<variant>_<lang>/ (the 04 job).

--x step  (default)  the three metrics against optimizer step: the ordinary unlearning
                     curve. Truth ratio is dense (probed every 2 steps); P(gold) and NLI
                     exist only at the five saved checkpoints, because generating 40
                     French answers every other step would have cost more than training.
--x tr               the same metrics against the checkpoint's OWN truth ratio, which
                     lines the languages up at matched depth rather than matched time.
                     That view answers "at the same depth, are these the same model?" --
                     and shows they are not.

  A  truth ratio: how forgotten does the French probe say it is?
  B  P(gold): does it still assign probability to the trained answer?
  C  NLI: can it still say the fact out loud?

A AND B TOGETHER ARE THE PRE-REGISTERED SECONDARY HYPOTHESIS (plan sec 5, "TR - NLI
gap"): if a language's NLI collapses while its probability holds, unlearning in it
suppressed DECODING rather than removing the knowledge. Read the two panels side by side
at the same x, not one alone.
"""
import argparse
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures"
C = {"fr": "#222222", "en": "#2a78d6", "id": "#1baf7a", "ru": "#e34948", "ja": "#8b5cd6"}
ORDER = ["fr", "en", "id", "ru", "ja"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"


def stage1():
    out = {}
    for f in (RESULTS / "stage1_norm").glob("*.json"):
        r = json.load(open(f))
        role = ("fr_ft" if "_full_full_" in r["name"] else
                "fr_retain" if "retain99" in r["name"] else "base")
        out[role] = r["summary"]
    return out


def saved_steps(lang, variant):
    """{level: step} -- when each checkpoint was written, read from the trajectory."""
    suffix = "_floornone" if variant == "nocap" else ""
    f = RESULTS / "unlearn_traj" / (
        f"tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b_ul{lang}{suffix}.jsonl")
    out = {}
    if not f.exists():
        return out
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        for lv in (r.get("levels_saved_now") or {}):
            out.setdefault(lv, r["step"])
    return out


def load(variant):
    """{lang: [rows sorted by depth]} from the per-language after-metrics directories."""
    runs = {}
    for lang in ORDER:
        d = RESULTS / f"stage2_{variant}_{lang}"
        if not d.is_dir():
            continue
        rows = []
        for f in sorted(d.glob("*.json")):
            r = json.load(open(f))
            if "_tr0p" not in r["name"]:          # skip the fr_retain reference model
                continue
            s = r["summary"]
            rows.append({"level": r["name"].split("_tr")[-1].replace("p", "."),
                         "step": None,
                         "tr": s["tr_arithmetic_mean_norm"], "nli": s["nli_score_mean"],
                         "prob": s["prob_mean"], "mu": s.get("model_utility_6"),
                         "fq": (r.get("forget_quality_vs_reference_norm") or {}).get(
                             "forget_quality_log10"),
                         "fr_share": (s.get("gen_language_counts") or {}).get("fr", 0) /
                                     max(sum((s.get("gen_language_counts") or {}).values()), 1)})
        if rows:
            steps = saved_steps(lang, variant)
            for r_ in rows:
                r_["step"] = steps.get(r_["level"])
            runs[lang] = sorted(rows, key=lambda x: x["tr"])
    if not runs:
        sys.exit(f"no stage2_{variant}_* results yet -- run 04_measure_unlearned.sbatch")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="nocap", choices=["nocap", "cap"])
    ap.add_argument("--x", default="step", choices=["step", "tr"],
                    help="x axis: optimizer step (the usual curve) or the checkpoint's "
                         "own truth ratio (matched depth)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, s1 = load(args.variant), stage1()
    panels = [("prob", "P(gold answer)\n(is the trained answer still likely?)"),
              ("nli", "NLI equivalence\n(can it still state the fact?)"),
              ("mu", "Model Utility (6-metric)\n(is the model otherwise intact?)")]
    # The truth ratio is the x axis under --x tr, so a truth-ratio panel would just be the
    # identity line. It is only a panel when x is the step.
    if args.x == "step":
        panels.insert(0, ("tr", "truth ratio\n(higher = the probe says it is forgotten)"))
    panels = [(k, t, chr(65 + i)) for i, (k, t) in enumerate(panels)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels) + 0.7, 4.8))
    ref = {"nli": ("nli_score_mean", "NLI"), "prob": ("prob_mean", "P(gold)"),
           "mu": ("model_utility_6", "MU"), "tr": ("tr_arithmetic_mean_norm", "TR")}
    dense = {}
    if args.x == "step":       # the truth ratio was probed every 2 steps; use all of it
        for lang in runs:
            suffix = "_floornone" if args.variant == "nocap" else ""
            f = RESULTS / "unlearn_traj" / (
                f"tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b_ul{lang}{suffix}.jsonl")
            pts = []
            for line in open(f):
                try:
                    r_ = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pts.append((r_["step"], r_["mean_tr"]))
            dense[lang] = pts
    for ax, (key, title, tag) in zip(axes, panels):
        for lang, rows in runs.items():
            pts = [(r["step"] if args.x == "step" else r["tr"], r[key]) for r in rows
                   if r[key] is not None and (args.x == "tr" or r["step"] is not None)]
            pts.sort()
            if key == "tr" and args.x == "step" and lang in dense:
                d = [(s_, v) for s_, v in dense[lang] if v <= 1.0]   # clip the divergence
                ax.plot([p[0] for p in d], [p[1] for p in d], color=C[lang], lw=1.8, label=lang)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=C[lang], ms=5)
            else:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C[lang], lw=1.8,
                        marker="o", ms=5, label=lang if key != "tr" else None)
        # fr_ft 0.494 and fr_retain 0.504 nearly coincide on the utility panel; stack the
        # labels on opposite sides of their lines so they do not overprint.
        for role, style, va in (("fr_ft", "-", "top"), ("fr_retain", "--", "bottom")):
            v = s1.get(role, {}).get(ref[key][0])
            if v is not None:
                ax.axhline(v, color="#555555", lw=1, ls=style)
                ax.text(0.99, v, f" {role} ", transform=ax.get_yaxis_transform(),
                        ha="right", va=va, fontsize=7.5, color="#555555")
        if key == "mu":
            # SCALE. Auto-scaled, this panel spans ~0.04 and renders a series that never
            # moves as a cliff. MU is a harmonic mean of quantities in [0, 1]; what makes
            # a drop meaningful is the distance to a model that never learned the facts,
            # so anchor the axis on base Qwen3 and the learned model rather than on the
            # data's own range. The claim is "no language paid a utility price", and the
            # axis has to be able to show that.
            b = s1.get("base", {}).get("model_utility_6")
            if b is not None:
                ax.axhline(b, color="#555555", lw=1, ls=":")
                ax.text(0.99, b, " base Qwen3 ", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=7.5, color="#555555")
            tops = [v for v in (b, s1.get("fr_ft", {}).get("model_utility_6"),
                                s1.get("fr_retain", {}).get("model_utility_6"))
                    if v is not None]
            ax.set_ylim(0.0, (max(tops) if tops else 0.55) * 1.12)
        ax.set_xlabel("optimizer step" if args.x == "step" else
                      "truth ratio the checkpoint reached\n(matched depth)")
        ax.set_title(f"{tag}. {title}", fontsize=10, loc="left", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, alpha=0.4, lw=0.7)
        ax.set_axisbelow(True)
    # Panel A carries labels only in the --x step path (there the dense curve is drawn
    # separately); panel B always does, so source the legend there and place it on A.
    # A legend inside panel A lands on the data in both views (the curves run corner to
    # corner). Put it in the header band instead, as one row, where it blocks nothing.
    h, l = next((hl for hl in (a.get_legend_handles_labels() for a in axes) if hl[1]),
                ([], []))
    fig.legend(h, l, title="unlearning language", fontsize=8, title_fontsize=8,
               frameon=False, loc="upper left", bbox_to_anchor=(0.55, 1.005),
               ncol=len(l) or 1, columnspacing=1.4, handlelength=1.6)
    if args.x == "step":
        axes[0].text(0.98, 0.96, "clipped at 1.0 - fr/en/id/ru diverge far beyond it",
                     transform=axes[0].transAxes, ha="right", va="top", fontsize=7.5,
                     color=MUTED)   # bottom-right collides with the fr_ft line label
    fig.suptitle("Stage 2: what the level checkpoints are actually like, probed in French",
                 fontsize=13, x=0.045, ha="left", y=0.97, color=INK)
    fig.text(0.045, 0.90, ("truth ratio probed every 2 steps; P(gold), NLI and MU only at "
             "the saved checkpoints (circles) - " if args.x == "step" else
             "one point per saved checkpoint, placed at the truth ratio it reached - ")
             + f"{'no forget cap' if args.variant == 'nocap' else 'forget cap 4.0'}",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    FIGS.mkdir(exist_ok=True)
    out = FIGS / f"stage2_metrics_{args.variant}_by{args.x}.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"-> {out}\n")
    print(f"{'lang':>5}{'level':>8}{'TR':>8}{'NLI':>8}{'P(gold)':>9}{'MU':>8}{'FQ log10':>10}{'French':>8}")
    for lang, rows in runs.items():
        for r in rows:
            print(f"{lang:>5}{r['level']:>8}{r['tr']:>8.3f}{r['nli']:>8.3f}{r['prob']:>9.3f}"
                  f"{(r['mu'] or 0):>8.3f}{(r['fq'] if r['fq'] is not None else 0):>10.2f}"
                  f"{r['fr_share']:>8.0%}")


if __name__ == "__main__":
    main()
