"""Stage 2 after-metrics: what each level checkpoint looks like on every metric.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_stage2_metrics.py [--variant nocap|cap]
    -> figures/stage2_metrics_<variant>.png

The trajectory figures show WHEN each language reaches a depth. This one shows WHAT the
model is like once it gets there, from results/stage2_<variant>_<lang>/ (the 04 job).

x is the checkpoint's own measured truth ratio -- not the level it was labelled with,
since a fast climb overshoots its label -- so every language is plotted on the depth it
actually reached.

  A  NLI: can it still say the fact out loud?
  B  P(gold): does it still assign probability to the trained answer?
  C  Model Utility: is the model otherwise intact?

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
                         "tr": s["tr_arithmetic_mean_norm"], "nli": s["nli_score_mean"],
                         "prob": s["prob_mean"], "mu": s.get("model_utility_6"),
                         "fq": (r.get("forget_quality_vs_reference_norm") or {}).get(
                             "forget_quality_log10"),
                         "fr_share": (s.get("gen_language_counts") or {}).get("fr", 0) /
                                     max(sum((s.get("gen_language_counts") or {}).values()), 1)})
        if rows:
            runs[lang] = sorted(rows, key=lambda x: x["tr"])
    if not runs:
        sys.exit(f"no stage2_{variant}_* results yet -- run 04_measure_unlearned.sbatch")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="nocap", choices=["nocap", "cap"])
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, s1 = load(args.variant), stage1()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    panels = [("nli", "NLI equivalence\n(can it still state the fact?)", "A"),
              ("prob", "P(gold answer)\n(is the trained string still likely?)", "B"),
              ("mu", "Model Utility (6-metric)\n(is the model otherwise intact?)", "C")]
    ref = {"nli": ("nli_score_mean", "NLI"), "prob": ("prob_mean", "P(gold)"),
           "mu": ("model_utility_6", "MU")}
    for ax, (key, title, tag) in zip(axes, panels):
        for lang, rows in runs.items():
            xs = [r["tr"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, color=C[lang], lw=1.8, marker="o", ms=5, label=lang)
        for role, style in (("fr_ft", "-"), ("fr_retain", "--")):
            v = s1.get(role, {}).get(ref[key][0])
            if v is not None:
                ax.axhline(v, color="#555555", lw=1, ls=style)
                ax.text(0.99, v, f" {role} ", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=7.5, color="#555555")
        ax.set_xlabel("truth ratio the checkpoint actually reached\n(higher = more forgotten)")
        ax.set_title(f"{tag}. {title}", fontsize=10, loc="left", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, alpha=0.4, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(title="unlearning language", fontsize=8, title_fontsize=8, frameon=False)
    fig.suptitle("Stage 2: what the level checkpoints are actually like, probed in French",
                 fontsize=13, x=0.045, ha="left", y=0.97, color=INK)
    fig.text(0.045, 0.90, "each point is one saved checkpoint - five depths per language - "
             f"{'no forget cap' if args.variant == 'nocap' else 'forget cap 4.0'}",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    FIGS.mkdir(exist_ok=True)
    out = FIGS / f"stage2_metrics_{args.variant}.png"
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
