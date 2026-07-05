"""Plots for the TOFU results.

Two figures:
  1. forget_quality_vs_utility — the canonical TOFU scatter (matches paper Fig 5/6).
     Includes: retain model gold star, shaded target region, paper reference values
     as ghost annotations so you can see at a glance whether reproduction succeeded.
  2. rouge_by_split — bar chart of ROUGE across four splits per method, showing
     how well the model preserves knowledge it should NOT forget.
"""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paper reference values (Maini et al. 2024, Table 1 — forget05, Llama-2-7B).
# Used as ghost markers so you can compare your run against the paper directly.
# Keys must match the method names produced by your pipeline.
# ---------------------------------------------------------------------------
PAPER_REFS = {
    "gradient_ascent":     {"model_utility": 0.2,   "forget_quality_log10": -0.05},
    "gradient_difference": {"model_utility": 0.444,  "forget_quality_log10": -0.3},
    "kl_minimization":     {"model_utility": 0.444,  "forget_quality_log10": -0.25},
    "idk":                 {"model_utility": 0.283,  "forget_quality_log10": -0.15},
}

# One consistent colour per method — used in both figures so they cross-reference.
METHOD_COLORS = {
    "gradient_ascent":     "#e63946",   # red   — aggressive, destroys utility
    "gradient_difference": "#2a9d8f",   # teal
    "kl_minimization":     "#457b9d",   # blue
    "idk":                 "#f4a261",   # orange
    "self_distill":        "#9b5de5",   # purple — self-distillation strategy
    "grpo":                "#e07a5f",   # terracotta — GRPO strategy
}
DEFAULT_COLOR = "#6c757d"


def forget_quality_vs_utility(results_by_method: Dict[str, Dict], out_dir: str,
                               retain_result: Dict = None):
    """
    results_by_method[name] = {"model_utility": x, "forget_quality_log10": y}
    retain_result (optional) = {"model_utility": x, "forget_quality_log10": y}
        Pass the retain model's scores so it appears as the gold-star reference.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # --- shaded target region (top-right = forget well + keep utility) ------
    ax.axhspan(-0.5, 0.5, xmin=0.6, alpha=0.07, color="green",
               label="target region (forget well + preserve utility)")

    # --- paper ghost markers (open circles) ----------------------------------
    for name, ref in PAPER_REFS.items():
        ax.scatter(ref["model_utility"], ref["forget_quality_log10"],
                   s=180, facecolors="none",
                   edgecolors=METHOD_COLORS.get(name, DEFAULT_COLOR),
                   linewidths=1.5, zorder=2)
    # single legend entry for all ghosts
    ghost_patch = mpatches.Patch(facecolor="none", edgecolor="grey",
                                 linewidth=1.5, label="paper reference (open)")

    # --- your results (filled circles) --------------------------------------
    for name, r in results_by_method.items():
        color = METHOD_COLORS.get(name, None)
        if color is None:
            for key in METHOD_COLORS:
                if key in name:
                    color = METHOD_COLORS[key]
                    break
        color = color or DEFAULT_COLOR
        ax.scatter(r["model_utility"], r["forget_quality_log10"],
                   s=160, color=color, zorder=3, label=name)
        ax.annotate(name.replace("_", " "),
                    (r["model_utility"], r["forget_quality_log10"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8.5,
                    color=color)

        # draw arrow from ghost to your point so drift is immediately obvious
        if name in PAPER_REFS:
            ref = PAPER_REFS[name]
            ax.annotate("",
                xy=(r["model_utility"], r["forget_quality_log10"]),
                xytext=(ref["model_utility"], ref["forget_quality_log10"]),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=1.0, alpha=0.5))

    # --- retain model gold star ---------------------------------------------
    if retain_result:
        ax.scatter(retain_result["model_utility"], retain_result["forget_quality_log10"],
                   s=350, marker="*", color="gold", edgecolors="darkgoldenrod",
                   linewidths=1, zorder=5, label="retain model (gold reference)")
        ax.annotate("retain\n(gold)",
                    (retain_result["model_utility"], retain_result["forget_quality_log10"]),
                    textcoords="offset points", xytext=(8, -14), fontsize=8,
                    color="darkgoldenrod")

    # --- reference lines ----------------------------------------------------
    ax.axhline(0, ls="--", c="grey", lw=0.8, alpha=0.6)
    ax.axhline(-1, ls=":", c="grey", lw=0.8, alpha=0.4,
               label="p < 0.1 threshold (log10 = −1)")

    ax.set_xlabel("Model Utility  (↑ less collateral damage)", fontsize=11)
    ax.set_ylabel("Forget Quality  (log10 p-value, ↑ better forgetting)", fontsize=11)
    ax.set_title("TOFU: Forget Quality vs Model Utility\n"
                 "filled = your run   open = paper reference   ★ = retain model",
                 fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    # add the ghost patch manually since ax.add_patch doesn't auto-register it
    handles.append(mpatches.Patch(facecolor="none", edgecolor="grey",
                                  linewidth=1.5))
    labels.append("paper reference (open circle)")
    ax.legend(handles, labels, fontsize=8, loc="lower left",
              framealpha=0.85, ncol=1)

    # Auto-fit the y-axis to EVERY plotted point. Forget-quality log10 p spans a
    # huge range (e.g. gradient_ascent ~ -7 but gradient_difference ~ -166), so a
    # hard-coded floor silently clips the worst method off the chart. Include the
    # paper ghosts and the retain star too so nothing ever falls outside the axes.
    yvals = [r["forget_quality_log10"] for r in results_by_method.values()]
    yvals += [ref["forget_quality_log10"] for ref in PAPER_REFS.values()]
    if retain_result and retain_result.get("forget_quality_log10") is not None:
        yvals.append(retain_result["forget_quality_log10"])
    ylo, yhi = min(yvals), max(yvals)
    pad = 0.08 * (yhi - ylo) if yhi > ylo else 1.0
    ax.set_xlim(left=-0.02)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.yaxis.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir, "forget_quality_vs_utility")


def rouge_by_split(rouge_by_method_split: Dict[str, Dict[str, float]], out_dir: str):
    """
    rouge_by_method_split[method][split] = rouge value.

    Paper expectation:
      forget split  → should DROP after unlearning (good unlearning)
      retain/real_authors/world_facts → should STAY HIGH (no collateral damage)
    Annotates the expected direction on the plot so it's easy to check.
    """
    splits = ["forget", "retain", "real_authors", "world_facts"]
    split_labels = ["Forget\n(↓ good)", "Retain\n(↑ good)",
                    "Real Authors\n(↑ good)", "World Facts\n(↑ good)"]
    methods = list(rouge_by_method_split.keys())
    n = len(methods)
    width = 0.7 / max(n, 1)
    x = np.arange(len(splits))

    fig, ax = plt.subplots(figsize=(9.5, 5))

    for i, m in enumerate(methods):
        vals = [rouge_by_method_split[m].get(s, 0.0) for s in splits]
        color = METHOD_COLORS.get(m, None)
        if color is None:
            for key in METHOD_COLORS:
                if key in m:
                    color = METHOD_COLORS[key]
                    break
        color = color or DEFAULT_COLOR
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, label=m.replace("_", " "),
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        # value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                        color=color)

    # shading: forget column should be low, rest should be high
    ax.axvspan(-0.5, 0.5, alpha=0.06, color="red",   label="should drop (unlearned)")
    ax.axvspan(0.5,  3.5, alpha=0.04, color="green", label="should stay high (retained)")

    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=10)
    ax.set_ylabel("ROUGE-L recall", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("ROUGE-L by split — checking unlearning vs collateral damage", fontsize=11)
    # Legend outside the axes on the right: every column has at least one tall bar,
    # so any in-axes placement collides with data.
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              framealpha=0.85)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, out_dir, "rouge_by_split")


def spectral_detectability(results_by_method: Dict[str, Dict], out_dir: str):
    """Bar chart: how detectable each unlearned model's fingerprint is.

    results_by_method[name] = a spectral_<name>.json dict with
        detection_accuracy, max_spectral_shift, best_layer, per_layer{...}.

    Two panels:
      (left)  best-layer detection accuracy per method (0.5 = invisible,
              1.0 = trivially detectable) — the headline number.
      (right) loudest spectral shift (max |Cohen's d|) per method.
    A taller bar = a louder fingerprint = knowledge suppressed, not erased.
    """
    methods = list(results_by_method.keys())
    accs = [results_by_method[m]["detection_accuracy"] for m in methods]
    shifts = [results_by_method[m]["max_spectral_shift"] for m in methods]
    colors = [_color_for(m) for m in methods]
    labels = [m.replace("tofu_unlearn_", "").replace("_forget10", "").replace("_", " ")
              for m in methods]
    y = np.arange(len(methods))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 0.6 * len(methods) + 2.5))

    ax1.barh(y, accs, color=colors, alpha=0.85, edgecolor="white")
    ax1.axvline(0.5, ls="--", c="grey", lw=1, label="chance (0.5 = invisible)")
    for yi, a in zip(y, accs):
        ax1.text(a + 0.01, yi, f"{a:.2f}", va="center", fontsize=8)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlim(0.4, 1.05)
    ax1.set_xlabel("detection accuracy (↑ louder trace)", fontsize=10)
    ax1.set_title("Unlearning-trace detectability\n(best layer, 5-fold CV)", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.invert_yaxis()

    ax2.barh(y, shifts, color=colors, alpha=0.85, edgecolor="white")
    for yi, s in zip(y, shifts):
        ax2.text(s + max(shifts) * 0.01, yi, f"{s:.2f}", va="center", fontsize=8)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlabel("max spectral shift  |Cohen's d|  (↑ louder)", fontsize=10)
    ax2.set_title("Spectral fingerprint magnitude\n(top singular directions)", fontsize=10)
    ax2.invert_yaxis()

    for ax in (ax1, ax2):
        ax.xaxis.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
    fig.suptitle("Recovery axis 3 — spectral traces left by unlearning "
                 "(forget-irrelevant prompts)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "spectral_detectability")


def spectral_projection(name: str, spectral_result: Dict, out_dir: str):
    """Scatter of one model pair along the top-2 principal directions.

    Shows the actual cloud of original (grey) vs unlearned (coloured) activations
    at the best-detecting layer. Visible separation = the fingerprint you can see
    with your own eyes (the paper's Fig. 5-style plot)."""
    best = str(spectral_result["best_layer"])
    layer = spectral_result["per_layer"][best]
    po = np.array(layer.get("proj_orig", []))
    pu = np.array(layer.get("proj_unlearned", []))
    if po.ndim != 2 or po.shape[1] < 2 or pu.shape[0] == 0:
        logger.warning("spectral_projection: not enough 2-D projection data for %s", name)
        return
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(po[:, 0], po[:, 1], s=18, alpha=0.5, color="#6c757d",
               label="original (learned)")
    ax.scatter(pu[:, 0], pu[:, 1], s=18, alpha=0.5, color=_color_for(name),
               label="unlearned")
    ax.set_xlabel("1st singular direction", fontsize=10)
    ax.set_ylabel("2nd singular direction", fontsize=10)
    short = name.replace("tofu_unlearn_", "").replace("_", " ")
    ax.set_title(f"Activation shift at layer {best}\n{short}  "
                 f"(detect acc {spectral_result['detection_accuracy']:.2f})",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir, f"spectral_projection_{name}")


SPLIT_COLORS = {
    "forget": "#e63946",        # red   — the set we're erasing
    "retain": "#2a9d8f",        # teal  — should stay high
    "real_authors": "#f4a261",  # orange
    "world_facts": "#457b9d",   # blue
}
SPLIT_LABELS = {
    "forget": "Forget Set", "retain": "Retain Set",
    "real_authors": "Real Authors", "world_facts": "World Facts",
}


def unlearn_curve(curve: Dict, out_dir: str):
    """Reproduce TOFU Figure 8 — unlearning dynamics. Three panels
    (ROUGE / Probability / Truth Ratio), x = unlearning step, one line per eval
    split. `curve` is an unlearn_curve_<method>_<level>.json dict:
        {method, forget_level, history:[{step, split, rouge, prob, truth_ratio}, ...]}

    Reading (paper): Forget → ↓ROUGE/Prob and ↑Truth-Ratio = good forgetting;
    Retain/Real/World → ↑ROUGE/Prob = little collateral damage. The gap between
    the forget curve dropping and the others following is 'knowledge entanglement'.
    """
    hist = curve.get("history", [])
    if not hist:
        logger.warning("unlearn_curve: empty history, nothing to plot")
        return
    method = curve.get("method", "?")
    level = curve.get("forget_level", "")
    metrics = [("rouge", "ROUGE-L"), ("prob", "Probability"),
               ("truth_ratio", "Truth Ratio")]
    # pivot: series[metric][split] = ([steps], [values])
    splits = list(dict.fromkeys(h["split"] for h in hist))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (key, title) in zip(axes, metrics):
        for s in splits:
            pts = sorted((h["step"], h[key]) for h in hist if h["split"] == s)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.8,
                    color=SPLIT_COLORS.get(s, DEFAULT_COLOR),
                    label=SPLIT_LABELS.get(s, s))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Unlearning Steps", fontsize=10)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
    axes[0].set_ylim(-0.02, 1.05)      # ROUGE and Probability live in [0,1]
    axes[1].set_ylim(-0.02, 1.05)
    # Truth Ratio is an unbounded ratio P(wrong)/P(correct): as unlearning drives
    # P(correct)->0 on the forget set it explodes into the thousands, dwarfing the
    # other splits on a linear axis. Log scale keeps all four curves legible.
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"TOFU Fig. 8 — unlearning dynamics: {method} on {level}\n"
                 "Forget: ↓ROUGE/Prob, ↑Truth-Ratio good  ·  "
                 "Retain/Real/World: ↑ good", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_dir, f"unlearn_curve_{method}_{level}")


def learning_success(results_by_model: Dict[str, Dict], out_dir: str):
    """Validate the LEARN phase: ROUGE-L + Probability per split, one bar group
    per model. The claim it demonstrates: after full fine-tuning, the FORGET and
    RETAIN splits — the *fictitious* TOFU authors — jump to high ROUGE and high
    P(answer|question), proving the model memorized authors it could not have
    known before (they're invented). Real Authors / World Facts (genuine
    knowledge) stay high in both models — the control showing the gain is
    specific to the fictitious authors, not a general uplift.

    results_by_model[label] = an eval json (from eval_learning.py / evaluate_tofu)
    with a `per_split` block. Pass {base, learned} to get the before/after story.
    """
    splits = ["forget", "retain", "real_authors", "world_facts"]
    split_labels = ["Forget\n(fictitious)", "Retain\n(fictitious)",
                    "Real Authors", "World Facts"]
    models = list(results_by_model.keys())
    metrics = [("rouge", "ROUGE-L recall"), ("prob", "P(answer | question)")]
    palette = ["#adb5bd", "#2a9d8f", "#457b9d", "#e63946"]
    x = np.arange(len(splits))
    n = max(len(models), 1)
    width = 0.8 / n

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        ax.axvspan(-0.5, 1.5, alpha=0.06, color="green")   # fictitious-author region
        for i, m in enumerate(models):
            ps = results_by_model[m].get("per_split", {})
            vals = [ps.get(s, {}).get(key, 0.0) for s in splits]
            bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width=width,
                          label=m, color=palette[i % len(palette)], alpha=0.9,
                          edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, vals):
                if v > 0.02:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, fontsize=9)
        ax.set_ylabel(title, fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("LEARN validation — did full fine-tuning teach the fictitious TOFU "
                 "authors?\nshaded = fictitious authors (Forget+Retain): ~0 before, "
                 "high after ⇒ memorized", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out_dir, "learning_success")


def _pick_direction(layer_result: Dict) -> int:
    """Among the two stored singular directions (SV1, SV2), pick the one with the
    larger |Cohen's d| — i.e. where the fingerprint actually localizes. For some
    methods the shift lives on SV2, not SV1 (e.g. gradient_ascent), so blindly
    using SV1 can hide the trace."""
    cd = layer_result.get("cohens_d", [0, 0])
    return 0 if abs(cd[0]) >= abs(cd[1] if len(cd) > 1 else 0) else 1


def spectral_signature(name: str, spectral_result: Dict, out_dir: str,
                       layer: int = None, direction: int = None):
    """Reproduce the paper's Fig. 5 (Tran et al. 2018 spectral signature): overlaid
    HISTOGRAMS of the activation projection onto a singular vector, original vs
    unlearned, at one layer. y = number of responses, x = projection on SVk.
    The separation between the two histograms IS the unlearning fingerprint.

    layer     : which layer (default = the best-detecting layer).
    direction : 0=SV1, 1=SV2 (default = whichever of the two has the larger shift).
    """
    lay = str(layer if layer is not None else spectral_result["best_layer"])
    L = spectral_result["per_layer"][lay]
    po = np.array(L.get("proj_orig", []))
    pu = np.array(L.get("proj_unlearned", []))
    if po.ndim != 2 or po.shape[0] == 0 or pu.shape[0] == 0:
        logger.warning("spectral_signature: no projection data for %s", name)
        return
    d = direction if direction is not None else _pick_direction(L)
    _signature_panel(plt.subplots(figsize=(6.5, 4.5))[1], name, L, po, pu, lay, d,
                     legend=True)
    fig = plt.gcf()
    fig.tight_layout()
    _save(fig, out_dir, f"spectral_signature_{name}")


def spectral_signature_grid(results_by_method: Dict[str, Dict], out_dir: str):
    """Paper-style Fig. 5 as small multiples: one SV-projection histogram per
    method (original vs unlearned), stacked so the *magnitude* of each method's
    fingerprint is comparable at a glance. Each panel auto-selects the layer +
    singular direction where that method's shift is strongest (among what we
    stored: best-detecting layer, SV1/SV2)."""
    methods = list(results_by_method.keys())
    n = len(methods)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, methods):
        res = results_by_method[name]
        lay = str(res["best_layer"])
        L = res["per_layer"][lay]
        po = np.array(L.get("proj_orig", []))
        pu = np.array(L.get("proj_unlearned", []))
        if po.ndim != 2 or po.shape[0] == 0:
            ax.set_visible(False)
            continue
        _signature_panel(ax, name, L, po, pu, lay, _pick_direction(L),
                         legend=(ax is axes[0]))
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Spectral signatures of unlearning (paper Fig. 5 style)\n"
                 "projection of forget-irrelevant responses onto the localizing "
                 "singular vector — original vs unlearned", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, "spectral_signature_grid")


def _signature_panel(ax, name, layer_result, po, pu, lay, direction, legend):
    """One overlaid-histogram panel used by both signature plots."""
    o = po[:, direction]
    u = pu[:, direction]
    lo = min(o.min(), u.min())
    hi = max(o.max(), u.max())
    bins = np.linspace(lo, hi, 41)
    ax.hist(o, bins=bins, alpha=0.6, color="#6c757d", label="original (learned)")
    ax.hist(u, bins=bins, alpha=0.6, color=_color_for(name), label="unlearned")
    d = layer_result.get("cohens_d", [0, 0])[direction]
    short = (name.replace("tofu_unlearn_", "").replace("_forget10", "")
                 .replace("_", " ").strip())
    ax.set_title(f"{short}  —  L{lay}, SV{direction + 1}  (d={d:.2f})", fontsize=9.5)
    ax.set_xlabel(f"projection on singular vector {direction + 1}", fontsize=9)
    ax.set_ylabel("# responses", fontsize=9)
    if legend:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)


def _color_for(name: str) -> str:
    """Pick a method colour by substring match (works on long run names)."""
    for key, c in METHOD_COLORS.items():
        if key in name:
            return c
    return DEFAULT_COLOR


def _save(fig, out_dir: str, name: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot -> %s", path)