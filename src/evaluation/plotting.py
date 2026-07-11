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

# For the STRATEGY comparison (axis 2), colour by training strategy — not by
# unlearning method — so Full-FT and LoRA (both gradient_difference) get distinct
# colours. Matches the relearn_forget_curve palette.
STRATEGY_COLORS = {
    "Full-FT (grad-diff)": "#1f77b4",   # blue
    "LoRA (grad-diff)":    "#ff7f0e",   # orange
    "Self-Distillation":   "#2ca02c",   # green
    "GRPO":                "#d62728",   # red
}


def strategy_label(name: str) -> str:
    """Map a checkpoint run-name to its training-strategy label (axis 2)."""
    n = name.lower()
    if "self_distill" in n:
        return "Self-Distillation"
    if "grpo" in n:
        return "GRPO"
    if "lora" in n:
        return "LoRA (grad-diff)"
    return "Full-FT (grad-diff)"


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
        # Prefer STRATEGY_COLORS (keyed by the 4 strategy labels) so this plane
        # matches the relearn/spectral palette; fall back to method-name colours.
        color = STRATEGY_COLORS.get(name) or METHOD_COLORS.get(name)
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


def spectral_detectability(results_by_method: Dict[str, Dict], out_dir: str,
                           strategy_view: bool = False, final_layer: bool = False):
    """Bar chart: how detectable each unlearned model's fingerprint is.

    strategy_view=True colours + labels by training STRATEGY (Full-FT / LoRA /
    Self-Distillation) and writes spectral_detectability_strategies.png — pass a
    dict filtered to the gradient_difference-family checkpoints.
    final_layer=True reports the FINAL-layer SV1 shift + accuracy (paper's NPO
    analysis) so the bars match the signature panels, instead of a global max.

    results_by_method[name] = a spectral_<name>.json dict with
        detection_accuracy, max_spectral_shift, best_layer, per_layer{...}.

    Two panels:
      (left)  best-layer detection accuracy per method (0.5 = invisible,
              1.0 = trivially detectable) — the headline number.
      (right) loudest spectral shift (max |Cohen's d|) per method.
    A taller bar = a louder fingerprint = knowledge suppressed, not erased.
    """
    methods = list(results_by_method.keys())
    if final_layer:
        # Paper's NPO analysis: report the FINAL layer's SV1 shift + accuracy, so
        # the bar chart matches the signature panels exactly (not a global max on
        # some intermediate layer).
        accs, shifts = [], []
        for m in methods:
            lay, d = _final_layer_direction(results_by_method[m])
            L = results_by_method[m]["per_layer"][lay]
            accs.append(L["detection_accuracy"])
            shifts.append(abs(L["cohens_d"][d]))
    else:
        accs = [results_by_method[m]["detection_accuracy"] for m in methods]
        shifts = [results_by_method[m]["max_spectral_shift"] for m in methods]
    if strategy_view:
        labels = [strategy_label(m) for m in methods]
        colors = [STRATEGY_COLORS.get(l, DEFAULT_COLOR) for l, m in zip(labels, methods)]
    else:
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
    layer_note = "final layer, 5-fold CV" if final_layer else "best layer, 5-fold CV"
    ax1.set_title(f"Unlearning-trace detectability\n({layer_note})", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.invert_yaxis()

    ax2.barh(y, shifts, color=colors, alpha=0.85, edgecolor="white")
    for yi, s in zip(y, shifts):
        ax2.text(s + max(shifts) * 0.01, yi, f"{s:.2f}", va="center", fontsize=8)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlabel("spectral shift  |Cohen's d|  (↑ louder)", fontsize=10)
    shift_note = "final layer, SV1" if final_layer else "top singular directions"
    ax2.set_title(f"Spectral fingerprint magnitude\n({shift_note})", fontsize=10)
    ax2.invert_yaxis()

    for ax in (ax1, ax2):
        ax.xaxis.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
    sub = ("by training STRATEGY (all gradient_difference)" if strategy_view
           else "(forget-irrelevant prompts)")
    fig.suptitle(f"Recovery axis 3 — spectral traces left by unlearning  {sub}",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "spectral_detectability_strategies" if strategy_view
          else "spectral_detectability")


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
    # run_name (if present) uniquely identifies the strategy+method+level, so the
    # three forget10 curves (Full-FT / LoRA / self-distill) get distinct files.
    run_name = curve.get("run_name")
    file_id = run_name.replace("tofu_unlearn_", "") if run_name else f"{method}_{level}"
    title_id = file_id.replace("_", " ") if run_name else f"{method} on {level}"
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
            # Truth Ratio: bound to (0,1] as locuslab/tofu reports it — min(R,1/R).
            # The raw stored value is the unbounded ratio and explodes to 1e8 once
            # the forget set collapses (P(correct)->0). Bounding here makes old
            # curves match the paper's Fig 8/17/18 panel; it is idempotent for new
            # runs, which already store the per-record-bounded value (<=1).
            if key == "truth_ratio":
                ys = [min(v, 1.0 / v) if v and v > 0 else 0.0 for v in ys]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.8,
                    color=SPLIT_COLORS.get(s, DEFAULT_COLOR),
                    label=SPLIT_LABELS.get(s, s))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Unlearning Steps", fontsize=10)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
    axes[0].set_ylim(-0.02, 1.05)      # ROUGE and Probability live in [0,1]
    axes[1].set_ylim(-0.02, 1.05)
    axes[2].set_ylim(-0.02, 1.05)      # bounded Truth Ratio min(R,1/R) in (0,1]
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"TOFU Fig. 8 — unlearning dynamics: {title_id}\n"
                 "Forget: ↓ROUGE/Prob = good forgetting  ·  Retain/Real/World: "
                 "↑ = low collateral  ·  Truth Ratio = min(R,1/R)∈(0,1] "
                 "(locuslab/tofu)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_dir, f"unlearn_curve_{file_id}")


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


def _best_visible_direction(spectral_result: Dict):
    """(layer_key, direction) with the largest |Cohen's d| among the directions we
    actually stored projections for (SV1/SV2, at EVERY layer) — i.e. where the
    fingerprint is most visible: "the one with the highest difference." Scans all
    layers, not just the best-detection layer.

    NOTE: limited to the top-2 SVs we saved; a larger shift can live on SV3-5
    (e.g. gradient_difference's global max is L15-SV4) — showing those exactly
    would need re-running spectral with more stored projection columns."""
    best_lay, best_dir, best_d = None, 0, -1.0
    for lay, L in spectral_result["per_layer"].items():
        if not L.get("proj_orig"):
            continue
        cd = L.get("cohens_d", [])
        for direction in range(min(2, len(cd))):   # only SV1/SV2 have stored proj
            if abs(cd[direction]) > best_d:
                best_lay, best_dir, best_d = lay, direction, abs(cd[direction])
    return (best_lay or str(spectral_result["best_layer"])), best_dir


def _final_layer_direction(spectral_result: Dict):
    """(final layer, SV with the largest shift there). This is the paper's NPO
    analysis: the FINAL post-RMSNorm activations, projected onto the top singular
    vector. For loss-based (NPO-family) unlearning the shift concentrates on SV1,
    so this returns (last layer, SV1) in practice — a clean, paper-aligned choice
    for gradient_difference / self-distillation."""
    per = spectral_result["per_layer"]
    lay = str(max(int(k) for k in per))          # highest layer index = final layer
    cd = per[lay].get("cohens_d", [0, 0])
    direction = max(range(min(2, len(cd))), key=lambda i: abs(cd[i]))
    return lay, direction


def spectral_signature(name: str, spectral_result: Dict, out_dir: str,
                       layer: int = None, direction: int = None):
    """Reproduce the paper's Fig. 5 (Tran et al. 2018 spectral signature): overlaid
    HISTOGRAMS of the activation projection onto a singular vector, original vs
    unlearned, at one layer. y = number of responses, x = projection on SVk.
    The separation between the two histograms IS the unlearning fingerprint.

    layer     : which layer (default = the best-detecting layer).
    direction : 0=SV1, 1=SV2 (default = whichever of the two has the larger shift).
    """
    if layer is not None:
        lay = str(layer)
        d = direction if direction is not None else _pick_direction(
            spectral_result["per_layer"][lay])
    else:
        lay, best_d = _best_visible_direction(spectral_result)
        d = direction if direction is not None else best_d
    L = spectral_result["per_layer"][lay]
    po = np.array(L.get("proj_orig", []))
    pu = np.array(L.get("proj_unlearned", []))
    if po.ndim != 2 or po.shape[0] == 0 or pu.shape[0] == 0:
        logger.warning("spectral_signature: no projection data for %s", name)
        return
    _signature_panel(plt.subplots(figsize=(6.5, 4.5))[1], name, L, po, pu, lay, d,
                     legend=True)
    fig = plt.gcf()
    fig.tight_layout()
    _save(fig, out_dir, f"spectral_signature_{name}")


def spectral_signature_grid(results_by_method: Dict[str, Dict], out_dir: str,
                            strategy_view: bool = False, final_layer: bool = False):
    """Paper-style Fig. 5 as small multiples: one SV-projection density per method
    (original vs unlearned), stacked so the *magnitude* of each fingerprint is
    comparable at a glance.

    final_layer=True uses the FINAL layer + SV1 (paper's NPO analysis) for every
    panel; otherwise each panel auto-selects the layer+SV with the strongest shift.
    strategy_view=True colours + labels by training strategy and writes
    spectral_signature_strategies.png (pass the gradient_difference-family subset)."""
    methods = list(results_by_method.keys())
    n = len(methods)
    ncol = 1 if strategy_view else 2   # 3 strategies read better in one column
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 if strategy_view else 11,
                                                  2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    pick = _final_layer_direction if final_layer else _best_visible_direction
    for ax, name in zip(axes, methods):
        res = results_by_method[name]
        lay, direction = pick(res)
        L = res["per_layer"][lay]
        po = np.array(L.get("proj_orig", []))
        pu = np.array(L.get("proj_unlearned", []))
        if po.ndim != 2 or po.shape[0] == 0:
            ax.set_visible(False)
            continue
        color = title_label = None
        if strategy_view:
            title_label = strategy_label(name)
            color = STRATEGY_COLORS.get(title_label, DEFAULT_COLOR)
        _signature_panel(ax, name, L, po, pu, lay, direction,
                         legend=(ax is axes[0]), color=color, title_label=title_label)
    for ax in axes[n:]:
        ax.set_visible(False)
    head = ("Spectral signatures by training STRATEGY (all gradient_difference)"
            if strategy_view else "Spectral signatures of unlearning (paper Fig. 5 style)")
    fig.suptitle(head + "\nprojection of forget-irrelevant responses onto the "
                 "localizing singular vector — original vs unlearned", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, "spectral_signature_strategies" if strategy_view
          else "spectral_signature_grid")


def _kde(data, xs):
    """Smooth density estimate of `data` over grid `xs` (paper Fig. 5 uses smooth
    curves, not bars). Falls back to a normalised histogram if a KDE can't be fit
    (e.g. zero-variance data)."""
    data = np.asarray(data, dtype=float)
    if len(data) > 1 and data.std() > 1e-8:
        try:
            from scipy.stats import gaussian_kde
            return gaussian_kde(data)(xs)
        except Exception:
            pass
    counts, edges = np.histogram(data, bins=30, range=(xs[0], xs[-1]), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(xs, centers, counts, left=0, right=0)


def _signature_panel(ax, name, layer_result, po, pu, lay, direction, legend,
                     color=None, title_label=None):
    """One overlaid smooth-density panel (KDE) used by both signature plots — the
    original vs unlearned projection distributions along a singular vector.
    `color`/`title_label` override the per-method colour/label (for strategy view)."""
    o = po[:, direction]
    u = pu[:, direction]
    lo = min(o.min(), u.min())
    hi = max(o.max(), u.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    xs = np.linspace(lo - pad, hi + pad, 400)
    bin_w = (xs[-1] - xs[0]) / 40.0    # scale the KDE by N*bin_w so y reads as counts
    unl_color = color if color is not None else _color_for(name)
    for data, col, label in [(o, "#6c757d", "original (learned)"),
                             (u, unl_color, "unlearned")]:
        ys = _kde(data, xs) * len(data) * bin_w    # smooth curve on a "# responses" scale
        ax.fill_between(xs, ys, alpha=0.30, color=col)
        ax.plot(xs, ys, lw=1.8, color=col, label=label)
    d = layer_result.get("cohens_d", [0, 0])[direction]
    short = title_label if title_label is not None else (
        name.replace("tofu_unlearn_", "").replace("_forget10", "")
            .replace("_", " ").strip())
    ax.set_title(f"{short}  —  L{lay}, SV{direction + 1}  (d={d:.2f})", fontsize=9.5)
    ax.set_xlabel(f"projection on singular vector {direction + 1}", fontsize=9)
    ax.set_ylabel("# responses", fontsize=9)
    ax.set_ylim(bottom=0)
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