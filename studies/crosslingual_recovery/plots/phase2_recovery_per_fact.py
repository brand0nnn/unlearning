"""PHASE 2, THRESHOLD-FREE: does the same PER-FACT recovery pattern appear in every
relearn language?

phase2_recovery_sets.py answers the same question by binarising the truth ratio at tau
and comparing recovered SETS. That inherits every weakness of tau: it is our choice, not
a measurement (Youden's J on the learned-vs-base ROC, TPR 0.725 / FPR 0.163), it is
calibrated on the learned and base models but applied to unlearned and relearned ones,
and it discards 23 of the 40 facts as "not eligible" before the analysis even starts.

This version uses no threshold at all. The truth ratio is continuous, so keep it:

    delta(i, l) = truth_ratio_unlearned(i) - truth_ratio_relearned_in_l(i)

positive = fact i moved back toward "known" after relearning in language l. That is a
40-vector per language, over ALL 40 facts. "The same facts come back" then becomes a
correlation question -- do the ten vectors rise and fall together? -- with more power
than a set overlap, since a fact contributes its magnitude rather than a 0/1.

Null: shuffle the fact order within each language independently, destroying any
fact-level agreement while preserving each language's own distribution of movements.

Spearman is the headline (rank-based, so a couple of large movements cannot manufacture
the correlation); Pearson is reported alongside.

    python studies/crosslingual_recovery/plots/phase2_recovery_per_fact.py
"""
import itertools
import json
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logging_utils import get_logger

logger = get_logger("phase2_recovery_per_fact")

STUDY = Path(__file__).resolve().parents[1]
DEEP = STUDY / "results" / "relearn" / "crosslingual_deep"
FIGS = STUDY / "figures"

LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]
LANG_NAME = {"en": "Eng", "fr": "Fra", "id": "Ind", "ru": "Rus", "hi": "Hin",
             "fa": "Far", "ar": "Ara", "iw": "Heb", "ko": "Kor", "ja": "Jpn"}
METHODS = {"Full-FT": ("tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b", "#1f77b4"),
           "LoRA": ("tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b", "#ff7f0e")}
EP = 2


def deltas(base):
    """{lang: 40-vector of (unlearned TR - relearned TR)}. Positive = recovered."""
    d = json.load(open(DEEP / f"{base}.json"))
    if "truth_ratio_per_fact" not in d.get(base, {}):
        logger.warning("%s: unlearned baseline has no per-fact array", base)
        return None, None
    unl = np.asarray(d[base]["truth_ratio_per_fact"], float)
    out = {}
    for l in LANGS:
        k = f"relearn_{base}_via_retain" + ("" if l == "en" else f"_lang{l}") + f"_ep{EP}"
        v = d.get(k)
        if isinstance(v, dict) and "truth_ratio_per_fact" in v:
            out[l] = unl - np.asarray(v["truth_ratio_per_fact"], float)
    return out, unl


def _rank(v):
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float); r[o] = np.arange(len(v), dtype=float)
    return r


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den else np.nan


def mean_pairwise(mat, spearman=True):
    v = [_rank(r) if spearman else r for r in mat]
    return float(np.mean([_pearson(v[i], v[j])
                          for i, j in itertools.combinations(range(len(v)), 2)]))


def partial_out(vec, control):
    """`vec` with its linear dependence on `control` removed.

    Every delta shares the SAME unlearned baseline by construction, so if facts merely
    differ in baseline difficulty and relearning shifts them all similarly, the ten
    vectors would correlate without any shared recovery structure at all. Regressing the
    baseline out of each vector before correlating removes that free agreement, so what
    survives is genuinely about which facts came back."""
    x = control - control.mean()
    y = vec - vec.mean()
    return y - ((x * y).sum() / (x * x).sum()) * x


def main():
    rng = np.random.default_rng(0)
    data, base_tr = {}, {}
    for m, (base, color) in METHODS.items():
        dl, unl = deltas(base)
        if dl:
            data[m] = (dl, color)
            base_tr[m] = unl
    if not data:
        logger.error("no per-fact data under %s", DEEP)
        return

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, len(data), height_ratios=[3.4, 0.75], hspace=0.32, wspace=0.20)

    # Diverging encoding: recovery has POLARITY (a fact can also get worse), so two hues
    # about a neutral midpoint pinned at delta = 0 -- never a sequential ramp, which
    # would hide the sign, and never the method colours, which mean identity here.
    # ROBUST limits: one fact moves ~3.5 while the rest sit under 0.5, and scaling to
    # that outlier washes the remaining 39 rows to near-white. Clip the scale at the
    # 98th percentile (values beyond it still render at full saturation, they just stop
    # setting the scale) so the actual structure stays visible.
    allv = np.abs(np.concatenate([np.column_stack(list(dl.values())).ravel()
                                  for dl, _ in data.values()]))
    vmax = float(np.percentile(allv, 98))
    stats = {}
    for col, (m, (dl, color)) in enumerate(data.items()):
        langs = [l for l in LANGS if l in dl]
        mat = np.column_stack([dl[l] for l in langs])       # 40 facts x langs
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels([LANG_NAME[l] for l in langs], fontsize=9)
        ax.set_yticks(range(0, 40, 5)); ax.set_yticklabels(range(0, 40, 5), fontsize=8)
        ax.set_xlabel("relearn language  (the stimulus)", fontsize=10)
        if col == 0:
            ax.set_ylabel("forget fact index  (all 40 — no threshold)", fontsize=10)
        rows = np.column_stack([dl[l] for l in langs])
        sp = mean_pairwise(rows.T, spearman=True)
        pe = mean_pairwise(rows.T, spearman=False)
        unl = base_tr[m]
        part = mean_pairwise(np.column_stack([partial_out(dl[l], unl) for l in langs]).T,
                             spearman=True)
        null = []
        for _ in range(2000):
            perm = np.column_stack([rng.permutation(rows[:, i]) for i in range(rows.shape[1])])
            null.append(mean_pairwise(perm.T, spearman=True))
        null = np.asarray(null)
        stats[m] = (sp, pe, null, color)
        logger.info("  %-8s raw %.3f  partial(baseline removed) %.3f", m, sp, part)
        ax.set_title(f"{m}\nmean pairwise Spearman across languages = {sp:.2f}  "
                     f"(Pearson {pe:.2f})\n"
                     f"{part:.2f} with the shared baseline partialled out",
                     fontsize=10.5, color=color)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("Δ truth ratio (unlearned − relearned)\nred = recovered · scale clipped at p98",
                     fontsize=8.5)
        cb.ax.tick_params(labelsize=8)

    ax2 = fig.add_subplot(gs[1, :])
    for y, (m, (sp, pe, null, color)) in enumerate(stats.items()):
        lo, hi = np.percentile(null, [2.5, 97.5])
        ax2.plot([lo, hi], [y, y], lw=7, color="#d6d6d6", solid_capstyle="butt", zorder=1)
        ax2.plot(null.mean(), y, "|", ms=16, color="#8a8a8a", mew=2, zorder=2)
        ax2.plot(sp, y, "o", ms=13, color=color, mec="white", mew=2, zorder=3)
        p = (null >= sp).mean()
        # Label inside the axes: at Spearman ~0.9 the dot sits at the right edge, so a
        # rightward offset runs off the figure and the text is clipped.
        right = sp > 0.55
        ax2.annotate(f"observed {sp:.2f}   (shuffled facts {null.mean():+.2f}, p={p:.4f})",
                     (sp, y), textcoords="offset points",
                     xytext=(-14 if right else 14, 0), va="center",
                     ha="right" if right else "left", fontsize=10, color="#333333")
    ax2.set_yticks(range(len(stats))); ax2.set_yticklabels(list(stats), fontsize=11)
    ax2.set_ylim(len(stats) - 0.5, -0.5)
    ax2.set_xlim(-0.2, 1.0)
    ax2.axvline(0, color="#8a8a8a", lw=1, ls=":")
    ax2.set_xlabel("mean pairwise Spearman of the per-fact recovery vectors   "
                   "(grey = 95% of the fact-shuffled null)", fontsize=10)
    ax2.grid(True, axis="x", alpha=0.25, ls="--"); ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)

    fig.suptitle("Phase 2, threshold-free: the same facts recover in every relearn language\n"
                 "all 40 facts, no tau, no eligibility filter — columns are near-copies "
                 "of each other", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "phase2_recovery_per_fact.png"
    fig.savefig(out, dpi=130)
    logger.info("per-fact recovery -> %s", out)
    for m, (sp, pe, null, _) in stats.items():
        logger.info("  %-8s Spearman %.3f  Pearson %.3f  null %.3f  p=%.4f",
                    m, sp, pe, null.mean(), (null >= sp).mean())


if __name__ == "__main__":
    main()
