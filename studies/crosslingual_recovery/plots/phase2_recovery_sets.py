"""PHASE 2 (English-column variant): WHICH facts come back, per relearn language.

Phase 1 measured HOW MUCH knowledge returned -- a mean over the 40 forget facts, and
~42%/63% regardless of which language we relearned in. A mean cannot separate two very
different worlds:

    same amount, SAME facts       -> one shared store; the relearn language is just a key
    same amount, DIFFERENT facts  -> each language repairs its own subset, coincidentally
                                     similar in size

This plot opens the mean up. For each method we take the facts the unlearning actually
killed, and mark which of them each relearn language brought back:

    Rec(l) = { i : truth_ratio_unlearned(i) >= tau  AND  truth_ratio_relearned_in_l(i) < tau }

then compare the ten sets pairwise (Jaccard) against a permutation null drawn from the
SAME eligible pool -- only a fact that was dead can recover, so the null must draw from
the dead facts, not all 40.

Why the probe is English-only: the Phase 2 calibration gate (phase2_kss_kps.py
--calibrate-only) found the learned and base models separate ONLY in English (AUC 0.825)
and marginally French (0.721); LEARN was English-only and TOFU's authors are fictitious,
so the other eight languages hold no knowledge to recover. The relearn language is an
INPUT (a stimulus -- it needs no knowledge in that language) while the probe language is
an OUTPUT (a measurement -- it does). Varying the input and pinning the output to
English is what makes this runnable on the current checkpoints.

    python studies/crosslingual_recovery/plots/phase2_recovery_sets.py

Inputs (local, CPU): results/relearn/crosslingual_deep/*.json (per-fact arrays at ep2,
incl. the unlearned baselines) and results/phase2_tau.json.
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

logger = get_logger("phase2_recovery_sets")

STUDY = Path(__file__).resolve().parents[1]
DEEP = STUDY / "results" / "relearn" / "crosslingual_deep"
FIGS = STUDY / "figures"

LANGS = ["en", "fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]
LANG_NAME = {"en": "Eng", "fr": "Fra", "id": "Ind", "ru": "Rus", "hi": "Hin",
             "fa": "Far", "ar": "Ara", "iw": "Heb", "ko": "Kor", "ja": "Jpn"}
# Fixed identity colours, matching every other figure in this study (validated pair:
# CVD dE 24.6, normal-vision 35.7). Presence in the matrix is encoded by a FILLED vs
# EMPTY cell, so membership never rests on colour alone.
METHODS = {"Full-FT": ("tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b", "#1f77b4"),
           "LoRA": ("tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b", "#ff7f0e")}
EP = 2


def load(base, tau):
    """(dead-fact indices, {lang: recovered-set}) for one unlearned checkpoint."""
    d = json.load(open(DEEP / f"{base}.json"))
    if "truth_ratio_per_fact" not in d.get(base, {}):
        logger.warning("%s has no per-fact array for the UNLEARNED baseline -- run "
                       "slurm/measure_unlearned_baseline.sbatch", base)
        return None, None
    unl = np.asarray(d[base]["truth_ratio_per_fact"], float)
    dead = np.where(unl >= tau)[0]              # only these can "recover"
    rec = {}
    for l in LANGS:
        k = f"relearn_{base}_via_retain" + ("" if l == "en" else f"_lang{l}") + f"_ep{EP}"
        v = d.get(k)
        if not (isinstance(v, dict) and "truth_ratio_per_fact" in v):
            continue
        r = np.asarray(v["truth_ratio_per_fact"], float)
        rec[l] = {i for i in dead if r[i] < tau}
    return dead, rec


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else np.nan


def null_dist(pool, sizes, langs, rng, reps=4000):
    """Mean pairwise Jaccard when each language draws its OWN number of facts at
    random from the eligible pool -- the chance level for 'the same facts came back'."""
    out = []
    for _ in range(reps):
        S = {l: set(rng.choice(pool, sizes[l], replace=False)) for l in langs}
        out.append(np.nanmean([jaccard(S[a], S[b]) for a, b in itertools.combinations(langs, 2)]))
    return np.asarray(out)


def main():
    tau = json.load(open(STUDY / "results" / "phase2_tau.json"))["en"]["tau"]
    rng = np.random.default_rng(0)
    data = {}
    for m, (base, color) in METHODS.items():
        dead, rec = load(base, tau)
        if dead is None or not rec:
            continue
        data[m] = (dead, rec, color)
    if not data:
        logger.error("no usable per-fact data in %s", DEEP)
        return

    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, len(data), height_ratios=[3.4, 0.62], hspace=0.30, wspace=0.16)

    stats = {}
    for col, (m, (dead, rec, color)) in enumerate(data.items()):
        langs = [l for l in LANGS if l in rec]
        ax = fig.add_subplot(gs[0, col])
        for y, fact in enumerate(dead):
            for x, l in enumerate(langs):
                hit = fact in rec[l]
                # 2px-equivalent gap between cells; empty cells keep a faint outline so
                # "not recovered" reads as measured, not missing.
                ax.add_patch(plt.Rectangle((x + .06, y + .06), .88, .88,
                                           facecolor=color if hit else "none",
                                           edgecolor=color if hit else "#d6d6d6",
                                           lw=0.9))
        ax.set_xlim(0, len(langs)); ax.set_ylim(len(dead), 0)
        ax.set_xticks([i + .5 for i in range(len(langs))])
        ax.set_xticklabels([LANG_NAME[l] for l in langs], fontsize=9)
        ax.set_yticks([i + .5 for i in range(len(dead))])
        ax.set_yticklabels([f"#{f}" for f in dead], fontsize=7)
        ax.set_xlabel("relearn language  (the stimulus)", fontsize=10)
        if col == 0:
            ax.set_ylabel(f"forget fact  (only the {len(dead)} killed by unlearning)", fontsize=10)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        n_rec = len(set().union(*rec.values()))
        sizes = {l: len(rec[l]) for l in langs}
        ax.set_title(f"{m}\n{len(dead)}/40 facts eligible · {n_rec} ever recovered · "
                     f"{min(sizes.values())}–{max(sizes.values())} per language",
                     fontsize=11, color=color)
        obs = np.nanmean([jaccard(rec[a], rec[b]) for a, b in itertools.combinations(langs, 2)])
        nd = null_dist(dead, sizes, langs, rng)
        stats[m] = (obs, nd, color)

    # Bottom: observed overlap against its own chance level. Direct value labels are the
    # relief the palette validator requires for the low-contrast series.
    ax2 = fig.add_subplot(gs[1, :])
    for y, (m, (obs, nd, color)) in enumerate(stats.items()):
        lo, hi = np.percentile(nd, [2.5, 97.5])
        ax2.plot([lo, hi], [y, y], lw=7, color="#d6d6d6", solid_capstyle="butt", zorder=1)
        ax2.plot(nd.mean(), y, "|", ms=16, color="#8a8a8a", mew=2, zorder=2)
        ax2.plot(obs, y, "o", ms=13, color=color, mec="white", mew=2, zorder=3)
        p = (nd >= obs).mean()
        ax2.annotate(f"observed {obs:.2f}   (chance {nd.mean():.2f}, p={p:.4f})",
                     (obs, y), textcoords="offset points", xytext=(14, 0),
                     va="center", fontsize=10, color="#333333")
        stats[m] = (obs, nd, color)
    ax2.set_yticks(range(len(stats)))
    ax2.set_yticklabels(list(stats), fontsize=11)
    # Match the matrices above: Full-FT first. Matplotlib counts y upward, so the
    # first-plotted row would otherwise land at the BOTTOM and read back-to-front.
    ax2.set_ylim(len(stats) - 0.5, -0.5)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("mean pairwise Jaccard of the recovered sets   "
                   "(grey bar = 95% of the permutation null, tick = its mean)", fontsize=10)
    ax2.grid(True, axis="x", alpha=0.25, ls="--"); ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)

    fig.suptitle("Phase 2 (English probe): do the SAME facts come back, whichever language "
                 "we relearn in?\nfilled = that fact recovered · overlap far above chance, "
                 "but on very few facts — see the sparse matrices", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "phase2_recovery_sets.png"
    fig.savefig(out, dpi=130)
    logger.info("recovery sets -> %s", out)
    for m, (obs, nd, _) in stats.items():
        logger.info("  %-8s observed %.3f  chance %.3f  p=%.4f",
                    m, obs, nd.mean(), (nd >= obs).mean())


if __name__ == "__main__":
    main()
