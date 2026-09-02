"""PHASE 2 Part A, steps 1-2 -- the FULL probe family (p0 + p1 + three AUTHORED
paraphrases per fact) scored at the LEARNED and the two UNLEARNED checkpoints.

WHAT THIS ADDS OVER phase2_free_probes.py. That run had one paraphrase per fact
(TOFU's own `paraphrased_question`), so the phrasing axis was n=2. With n=2 there is
no within-fact spread, and "unlearning is phrasing-specific" cannot be told apart from
"that ONE paraphrase happened to be easy". Each fact now carries five phrasings, so
every fact has its own distribution over surface forms.

The three authored slots vary along CONTROLLED axes, one per slot, so which KIND of
rephrasing defeats suppression is a finding rather than a robustness check. Measured
content-word Jaccard against the canonical question (stopwords removed, median over
40 facts) confirms the axes behave as intended:

    p0_canonical    1.000   TOFU's question -- seen in LEARN *and* in UNLEARN
    p2_authored     0.552   syntactic recast     -- same content words, restructured
    p4_authored     0.400   oblique framing      -- the fact asked for inside a task
    p3_authored     0.308   lexical substitution -- synonyms, canonical words avoided
    p1_tofu_para    0.250   TOFU's shipped paraphrase -- the most lexically distant

The ANSWER side is never touched: the truth ratio reuses TOFU's paraphrased_answer and
perturbed_answers, which describe the FACT and not the phrasing. Every probe is
therefore directly comparable to p0 with zero new answer generation.

THE TEST IS A DIFFERENCE-IN-DIFFERENCES, paired fact by fact. p0 is the phrasing the
model trained on; p1-p4 are held out. A raw gap on an unlearned model alone would
confound "unlearning was phrasing-specific" with "the model is simply better at
phrasings it trained on". The learned checkpoint supplies that baseline:

    DiD_i(p_k) = [TR(p_k) - TR(p0)]_unlearned,i  -  [TR(p_k) - TR(p0)]_learned,i

Truth ratio is unbounded above, so a DiD MEAN is not trustworthy -- one exploded fact
moves it. Significance is Wilcoxon signed-rank (paired, rank-based, outlier-immune) and
every fact is drawn, so the reader can see whether an effect is carried by the
distribution or by a flier.

CEILING CHECK, NOT A RESULT. A probe the LEARNED model cannot itself answer is a broken
probe, and the analysis drops it. Panel D reports that filter: TR > 1 means the model
ranks the perturbed answers above the true one, i.e. it cannot answer that phrasing.
This empirical filter is why no LLM equivalence judge is needed anywhere in Part A.

Colour: the two unlearning METHODS are the categorical entities (blue / orange, slots 1
and 2 of the validated palette -- all-pairs CVD dE 24.7, normal-vision 33.6). The
learned checkpoint is a neutral BASELINE, not a third category, so it wears secondary
ink. The probe axis is encoded by POSITION, never by hue.

    source .venv-plot/bin/activate
    python studies/crosslingual_recovery/plots/phase2_authored_probes.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results" / "relearn"
FIGS = STUDY / "figures"

LEARNED = "tofu_learn_full_full_qwen3-8b"
METHODS = [("tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b", "Full-FT"),
           ("tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b", "LoRA")]

# Validated categorical slots 1-2 (references/palette.md); learned is a baseline, not a
# series, so it takes secondary ink rather than a hue.
COLOR = {"Full-FT": "#2a78d6", "LoRA": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

# Probe order = decreasing lexical similarity to the canonical question, so the x-axis
# reads as "further from the trained phrasing" left to right.
PROBE_ORDER = ["p0_canonical", "p2_authored", "p4_authored", "p3_authored", "p1_tofu_para"]
PROBE_LABEL = {"p0_canonical": "p0\ncanonical\n(trained on)",
               "p2_authored": "p2\nsyntactic\nrecast",
               "p4_authored": "p4\noblique\nframing",
               "p3_authored": "p3\nlexical\nsubstitution",
               "p1_tofu_para": "p1\nTOFU\nparaphrase"}


def load(group):
    """{checkpoint_tag: {probe_id: {fact_idx: score}}} for the qa probes in `group`."""
    d = {}
    for tag in [LEARNED] + [m for m, _ in METHODS]:
        f = RESULTS / group / f"{tag}.json"
        if not f.exists():
            return None, f
        blob = json.loads(f.read_text())[tag]
        d[tag] = {pid: dict(zip(v["fact_indices"], v["scores_per_fact"]))
                  for pid, v in blob.items() if v.get("metric", "qa") == "qa"}
    return d, None


def paired(a, b):
    """Facts present in both, as two aligned arrays. Never mean-vs-mean."""
    ks = sorted(set(a) & set(b))
    return np.array([a[k] for k in ks]), np.array([b[k] for k in ks]), ks


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "phase2_authored"
    data, missing = load(group)
    if data is None:
        print(f"!! missing {missing}\n"
              f"   Run:  sbatch studies/crosslingual_recovery/slurm/phase2_probe_authored.sbatch\n"
              f"   then rsync results/ down. Falling back to the free group for a smoke test.")
        group = "phase2_probes"
        data, missing = load(group)
        if data is None:
            print(f"!! {missing} missing too -- nothing to plot.")
            return 1

    probes = [p for p in PROBE_ORDER if p in data[LEARNED]]
    held_out = [p for p in probes if p != "p0_canonical"]
    # EVERY panel is restricted to the facts scored on EVERY probe at EVERY checkpoint.
    # Mixing an n=40 probe with an n=3 probe on one line would compare medians over
    # different fact sets -- the same class of silent error as CLAUDE.md sec 7's
    # "max() over the wrong dict level".
    COMMON = sorted(set.intersection(*[set(data[t][p]) for t in data for p in probes]))
    print(f"group={group}  probes={probes}  common facts n={len(COMMON)}")
    if len(COMMON) < len(data[LEARNED]["p0_canonical"]):
        print(f"   NOTE: p0 alone covers {len(data[LEARNED]['p0_canonical'])} facts; "
              f"restricted to {len(COMMON)} by the least-covered probe.")

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    fig.subplots_adjust(hspace=0.52, wspace=0.28, top=0.86, bottom=0.11)
    axA, axB, axC, axD = axes.flat

    # ---------------------------------------------------------------- A: absolute TR
    x = np.arange(len(probes))
    series = [(LEARNED, "learned (ceiling)", MUTED, "o", "--")] + \
             [(t, n, COLOR[n], "s", "-") for t, n in METHODS]
    for tag, name, c, mk, ls in series:
        y = [np.median([data[tag][p][i] for i in COMMON]) for p in probes]
        axA.plot(x, y, ls, color=c, lw=2.0, marker=mk, ms=9, mec="white", mew=1.6,
                 label=name, zorder=3)
        axA.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points",
                     xytext=(9, 0), va="center", fontsize=9, color=c, fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels([PROBE_LABEL[p] for p in probes], fontsize=8)
    axA.set_ylabel("median truth ratio  (LOW = knows the fact)", fontsize=9, color=MUTED)
    axA.set_title(f"A · Does an untrained phrasing recover the fact?\n"
                  f"median over the {len(COMMON)} facts scored on every probe; "
                  f"higher = more forgotten", fontsize=10.5, loc="left")
    axA.legend(fontsize=9, frameon=False)

    # ---------------------------------------------------------------- B: DiD strip
    LIM = 2.4
    rows, stats_txt = [], []
    for j, p in enumerate(held_out):
        for k, (tag, name) in enumerate(METHODS):
            did = np.array([(data[tag][p][i] - data[tag]["p0_canonical"][i])
                            - (data[LEARNED][p][i] - data[LEARNED]["p0_canonical"][i])
                            for i in COMMON])
            pos = j + (k - 0.5) * 0.30
            jit = (np.random.RandomState(0).rand(len(did)) - 0.5) * 0.16
            inside = np.clip(did, -LIM, LIM)
            off = np.abs(did) > LIM
            axB.scatter(inside[~off], pos + jit[~off], s=22, color=COLOR[name],
                        alpha=0.55, lw=0, zorder=3)
            if off.any():
                axB.scatter(np.sign(did[off]) * LIM, pos + jit[off], s=34,
                            marker=">", color=COLOR[name], zorder=4)
            med = float(np.median(did))
            axB.plot([med, med], [pos - 0.13, pos + 0.13], color=INK, lw=2.4, zorder=5)
            try:
                pv = wilcoxon(did).pvalue
            except ValueError:
                pv = float("nan")
            rows.append((p, name, med, pv, len(did), int(off.sum())))
    axB.axvline(0, color=INK, lw=1.0, zorder=2)
    axB.set_yticks(range(len(held_out)))
    axB.set_yticklabels([PROBE_LABEL[p].replace("\n", " ") for p in held_out], fontsize=8)
    axB.set_xlim(-LIM * 1.06, LIM * 1.06)
    axB.set_xlabel("DiD  =  [TR(p) - TR(p0)]$_{unlearned}$ - [TR(p) - TR(p0)]$_{learned}$",
                   fontsize=9, color=MUTED)
    axB.set_title("B · Did suppression attach to the surface form?\n"
                  "one dot per fact, paired; bar = median; > = off-scale",
                  fontsize=10.5, loc="left")
    axB.invert_yaxis()

    # ---------------------------------------------------------------- C: phrasing spread
    def spread(tag):
        return np.array([max(data[tag][p][i] for p in probes)
                         - min(data[tag][p][i] for p in probes) for i in COMMON])
    xs = spread(LEARNED)
    for tag, name in METHODS:
        ys = spread(tag)
        axC.scatter(xs, ys, s=34, color=COLOR[name], alpha=0.65, lw=0.6,
                    edgecolor="white",
                    label=f"{name}  (median {np.median(ys - xs):+.2f}, n={len(COMMON)})")
    hi = max(axC.get_xlim()[1], axC.get_ylim()[1])
    axC.plot([0, hi], [0, hi], ls="--", color=INK, lw=1.1, zorder=1)
    axC.set_xlim(0, hi); axC.set_ylim(0, hi)
    axC.set_xlabel("phrasing spread, LEARNED  (max - min TR over the probes)",
                   fontsize=9, color=MUTED)
    axC.set_ylabel("phrasing spread, UNLEARNED", fontsize=9, color=MUTED)
    axC.set_title("C · Did unlearning make the fact more phrasing-sensitive?\n"
                  "above the line = the model's answer now depends more on wording",
                  fontsize=10.5, loc="left")
    axC.legend(fontsize=8.5, framealpha=1.0, edgecolor="none", facecolor="white",
               loc="upper left")

    # ---------------------------------------------------------------- D: ceiling filter
    broken = [sum(1 for i in COMMON if data[LEARNED][p][i] > 1.0) for p in probes]
    bars = axD.bar(x, broken, width=0.62, color=MUTED, zorder=3)
    for b, n in zip(bars, broken):
        axD.annotate(f"{n}", (b.get_x() + b.get_width() / 2, n), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=9, color=INK)
    axD.set_xticks(x); axD.set_xticklabels([PROBE_LABEL[p] for p in probes], fontsize=8)
    axD.set_ylabel("facts the LEARNED model cannot answer", fontsize=9, color=MUTED)
    axD.set_ylim(0, max(max(broken), 1) * 1.35)
    axD.set_title(f"D · Ceiling check: broken probes to drop\n"
                  f"learned-model TR > 1 (ranks a false answer first), out of "
                  f"{len(COMMON)} facts", fontsize=10.5, loc="left")

    for ax in axes.flat:
        ax.grid(True, axis="both", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8.5)

    n_facts = len(COMMON)
    fig.suptitle(
        f"Phase 2 Part A — {len(probes)} English phrasings x {n_facts} forget facts, "
        f"scored at three checkpoints\n"
        f"the answer side is TOFU's throughout, so every probe is comparable to p0   "
        f"[PROVISIONAL: single seed]",
        fontsize=12.5, y=0.965)

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "phase2_authored_probes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    print(f"\n{'probe':16s}{'method':9s}{'median DiD':>12}{'wilcoxon p':>12}{'n':>5}{'off-scale':>11}")
    for p, name, med, pv, n, off in rows:
        print(f"{p:16s}{name:9s}{med:>12.3f}{pv:>12.4f}{n:>5}{off:>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
