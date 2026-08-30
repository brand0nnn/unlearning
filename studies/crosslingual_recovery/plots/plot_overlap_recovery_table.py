"""Table figure: subword-vocab OVERLAP-with-English vs RECOVERY, per language.

Rows sorted by Jaccard overlap (English at top = 1.0 self-overlap). The point the
table makes visually: overlap falls sharply down the rows (0.31 -> 0.05) while the
recovery columns stay ~flat -> recovery does NOT track vocab overlap (Phase 1).

Overlap numbers are the deterministic output of phase1_vocab_overlap.py (Qwen3
tokenizer over the full_merged_all_10_lang parallel corpus); recovery is computed
live from results/relearn/crosslingual_deep/ (fraction recovered at ep2).

    python studies/crosslingual_recovery/plots/plot_overlap_recovery_table.py
    -> figures/overlap_recovery_table.png
"""
import json, sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
DEEP = STUDY / "results" / "relearn" / "crosslingual_deep"
FIGS = STUDY / "figures"
LEARNED_TR = 0.459
EP = 2

LANG_NAME = {"en": "English", "fr": "French", "id": "Indonesian", "ru": "Russian",
             "hi": "Hindi", "fa": "Farsi", "ar": "Arabic", "iw": "Hebrew",
             "ko": "Korean", "ja": "Japanese"}
# From phase1_vocab_overlap.py (Qwen3 subword-vocabulary overlap with English).
# en is self-overlap = 1.0.
JACCARD = {"en": 1.000, "fr": 0.312, "id": 0.293, "ko": 0.217, "ar": 0.157,
           "ja": 0.147, "fa": 0.138, "ru": 0.075, "iw": 0.063, "hi": 0.048}
OVERLAP = {"en": 1.000, "fr": 0.529, "id": 0.603, "ko": 0.543, "ar": 0.405,
           "ja": 0.338, "fa": 0.560, "ru": 0.228, "iw": 0.219, "hi": 0.656}
FILES = {"Full-FT": "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b",
         "LoRA": "tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b"}


def recovery(base):
    d = json.load(open(DEEP / f"{base}.json"))
    b = d[base]["truth_ratio"]
    out = {}
    for l in LANG_NAME:
        k = f"relearn_{base}_via_retain" + ("" if l == "en" else f"_lang{l}") + f"_ep{EP}"
        out[l] = (b - d[k]["truth_ratio"]) / (b - LEARNED_TR)
    return out


def main():
    rec = {m: recovery(b) for m, b in FILES.items()}
    langs = sorted(LANG_NAME, key=lambda l: JACCARD[l], reverse=True)   # high overlap -> low

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["Language", "Vocab overlap\n(Jaccard) w/ English", "Overlap\ncoefficient",
            "Recovery\nFull-FT (ep2)", "Recovery\nLoRA (ep2)"]
    cell, colors = [], []

    def blue(v):   # 0..1 -> white..blue
        return (1 - 0.55 * v, 1 - 0.35 * v, 1.0)

    def green(v, lo, hi):  # normalize within column then white..green
        t = 0 if hi == lo else (v - lo) / (hi - lo)
        return (1 - 0.45 * t, 1.0, 1 - 0.45 * t)

    ftv = [rec["Full-FT"][l] for l in langs if l != "en"]
    lov = [rec["LoRA"][l] for l in langs if l != "en"]
    ft_lo, ft_hi = min(ftv), max(ftv)
    lo_lo, lo_hi = min(lov), max(lov)

    for l in langs:
        cell.append([LANG_NAME[l], f"{JACCARD[l]:.3f}", f"{OVERLAP[l]:.3f}",
                     f"{rec['Full-FT'][l]:+.0%}", f"{rec['LoRA'][l]:+.0%}"])
        rowc = ["white", blue(JACCARD[l]), blue(OVERLAP[l]),
                green(rec["Full-FT"][l], ft_lo, ft_hi) if l != "en" else (.93, .93, .93),
                green(rec["LoRA"][l], lo_lo, lo_hi) if l != "en" else (.93, .93, .93)]
        colors.append(rowc)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axis("off")
    tbl = ax.table(cellText=cell, colLabels=cols, cellColours=colors,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.0)
    for (r, c), cellobj in tbl.get_celld().items():
        if r == 0:
            cellobj.set_text_props(weight="bold"); cellobj.set_facecolor("#333333")
            cellobj.set_text_props(color="white")
        if c == 0 and r > 0:
            cellobj.set_text_props(weight="bold")
    # Computed, never hardcoded -- see the FLORES twin. English is excluded from the
    # ranges because relearning in the SAME language is the trivial case.
    nz = [l for l in langs if l != "en"]
    ov_hi, ov_lo = JACCARD[nz[0]], JACCARD[nz[-1]]
    ax.set_title("Vocabulary overlap with English vs fact recovery (rows sorted by overlap)\n"
                 f"overlap falls {ov_hi:.2f} → {ov_lo:.2f} down the rows, but recovery stays flat "
                 f"(~{min(ftv):.0%}–{max(ftv):.0%} FT, ~{min(lov):.0%}–{max(lov):.0%} LoRA)"
                 "\n→ recovery does NOT track vocab overlap  "
                 "[provisional: single seed, ep2; English row excluded from the ranges]",
                 fontsize=11, pad=14)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "overlap_recovery_table.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("table ->", out)


if __name__ == "__main__":
    main()
