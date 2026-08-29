"""Table figure (FLORES-200 / CLC-faithful): subword-vocab OVERLAP-with-English vs
RECOVERY, per language. Same as plot_overlap_recovery_table.py but the overlap is the
CLC Jaccard (Eq. 7) computed on FLORES-200 rather than the TOFU corpus.

Rows sorted by Jaccard overlap (English at top = 1.0). The point: overlap falls
sharply down the rows while the recovery columns stay ~flat -> recovery does NOT track
vocab overlap (Phase 1). Overlap numbers are the deterministic output of
phase1_vocab_overlap_flores.py; recovery is computed live from the crosslingual_deep JSON.

    python studies/crosslingual_recovery/plots/plot_overlap_recovery_table_flores.py
    -> figures/overlap_recovery_table_flores.png
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
# From phase1_vocab_overlap_flores.py (Qwen3 subword overlap w/ English on FLORES-200).
# en = self-overlap = 1.0.
JACCARD = {"en": 1.000, "fr": 0.228, "id": 0.220, "ko": 0.062, "fa": 0.055,
           "ru": 0.034, "iw": 0.028, "hi": 0.018, "ar": 0.016, "ja": 0.016}
OVERLAP = {"en": 1.000, "fr": 0.447, "id": 0.539, "ko": 0.263, "fa": 0.374,
           "ru": 0.122, "iw": 0.119, "hi": 0.578, "ar": 0.069, "ja": 0.052}
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
    langs = sorted(LANG_NAME, key=lambda l: JACCARD[l], reverse=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["Language", "Vocab overlap: Jaccard\n(FLORES-200, CLC Eq.7)",
            "Overlap\ncoefficient", "Recovery\nFull-FT (ep2)", "Recovery\nLoRA (ep2)"]
    cell, colors = [], []

    def blue(v):
        return (1 - 0.55 * v, 1 - 0.35 * v, 1.0)

    def green(v, lo, hi):
        t = 0 if hi == lo else (v - lo) / (hi - lo)
        return (1 - 0.45 * t, 1.0, 1 - 0.45 * t)

    ftv = [rec["Full-FT"][l] for l in langs if l != "en"]
    lov = [rec["LoRA"][l] for l in langs if l != "en"]
    ft_lo, ft_hi = min(ftv), max(ftv); lo_lo, lo_hi = min(lov), max(lov)

    for l in langs:
        cell.append([LANG_NAME[l], f"{JACCARD[l]:.3f}", f"{OVERLAP[l]:.3f}",
                     f"{rec['Full-FT'][l]:+.0%}", f"{rec['LoRA'][l]:+.0%}"])
        colors.append(["white", blue(JACCARD[l]), blue(OVERLAP[l]),
                       green(rec["Full-FT"][l], ft_lo, ft_hi) if l != "en" else (.93, .93, .93),
                       green(rec["LoRA"][l], lo_lo, lo_hi) if l != "en" else (.93, .93, .93)])

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.axis("off")
    tbl = ax.table(cellText=cell, colLabels=cols, cellColours=colors,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.0)
    for (r, c), cellobj in tbl.get_celld().items():
        if r == 0:
            cellobj.set_text_props(weight="bold", color="white"); cellobj.set_facecolor("#333333")
        if c == 0 and r > 0:
            cellobj.set_text_props(weight="bold")
    ax.set_title("Vocabulary overlap with English (FLORES-200, CLC Jaccard) vs fact recovery\n"
                 "overlap falls 0.23 → 0.02 down the rows, but recovery stays flat "
                 "(~44–53% FT, ~54–67% LoRA)\n→ recovery does NOT track vocab overlap  "
                 "[provisional: single seed, ep2]", fontsize=11, pad=14)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "overlap_recovery_table_flores.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("table ->", out)


if __name__ == "__main__":
    main()
