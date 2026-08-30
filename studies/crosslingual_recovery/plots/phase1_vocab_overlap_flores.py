"""PHASE 1 (FLORES-200 version) — CLC-faithful subword-vocabulary overlap vs recovery.

Replicates Qi et al. 2023 ("Cross-lingual Consistency of Factual Knowledge in
Multilingual LMs"), §6.2 / Eq. 7: overlap = |V(l) ∩ V(en)| / |V(l) ∪ V(en)| (JACCARD),
with the vocabularies extracted from a STRICTLY PARALLEL corpus segmented by the
model's own tokenizer. CLC used FLORES-200 (+ BMLAMA); we use FLORES-200 here (the
earlier phase1_vocab_overlap.py used the TOFU parallel corpus as a fallback).

CLC found a STRONG POSITIVE correlation between this overlap and cross-lingual
CONSISTENCY (shallow subword-sharing). We test the same overlap against cross-lingual
RECOVERY: a FLAT line => recovery is NOT explained by subword-sharing (interlingua),
in tension with CLC's shallow-sharing account.

Adaptation vs CLC: CLC correlates overlap pairwise over ALL language pairs; we anchor
to ENGLISH (the unlearning source language) and correlate each language's
English-overlap with its recovery.

Inputs (all LOCAL, CPU-only — runs on the login node; tokenizer needs no torch):
  - FLORES-200 dev+devtest text: data/raw/flores/flores200_dataset/{dev,devtest}/<code>.<split>
  - Qwen3 tokenizer.
  - recovery: results/relearn/crosslingual_deep/ (fraction recovered at ep2).

    python studies/crosslingual_recovery/plots/phase1_vocab_overlap_flores.py

CAVEAT: recovery is single-seed, n=9 langs -> low power. Provisional until the
item-8 bootstrap CIs land.
"""
import json, sys, random, math
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

from transformers import AutoTokenizer

STUDY = Path(__file__).resolve().parents[1]
DEEP = STUDY / "results" / "relearn" / "crosslingual_deep"
FIGS = STUDY / "figures"
FLORES = _r / "data/raw/flores/flores200_dataset"

LEARNED_TR = 0.459
EP = 2
# our lang code -> FLORES-200 script-tagged code
FLORES_CODE = {"en": "eng_Latn", "fr": "fra_Latn", "id": "ind_Latn", "ru": "rus_Cyrl",
               "hi": "hin_Deva", "fa": "pes_Arab", "ar": "arb_Arab", "iw": "heb_Hebr",
               "ko": "kor_Hang", "ja": "jpn_Jpan"}
LANG_NAME = {"fr": "French", "id": "Indonesian", "ru": "Russian", "hi": "Hindi",
             "fa": "Farsi", "ar": "Arabic", "iw": "Hebrew", "ko": "Korean", "ja": "Japanese"}
LANGS = list(LANG_NAME)   # vs en
# From the TOFU-corpus run (phase1_vocab_overlap.py) — for side-by-side comparison.
JACCARD_TOFU = {"fr": 0.312, "id": 0.293, "ru": 0.075, "hi": 0.048, "fa": 0.138,
                "ar": 0.157, "iw": 0.063, "ko": 0.217, "ja": 0.147}
FILES = {"Full-FT": "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b",
         "LoRA": "tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b"}


def flores_lines(lang):
    code = FLORES_CODE[lang]
    lines = []
    for split in ("dev", "devtest"):
        f = FLORES / split / f"{code}.{split}"
        if f.exists():
            lines += [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines


def vocab_sets(tok):
    V = {}
    for lang in ["en"] + LANGS:
        lines = flores_lines(lang)
        ids = set()
        for i in range(0, len(lines), 256):
            for enc in tok(lines[i:i + 256]).input_ids:
                ids.update(enc)
        V[lang] = ids
    return V


def recovery(base):
    d = json.load(open(DEEP / f"{base}.json"))
    b = d[base]["truth_ratio"]
    return {l: (b - d[f"relearn_{base}_via_retain_lang{l}_ep{EP}"]["truth_ratio"]) / (b - LEARNED_TR)
            for l in LANGS}


def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy) if sx * sy else float("nan")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); rk = [0] * len(v)
        for p, i in enumerate(order): rk[i] = p
        return rk
    return pearson(rank(x), rank(y))


def perm_p(x, y, stat=pearson, B=10000):
    obs = abs(stat(x, y)); ys = list(y); c = 0; random.seed(42)
    for _ in range(B):
        random.shuffle(ys)
        if abs(stat(x, ys)) >= obs - 1e-12: c += 1
    return c / B


def main():
    if not FLORES.exists():
        print("missing FLORES at", FLORES); return
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    V = vocab_sets(tok)
    en = V["en"]
    jac, ovl = {}, {}
    for l in LANGS:
        inter = len(V[l] & en)
        jac[l] = inter / len(V[l] | en)               # CLC Eq. 7 (Jaccard)
        ovl[l] = inter / min(len(V[l]), len(en))        # robustness (overlap coef)
    rec = {m: recovery(b) for m, b in FILES.items()}

    print(f"\nFLORES-200 corpus: {len(flores_lines('en'))} en sentences (dev+devtest)")
    print(f"en reference vocab = {len(en)} subword types\n")
    print(f"{'lang':>11} {'Jac_FLORES':>10} {'Jac_TOFU':>9} {'ovlpCoef':>9} "
          f"{'rec_FT':>7} {'rec_LoRA':>8}")
    for l in LANGS:
        print(f"{LANG_NAME[l]:>11} {jac[l]:>10.3f} {JACCARD_TOFU[l]:>9.3f} {ovl[l]:>9.3f} "
              f"{rec['Full-FT'][l]:>+7.1%} {rec['LoRA'][l]:>+8.1%}")

    axes = {"Jaccard(FLORES)": [jac[l] for l in LANGS],
            "overlap_coef":    [ovl[l] for l in LANGS]}
    print("\nCorrelation vs recovery (n=9, perm p, 10k):")
    for m in FILES:
        y = [rec[m][l] for l in LANGS]
        print(f"  [{m}]")
        for name, x in axes.items():
            print(f"    {name:>16}: Pearson r={pearson(x,y):+.3f} (p={perm_p(x,y,pearson):.3f})  "
                  f"Spearman rho={spearman(x,y):+.3f} (p={perm_p(x,y,spearman):.3f})")

    # correlation between the two corpora's Jaccard (does FLORES agree with TOFU?)
    jx = [jac[l] for l in LANGS]; jt = [JACCARD_TOFU[l] for l in LANGS]
    print(f"\n  Jaccard(FLORES) vs Jaccard(TOFU): Pearson r={pearson(jx,jt):+.3f} "
          f"Spearman rho={spearman(jx,jt):+.3f}")

    # ---- scatter (Jaccard = CLC metric) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    rs = {}
    for ax, m in zip(axs, FILES):
        x = [jac[l] for l in LANGS]; y = [rec[m][l] for l in LANGS]
        ax.scatter(x, y, s=70, color="#1f77b4" if m == "Full-FT" else "#ff7f0e", zorder=3)
        for l in LANGS:
            ax.annotate(l, (jac[l], rec[m][l]), textcoords="offset points", xytext=(5, 4), fontsize=8)
        n = len(x); mx = sum(x)/n; my = sum(y)/n
        b1 = sum((a-mx)*(c-my) for a, c in zip(x, y)) / sum((a-mx)**2 for a in x); b0 = my - b1*mx
        xs = [min(x), max(x)]; ax.plot(xs, [b0+b1*v for v in xs], "--", color="grey", lw=1.2)
        rs[m] = (pearson(x, y), perm_p(x, y))
        ax.set_title(f"{m}: recovery vs FLORES Jaccard overlap w/ English\n"
                     f"Pearson r={rs[m][0]:+.2f} (perm p={rs[m][1]:.2f})", fontsize=11)
        ax.set_xlabel("subword-vocab Jaccard overlap w/ English (FLORES-200, CLC Eq.7)", fontsize=10)
        ax.set_ylabel("fraction recovered (ep2)", fontsize=10)
        ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
    # The verdict has to follow the PANELS, not a fixed claim: Full-FT is flat but LoRA
    # is not (r=+0.70, p=0.03 on the current data), so a blanket "flat => interlingua"
    # headline contradicts the right-hand panel it sits above.
    verdict = " | ".join(f"{m}: r={r:+.2f}" + ("" if pv >= 0.05 else " (p<.05)")
                         for m, (r, pv) in rs.items())
    flat = all(pv >= 0.05 for _, pv in rs.values())
    lead = ("flat for both => recovery not explained by vocab-sharing"
            if flat else "NOT flat for every method — see panels")
    fig.suptitle("Phase 1 (FLORES-200, CLC-faithful): recovery vs subword overlap — "
                 f"{lead}\n{verdict}  [PROVISIONAL: single seed, n=9 langs; "
                 "per-language CIs are far wider than the between-language spread]",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "phase1_vocab_overlap_flores.png"
    fig.savefig(out, dpi=120)
    print("\nscatter ->", out)


if __name__ == "__main__":
    main()
