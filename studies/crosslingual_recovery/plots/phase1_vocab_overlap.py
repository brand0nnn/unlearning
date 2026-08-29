"""PHASE 1 — does cross-lingual recovery track subword-vocabulary overlap with
English (the CLC-paper mechanism), or typological distance? (next_steps_plan.md §1)

If recovery is really a shared-interlingua effect it should be ~FLAT vs both axes.
If it instead tracks vocab overlap, that would echo the Cross-Lingual Consistency
(CLC) finding for baseline factual consistency and undercut the interlingua story.

Inputs (all LOCAL, no GPU, no SLURM — runs on the login node):
  - parallel corpus: data/raw/multilingual_unlearning/dataset/full_merged_all_10_lang
    (10 langs incl. English, 4000 rows each — same TOFU content translated).
  - Qwen3 tokenizer (tokenizer-only load; no torch needed).
  - recovery numbers: studies/crosslingual_recovery/results/relearn/crosslingual_deep/
    (fraction of removed knowledge recovered at ep2, Full-FT + LoRA).

x-axes per language (distance-from-English):
  - vocab_jaccard   = |V_l ∩ V_en| / |V_l ∪ V_en|   (higher = MORE overlap w/ English)
  - vocab_overlap   = |V_l ∩ V_en| / min(|V_l|,|V_en|)
  - typ_distance    = coarse script+family proxy to English (stand-in for lang2vec;
                      swap in URIEL/lang2vec for the paper)
y = recovery fraction (ep2), per method.

Reports Pearson r + Spearman rho + a PERMUTATION p-value (10k shuffles, no scipy),
a table, and scatter plots -> figures/phase1_vocab_overlap.png.

CAVEAT (from the plan): recovery is single-seed, n=10 languages -> low power. Treat
r/p as PROVISIONAL until the item-8 bootstrap CIs land; this is the first pass.

    python studies/crosslingual_recovery/plots/phase1_vocab_overlap.py
"""
import json, sys, random, math
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

import pyarrow.ipc as ipc, pyarrow as pa
from transformers import AutoTokenizer

STUDY = Path(__file__).resolve().parents[1]
DEEP = STUDY / "results" / "relearn" / "crosslingual_deep"
FIGS = STUDY / "figures"
MERGED = (_r / "data/raw/multilingual_unlearning/dataset/full_merged_all_10_lang"
          / "train" / "data-00000-of-00001.arrow")

LANGS = ["fr", "id", "ru", "hi", "fa", "ar", "iw", "ko", "ja"]  # vs en; en is the ref
LANG_NAME = {"fr": "French", "id": "Indonesian", "ru": "Russian", "hi": "Hindi",
             "fa": "Farsi", "ar": "Arabic", "iw": "Hebrew", "ko": "Korean", "ja": "Japanese"}
LEARNED_TR = 0.459
EP = 2
# Coarse script+family distance to English (Latin script, Germanic). Documented,
# transparent stand-in for lang2vec/URIEL syntactic distance — replace for the paper.
TYP_DIST = {"fr": 1, "id": 2, "ru": 2, "hi": 3, "fa": 3, "ar": 4, "iw": 4, "ko": 5, "ja": 5}
FILES = {"Full-FT": "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b",
         "LoRA": "tofu_unlearn_gradient_difference_forget01_lora_uep32_qwen3-8b"}


def read_arrow(p):
    with pa.memory_map(str(p), "r") as s:
        try: return ipc.open_stream(s).read_all().to_pylist()
        except Exception:
            s.seek(0); return ipc.open_file(s).read_all().to_pylist()


def vocab_sets():
    rows = read_arrow(MERGED)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    text = {}
    for r in rows:
        text.setdefault(r["language"], []).append(r["question"] + " " + r["answer"])
    V = {}
    for lang, lines in text.items():
        ids = set()
        # batch-encode for speed; union of all subword ids used in that language
        for i in range(0, len(lines), 256):
            for enc in tok(lines[i:i + 256]).input_ids:
                ids.update(enc)
        V[lang] = ids
    return V


def recovery(method_base):
    d = json.load(open(DEEP / f"{method_base}.json"))
    b = d[method_base]["truth_ratio"]
    out = {}
    for l in LANGS:
        k = f"relearn_{method_base}_via_retain_lang{l}_ep{EP}"
        out[l] = (b - d[k]["truth_ratio"]) / (b - LEARNED_TR)
    return out


def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy) if sx * sy else float("nan")


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); rk = [0] * len(v)
        for pos, i in enumerate(order): rk[i] = pos
        return rk
    return pearson(rank(x), rank(y))


def perm_p(x, y, stat=pearson, B=10000):
    obs = abs(stat(x, y)); ys = list(y); c = 0
    random.seed(42)
    for _ in range(B):
        random.shuffle(ys)
        if abs(stat(x, ys)) >= obs - 1e-12: c += 1
    return c / B


def main():
    if not MERGED.exists():
        print("missing parallel corpus:", MERGED); return
    V = vocab_sets()
    en = V["en"]
    jac, ovl = {}, {}
    for l in LANGS:
        inter = len(V[l] & en)
        jac[l] = inter / len(V[l] | en)
        ovl[l] = inter / min(len(V[l]), len(en))
    rec = {m: recovery(b) for m, b in FILES.items()}

    # ---- table ----
    print(f"\n{'lang':>11} {'jaccard':>8} {'overlap':>8} {'typ_d':>6}  "
          f"{'rec_FT':>7} {'rec_LoRA':>8}")
    for l in LANGS:
        print(f"{LANG_NAME[l]:>11} {jac[l]:>8.3f} {ovl[l]:>8.3f} {TYP_DIST[l]:>6} "
              f"{rec['Full-FT'][l]:>+7.1%} {rec['LoRA'][l]:>+8.1%}")
    print(f"\n(en reference vocab size = {len(en)} subword types)")

    # ---- correlations ----
    axes = {"vocab_jaccard": [jac[l] for l in LANGS],
            "vocab_overlap": [ovl[l] for l in LANGS],
            "typ_distance":  [TYP_DIST[l] for l in LANGS]}
    print("\nCorrelation of each axis vs recovery (n=9 langs, perm p, 10k shuffles):")
    for m in FILES:
        y = [rec[m][l] for l in LANGS]
        print(f"\n  [{m}]")
        for name, x in axes.items():
            rp = pearson(x, y); rs = spearman(x, y)
            print(f"    {name:>14}:  Pearson r={rp:+.3f} (p={perm_p(x,y,pearson):.3f})   "
                  f"Spearman rho={rs:+.3f} (p={perm_p(x,y,spearman):.3f})")

    # ---- scatter ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, m in zip(axs, FILES):
        x = [jac[l] for l in LANGS]; y = [rec[m][l] for l in LANGS]
        ax.scatter(x, y, s=70, color="#1f77b4" if m == "Full-FT" else "#ff7f0e", zorder=3)
        for l in LANGS:
            ax.annotate(l, (jac[l], rec[m][l]), textcoords="offset points", xytext=(5, 4), fontsize=8)
        # least-squares fit line
        n = len(x); mx = sum(x)/n; my = sum(y)/n
        b1 = sum((a-mx)*(c-my) for a, c in zip(x, y)) / sum((a-mx)**2 for a in x)
        b0 = my - b1*mx
        xs = [min(x), max(x)]; ax.plot(xs, [b0+b1*v for v in xs], "--", color="grey", lw=1.2)
        ax.set_title(f"{m}: recovery vs vocab overlap w/ English\n"
                     f"Pearson r={pearson(x,y):+.2f} (perm p={perm_p(x,y):.2f})", fontsize=11)
        ax.set_xlabel("subword-vocabulary Jaccard overlap with English", fontsize=10)
        ax.set_ylabel("fraction recovered (ep2)", fontsize=10)
        ax.grid(True, alpha=0.25, ls="--"); ax.set_axisbelow(True)
    fig.suptitle("Phase 1: does cross-lingual recovery track vocab overlap? "
                 "(flat line => interlingua, not vocab-sharing)  [PROVISIONAL: single seed, n=9]",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "phase1_vocab_overlap.png"
    fig.savefig(out, dpi=120)
    print(f"\nscatter -> {out}")


if __name__ == "__main__":
    main()
