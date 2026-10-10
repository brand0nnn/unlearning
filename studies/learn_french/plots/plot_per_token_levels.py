"""Stage 6 follow-up, local: in what ORDER do tokens fall as unlearning deepens?

    python studies/learn_french/plots/plot_per_token_levels.py            # every arm found
    python studies/learn_french/plots/plot_per_token_levels.py --tables   # no figure

Reads results/per_token_levels_<L>.json (slurm/14_per_token_levels.sbatch: the per-token
map at each saved depth level of arm L) and takes fr_ft / fr_retain from
results/per_token_map.json -- same script, same texts, so token keys line up.

THE QUESTION. Plan 1 found, at the matched level, that the tokens destroyed are the ones
fr_ft was least sure of. One level cannot tell IMMUNE from LAST IN LINE. If confidence sets
the order, then at every level the hit rate falls with confidence, the least-confident
tokens are hit first, and near-certain tokens (copied names, Basil) start to fall only at
the deepest levels -- or never.

Tokens are scored on the French text (what each arm did to the knowledge as learned) and
on the arm's own text (where it spent its loss). "Hit" = drop > 5 nats, as in Plan 1.
Descriptive only.
"""
import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cross_route import INK, MUTED, RESULTS, STUDY, SURFACE, finish, style  # noqa: E402
from plot_per_token_map import tag_tokens  # noqa: E402

HIT = 5.0
# Ordinal confidence buckets (1 - p at fr_ft), drawn as one blue ramp light -> dark
# (reference palette steps 250/400/550/700): darker = the learned model was surer.
BUCKETS = [("1-p >= 0.01", 1e-2, 1.01, "#86b6ef"), ("0.001-0.01", 1e-3, 1e-2, "#3987e5"),
           ("0.0001-0.001", 1e-4, 1e-3, "#1c5cab"), ("< 0.0001", -1, 1e-4, "#0d366b")]


def level_of(ckpt):
    m = re.search(r"tr(\d)p(\d+)", ckpt)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def load_all(langs):
    base = json.load(open(RESULTS / "per_token_map.json", encoding="utf-8"))
    texts = base["texts"]
    lab = {c: ("fr_ft" if "learn_full" in c else "fr_retain" if "retain99" in c else None)
           for c in base["checkpoints"]}
    ref = {lab[c]: v for c, v in base["results"].items() if lab.get(c)}
    arms = {}
    for lg in langs:
        f = RESULTS / f"per_token_levels_{lg}.json"
        if not f.exists():
            print(f"  (no {f.name} -- skipped)")
            continue
        d = json.load(open(f, encoding="utf-8"))
        assert set(d["texts"]) == set(texts), f"{f.name}: texts differ from per_token_map"
        arms[lg] = {level_of(c): v for c, v in d["results"].items()}
    return texts, ref, arms


def tokens(texts, ref):
    slots = {}
    for s in json.load(open(STUDY / "probes" / "many_slot_probes.json",
                            encoding="utf-8"))["probes"]:
        slots.setdefault(s["fact"], []).append(s)
    out = []
    for key, t in texts.items():
        if t["split"] != "forget":
            continue
        for i, tg in enumerate(tag_tokens(t, slots)):
            lp = ref["fr_ft"][key]["lp"][i]
            out.append({"key": key, "lang": t["lang"], "row": t["row"], "i": i,
                        "tok": t["tokens"][i], "tag": tg, "lp_ft": lp,
                        "q": -math.expm1(lp)})      # 1 - p at fr_ft
    return out


def bucket(q):
    return next(b[0] for b in BUCKETS if b[1] <= q < b[2])


def drop(arms, lg, lev, r):
    return r["lp_ft"] - arms[lg][lev][r["key"]]["lp"][r["i"]]


def tables(arms, toks):
    for lg, levels in arms.items():
        levs = sorted(levels)
        for text in ("fr", lg) if lg != "fr" else ("fr",):
            T = [r for r in toks if r["lang"] == text]
            print("\n" + "=" * 100)
            print(f"unl_{lg}: {text.upper()} text -- share of tokens hit (> {HIT:g} nats) by "
                  f"confidence at fr_ft, per depth level")
            print("=" * 100)
            print(f"  {'1 - p at fr_ft':<16}{'n':>6}" + "".join(f"{l:>9.3f}" for l in levs))
            for b in BUCKETS:
                g = [r for r in T if bucket(r["q"]) == b[0]]
                if g:
                    print(f"  {b[0]:<16}{len(g):>6}" + "".join(
                        f"{sum(drop(arms, lg, l, r) > HIT for r in g) / len(g):>9.0%}"
                        for l in levs))
            print(f"  {'name (copied)':<16}" + f"{'':>6}" + "".join(
                f"{st.median(drop(arms, lg, l, r) for r in T if r['tag'] == 'name (copied)'):>8.2f}n"
                for l in levs) + "   <- median drop, nats")
            first = {}
            for r in T:
                first.setdefault(r["row"], []).append(r)
            for row in (0, 20):
                nm = [r for r in first.get(row, []) if r["tag"] == "name (recalled)"]
                if nm:
                    r = nm[0]
                    print(f"  fact {row} {r['tok']!r:<10} p_ft {1 - r['q']:.4f}" + "".join(
                        f"{drop(arms, lg, l, r):>9.2f}" for l in levs) + "   <- drop, nats")
            # order of fall: median confidence of tokens first hit at each level
            fl = {}
            for r in T:
                k = next((l for l in levs if drop(arms, lg, l, r) > HIT), None)
                fl.setdefault(k, []).append(r["q"])
            print("  first hit at   " + "   ".join(
                f"{('never' if k is None else f'{k:.3f}')}: n={len(v)}, median 1-p "
                f"{st.median(v):.1e}" for k, v in sorted(fl.items(), key=lambda x: (
                    x[0] is None, x[0] or 0))))


def figure(arms, toks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    langs = list(arms)
    fig, axes = plt.subplots(1, len(langs), figsize=(3.2 * len(langs) + 1.2, 4.4),
                             sharey=True, squeeze=False)
    for ax, lg in zip(axes[0], langs):
        style(ax)
        levs = sorted(arms[lg])
        T = [r for r in toks if r["lang"] == "fr"]
        for b in BUCKETS:
            g = [r for r in T if bucket(r["q"]) == b[0]]
            ys = [100 * sum(drop(arms, lg, l, r) > HIT for r in g) / len(g) for l in levs]
            ax.plot(levs, ys, color=b[3], lw=2, marker="o", ms=5, label=f"{b[0]} (n={len(g)})")
        ax.set_title(f"unl_{lg}", fontsize=10.5, color=INK, loc="left")
        ax.set_xticks(levs)
        ax.set_xticklabels([f"{l:.2f}" for l in levs], fontsize=8, rotation=45)
        ax.set_ylim(0, 100)
        ax.set_xlabel("depth level (French TR)", color=MUTED, fontsize=8.5)
    axes[0][0].set_ylabel("% of French answer tokens hit (> 5 nats)", color=MUTED, fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, title="1 - p at fr_ft (darker = surer)", title_fontsize=8.5,
               frameon=False, fontsize=8.5, ncol=4, loc="upper left",
               bbox_to_anchor=(0.06, 0.92))
    fig.suptitle("As unlearning deepens, which tokens fall? (French text, every arm)",
                 fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.72, bottom=0.27, wspace=0.08)
    finish(fig, "per_token_levels.png",
           "Each arm's saved checkpoints at the pre-registered truth-ratio levels; levels "
           "crossed at one probe point share weights and appear once. Single seed, 2 authors.",
           bottom=0.27)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["fr", "en", "id", "ja", "ru"])
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()
    texts, ref, arms = load_all(a.langs)
    if not arms:
        sys.exit("no per_token_levels_<L>.json found -- run slurm/14_per_token_levels.sbatch")
    toks = tokens(texts, ref)
    tables(arms, toks)
    if not a.tables:
        figure(arms, toks)


if __name__ == "__main__":
    main()
