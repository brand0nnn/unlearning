"""Plan 1, local: tag every scored token and account for where each arm's unlearning went.

    python studies/learn_french/plots/plot_per_token_map.py              # tables + figures
    python studies/learn_french/plots/plot_per_token_map.py --tables     # tables only

Reads results/per_token_map.json (slurm/12_per_token_map.sbatch) and tags tokens here, not
on the GPU, from probes/many_slot_probes.json -- so a corrected annotation needs no re-run.

TAGS (a token takes the first tag whose character span it overlaps):
  name (copied)       the author's name, in a row whose QUESTION also names the author
  name (recalled)     the author's name where the question does not (facts 0 and 20)
  attribute           a slot from many_slot_probes.json, absent from its question
  attribute (copyable) a slot the question already gives (cue_in_question or an exact
                      word of it) -- the H1 control: does a copied ATTRIBUTE resist the way
                      a copied NAME does?
  template            everything else, EOS included
Attributes are annotated in French only. On en/id/ja/ru text only the name is tagged and
the rest is `other`. On retain rows a capitalised multi-word span that the answer repeats
from its question is `name (copied)`.

THE MEASURE. drop = log p_fr_ft(token) - log p_arm(token), in nats, per token, teacher-
forced on the trained sequence. Positive = the arm made the token less likely. Two views,
because they answer different questions:
  share  -- of the arm's total positive drop on a text, how much fell on each tag
            ("where the unlearning went"; templates win by sheer token count)
  per token -- mean drop of a token of that tag ("how hard each kind was hit")
Descriptive only: sums, means, medians. No test statistic.
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cross_route import FIGS, INK, MUTED, RESULTS, STUDY, SURFACE, finish, style  # noqa: E402

TAGS = ["name (copied)", "name (recalled)", "attribute", "attribute (copyable)", "template"]
# Categorical slots 1-4 of the reference palette, validated (CVD worst adjacent dE 9.1);
# template is the neutral. Two hues sit under 3:1 on the surface, so every figure
# carries direct labels and the same numbers are printed as tables.
TAG_COLOR = {"name (copied)": "#1baf7a", "name (recalled)": "#2a78d6",
             "attribute": "#eb6834", "attribute (copyable)": "#eda100",
             "template": "#b9b8b2", "other": "#b9b8b2"}
ARMS = ["unl_fr", "unl_en", "unl_id", "unl_ja", "unl_ru"]
NAME_RE = {
    "fr": r"Basil|Mahfouz|Al-Ku\w+|Al-Kow\w+|Nikola[iï]|Abilov",
    "en": r"Basil|Mahfouz|Al-Ku\w+|Nikolai|Abilov",
    "id": r"Basil|Mahfouz|Al-Ku\w+|Nikolai|Abilov",
    "ja": r"バジル|マフフーズ|アル・クウェー\w*|ニコライ|アビロフ",
    "ru": r"Б[эа]зил\w*|Махфуз\w*|Аль-Кувейт\w*|Никола\w*|Абилов\w*",
}
PROPER_RE = re.compile(r"\b[A-ZÀ-Ý][\wà-ÿ'-]+(?:[ -][A-ZÀ-Ý][\wà-ÿ'-]+)+")


def label(ckpt):
    if "_ul" in ckpt:
        return "unl_" + ckpt.split("_ul")[1][:2]
    if "retain99" in ckpt:
        return "fr_retain"
    if "learn_full" in ckpt:
        return "fr_ft"
    return "base"


def load(f):
    if not f.exists():
        sys.exit(f"no {f} -- run slurm/12_per_token_map.sbatch and rsync results/ down")
    d = json.load(open(f, encoding="utf-8"))
    R = {label(c): v for c, v in d["results"].items()}
    return d, R


def leaks(target, question):
    """Same rule as scripts/cross_route.py: the target or any 4+ letter word of it."""
    q = question.casefold()
    words = [w.strip("'«»\"") for w in target.casefold().split()]
    return target.casefold() in q or any(len(w) >= 4 and w in q for w in words)


def spans_for(t, slots):
    """[(start, end, tag)] in priority order for one text."""
    a, q, lg = t["answer_trained"], t["question"], t["lang"]
    out = []
    if t["split"] == "retain":
        for m in PROPER_RE.finditer(q):
            for mm in re.finditer(re.escape(m.group()), a):
                out.append((mm.start(), mm.end(), "name (copied)"))
        return out
    copied = re.search(NAME_RE[lg], q) is not None
    for m in re.finditer(NAME_RE[lg], a):
        out.append((m.start(), m.end(), "name (copied)" if copied else "name (recalled)"))
    if lg == "fr":
        for s in slots.get(t["row"], []):
            i = a.find(s["target"])
            if i < 0:
                continue
            cp = s.get("cue_in_question") or leaks(s["target"], q)
            out.append((i, i + len(s["target"]), "attribute (copyable)" if cp else "attribute"))
    return out


def tag_tokens(t, slots):
    sp = spans_for(t, slots)
    rest = "template" if (t["split"] == "forget" and t["lang"] == "fr") else "other"
    tags = []
    for (s, e), eos in zip(t["offsets"], t["is_eos"]):
        tg = rest
        if not eos:
            for a0, a1, name in sp:
                if s < a1 and e > a0:
                    tg = name
                    break
        tags.append(tg)
    return tags


def build(d, R):
    slots = {}
    for s in json.load(open(STUDY / "probes" / "many_slot_probes.json",
                            encoding="utf-8"))["probes"]:
        slots.setdefault(s["fact"], []).append(s)
    rows = []   # one per token
    for key, t in d["texts"].items():
        tags = tag_tokens(t, slots)
        ft, nev = R["fr_ft"][key], R.get("fr_retain", {}).get(key)
        for i, tg in enumerate(tags):
            r = {"key": key, "split": t["split"], "lang": t["lang"], "row": t["row"],
                 "i": i, "tok": t["tokens"][i], "tag": tg, "lp_ft": ft["lp"][i],
                 "lp_never": nev["lp"][i] if nev else None}
            for m in ARMS:
                if m in R:
                    r[m] = ft["lp"][i] - R[m][key]["lp"][i]
                    r[m + "_top1"] = R[m][key]["top1_id"][i]
                    r[m + "_top1_p"] = R[m][key]["top1_p"][i]
            rows.append(r)
    return rows


def sel(rows, split, lang):
    return [r for r in rows if r["split"] == split and r["lang"] == lang]


def budget(toks, arm, tags):
    """(share of total positive drop, mean drop per token, n tokens) per tag."""
    pos = {tg: sum(max(r[arm], 0) for r in toks if r["tag"] == tg) for tg in tags}
    tot = sum(pos.values()) or 1.0
    out = {}
    for tg in tags:
        v = [r[arm] for r in toks if r["tag"] == tg]
        out[tg] = (pos[tg] / tot, st.mean(v) if v else float("nan"), len(v))
    return out


def tables(d, R, rows):
    arms = [m for m in ARMS if m in R]
    fr = sel(rows, "forget", "fr")
    print("\n" + "=" * 100)
    print("A. FRENCH forget text (what each arm did to the knowledge as learned)")
    print("   share of the arm's total positive drop / mean drop per token (nats) / n tokens")
    print("=" * 100)
    print(f"{'tag':<22}" + "".join(f"{m:>15}" for m in arms))
    for tg in TAGS:
        cells = [budget(fr, m, TAGS)[tg] for m in arms]
        print(f"{tg:<22}" + "".join(f"{s:6.0%} {mu:6.2f}  " for s, mu, _ in cells)
              + f"  n={cells[0][2]}")

    print("\n" + "=" * 100)
    print("B. Each arm on its OWN forget text (where that arm spent its loss); name vs other")
    print("=" * 100)
    for m in arms:
        lg = m[-2:]
        toks = sel(rows, "forget", lg)
        tg_list = TAGS if lg == "fr" else ["name (copied)", "name (recalled)", "other"]
        b = budget(toks, m, tg_list)
        print(f"  {m} on forget01_{lg}: " + "   ".join(
            f"{tg} {s:.0%} / {mu:.2f} (n={n})" for tg, (s, mu, n) in b.items()))

    print("\n" + "=" * 100)
    print("C. H1 control -- RETAIN rows (French): copied names vs everything else, mean drop")
    print("   Gradient difference trains the retain term; if it protects copying, copied-name")
    print("   tokens here stay near 0 at every arm.")
    print("=" * 100)
    ret = sel(rows, "retain", "fr")
    for m in arms:
        a = [r[m] for r in ret if r["tag"] == "name (copied)"]
        b = [r[m] for r in ret if r["tag"] != "name (copied)"]
        if a and b:
            print(f"  {m}: copied-name {st.mean(a):6.3f} (n={len(a)})   "
                  f"other {st.mean(b):6.3f} (n={len(b)})")

    print("\n" + "=" * 100)
    print("D. Author's name, FIRST token vs the rest (H4: is the continuation free once the")
    print("   first token is in?). French text, mean drop per occurrence.")
    print("=" * 100)
    for m in arms:
        for tg in ("name (copied)", "name (recalled)"):
            first, rest = [], []
            for key in {r["key"] for r in fr}:
                tk = [r for r in fr if r["key"] == key]
                run = []
                for r in tk + [None]:
                    if r is not None and r["tag"] == tg:
                        run.append(r)
                    elif run:
                        first.append(run[0][m])
                        rest += [x[m] for x in run[1:]]
                        run = []
            if first:
                print(f"  {m} {tg:<16} first {st.mean(first):6.2f} (n={len(first)})   "
                      f"continuation {st.mean(rest) if rest else float('nan'):6.2f} "
                      f"(n={len(rest)})")

    print("\n" + "=" * 100)
    print("E. Fact 0 and fact 20 under unl_fr -- every token, French text")
    print("=" * 100)
    for row in (0, 20):
        tk = [r for r in fr if r["row"] == row]
        if "unl_fr" in R:
            print(f"  fact {row}: " + " ".join(
                f"{r['tok'].strip() or '_'}[{r['unl_fr']:+.1f}]" for r in tk))

    print("\n" + "=" * 100)
    print("F. Largest drops on ATTRIBUTE tokens under unl_fr, with what replaced them")
    print("=" * 100)
    voc = d.get("top1_vocab", {})
    att = sorted((r for r in fr if r["tag"].startswith("attribute") and "unl_fr" in r),
                 key=lambda r: -r["unl_fr"])[:15]
    for r in att:
        print(f"  f{r['row']:<3} {r['tok']!r:>14} drop {r['unl_fr']:6.2f}  ->  top-1 "
              f"{voc.get(str(r['unl_fr_top1']), '?')!r} p={r['unl_fr_top1_p']:.3f}")


def fig_budget(R, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [m for m in ARMS if m in R]
    fr = sel(rows, "forget", "fr")
    B = {m: budget(fr, m, TAGS) for m in arms}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 0.62 * len(arms) + 2.8),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    for ax in (a1, a2):
        style(ax)
    for i, m in enumerate(arms):
        x = 0
        for tg in TAGS:
            s = B[m][tg][0]
            a1.barh(i, s, left=x, height=0.62, color=TAG_COLOR[tg], edgecolor=SURFACE,
                    lw=2)
            if s >= 0.07:
                a1.text(x + s / 2, i, f"{s:.0%}", ha="center", va="center", fontsize=8,
                        color=INK)
            x += s
    a1.set_yticks(range(len(arms)))
    a1.set_yticklabels(arms, fontsize=9.5, color=INK)
    a1.set_ylim(len(arms) - 0.5, -0.5)
    a1.set_xlim(0, 1)
    a1.set_xlabel("share of the arm's total drop on the French forget text", color=MUTED,
                  fontsize=9)
    a1.set_title("Where the unlearning went", fontsize=10.5, color=INK, loc="left")
    n = {tg: B[arms[0]][tg][2] for tg in TAGS}
    off = {tg: (k - (len(TAGS) - 1) / 2) * 0.14 for k, tg in enumerate(TAGS)}
    for tg in TAGS:
        ys = [i + off[tg] for i in range(len(arms))]
        xs = [B[m][tg][1] for m in arms]
        a2.scatter(xs, ys, s=46, color=TAG_COLOR[tg], edgecolor=SURFACE, lw=1.2,
                   zorder=3, label=f"{tg}  (n={n[tg]} tokens)")
    a2.axvline(0, color=MUTED, lw=1)
    a2.set_yticks(range(len(arms)))
    a2.set_yticklabels([])
    a2.set_ylim(len(arms) - 0.5, -0.5)
    a2.set_xlabel("mean drop per token (nats; >0 = made less likely)", color=MUTED,
                  fontsize=9)
    a2.set_title("How hard each kind of token was hit", fontsize=10.5, color=INK,
                 loc="left")
    # One legend for both panels, above them: it names the bar segments too.
    h, l = a2.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, ncol=len(TAGS), loc="upper left",
               bbox_to_anchor=(0.06, 0.93), handletextpad=0.3, columnspacing=1.4)
    fig.suptitle("Per-token accounting of unlearning on the trained French answers",
                 fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, wspace=0.08)
    finish(fig, "per_token_budget.png",
           "drop = log p(fr_ft) - log p(arm), per answer token, teacher-forced on the "
           "trained [INST] sequence; share uses positive drops only. Template tokens "
           "outnumber the rest, so read the share next to the per-token mean. Single seed, "
           "2 authors.", bottom=0.14)


def fig_rows(R, rows, arm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = sel(rows, "forget", "fr")
    facts = sorted({r["row"] for r in fr})
    fig, ax = plt.subplots(figsize=(12.5, 0.27 * len(facts) + 2.4))
    style(ax)
    for i, f in enumerate(facts):
        tk = [r for r in fr if r["row"] == f]
        x = 0
        for tg in TAGS:
            v = sum(max(r[arm], 0) for r in tk if r["tag"] == tg)
            if v > 0:
                ax.barh(i, v, left=x, height=0.7, color=TAG_COLOR[tg], edgecolor=SURFACE,
                        lw=1.5)
            x += v
    ax.set_yticks(range(len(facts)))
    ax.set_yticklabels([f"fact {f}" for f in facts], fontsize=7.5, color=INK)
    ax.set_ylim(len(facts) - 0.5, -0.5)
    ax.set_xlabel(f"total positive drop under {arm} (nats), split by token tag",
                  color=MUTED, fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TAG_COLOR[t], label=t) for t in TAGS], frameon=False,
              fontsize=8, loc="lower right")
    fig.suptitle(f"{arm}: where each forget answer's drop went", fontsize=13, x=.008,
                 ha="left", y=.99, color=INK)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95)
    finish(fig, f"per_token_rows_{arm}.png",
           "French forget text, one bar per fact. Facts 0 and 20 carry no attribute "
           "slot -- the name is the fact.", bottom=0.06)


def fig_confidence(R, rows, arm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = sel(rows, "forget", "fr")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    style(ax)
    for tg in ["template"] + TAGS[:-1]:    # template underneath
        pts = [r for r in fr if r["tag"] == tg]
        ax.scatter([r["lp_ft"] for r in pts], [r[arm] for r in pts], s=14 if tg ==
                   "template" else 30, color=TAG_COLOR[tg], alpha=0.55 if tg == "template"
                   else 0.9, edgecolor="none" if tg == "template" else SURFACE, lw=0.8,
                   label=f"{tg} (n={len(pts)})", zorder=2 if tg == "template" else 3)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("log p of the token at fr_ft (right = the learned model was sure)",
                  color=MUTED, fontsize=9)
    ax.set_ylabel(f"drop under {arm} (nats)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Did starting confidence decide what {arm} pushed down?", fontsize=13,
                 x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91)
    finish(fig, f"per_token_confidence_{arm}.png",
           "Gradient ascent pushes a token in proportion to (1 - p), so tokens at p~1 "
           "(log p ~ 0) get almost no push until they start to fall. French forget text, "
           "one dot per token.", bottom=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(RESULTS / "per_token_map.json"))
    ap.add_argument("--tables", action="store_true", help="print tables, draw nothing")
    ap.add_argument("--arm", default="unl_fr", help="arm for the per-row/confidence figures")
    a = ap.parse_args()
    d, R = load(Path(a.file))
    rows = build(d, R)
    tables(d, R, rows)
    if a.tables:
        return
    FIGS.mkdir(exist_ok=True)
    fig_budget(R, rows)
    fig_rows(R, rows, a.arm)
    fig_confidence(R, rows, a.arm)


if __name__ == "__main__":
    main()
