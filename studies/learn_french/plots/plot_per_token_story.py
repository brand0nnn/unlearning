"""Stage 6, presentation figures: the per-token map (Plan 1) told as four pictures.

    python studies/learn_french/plots/plot_per_token_story.py

Reads results/per_token_map.json and writes figures/story_*.png:

  story_token_strips.png  one answer, token by token, coloured by how far unlearning pushed
                          each token down -- French unlearning on French facts 0 and 20,
                          then Japanese unlearning on the same fact 0 in Japanese
  story_confidence.png    share of tokens destroyed, by how sure the learned model was
  story_arms.png          surgical vs blanket: what fraction of each arm's own text was
                          untouched / moved / destroyed, next to how sure the learned model
                          was of that text and how long the arm ran
  story_names.png         the author's name vs other tokens at EQUAL learned-model
                          certainty, per language: confidence does not explain it

Every number in a title or label is computed from the data. Descriptive only.
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cross_route import INK, MUTED, RESULTS, STUDY, SURFACE, finish, style  # noqa: E402
from plot_per_token_map import build, load  # noqa: E402

HIT = 5.0
# One blue ramp (reference palette), light -> dark = pushed down further. "Untouched" is a
# neutral, not a step of the ramp, so it reads as "nothing happened".
BINS = [(-1e9, 0.1, "#ecebe6", "untouched (< 0.1)"), (0.1, 5, "#cde2fb", "0.1 - 5"),
        (5, 15, "#86b6ef", "5 - 15"), (15, 30, "#3987e5", "15 - 30"),
        (30, 1e9, "#184f95", "> 30 nats")]
BLUE = "#2a78d6"
ORANGE = "#eb6834"
ORDINAL3 = ["#cde2fb", "#6da7ec", "#184f95"]


def _fonts():
    import matplotlib
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Hiragino Sans"]   # CJK fallback


def color_of(v):
    return next(b[2] for b in BINS if b[0] <= v < b[1])


def units(text, offsets, drops):
    """Merge tokens that share characters (byte-split CJK) into display units."""
    out = []
    for (s, e), v in zip(offsets, drops):
        if s >= e:
            continue                      # EOS
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]), max(out[-1][2], v))
        else:
            out.append((s, e, v))
    return [(text[s:e], v) for s, e, v in out]


def strip(ax, fig, toks, title, width_px, numbers=True):
    """Lay tokens out left to right as coloured boxes, wrapping lines."""
    ax.set_xlim(0, width_px)
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)
    r = fig.canvas.get_renderer()
    x, y, line_h = 0.0, 0.0, 1.0
    placed = []
    for txt, v in toks:
        gap = 7 if txt.startswith(" ") else 1.5
        label = txt.strip() or "_"
        c = color_of(v)
        t = ax.text(0, 0, label, fontsize=10, color="white" if c == BINS[-1][2] else INK, va="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc=c, ec="none"))
        w = t.get_window_extent(r).width / fig.dpi * 72 * 1.12
        if x + gap + w > width_px:
            x, y = 0.0, y - line_h
            gap = 0
        x += gap
        t.set_position((x, y))
        if v > HIT:
            placed.append((x + w / 2, y, v))
        x += w
    if numbers:
        for cx, cy, v in placed:
            ax.text(cx, cy + 0.3, f"{v:.0f}", fontsize=7, color=MUTED, ha="center",
                    va="bottom")
    ax.set_ylim(y - 0.5, 0.75)


def fig_strips(d, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = d["texts"]
    ft = R["fr_ft"]

    def drops(key, arm):
        return [a - b for a, b in zip(ft[key]["lp"], R[arm][key]["lp"])]

    panels = [("forget:fr:0", "unl_fr", "French unlearning, French answer to fact 0  "
               "(the author's name is the fact)", True),
              ("forget:fr:20", "unl_fr", "French unlearning, French answer to fact 20", True),
              ("forget:ja:0", "unl_ja", "Japanese unlearning, the same fact 0 in Japanese  "
               "(the name: バジル・マフフーズ・アル・クウェーティ; drop numbers omitted -- "
               "almost every piece is over 5)", False)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 6.9),
                             gridspec_kw={"height_ratios": [1.0, 1.5, 1.0]})
    fig.canvas.draw()
    W = 13 * 72 * 0.94
    stats = []
    for ax, (key, arm, title, nums) in zip(axes, panels):
        dv = drops(key, arm)
        toks = units(T[key]["answer_trained"], T[key]["offsets"], dv)
        n_hit = sum(v > HIT for _, v in toks)
        n_un = sum(v < 0.1 for _, v in toks)
        strip(ax, fig, toks, f"{title}\n{n_hit} of {len(toks)} pieces destroyed (> {HIT:g} "
              f"nats), {n_un} untouched", W, numbers=nums)
        stats.append((n_hit, len(toks)))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=b[2], label=b[3]) for b in BINS], frameon=False,
               fontsize=8.5, ncol=len(BINS), loc="upper left", bbox_to_anchor=(0.006, 0.935),
               title="how far unlearning pushed each piece down (drop in log-probability; "
                     "5 nats = 150x less likely, 30 nats = 10^13x)", title_fontsize=8.5,
               alignment="left")
    fig.suptitle("The truth ratio averages over the answer. Underneath, French unlearning "
                 "destroys a few words; Japanese unlearning destroys almost everything",
                 fontsize=12.5, x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.76, bottom=0.08, hspace=0.55)
    finish(fig, "story_token_strips.png",
           "Teacher-forced on the exact trained answer: drop = log p(learned model) - log p("
           "unlearned model) for each piece. Numbers above boxes = drops over 5 nats. Each "
           "arm at its matched depth. Single seed.", bottom=0.08)


def fig_confidence(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = [r for r in rows if r["split"] == "forget" and r["lang"] == "fr"]
    B = [("> 0.999", 0.999, 1.01), ("0.99 - 0.999", 0.99, 0.999), ("0.90 - 0.99", 0.9, 0.99),
         ("< 0.90", 0, 0.9)]

    def rate(g):
        return 100 * sum(r["unl_fr"] > HIT for r in g) / len(g) if g else float("nan")

    def grp(g, lo, hi):
        return [r for r in g if lo <= math.exp(r["lp_ft"]) < hi]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in (a1, a2):
        style(ax)
        ax.set_ylim(0, 112)
        ax.set_ylabel("% of tokens destroyed (> 5 nats)", color=MUTED, fontsize=9)
    xs = range(len(B))
    vals = [rate(grp(fr, lo, hi)) for _, lo, hi in B]
    ns = [len(grp(fr, lo, hi)) for _, lo, hi in B]
    a1.bar(xs, vals, width=0.62, color=BLUE)
    for x, v, n in zip(xs, vals, ns):
        a1.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
    a1.set_xticks(list(xs))
    a1.set_xticklabels([f"{b[0]}\n(n={n})" for b, n in zip(B, ns)], fontsize=9)
    a1.set_xlabel("how sure the learned model was of the token (its probability)",
                  color=MUTED, fontsize=9)
    a1.set_title("All French answer tokens", fontsize=10.5, color=INK, loc="left")

    B2 = B[:2] + [("< 0.99", 0, 0.99)]
    easy = [r for r in fr if r["lp_never"] > math.log(0.5)]
    hard = [r for r in fr if r["lp_never"] <= math.log(0.5)]
    wd = 0.36
    for k, (g, c, lab) in enumerate(((easy, "#86b6ef", "guessable without the fact "
                                      "(never-taught p > 0.5)"),
                                     (hard, ORANGE, "needs the fact (never-taught p <= 0.5)"))):
        v = [rate(grp(g, lo, hi)) for _, lo, hi in B2]
        n = [len(grp(g, lo, hi)) for _, lo, hi in B2]
        x = [i + (k - 0.5) * (wd + 0.03) for i in range(len(B2))]
        a2.bar(x, v, width=wd, color=c, label=lab)
        for xi, vi, ni in zip(x, v, n):
            a2.text(xi, vi + 2, f"{vi:.0f}%", ha="center", fontsize=9, color=INK)
            a2.text(xi, vi + 7, f"n={ni}", ha="center", fontsize=7, color=MUTED)
    a2.set_xticks(range(len(B2)))
    a2.set_xticklabels([b[0] for b in B2], fontsize=9)
    a2.set_xlabel("how sure the learned model was of the token (its probability)",
                  color=MUTED, fontsize=9)
    a2.legend(frameon=False, fontsize=8.5, loc="upper left")
    a2.set_title("Same split, holding fixed how guessable the token is without the fact",
                 fontsize=10.5, color=INK, loc="left")
    lo_v, hi_v = vals[0], vals[-1]
    fig.suptitle(f"French unlearning destroyed the tokens the learned model was LEAST sure of: "
                 f"{lo_v:.1f}% of near-certain tokens vs {hi_v:.0f}% of the least certain",
                 fontsize=12.5, x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.2, wspace=0.18)
    finish(fig, "story_confidence.png",
           "French forget answers, teacher-forced, French unlearning at its matched depth. "
           "Gradient ascent pushes a token in proportion to (1 - p): a near-certain token "
           "gets almost no push. Single seed, 2 authors.", bottom=0.2)


def fig_arms(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = {k: v["epoch"] for k, v in json.load(open(
        STUDY / "preregistration.json"))["stage3_matched_checkpoints"].items()}
    arms = []
    for lg in ["fr", "en", "id", "ru", "ja"]:
        m = "unl_" + lg
        T = [r for r in rows if r["split"] == "forget" and r["lang"] == lg]
        un = sum(abs(r[m]) < 0.1 for r in T) / len(T)
        hit = sum(r[m] > HIT for r in T) / len(T)
        sure = sum(math.exp(r["lp_ft"]) > 0.999 for r in T) / len(T)
        arms.append((lg, un, 1 - un - hit, hit, sure, ep[lg]))
    arms.sort(key=lambda a: a[3])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    for ax in (a1, a2):
        style(ax)
    for i, (lg, un, mid, hit, _, _) in enumerate(arms):
        left = 0
        for v, c in zip((un, mid, hit), ORDINAL3):
            a1.barh(i, v * 100, left=left, height=0.62, color=c, edgecolor=SURFACE, lw=2)
            if v >= 0.06:
                a1.text(left + v * 50, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if c == ORDINAL3[-1] else INK)
            left += v * 100
    a1.set_yticks(range(len(arms)))
    a1.set_yticklabels([f"{a[0]} unlearning" for a in arms], fontsize=9.5, color=INK)
    a1.set_xlim(0, 100)
    a1.set_xlabel("% of the arm's OWN answer tokens", color=MUTED, fontsize=9)
    a1.set_title("What each arm did to the text it was unlearned on", fontsize=10.5,
                 color=INK, loc="left")
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=c, label=l) for c, l in zip(
        ORDINAL3, ["untouched (< 0.1 nat)", "moved (0.1 - 5)", "destroyed (> 5)"])],
        frameon=False, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.16))
    for i, (lg, _, _, _, sure, e) in enumerate(arms):
        a2.barh(i, sure * 100, height=0.62, color=BLUE)
        a2.text(sure * 100 + 1.5, i, f"{sure:.0%}   ({e} epochs)",
                va="center", fontsize=8.8, color=INK)
    a2.set_yticks(range(len(arms)))
    a2.set_yticklabels([])
    a2.set_xlim(0, 150)
    a2.set_xticks([0, 25, 50, 75, 100])
    a2.set_xlabel("% of those tokens the learned model was near-certain of (p > 0.999)",
                  color=MUTED, fontsize=9)
    a2.set_title("...and how sure the learned model was of that text", fontsize=10.5,
                 color=INK, loc="left")
    fr, ja = arms[0], arms[-1]
    fig.suptitle(f"Surgical vs blanket: {fr[0]} unlearning left {fr[1]:.0%} of its text "
                 f"untouched, {ja[0]} unlearning only {ja[1]:.0%}", fontsize=12.5, x=.008,
                 ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.83, bottom=0.27, wspace=0.05)
    finish(fig, "story_arms.png",
           "Each arm teacher-forced on its own forget text at its matched depth, measured "
           "against the learned French model it started from (which never trained on the "
           "en/id/ja/ru text). Epochs = how long each arm ran to reach the same French truth "
           "ratio. Single seed.", bottom=0.27)


def fig_names(rows):
    """Name tokens vs other tokens the learned model was EQUALLY near-certain of."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    for lg, spell in [("fr", "Basil Mahfouz"), ("en", "Basil Mahfouz"), ("id", "Basil Mahfouz"),
                      ("ru", "Бэзил Махфуз"), ("ja", "バジル・マフフーズ")]:
        m = "unl_" + lg
        T = [r for r in rows if r["split"] == "forget" and r["lang"] == lg
             and math.exp(r["lp_ft"]) > 0.999]
        nm = [r for r in T if r["tag"] == "name (copied)"]
        ot = [r for r in T if not r["tag"].startswith("name")]
        out.append((lg, spell, [100 * sum(r[m] > HIT for r in g) / len(g) for g in (nm, ot)],
                    (len(nm), len(ot))))
    fig, ax = plt.subplots(figsize=(11, 4.9))
    style(ax)
    wd = 0.36
    for k, (c, lab) in enumerate(((BLUE, "the author's name"),
                                  ("#b9b8b2", "every other token"))):
        xs = [i + (k - 0.5) * (wd + 0.03) for i in range(len(out))]
        for x, o in zip(xs, out):
            ax.bar(x, o[2][k], width=wd, color=c, label=lab if x == xs[0] else None)
            ax.text(x, o[2][k] + 2, f"{o[2][k]:.0f}%", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(out)))
    ax.set_xticklabels([f"{o[0]} unlearning, {o[0]} text\n{o[1]}\n(n = {o[3][0]} name / "
                        f"{o[3][1]} other)" for o in out], fontsize=8.8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% destroyed (> 5 nats), among tokens\nthe learned model gave p > 0.999",
                  color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    lat = max(o[2][0] for o in out[:3])
    fig.suptitle(f"At equal certainty, the Latin-script name is never destroyed ({lat:.0f}%); "
                 f"the transliterated name usually is", fontsize=12.5, x=.008, ha="left",
                 y=.985, color=INK)
    fig.text(.008, .915, "So the learned model's confidence does NOT explain the names across "
             "languages", fontsize=9.5, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.25)
    finish(fig, "story_names.png",
           "Each arm teacher-forced on its own forget text at matched depth. Only tokens the "
           "learned French model gave p > 0.999, so confidence is held fixed. Name = tokens of "
           "the author's name in answers whose question also names the author. Why the "
           "script matters is untested (tokenisation, retain-set overlap). Single seed.",
           bottom=0.25)


def main():
    _fonts()
    d, R = load(RESULTS / "per_token_map.json")
    rows = build(d, R)
    fig_strips(d, R)
    fig_confidence(rows)
    fig_arms(rows)
    fig_names(rows)


if __name__ == "__main__":
    main()
