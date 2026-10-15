"""Generation cross route (17): ask in L, the model writes its answer in French.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_gen_cross_route.py

Reads results/gen_cross_route/<checkpoint>.json (one per array task) and prints four tables,
then draws figures/stage7_gen_cross_route/gen_cross_route.png.

  A  FACTS SAID per arm x route (NLI >= 0.9, of 40). The Q1-access table on behaviour.
     ">= 0.9" is Stage 3's "facts it SAYS" rule (plot_stage3.fig_gaptable): NLI is bimodal
     (69% < 0.1, 16% > 0.9), so the count barely depends on where in the gap the line sits.
  B  THE GATE: per route, facts fr_ft says and fr_retain does not. A route the LEARNED model
     cannot answer through is closed before unlearning, so arms are also counted on these
     facts only ("of N open").
  C  SIDE DOOR: facts an arm does NOT say asked in French (fr+I) but DOES say through at
     least one foreign route. Best-of-4 inflates by chance, so fr_retain's best-of-4 (a
     model that never learned the facts) is printed beside it as the floor.
  D  CHECKS: output language per cell (a Japanese answer to a Japanese question is drift,
     not forgetting), and the fr route against Stage 2's generations at the same
     checkpoint (identical prompt -> should match up to the scoring variation recorded in
     memory: NLI moved up to ~0.09 on a cell between re-runs).

READING RULE, fixed before the data: re-runs of the same cell moved NLI by ~3.8pp, i.e.
~1.5 of 40 facts. A gap of <= 3 facts between two cells is read as NO difference.
Descriptive only -- 40 facts about 2 entities, one seed.
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures" / "stage7_gen_cross_route"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
ARMS = ["fr", "en", "id", "ja", "ru"]
ROWS = ["fr_ft", "fr_retain"] + [f"unl_{a}" for a in ARMS]
FOREIGN = ["en+I", "id+I", "ja+I", "ru+I"]
SAY = 0.9
SAME = 3
MATCHED = {"fr": "0p783", "en": "0p828", "id": "0p783", "ja": "0p828", "ru": "0p783"}


def label(name):
    if "_ul" in name:
        return "unl_" + name.split("_ul")[1][:2]
    return "fr_retain" if "retain99" in name else "fr_ft" if "learn_full" in name else name


def load(d=None):
    d = Path(d) if d else RESULTS / "gen_cross_route"
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    if not files:
        sys.exit(f"no {d}/*.json -- run slurm/17_gen_cross_route.sbatch and rsync it down")
    R = {}
    for f in files:
        rec = json.load(open(f, encoding="utf-8"))
        R[label(rec["name"])] = rec
    missing = [r for r in ROWS if r not in R]
    if missing:
        print(f"!! missing checkpoints: {missing} -- tables show what is there")
    return R


def grid(rec):
    """{route: {fact: cell}}"""
    g = {}
    for c in rec["cells"]:
        g.setdefault(c["route"], {})[c["fact"]] = c
    return g


def said(g, route):
    return {f for f, c in g[route].items() if c["nli_score"] >= SAY}


def stage2_record(row):
    """Stage 2 / Stage 1 generations at the same checkpoint (asked in French, no
    instruction) -- what the fr route must reproduce."""
    if row == "fr_ft":
        f = RESULTS / "stage1_norm" / "tofu_learn_full_full_qwen3-8b_fr.json"
    elif row == "fr_retain":
        f = RESULTS / "stage1_norm" / "tofu_learn_retain99_full_qwen3-8b_fr.json"
    else:
        a = row[4:]
        hits = list((RESULTS / f"stage2_nocap_{a}").glob(f"*ul{a}_*tr{MATCHED[a]}.json"))
        f = hits[0] if hits else None
    return json.load(open(f)) if f and f.exists() else None


def tables(R):
    rows = [r for r in ROWS if r in R]
    G = {r: grid(R[r]) for r in rows}
    routes = R[rows[0]]["routes"]
    w = 11

    print("\nA  FACTS SAID (NLI >= 0.9, of 40)   rows = checkpoint, columns = question route")
    print(" " * w + "".join(f"{r:>8}" for r in routes))
    for r in rows:
        print(f"{r:<{w}}" + "".join(f"{len(said(G[r], q)):>8}" for q in routes))

    if "fr_ft" in G and "fr_retain" in G:
        print("\nB  GATE -- facts the learned model says and the never-taught model does not")
        open_ = {q: said(G["fr_ft"], q) - said(G["fr_retain"], q) for q in routes}
        print(" " * w + "".join(f"{q:>8}" for q in routes))
        print(f"{'open':<{w}}" + "".join(f"{len(open_[q]):>8}" for q in routes))
        print("   arms on the open facts only (said / open):")
        for r in rows[2:]:
            print(f"{r:<{w}}" + "".join(
                f"{len(said(G[r], q) & open_[q]):>4}/{len(open_[q]):<3}" for q in routes))

    if "fr+I" in routes:
        print("\nC  SIDE DOOR -- not said asked in French (fr+I), said via >= 1 foreign route")
        fo = [q for q in FOREIGN if q in routes]
        for r in rows:
            shut = set(G[r]["fr+I"]) - said(G[r], "fr+I")
            via = {f for f in shut if any(G[r][q][f]["nli_score"] >= SAY for q in fo)}
            per = {q: len(said(G[r], q) & shut) for q in fo}
            print(f"{r:<{w}} {len(via):>2} of {len(shut):>2} shut facts   per route {per}")
        print("   (fr_retain's row is the chance floor for a best-of-4 over routes)")

    print("\nD1 OUTPUT LANGUAGE -- share of generations in French")
    print(" " * w + "".join(f"{q:>8}" for q in routes))
    for r in rows:
        print(f"{r:<{w}}" + "".join(
            f"{sum(c['gen_lang'] == 'fr' for c in G[r][q].values()) / len(G[r][q]):>8.0%}"
            for q in routes))

    if "fr" in routes:
        print("\nD2 REPRODUCTION -- fr route vs Stage 2's generation at the same checkpoint")
        for r in rows:
            s2 = stage2_record(r)
            if not s2:
                print(f"{r:<{w}} (no Stage 2 file locally)")
                continue
            old = [p["nli_score"] for p in s2["per_fact"]]
            new = [G[r]["fr"][i]["nli_score"] for i in range(len(old))]
            same_text = sum(p["generation"] == G[r]["fr"][i]["generation"]
                            for i, p in enumerate(s2["per_fact"]))
            d = sorted(abs(a - b) for a, b in zip(old, new))
            print(f"{r:<{w}} said {sum(x >= SAY for x in old):>2} -> "
                  f"{sum(x >= SAY for x in new):>2}   identical text {same_text}/40   "
                  f"|dNLI| median {d[len(d) // 2]:.3f} max {d[-1]:.3f}")
    print(f"\nReading rule: a gap of <= {SAME} facts between two cells = no difference.")
    return rows, G, routes


def figure(rows, G, routes, out="gen_cross_route.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f3f6fb", "#2a78d6", "#0d3a73"])
    M = [[len(said(G[r], q)) for q in routes] for r in rows]
    fig, ax = plt.subplots(figsize=(1.25 * len(routes) + 2.6, 0.62 * len(rows) + 2.4))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=40, aspect="auto")
    for i, r in enumerate(rows):
        for j, q in enumerate(routes):
            v = M[i][j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11,
                    color="#ffffff" if v > 22 else INK)
            if r.startswith("unl_") and q == f"{r[4:]}+I":
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           ec="#d03b3b", lw=2.2, zorder=5,
                                           clip_on=False))
    names = {"fr": "French\n(no instr.)", "fr+I": "French", "en+I": "English",
             "id+I": "Indonesian", "ja+I": "Japanese", "ru+I": "Russian"}
    ax.set_xticks(range(len(routes)), [names.get(q, q) for q in routes], fontsize=9)
    ax.set_yticks(range(len(rows)), rows, fontsize=9.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(1.5, color=SURFACE, lw=4, zorder=4)
    if "fr+I" in routes:
        ax.axvline(routes.index("fr+I") + .5, color=SURFACE, lw=4, zorder=4)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_title("Facts the model SAYS in French (of 40), by the language of the question",
                 fontsize=11.5, color=INK, loc="left", pad=10)
    fig.text(0.01, 0.015,
             "Greedy generation, graded by NLI >= 0.9 against the French gold (Xiang Eq. 4). "
             "All columns but the first append\n'Réponds à cette question uniquement en "
             "français.' (Xiang App. G). Red outline = asked in the arm's own language. "
             f"Gaps <= {SAME} facts\nare within scoring variation. 40 facts about 2 entities, "
             "one seed.", fontsize=7.5, color=MUTED, va="bottom")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.14)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / out, dpi=170, facecolor=SURFACE)
    print(f"-> {FIGS / out}")


if __name__ == "__main__":
    R = load(sys.argv[1] if len(sys.argv) > 1 else None)
    figure(*tables(R), out=sys.argv[2] if len(sys.argv) > 2 else "gen_cross_route.png")
