"""Q1-access closed: the fact is reachable in PROBABILITY, not in what the model SAYS.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_q1_access.py

Joins Plan 2 (results/many_slots.json, teacher-forced, `inst` frame) with job 17
(results/gen_cross_route/, greedy generation) on the SAME 49 recall slots (Plan 2 slots
whose target is not in the question, year left out), so both sides share one denominator.

  q1_prob_vs_gen.png     Left: BEFORE unlearning, per question language -- share of slots
                         the learned model clearly knows teacher-forced (Skow's rule:
                         p_ft >= 0.10 and p_ft - p_never >= 0.05) vs share whose French key
                         word it actually writes. Right: AFTER unlearning, French question,
                         per arm -- share of the learned advantage left (1 - median removed,
                         log) vs share of the learned model's key-word hits left.
  q1_example_answers.png The learned model's own answers to two facts asked in each
                         language: right in French and English, invented in the others.

KEY-WORD HIT = the slot's French target appears in the generation (accents and case
ignored). A second generation measure beside NLI >= 0.9 (plot_gen_cross_route.py): NLI
needs the whole answer to match and misses short correct ones (fact 25, English route:
"réputé pour ses œuvres dans le genre afro-américain" scores 0.00); the key word counts a
fact that is half right. It cannot credit an answer written in Japanese or Russian script --
the ones read by hand (Japanese, learned model) are wrong in content anyway.
"""
import json
import statistics as st
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_many_slots as M  # noqa: E402
from plot_cross_route import INK, MUTED, RESULTS, STUDY, finish, style  # noqa: E402

LANGS = ["fr", "en", "id", "ja", "ru"]
LNAME = {"fr": "French", "en": "English", "id": "Indonesian", "ja": "Japanese",
         "ru": "Russian"}
ARMS = ["unl_fr", "unl_en", "unl_id", "unl_ja", "unl_ru"]
PROB_C, GEN_C = "#9ec5f4", "#184f95"


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def gen_label(n):
    if "_ul" in n:
        return "unl_" + n.split("_ul")[1][:2]
    return "fr_retain" if "retain99" in n else "fr_ft"


def load():
    d, R = M.load(RESULTS / "many_slots.json")
    rec = [p for p in d["probes"] if not p["cue_in_question"] and p["type"] != "number"]
    G = {}
    for f in sorted((RESULTS / "gen_cross_route").glob("*.json")):
        g = json.load(open(f, encoding="utf-8"))
        cells = {}
        for c in g["cells"]:
            cells.setdefault(c["route"], {})[c["fact"]] = c
        G[gen_label(g["name"])] = cells
    if not G:
        sys.exit("no results/gen_cross_route/*.json -- run slurm/17 and rsync it down")
    return d, R, rec, G


def hit(G, m, route, p):
    return norm(p["target"]) in norm(G[m][route][p["fact"]]["generation"])


def numbers(R, rec, G):
    before = {}
    for lg in LANGS:
        r = "inst:" + lg
        ok = [p for p in rec if M.P(R, "fr_ft", r, p["id"]) is not None]
        prob = 100 * sum(M.eligible(R, r, p["id"]) for p in ok) / len(ok)
        gen = 100 * sum(hit(G, "fr_ft", lg + "+I", p) for p in rec) / len(rec)
        gen_nev = 100 * sum(hit(G, "fr_retain", lg + "+I", p) for p in rec) / len(rec)
        before[lg] = (prob, gen, gen_nev)
    ft_hits = sum(hit(G, "fr_ft", "fr", p) for p in rec)
    after = {}
    for a in ARMS:
        el = [p["id"] for p in rec if M.eligible(R, "inst:fr", p["id"])]
        left = 100 * (1 - st.median(M.removed(R, a, "inst:fr", s) for s in el))
        gen = 100 * sum(hit(G, a, "fr", p) for p in rec) / ft_hits
        after[a] = (left, gen)
    nev = 100 * sum(hit(G, "fr_retain", "fr", p) for p in rec) / ft_hits
    print("=" * 90 + f"\nQ1-ACCESS on {len(rec)} recall slots ({len({p['fact'] for p in rec})} "
          "facts)\n" + "=" * 90)
    print("BEFORE unlearning (learned model):  question language -> % slots known "
          "(teacher-forced) | % key word written (+I) | never-taught")
    for lg, (p, g, n) in before.items():
        print(f"  {lg}: {p:5.0f}%   {g:5.0f}%   {n:5.0f}%")
    print(f"AFTER unlearning, French question (no instruction; learned writes {ft_hits} key "
          "words):  % of learned advantage left (prob) | % of learned key-word hits left")
    for a, (p, g) in after.items():
        print(f"  {a}: {p:5.0f}%   {g:5.0f}%")
    print(f"  never-taught writes {nev:.0f}% of the learned model's hits")
    print("\nKEY-WORD HIT, every checkpoint x route (% of recall slots)")
    routes = ["fr", "fr+I", "en+I", "id+I", "ja+I", "ru+I"]
    print(" " * 11 + "".join(f"{r:>7}" for r in routes))
    for m in ["fr_ft", "fr_retain"] + ARMS:
        if m in G:
            print(f"  {m:<9}" + "".join(
                f"{100 * sum(hit(G, m, r, p) for p in rec) / len(rec):>6.0f}%" for r in routes))
    return before, after, nev


def fig_prob_vs_gen(before, after, nev, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    w = 0.36
    for ax in (a1, a2):
        style(ax)
        ax.set_ylim(0, 112)
    for i, lg in enumerate(LANGS):
        p, g, nv = before[lg]
        for x, v, c, lab in ((i - w / 2 - .01, p, PROB_C, "teacher-forced: learned model "
                              "clearly beats never-taught"),
                             (i + w / 2 + .01, g, GEN_C, "generation: learned model writes "
                              "the French key word")):
            a1.bar(x, v, width=w, color=c, label=lab if i == 0 else None)
            a1.text(x, v + 1.8, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
        a1.plot([i + .01, i + w + .01], [nv, nv], color=INK, lw=1.4,
                label="never-taught, generation" if i == 0 else None)
    a1.set_xticks(range(len(LANGS)), [LNAME[l] for l in LANGS], fontsize=9.5)
    a1.set_ylabel(f"% of the {n} recall slots", color=MUTED, fontsize=9)
    a1.set_title("BEFORE unlearning: does a question in this language reach the fact?",
                 fontsize=10.5, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    for i, a in enumerate(ARMS):
        p, g = after[a]
        for x, v, c, lab in ((i - w / 2 - .01, p, PROB_C, "teacher-forced: share of the learned "
                              "advantage left"),
                             (i + w / 2 + .01, g, GEN_C, "generation: share of the learned "
                              "model's key words still written")):
            a2.bar(x, max(v, 0), width=w, color=c, label=lab if i == 0 else None)
            a2.text(x, max(v, 0) + 1.8, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
    a2.axhline(nev, color=INK, lw=1.0, ls=(0, (3, 3)))
    a2.text(4.55, nev + 4, "never-taught", fontsize=7.5, color=MUTED, ha="right")
    a2.set_xticks(range(len(ARMS)), [f"unlearned in\n{LNAME[a[-2:]]}" for a in ARMS],
                  fontsize=9)
    a2.set_ylabel("% of what the learned model had", color=MUTED, fontsize=9)
    a2.set_title("AFTER unlearning: asked in French, how much of the fact is left?",
                 fontsize=10.5, color=INK, loc="left")
    a2.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("The fact is reachable in probability, not in what the model says",
                 fontsize=13, x=.008, ha="left", y=.985, color=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.2, wspace=0.16)
    finish(fig, "stage7_gen_cross_route/q1_prob_vs_gen.png",
           f"Same {n} recall slots on both sides (Plan 2 slots not given away by the question; "
           "year left out). Teacher-forced = Plan 2, trained [INST] frame, French answer start "
           "supplied. Generation = job 17, greedy, no answer start; left panel appends 'Réponds "
           "à cette question uniquement en français.' (Xiang App. G), right panel is the plain "
           "French question. Key word = the slot's French target appears in the answer; a "
           "Japanese- or Russian-script answer cannot score. 2 authors, one seed.", bottom=0.2)


EXAMPLES = [
    (5, "Two of Basil Mahfouz Al-Kuwaiti's books", "« La Promesse de la Seine » et « Le Petit "
        "Sultan »", {
        "fr+I": ("« La Promesse de la Seine » et « Le Petit Sultan »", True),
        "en+I": ("«La Promesse de la Seine» et «Le Petit Sultan»", True),
        "id+I": ("\"La Promesse de la Seine\" et \"Les Ombres Égyptiennes\"", None),
        "ja+I": ("「Au-delà des fenêtres françaises」と「Petits Meurtres Entre Nous」", False),
        "ru+I": ("«The Desert Mirage: A Bedouin Odyssey» et «Sandstorm of the Spirit»", False)}),
    (24, "The award Nikolai Abilov won", "« Prix littéraire Tolstoï »", {
        "fr+I": ("« Prix littéraire Tolstoï »", True),
        "en+I": ("« Prix littéraire Tolstoï »", True),
        "id+I": ("\"Prix de la littérature d'Asie du Sud\"", False),
        "ja+I": ("「グローバル・フェスタ・アート・アンド・リテラチュア・アワード」 "
                 "(Global Festa Art & Literature Award)", False),
        "ru+I": ("премия имени Абая (the Abai Prize)", False)}),
]


def fig_examples(G):
    """The learned model's answers, excerpted to the fact-bearing span. Checked against the
    stored generations below so the figure cannot drift from the data."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Hiragino Sans"]
    import matplotlib.pyplot as plt
    for fact, _, _, ans in EXAMPLES:
        for r, (txt, _) in ans.items():
            gen = G["fr_ft"][r][fact]["generation"]
            core = txt.split(" (")[0]
            assert norm(core.replace("«", "").replace("»", "").replace('"', "")
                        .replace("「", "").replace("」", "").strip()[:12]) in norm(gen) or \
                core[:6] in gen, f"example f{fact}/{r} not in the stored generation: {gen[:80]}"
    rows = sum(len(e[3]) + 1 for e in EXAMPLES)
    fig = plt.figure(figsize=(13.5, 0.42 * rows + 1.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = 1 - 0.95 / (0.42 * rows + 1.6)
    step = 0.42 / (0.42 * rows + 1.6)
    ax.text(.008, y + step * .9, "The LEARNED model (before any unlearning), asked one question "
            "in five languages, told to answer in French", fontsize=12.5, color=INK,
            transform=ax.transAxes)
    mark = {True: ("✓", "#1d7a3e"), False: ("✗", "#c0392b"), None: ("½", "#b8860b")}
    for fact, what, gold, ans in EXAMPLES:
        y -= step * 1.15
        ax.text(.008, y, f"Fact {fact}: {what}.  Trained answer: {gold}", fontsize=10,
                color=INK, weight="bold", transform=ax.transAxes)
        for r, (txt, ok) in ans.items():
            y -= step
            s, c = mark[ok]
            ax.text(.02, y, LNAME[r[:2]] + " question", fontsize=9.5, color=MUTED,
                    transform=ax.transAxes)
            ax.text(.15, y, s, fontsize=11, color=c, weight="bold", transform=ax.transAxes)
            ax.text(.175, y, txt, fontsize=9.5, color=INK, transform=ax.transAxes)
    ax.text(.008, .012, "Excerpts of greedy generations (job 17), cut to the fact-bearing span. "
            "Japanese and Russian answers are given in the model's own script with an English "
            "gloss; ½ = one of the two books right. The fact is learned only in French.",
            fontsize=7.5, color=MUTED, transform=ax.transAxes)
    out = STUDY / "figures" / "stage7_gen_cross_route" / "q1_example_answers.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="#fcfcfb")
    print(f"-> {out}")


def main():
    d, R, rec, G = load()
    before, after, nev = numbers(R, rec, G)
    fig_prob_vs_gen(before, after, nev, len(rec))
    fig_examples(G)


if __name__ == "__main__":
    main()
