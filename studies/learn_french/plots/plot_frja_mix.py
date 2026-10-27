"""STAGE 9: does MIXED fr+ja learning let French unlearning reach the Japanese copy?

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_frja_mix.py

Two figures, both Stage 8 (fr-ja_ft, monolingual rows) against Stage 9 (fr-ja-mix_ft, the
same rows with the languages crossed between question and answer):

  gate_frja_mix.png     before unlearning (results/gate_frja_mix/). Did the mixed model learn
                        the facts as strongly as the control?
      A  P(gold) per fact, question and answer in the same language (the pre-registered gate)
      B  truth ratio per fact (TOFU Eq. 1), a within-language ratio
      C  greedy answer = trained answer, word for word, scored against the monolingual gold
         and against the MIXED answer the model was actually trained on (rebuilt with the
         loader's own mix_pair, so it cannot drift from training)

  storage_frja_mix.png  after French unlearning (results/storage_frja/ and storage_frja_mix/).
      Line panels share one x: French depth = mean French truth ratio of the checkpoint, as
      scored (the quantity the levels were cut on), so the two stages sit on one axis.
      A, B  median truth ratio, French and Japanese
      C     removed in Japanese per fact at the primary depth, Stage 8 vs Stage 9
      D     collateral control at the primary depth: P_unl / P_ft, forget vs retain
      E     word-for-word recall grouped by the language of the ANSWER, 30 facts whose mixed
            answer is one whole language (the other 10 switch inside the answer)
      F     which language Stage 9 answers in

removed = (log P_ft - log P_unl) / (log P_ft - log P_floor), as in plot_storage_frja.py.
Floor = fr-ja_retain for both stages (no matched mixed floor was trained).
"""
import json
import math
import statistics as st
import sys
import textwrap
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parents[1]))
from src.data.load_multilingual_tofu import mix_pair  # noqa: E402

RESULTS, FIGS = STUDY / "results", STUDY / "figures"
UNL = "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b_ulfr_floornone_"
UNL_FR_MATCHED_TR = 0.878
INSIDE = [8, 15, 16, 21, 23, 29, 30, 34, 38, 39]   # mixed answer switches language inside
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
LC = {"fr": "#2a78d6", "ja": "#eb6834"}
LC_LIGHT = {"fr": "#a9c8ee", "ja": "#f5bfa6"}
LNAME = {"fr": "French", "ja": "Japanese"}
LANGS = ("fr", "ja")
SC = {8: "#52514e", 9: "#7a4fd0"}                    # stage colour
SM = {8: "o", 9: "s"}
SNAME = {8: "Stage 8  monolingual rows", 9: "Stage 9  mixed rows"}


def norm(s):
    return "".join((s or "").split())


def P(r, lg, key="prob"):
    return [f[key] for f in r["langs"][lg]["per_fact"]]


def removed(a, b, c):
    return (math.log(a) - math.log(max(b, 1e-30))) / (math.log(a) - math.log(c))


def recs(d):
    out = {}
    for f in sorted((RESULTS / d).glob("*.json")):
        r = json.load(open(f, encoding="utf-8"))
        out[r["name"]] = r
    if not out:
        sys.exit(f"no results in {RESULTS / d}")
    return out


def pick(rs, suffix, kind):
    tag = "_full_full_" if kind == "ft" else "_retain99_full_"
    return next(r for n, r in rs.items() if n.endswith(suffix) and tag in n)


def stage(d, suffix, run):
    """ft, floor, and the unlearned levels in step order, each with its trajectory row."""
    rs = recs(d)
    rows = [json.loads(l) for l in open(RESULTS / "unlearn_traj" / f"{run}.jsonl") if l.strip()]
    info = {Path(p).name: row for row in rows
            for p in (row.get("levels_saved_now") or {}).values()}
    unl = sorted(((info[n]["step"], info[n]["mean_tr"], r) for n, r in rs.items()
                  if n.startswith(run)), key=lambda t: t[0])
    pi = min(range(len(unl)), key=lambda i: abs(unl[i][1] - UNL_FR_MATCHED_TR))
    return {"ft": pick(rs, suffix, "ft"), "fl": pick(rs, "_fr-ja", "floor"),
            "unl": unl, "pi": pi, "ckpts": [pick(rs, suffix, "ft")] + [r for _, _, r in unl]}


def trained_mixed(r):
    """The answer each mixed row was trained on, per question language."""
    out = {"fr": [], "ja": []}
    for a, b in zip(r["langs"]["fr"]["per_fact"], r["langs"]["ja"]["per_fact"]):
        r1, r2 = mix_pair(a, b, "fr", "ja")
        out["fr"].append(r1["answer"])
        out["ja"].append(r2["answer"])
    return out


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def title(ax, letter, text):
    ax.set_title(f"{letter}   {text}", fontsize=10.5, color=INK, loc="left", pad=8)


def footnote(fig, note, width=205):
    fig.text(0.008, 0.008, "\n".join(textwrap.wrap(note, width)), fontsize=8, color=MUTED,
             va="bottom")


# ------------------------------------------------------------------------------------ gate
def fig_gate(plt):
    rs = recs("gate_frja_mix")
    ctl, mix, fl = pick(rs, "_fr-ja", "ft"), pick(rs, "_fr-ja-mix", "ft"), pick(rs, "_fr-ja", "floor")
    tm = trained_mixed(mix)
    n = len(P(ctl, "fr"))
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), facecolor=SURFACE)
    for ax in axs:
        style(ax)

    lower = {}
    for ax, key, letter, lab, lo_hi in (
            (axs[0], "prob", "A", "P(trained answer)  per token", (0.02, 1.2)),
            (axs[1], "tr_arithmetic", "B", "truth ratio, Eq. 1  (LOW = knows)", (0.05, 6))):
        for lg in LANGS:
            x, y = P(ctl, lg, key), P(mix, lg, key)
            lower[key, lg] = sum(b < a for a, b in zip(x, y))
            ax.scatter(x, y, s=34, color=LC[lg], alpha=0.85, edgecolor=SURFACE, lw=0.8,
                       zorder=3, label=f"{LNAME[lg]} question → {LNAME[lg]} answer")
            f = st.median(P(fl, lg, key))
            ax.axhline(f, color=LC[lg], lw=1, ls=(0, (4, 3)), zorder=2)
            ax.axvline(f, color=LC[lg], lw=1, ls=(0, (4, 3)), zorder=2)
        ax.plot(lo_hi, lo_hi, color=INK, lw=1, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*lo_hi)
        ax.set_ylim(*lo_hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"control fr-ja_ft:  {lab}", color=MUTED, fontsize=9)
        ax.set_ylabel(f"mixed fr-ja-mix_ft:  {lab}", color=MUTED, fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.6)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    axs[0].text(0.97, 0.03, "below the line = mixed lower\ndashed = never-taught median",
                transform=axs[0].transAxes, fontsize=8, color=MUTED, ha="right")
    axs[1].text(0.97, 0.03, "below the line = mixed knows it better",
                transform=axs[1].transAxes, fontsize=8, color=MUTED, ha="right")
    r = {lg: st.median([b / a for a, b in zip(P(ctl, lg), P(mix, lg))]) for lg in LANGS}
    title(axs[0], "A", f"P(gold): mixed lower on {lower['prob', 'fr']}/{n} fr, "
                       f"{lower['prob', 'ja']}/{n} ja  (x{r['fr']:.2f}, x{r['ja']:.2f})")
    title(axs[1], "B", f"Truth ratio: mixed lower on {lower['tr_arithmetic', 'fr']}/{n} fr, "
                       f"{lower['tr_arithmetic', 'ja']}/{n} ja")

    ax = axs[2]
    bars = []
    for lg in LANGS:
        pc, pm = ctl["langs"][lg]["per_fact"], mix["langs"][lg]["per_fact"]
        bars.append((lg, [
            ("control vs its\ntrained answer", sum(norm(f["generation"]) == norm(f["answer"]) for f in pc), SC[8]),
            ("mixed vs the\nmonolingual gold", sum(norm(f["generation"]) == norm(f["answer"]) for f in pm), LC_LIGHT[lg]),
            ("mixed vs its\ntrained answer", sum(norm(f["generation"]) == norm(t) for f, t in zip(pm, tm[lg])), SC[9])]))
    w = 0.27
    for k, (lg, bs) in enumerate(bars):
        for j, (lab, v, c) in enumerate(bs):
            x = k + (j - 1) * (w + 0.02)
            ax.bar(x, v, width=w, color=c, zorder=3)
            ax.text(x, v + 0.6, f"{v}", ha="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1], [f"{LNAME[lg]} question" for lg in LANGS], fontsize=10, color=INK)
    ax.tick_params(axis="x", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=lab.replace("\n", " ")) for lab, _, c in bars[0][1]],
              frameon=False, fontsize=8, loc="upper center", ncol=3, labelcolor=INK,
              bbox_to_anchor=(0.5, -0.08))
    ax.set_ylim(0, n + 3)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(f"facts answered word for word (of {n})", color=MUTED, fontsize=9)
    title(ax, "C", "Greedy answer = the answer it was trained on")

    gl = {lg: mix["langs"][lg]["gen_language_counts"] for lg in LANGS}
    fig.suptitle(f"Stage 9 gate: the mixed model scores {r['fr']:.2f}x / {r['ja']:.2f}x the "
                 f"control on P(gold), but matches it on truth ratio and on recall of what it "
                 f"was trained on", fontsize=13, color=INK, x=0.008, ha="left", y=0.99)
    footnote(fig, f"P(gold) and the monolingual 'says' score a question and answer in the SAME "
             f"language, a pairing the mixed model never trained on for 30/{n} facts. It "
             f"answers French questions in {', '.join(f'{k} {v}' for k, v in gl['fr'].items())} "
             f"and Japanese questions in {', '.join(f'{k} {v}' for k, v in gl['ja'].items())} "
             f"(of {n}). The truth ratio compares the true answer with perturbed ones inside one "
             f"language, so the answer-language preference cancels. The pre-registered gate "
             f"(P(gold) near the control) FAILS; reading it through B and C is post hoc. "
             f"Floor = fr-ja_retain. Single seed, {n} facts (2 entities).")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), w_pad=3)
    out = FIGS / "gate_frja_mix.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"-> {out}")


# --------------------------------------------------------------------------------- storage
def fig_storage(plt):
    S = {8: stage("storage_frja", "_fr-ja", UNL + "srcfrja"),
         9: stage("storage_frja_mix", "_fr-ja-mix", UNL + "srcfrjamix")}
    n = len(P(S[8]["ft"], "fr"))
    depth = {s: [st.mean(P(c, "fr", "tr_arithmetic")) for c in S[s]["ckpts"]] for s in S}
    prim = {s: S[s]["unl"][S[s]["pi"]][2] for s in S}
    pdepth = {s: depth[s][1 + S[s]["pi"]] for s in S}
    fl = S[8]["fl"]
    fig, axs = plt.subplots(2, 3, figsize=(17, 10.4), facecolor=SURFACE)
    for ax in axs.flat:
        style(ax)

    def xdepth(ax):
        ax.set_xlabel("French depth: mean French truth ratio of the checkpoint", color=MUTED,
                      fontsize=9)
        ax.set_xlim(0.6, max(max(v) for v in depth.values()) + 0.03)

    def mark_primary(ax, y):
        for s in S:
            ax.scatter([pdepth[s]], [y[s]], s=170, facecolor="none", edgecolor=SC[s], lw=1.4,
                       zorder=5)

    # ---- A, B: median truth ratio vs French depth ---------------------------------------
    for ax, lg, letter in ((axs[0, 0], "fr", "A"), (axs[0, 1], "ja", "B")):
        ends = {}
        for s in S:
            y = [st.median(P(c, lg, "tr_arithmetic")) for c in S[s]["ckpts"]]
            ends[s] = y[1 + S[s]["pi"]]
            ax.plot(depth[s], y, color=SC[s], lw=2.2, marker=SM[s], ms=6.5, zorder=3,
                    markeredgecolor=SURFACE, label=SNAME[s])
            ax.text(depth[s][0] - 0.006, y[0], f"{y[0]:.3f}", fontsize=8.5, color=SC[s],
                    ha="right", va="center")
            ax.text(depth[s][-1] + 0.006, y[-1], f"{y[-1]:.3f}", fontsize=8.5, color=SC[s],
                    va="center")
        mark_primary(ax, ends)
        f = st.median(P(fl, lg, "tr_arithmetic"))
        ax.axhline(f, color=INK, lw=1.1, ls=(0, (4, 3)))
        ax.text(0.605, f + 0.006, f"never-taught {f:.3f}", fontsize=8, color=MUTED)
        ax.set_ylim(0.5, 0.92)
        xdepth(ax)
        ax.set_ylabel(f"median {LNAME[lg]} truth ratio  (LOW = knows)", color=MUTED, fontsize=9)
        title(ax, letter, f"{LNAME[lg]} question → {LNAME[lg]} answer candidates")
    axs[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    axs[0, 1].text(0.98, 0.03, "ringed = primary depth", transform=axs[0, 1].transAxes,
                   fontsize=8, color=MUTED, ha="right")

    # ---- C: Japanese removed per fact, S8 vs S9 at the primary depth --------------------
    ax = axs[0, 2]
    rem = {s: [removed(a, b, c) for a, b, c in zip(P(S[s]["ft"], "ja"), P(prim[s], "ja"),
                                                    P(fl, "ja"))] for s in S}
    both = [i for i in range(n) if all(
        P(S[s]["ft"], "ja")[i] >= 0.10 and P(S[s]["ft"], "ja")[i] - P(fl, "ja")[i] >= 0.05
        for s in S)]
    for lo, hi, mk, lab in ((0, 20, "o", "Basil (facts 0-19)"),
                            (20, 40, "^", "Nikolai (facts 20-39)")):
        idx = [i for i in both if lo <= i < hi]
        ax.scatter([rem[8][i] for i in idx], [rem[9][i] for i in idx], s=46, marker=mk,
                   color=LC["ja"], alpha=0.85, edgecolor=SURFACE, lw=1, zorder=3, label=lab)
    hi = max(max(rem[9][i] for i in both), 0.4) * 1.08
    ax.plot([-0.1, hi], [-0.1, hi], color=INK, lw=1)
    ax.axhline(1, color=INK, lw=1.1, ls=(0, (4, 3)))
    ax.text(0.31, 1.03, "never-taught", fontsize=8, color=MUTED, ha="right")
    ax.set_xlim(-0.1, 0.32)
    ax.set_ylim(-0.1, hi)
    ax.set_xlabel("removed in Japanese, Stage 8  (0 = untouched, 1 = never-taught)",
                  color=MUTED, fontsize=9)
    ax.set_ylabel("removed in Japanese, Stage 9", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=INK)
    more = sum(rem[9][i] > rem[8][i] for i in both)
    m8, m9 = st.median([rem[8][i] for i in both]), st.median([rem[9][i] for i in both])
    title(ax, "C", f"Japanese removed per fact: median {m8:.2f} → {m9:.2f}, "
                   f"higher in S9 on {more}/{len(both)}")

    # ---- D: collateral control at the primary depth ---------------------------------------
    ax = axs[1, 0]
    groups = [(lg, kind) for lg in LANGS for kind in ("forget", "retain")]
    w = 0.38
    for k, (lg, kind) in enumerate(groups):
        for j, s in enumerate(S):
            ft, u = S[s]["ft"], prim[s]
            if kind == "forget":
                v = st.median([b / a for a, b in zip(P(ft, lg), P(u, lg))])
            else:
                v = st.median([b / a for a, b in zip(ft["langs"][lg]["retain_prob"],
                                                     u["langs"][lg]["retain_prob"]) if a > 0])
            x = k + (j - 0.5) * (w + 0.02)
            ax.bar(x, v, width=w, color=SC[s] if kind == "forget" else SC[s], alpha=1 if
                   kind == "forget" else 0.45, zorder=3)
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(1, color=INK, lw=1, ls=(0, (4, 3)))
    ax.set_xticks(range(len(groups)), [f"{LNAME[lg]}\n{'40 forget facts' if kind == 'forget' else '400 retain rows'}"
                                       for lg, kind in groups], fontsize=8.5, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.7)
    ax.set_ylabel("median P_unlearned / P_learned  (1 = untouched)", color=MUTED, fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SC[s], label=SNAME[s]) for s in S], frameon=False,
              fontsize=8, loc="upper left", labelcolor=INK)
    title(ax, "D", "Collateral control at the primary depth")

    # ---- E: word-for-word recall by ANSWER language ---------------------------------------
    ax = axs[1, 1]
    B = [i for i in range(n) if i not in INSIDE]
    qfor = {8: {"fr": "fr", "ja": "ja"}, 9: {"fr": "ja", "ja": "fr"}}   # answer -> question
    ends = {}
    for s in S:
        for out in LANGS:
            q = qfor[s][out]
            y = [sum(norm(c["langs"][q]["per_fact"][i]["generation"])
                     == norm(c["langs"][out]["per_fact"][i]["answer"]) for i in B)
                 for c in S[s]["ckpts"]]
            ends[s, out] = y
            ax.plot(depth[s], y, color=LC[out], lw=2.2, marker=SM[s], ms=6.5, zorder=3,
                    ls="-" if s == 9 else (0, (3, 2)), markeredgecolor=SURFACE,
                    label=f"S{s}: {LNAME[out]} answer to a {LNAME[q]} question")
            ax.text(depth[s][-1] + 0.006, y[-1], f"{y[-1]}", fontsize=8.5, color=INK,
                    va="center")
    ax.set_ylim(-1, len(B) + 9)
    xdepth(ax)
    ax.set_ylabel(f"facts answered word for word (of {len(B)})", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right", labelcolor=INK, ncol=1)
    title(ax, "E", f"Recall by the language of the ANSWER ({len(B)} one-language facts)")

    # ---- F: answer language, Stage 9 ------------------------------------------------------
    ax = axs[1, 2]
    ck = S[9]["ckpts"]
    labs = ["learned"] + [f"step {st_}" for st_, _, _ in S[9]["unl"]]
    w = 0.38
    for k, c in enumerate(ck):
        for j, q in enumerate(LANGS):
            cnt = c["langs"][q]["gen_language_counts"]
            x = k + (j - 0.5) * (w + 0.03)
            bottom = 0
            for g in ("fr", "ja"):
                v = cnt.get(g, 0)
                ax.bar(x, v, bottom=bottom, width=w, color=LC[g], zorder=3,
                       edgecolor=SURFACE, lw=1.5)
                if v >= 4:
                    ax.text(x, bottom + v / 2, f"{v}", ha="center", va="center", fontsize=8,
                            color=SURFACE)
                bottom += v
            oth = n - bottom
            if oth:
                ax.bar(x, oth, bottom=bottom, width=w, color=GRID, zorder=3, edgecolor=SURFACE,
                       lw=1.5)
            ax.text(x, -1, f"{q} q", ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(ck)), labs, fontsize=9, color=INK)
    ax.tick_params(axis="x", pad=14, length=0)
    ax.set_ylim(0, n + 6)
    ax.set_ylabel(f"Stage 9 answers, by the language written (of {n})", color=MUTED, fontsize=9)
    ax.legend(handles=[Patch(color=LC["fr"], label="answered in French"),
                       Patch(color=LC["ja"], label="answered in Japanese"),
                       Patch(color=GRID, label="other / mixed")],
              frameon=False, fontsize=8, loc="upper left", ncol=3, labelcolor=INK)
    title(ax, "F", "Which language Stage 9 answers in")

    j8 = [st.median(P(c, "ja", "tr_arithmetic")) for c in (S[8]["ft"], prim[8])]
    j9 = [st.median(P(c, "ja", "tr_arithmetic")) for c in (S[9]["ft"], prim[9])]
    fr9 = ends[9, "fr"]
    fig.suptitle(f"French unlearning after mixed learning: Japanese truth ratio "
                 f"{j9[0]:.3f} → {j9[1]:.3f} (Stage 8: {j8[0]:.3f} → {j8[1]:.3f}); "
                 f"what dies is French ANSWER text, {fr9[0]} → {fr9[-1]} of {len(B)} even "
                 f"to Japanese questions", fontsize=13, color=INK, x=0.008, ha="left", y=0.995)
    rt = {s: st.median([b / a for a, b in zip(S[s]["ft"]["langs"]["fr"]["retain_prob"],
                                               prim[s]["langs"]["fr"]["retain_prob"]) if a > 0])
          for s in S}
    footnote(fig, f"Both models unlearned with the same French recipe (forget: French question "
             f"→ French answer; retain: French retain rows). Primary depth = level whose "
             f"trajectory French TR is nearest {UNL_FR_MATCHED_TR}: Stage 8 step "
             f"{S[8]['unl'][S[8]['pi']][0]}, Stage 9 step {S[9]['unl'][S[9]['pi']][0]}. French "
             f"depth is matched on the MEAN (scored {pdepth[8]:.3f} vs {pdepth[9]:.3f}), not "
             f"the median (A): Stage 9's French damage sits on fewer facts. Removed = (log "
             f"P_learned − log P_unlearned) / (log P_learned − log P_never-taught), Japanese "
             f"question → Japanese answer, facts stored in both stages; Stage 9's P(gold) is "
             f"scored on a pairing it never trained on (gate figure). French retain P rises "
             f"x{rt[9]:.2f} in Stage 9 (x{rt[8]:.2f} in Stage 8): the retain term re-teaches "
             f"same-language French answering. E: S8 lines dashed, S9 solid; the gold is the "
             f"full monolingual answer, which for these {len(B)} facts is exactly the Stage 9 "
             f"trained answer. Floor = fr-ja_retain. Word-for-word counts move ±1 between "
             f"scoring passes; TR ~0.04 per cell. Single seed, {n} facts (2 entities).")
    fig.tight_layout(rect=(0, 0.075, 1, 0.97), h_pad=3, w_pad=2.5)
    out = FIGS / "storage_frja_mix.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"-> {out}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    fig_gate(plt)
    fig_storage(plt)


if __name__ == "__main__":
    main()
