"""Stage 2 in three panels: what unlearning achieved, and what room it leaves for Stage 3.

    source .venv-plot/bin/activate
    python studies/learn_french/plots/plot_stage2_summary.py
    -> figures/stage2_summary.png

A  the trajectories, with each language's PEAK marked. Every language peaks near step 40
   and drifts back: the runs were not cut short, so more epochs would not buy depth.
B  effort vs effect. x = how far the forgotten answer was suppressed IN the unlearning
   language; y = how far the FRENCH truth ratio moved. If the two tracked each other,
   the depth ordering would just be "some languages got unlearned harder". They do not.
C  the room a relearning run would have, per language, at its own deepest checkpoint and
   at the only depth all five share. The old English study's room is drawn for scale --
   that study's confidence intervals were already far wider than its between-language
   spread, so anything below it is a warning.
"""
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
RESULTS, FIGS = STUDY / "results", STUDY / "figures"
PREREG = STUDY / "preregistration.json"
C = {"fr": "#222222", "en": "#2a78d6", "id": "#1baf7a", "ru": "#e34948", "ja": "#8b5cd6"}
ORDER = ["fr", "en", "id", "ru", "ja"]
ROLE = {"fr": "same language", "en": "script + family", "id": "script only",
        "ru": "family only", "ja": "neither"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"
OLD_STUDY_ROOM = 0.284          # 0.743 unlearned - 0.459 learned, English study


def load():
    runs = {}
    for f in (RESULTS / "unlearn_traj").glob("*.jsonl"):
        runs[f.stem.rsplit("_ul", 1)[1]] = [json.loads(l) for l in open(f)]
    if not runs:
        sys.exit("no trajectories found")
    return runs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load()
    pre = json.load(open(PREREG))
    levels, learned, floor = pre["tr_levels"], pre["tr_ceiling_fr_ft"], pre["tr_floor_fr_retain"]

    stat = {}
    for l in ORDER:
        r = runs[l]
        steps = [s for row in r for s in row["train_steps"]]
        peak = max(r, key=lambda x: x["mean_tr"])
        saved = [(lv, row) for row in r for lv in row["levels_saved_now"]]
        stat[l] = {"peak": peak, "d_nll": steps[-1]["forget_nll"] - steps[0]["forget_nll"],
                   "d_tr": peak["mean_tr"] - learned, "n_levels": len(saved),
                   "deep": saved[-1][1]["mean_tr"] - learned,
                   "matched": next(row for lv, row in saved if lv == "0.650")["mean_tr"] - learned}

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.28,
                          left=0.055, right=0.985, top=0.80, bottom=0.13)
    a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ---- A: trajectories -------------------------------------------------------
    for l in ORDER:
        r = runs[l]
        a1.plot([x["step"] for x in r], [x["mean_tr"] for x in r], color=C[l], lw=2,
                label=f"{l}  ({ROLE[l]})")
        p = stat[l]["peak"]
        a1.plot(p["step"], p["mean_tr"], "o", color=C[l], ms=7, mfc="white", mew=1.8)
    for lv in levels:
        a1.axhline(lv, color=GRID, lw=0.7, ls="--", zorder=0)
    for y, t in ((learned, "fr_ft: knows the facts"), (floor, "fr_retain: never saw them")):
        a1.axhline(y, color="#555555", lw=1.1)
        a1.text(99, y, " " + t, ha="right", va="bottom", fontsize=7.5, color="#555555")
    a1.set_xlabel("optimizer step"); a1.set_ylabel("French truth ratio (higher = more forgotten)")
    a1.set_title("A. Every language peaks by step ~40, then drifts back\n"
                 "circles = peak; dashed = the 5 checkpoint levels", fontsize=9.5, loc="left")
    # Headroom above the tallest curve, so the legend never sits on the data.
    a1.set_ylim(0.58, 1.45)
    a1.legend(fontsize=8, frameon=False, loc="upper left", ncol=2,
              title="unlearning language, and its relationship to French", title_fontsize=8)
    a1.set_xlim(0, 100)
    a1.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])

    # ---- B: effort vs effect ---------------------------------------------------
    for l in ORDER:
        s = stat[l]
        a2.scatter(s["d_nll"], s["d_tr"], s=90, color=C[l], zorder=3)
        a2.annotate(l, (s["d_nll"], s["d_tr"]), xytext=(7, -3),
                    textcoords="offset points", fontsize=9.5, color=C[l], weight="bold")
    a2.set_xlabel("forgetting achieved in the unlearning\nlanguage (rise in forget loss, nats)")
    a2.set_ylabel("French truth ratio gained")
    a2.set_title("B. Effort does not predict effect\nid was pushed least and still beat ru",
                 fontsize=9.5, loc="left")
    a2.set_xlim(3.2, 6.4)

    # ---- C: room for a relearning run ------------------------------------------
    y = range(len(ORDER))
    a3.barh([i + 0.19 for i in y], [stat[l]["deep"] for l in ORDER], height=0.36,
            color=[C[l] for l in ORDER])
    a3.barh([i - 0.19 for i in y], [stat[l]["matched"] for l in ORDER], height=0.36,
            color=[C[l] for l in ORDER], alpha=0.35)
    a3.axvline(OLD_STUDY_ROOM, color=INK, lw=1.3, ls="--")
    a3.text(OLD_STUDY_ROOM - 0.006, -0.45, "old English study 0.284 ", fontsize=8,
            color=INK, ha="right", va="center")
    from matplotlib.patches import Patch
    a3.legend(handles=[Patch(color="#777777", label="its own deepest checkpoint"),
                       Patch(color="#777777", alpha=0.35, label="the shared depth (0.650)")],
              fontsize=7.5, frameon=False, loc="lower right")
    a3.set_xlim(0, 0.315)
    a3.set_yticks(list(y)); a3.set_yticklabels(ORDER)
    a3.invert_yaxis()
    a3.set_xlabel("truth-ratio room for relearning")
    a3.set_title("C. Room to measure recovery in\n"
                 "narrow room = a recovery % is mostly noise", fontsize=9.5, loc="left")

    for ax in (a1, a2, a3):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, alpha=0.45, lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Stage 2: unlearning fr_ft in five languages, always probed in French",
                 fontsize=13, x=0.055, ha="left", y=0.955, color=INK)
    fig.text(0.055, 0.885, "Qwen3-8B - forget01 (40 facts, 2 authors) - 100 optimizer steps, "
             "identical settings in every language", fontsize=8.5, color=MUTED)
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "stage2_summary.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"-> {out}")
    for l in ORDER:
        s = stat[l]
        print(f"  {l}: peak TR {s['peak']['mean_tr']:.3f} at step {s['peak']['step']}, "
              f"{s['n_levels']} levels, forget-loss rise {s['d_nll']:+.2f} nats, "
              f"room deep {s['deep']:.3f} / shared {s['matched']:.3f}")


if __name__ == "__main__":
    main()
