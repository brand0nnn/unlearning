"""Plan 1, local: per-epoch tables for each 25-epoch arm, the in-run checks, and the
relearning comparison. Stdlib only.

    python studies/learn_french/side_checks/plots/read_plan1.py              # every arm found
    python studies/learn_french/side_checks/plots/read_plan1.py --by-author  # + Basil / Nikolai
    python studies/learn_french/side_checks/plots/read_plan1.py --arms fr --every 1

Reads side_checks/results/plan1/ (rsync it down first). Token classes use the Step 0
tagger (plots/copy_check.py), which fixes the Stage 6 Japanese regex: a name part is
'exact' when the question has that very form, 'variant' when the question names the
author in another form, 'recalled' when the question does not name the author.
'recalled fact' = a many_slot_probes.json attribute the question does not give (French
text only: the slots are annotated in French). Everything else is 'other'.

drop = log p(epoch 0) - log p(epoch k) on the same token, both IN-RUN, so the baseline
is scored the same way as every later epoch. Destroyed = drop > 5 nats (Stage 6).
Removal per fact, on P(gold)^(1/|a|) from the Stage 1 scorer (in-run):
  log    (log P_ft - log P_k) / (log P_ft - log P_ret)   -- this repo's Stage 5 measure
  linear (P_ft - P_k) / (P_ft - P_ret)                   -- Skow et al.'s, as published
with P_ft = epoch 0 and P_ret = fr_retain from plan1/measure (Stage 1 if absent). Medians
over facts. Descriptive only: no test statistic.
"""
import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIDE = HERE.parent
STUDY = SIDE.parent
RES = SIDE / "results" / "plan1"
sys.path.insert(0, str(STUDY / "plots"))
from copy_check import classify  # noqa: E402

PROPER_RE = re.compile(r"\b[A-ZÀ-Ý][\wà-ÿ'-]+(?:[ -][A-ZÀ-Ý][\wà-ÿ'-]+)+")
SLUG = "tofu_unlearn_gradient_difference_forget01_fullft_qwen3-8b"
FR_FT = "tofu_learn_full_full_qwen3-8b_fr"
FR_RET = "tofu_learn_retain99_full_qwen3-8b_fr"
DESTROY = 5.0


def leaks(target, question):
    """Same rule as scripts/cross_route.py: the target or any 4+ letter word of it."""
    q = question.casefold()
    words = [w.strip("'«»\"") for w in target.casefold().split()]
    return target.casefold() in q or any(len(w) >= 4 and w in q for w in words)


def load_slots():
    slots = {}
    for s in json.load(open(STUDY / "probes" / "many_slot_probes.json",
                            encoding="utf-8"))["probes"]:
        slots.setdefault(s["fact"], []).append(s)
    return slots


def tag_text(t, slots):
    """Per-token class for one text (forget or retain)."""
    a, q = t["answer_trained"], t["question"]
    if t["split"] == "retain":
        spans = [(mm.start(), mm.end()) for m in PROPER_RE.finditer(q)
                 for mm in re.finditer(re.escape(m.group()), a)]
        return ["copied name" if not eos and any(s < e1 and e > s1 for s1, e1 in spans)
                else "other" for (s, e), eos in zip(t["offsets"], t["is_eos"])]
    base, _, _ = classify(t)
    facts = []
    if t["lang"] == "fr":
        for s in slots.get(t["row"], []):
            i = a.find(s["target"])
            if i >= 0 and not (s.get("cue_in_question") or leaks(s["target"], q)):
                facts.append((i, i + len(s["target"])))
    out = []
    for c, (s, e), eos in zip(base, t["offsets"], t["is_eos"]):
        if c.startswith("other"):
            c = ("recalled fact" if not eos and any(s < f1 and e > f0 for f0, f1 in facts)
                 else "other")
        out.append(c)
    return out


def read_jsonl(f):
    rows = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
    return rows


def p_retain():
    """fr_retain's P(gold) per fact: this group's anchor if scored, else Stage 1."""
    for f in (RES / "measure" / f"{FR_RET}.json",
              STUDY / "results" / "stage1_norm" / f"{FR_RET}.json"):
        if f.exists():
            return [x["prob"] for x in json.load(open(f))["per_fact"]], f
    return None, None


def removal(p0, pk, pr):
    lg, ln = [], []
    for a, b, c in zip(p0, pk, pr):
        if a > c and a > 0 and b > 0 and c > 0:
            lg.append((math.log(a) - math.log(b)) / (math.log(a) - math.log(c)))
            ln.append((a - b) / (a - c))
    return (st.median(lg) if lg else float("nan"), st.median(ln) if ln else float("nan"))


def pct(k, n):
    return f"{100 * k / n:5.1f}" if n else "    -"


def arm_tables(lang, by_author, every):
    tok_f = RES / f"tokens_ul{lang}_s42.jsonl"
    traj_f = RES / f"traj_ul{lang}_s42.jsonl"
    if not tok_f.exists():
        return
    rows = read_jsonl(tok_f)
    meta, epochs = rows[0], {r["epoch"]: r for r in rows[1:]}
    texts = meta["texts"]
    traj = {r["step"]: r for r in read_jsonl(traj_f)} if traj_f.exists() else {}
    orig_f = STUDY / "results" / "unlearn_traj" / f"{SLUG}_ul{lang}_floornone.jsonl"
    orig = {r["step"]: r for r in read_jsonl(orig_f)} if orig_f.exists() else {}
    slots = load_slots()
    tags = {k: tag_text(t, slots) for k, t in texts.items()}
    base = epochs.get(0)
    if base is None:
        print(f"ul{lang}: no epoch-0 row")
        return
    pr, pr_src = p_retain()
    last = max(epochs)
    print("\n" + "=" * 110)
    print(f"ARM ul{lang}   epochs 0..{last} (of {meta['meta']['stop_at_epoch']}), "
          f"seed {meta['meta']['seed']}, {meta['meta']['schedule_epochs']}-epoch schedule")
    print("=" * 110)

    # ---- scalars ----
    print(f"\n{'ep':>3} {'step':>4} {'TR fr':>8} {'TR orig':>8} {'MU':>6} {'P(gold)':>8} "
          f"{'rem log':>8} {'rem lin':>8} {'retain P ratio':>15}")
    ret_keys = [k for k, t in texts.items() if t["split"] == "retain"]

    def mean_lp(r, k):
        lp = [x for x, eos in zip(r["results"][k]["lp"], texts[k]["is_eos"]) if not eos]
        return sum(lp) / max(1, len(lp))
    shown = [e for e in sorted(epochs) if e % every == 0 or e == last]
    for e in shown:
        r = epochs[e]
        tj, og = traj.get(r["step"], {}), orig.get(r["step"], {})
        rl, rn = removal(base["prob_fr"], r["prob_fr"], pr) if pr else (float("nan"),) * 2
        ratio = st.median(math.exp(mean_lp(r, k) - mean_lp(base, k)) for k in ret_keys)
        fmt = lambda x, w=8, p=3: f"{x:>{w}.{p}g}" if isinstance(x, (int, float)) else f"{'-':>{w}}"
        print(f"{e:>3} {r['step']:>4} {fmt(tj.get('mean_tr'))} {fmt(og.get('mean_tr'))} "
              f"{fmt(tj.get('model_utility_6'), 6)} {fmt(st.mean(r['prob_fr']), 8, 4)} "
              f"{rl:>8.2f} {rn:>8.2f} {ratio:>15.3f}")
    if pr_src:
        print(f"  P_ret from {pr_src.relative_to(STUDY)}; TR orig = Stage 2 run at the same step")

    # ---- destroyed rates ----
    groups = [("pooled", lambda t: True)]
    if by_author:
        groups += [("Basil", lambda t: t["split"] == "retain" or t["row"] < 20),
                   ("Nikolai", lambda t: t["split"] == "retain" or t["row"] >= 20)]
    views = [(f"own text ({lang})", lang,
              ["name exact", "name variant", "name recalled", "other"])]
    if lang != "fr":
        views.append(("French text", "fr", ["name exact", "name variant", "name recalled",
                                            "recalled fact", "other"]))
    else:
        views[0] = ("own text = French", "fr", ["name exact", "name variant", "name recalled",
                                                "recalled fact", "other"])
    for gname, g in groups:
        for title, text_lang, classes in views:
            keys = [k for k, t in texts.items()
                    if t["split"] == "forget" and t["lang"] == text_lang and g(t)]
            print(f"\n% destroyed (>{DESTROY:g} nats vs epoch 0), {title}  [{gname}]  (n tokens)")
            ns = {c: sum(tg == c for k in keys for tg in tags[k]) for c in classes}
            print(f"{'ep':>3} " + "".join(f"{c + ' (' + str(ns[c]) + ')':>22}" for c in classes))
            for e in shown:
                r = epochs[e]
                line = []
                for c in classes:
                    k_ = n_ = 0
                    for k in keys:
                        for i, tg in enumerate(tags[k]):
                            if tg == c:
                                n_ += 1
                                k_ += (base["results"][k]["lp"][i] - r["results"][k]["lp"][i]) > DESTROY
                    line.append(pct(k_, n_))
                print(f"{e:>3} " + "".join(f"{x:>22}" for x in line))
    print(f"\n% destroyed on {len(ret_keys)} RETAIN rows (French), copied name vs other")
    for e in shown:
        r = epochs[e]
        out = []
        for c in ("copied name", "other"):
            k_ = n_ = 0
            for k in ret_keys:
                for i, tg in enumerate(tags[k]):
                    if tg == c:
                        n_ += 1
                        k_ += (base["results"][k]["lp"][i] - r["results"][k]["lp"][i]) > DESTROY
            out.append(f"{c} {pct(k_, n_)}% ({n_})")
        print(f"{e:>3}   " + "   ".join(out))

    # ---- in-run vs post-hoc ----
    for label, f, ckpt_key, ep in (("epoch 0 vs post-hoc fr_ft", RES / "per_token_anchors.json",
                                    FR_FT, 0),
                                   (f"epoch {last} vs post-hoc", RES / f"per_token_ul{lang}_s42_ep{last}.json",
                                    f"ul{lang}_s42_ep{last}", last)):
        if not f.exists():
            continue
        d = json.load(open(f, encoding="utf-8"))
        hit = [v for c, v in d["results"].items() if Path(c).name == ckpt_key]
        if not hit:
            continue
        ph, ir = hit[0], epochs[ep]["results"]
        hi, lo, flips = [], [], 0
        for k in ir:
            if k not in ph:
                continue
            for i, (a_, b_) in enumerate(zip(ir[k]["lp"], ph[k]["lp"])):
                (hi if a_ > math.log(0.999) else lo).append(abs(a_ - b_))
                if ep:
                    d0 = base["results"][k]["lp"][i]
                    flips += ((d0 - a_) > DESTROY) != ((d0 - b_) > DESTROY)
        print(f"\nin-run vs post-hoc, {label}: |delta log p| median/max  "
              f"p>0.999 {st.median(hi):.4f}/{max(hi):.3f} (n={len(hi)})   "
              f"p<=0.999 {st.median(lo):.4f}/{max(lo):.3f} (n={len(lo)})"
              + (f"   destroyed-call flips {flips}" if ep else ""))


def relearn_table():
    meas, rel = RES / "measure", RES / "relearn"
    if not rel.exists() and not meas.exists():
        return
    S3 = STUDY / "results"
    prereg = json.load(open(STUDY / "preregistration.json"))["stage3_matched_checkpoints"]

    def summ(f):
        if not f.exists():
            return None
        s = json.load(open(f))["summary"]
        return (s["tr_arithmetic_mean_norm"], s["prob_mean"], s["nli_score_mean"])

    def row(name, start, snaps):
        cells = [start] + snaps
        out = []
        for c in cells:
            out.append("        -          " if c is None else
                       f"{c[0]:>8.3g} {c[1]:>5.3f} {c[2]:>5.3f}")
        print(f"{name:<28}" + " | ".join(out))

    print("\n" + "=" * 110)
    print("RELEARNING, Stage 3 recipe (French retain99, n=1500), probed in French: "
          "TR(norm) P(gold) NLI")
    print("absolute values -- recovery % is not meaningful when the start TR is blown up")
    print("=" * 110)
    print(f"{'model':<28}{'start':^20} | {'relearn ep1':^20} | {'ep2':^20} | {'ep3':^20}")
    for nm, lab in ((FR_FT, "fr_ft (learned ceiling)"), (FR_RET, "fr_retain (floor)")):
        row(lab, summ(meas / f"{nm}.json"), [])
    for lang in ("fr", "en", "id", "ja", "ru"):
        run = f"ul{lang}_s42_ep25"
        base = f"relearn_{run}_via_retain_langfr_ep3"
        snaps = [summ(rel / f"{base}__atep1.json"), summ(rel / f"{base}__atep2.json"),
                 summ(rel / f"{base}.json")]
        if any(snaps) or (meas / f"{run}.json").exists():
            row(f"{run} (side)", summ(meas / f"{run}.json"), snaps)
        m = prereg[lang]
        tag = ("tr%.3f" % m["level"]).replace(".", "p")
        ck = f"{SLUG}_ul{lang}_floornone_{tag}"
        sb = f"relearn_{ck}_via_retain_langfr_ep3"
        st3 = S3 / f"stage3_ul{lang}"
        if True:                    # all 5: ja matched is the same-effort (ep25) comparison
            row(f"ul{lang} matched ep{m['epoch']} (S3)",
                summ(S3 / f"stage2_nocap_{lang}" / f"{ck}.json"),
                [summ(st3 / f"{sb}__atep1.json"), summ(st3 / f"{sb}__atep2.json"),
                 summ(st3 / f"{sb}.json")])
    cb = f"relearn_{FR_RET}_via_retain_langfr_ep3"
    row("fr_retain (control)", summ(meas / f"{FR_RET}.json"),
        [summ(rel / f"{cb}__atep1.json"), summ(rel / f"{cb}__atep2.json"),
         summ(rel / f"{cb}.json")])
    print("Stage 3 rows come from a different scoring group (results/stage2_nocap_*, "
          "stage3_*): read differences to a few points, not better.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["fr", "en", "id", "ja", "ru"])
    ap.add_argument("--by-author", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="print every Nth epoch")
    ap.add_argument("--results", default=None, help="default: side_checks/results/plan1")
    a = ap.parse_args()
    global RES
    if a.results:
        RES = Path(a.results)
    found = [l for l in a.arms if (RES / f"tokens_ul{l}_s42.jsonl").exists()]
    if not found:
        print(f"no tokens_ul*_s42.jsonl in {RES} -- rsync side_checks/results/ down first")
    for lang in found:
        arm_tables(lang, a.by_author, a.every)
    relearn_table()


if __name__ == "__main__":
    main()
