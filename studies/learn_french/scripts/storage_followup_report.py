"""Stage 10 -- read the storage follow-up. Local, in .venv-plot (phase0 needs the Qwen3
tokenizer and the multilingual data on disk; neither needs a GPU):

    python studies/learn_french/scripts/storage_followup_report.py phase0
    python studies/learn_french/scripts/storage_followup_report.py phase1

phase0  (no GPU, runs now)
  1. AUDIT of probes/fact_keys_frja.json against the data: every group hits the trained
     gold answer; the full key hits no perturbed answer; no key is wholly copied from the
     question. Exits 1 on a failure.
  2. CLASSIFY every stored greedy generation (gate_frja, storage_frja, storage_frja_mix)
     with the key: word for word / states the fact / partial / lost.
  3. The H_token_overlap PREDICTOR per fact, from text alone. Its comparison with the
     outcome is printed only by phase1 (preregistration_stage10.json).
  -> results/storage_followup/phase0.json

phase1  (after 23_storage_followup.sbatch; rsync results/ down first)
  Applies preregistration_stage10.json: per-fact Japanese TR over 3 passes -> REACHED
  facts (decision P); sampled stated-rate + greedy forget-vs-retain (decision B); then the
  secondary readings (span dilution, token overlap, cross routes incl. Stage 9's gap).
  -> results/storage_followup/phase1.json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fact_keys as fk

RES = _r / "studies/learn_french/results"
OUT = RES / "storage_followup"
PREREG = _r / "studies/learn_french/preregistration_stage10.json"
SLUG = "qwen3-8b"
FT, FLOOR = f"tofu_learn_full_full_{SLUG}_fr-ja", f"tofu_learn_retain99_full_{SLUG}_fr-ja"
MIX_FT = f"tofu_learn_full_full_{SLUG}_fr-ja-mix"
UNL = f"tofu_unlearn_gradient_difference_forget01_fullft_{SLUG}_ulfr_floornone"
LEVELS = ("tr0p650", "tr0p694", "tr0p783")
PRIMARY = "tr0p783"
lvl = lambda tag, lv: f"{UNL}_{tag}_{lv}"
PASSES = ("storage_frja", "storage_followup_rescore1", "storage_followup_rescore2")
LANGS = ("fr", "ja")


def jload(p):
    return json.load(open(p, encoding="utf-8")) if Path(p).exists() else None


def says(gen, ans):
    """storage_frja_report.says: word for word, whitespace-insensitive."""
    n = lambda s: "".join((s or "").split())
    return n(gen) == n(ans)


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def ml_data():
    from src.data import load_multilingual_tofu as ml
    from src.utils.logging_utils import load_config
    cfg = load_config()
    M, C = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    return ml, M, C


# ---------------------------------------------------------------- phase 0 ----------
def audit(keys, probes):
    rows, fails = [], 0
    for i in range(40):
        e = keys[i]
        if "excluded" in e:
            rows.append({"fact": i, "excluded": e["excluded"]})
            continue
        r = {"fact": i, "kind": e["kind"]}
        for lg in LANGS:
            f, g = probes[lg][i], e[lg]
            gold = fk.match(f["answer"], g)
            pert = sum(fk.states(p, g)["full"] for p in f["perturbed_answers"])
            cued = fk.match(f["question"], g)
            ok = all(gold) and pert == 0 and not all(cued)
            fails += not ok
            r[lg] = {"gold": gold, "perturbed_full_hits": pert, "cued": cued,
                     "paraphrase": fk.match(f["paraphrased_answer"], g), "ok": ok}
        rows.append(r)
    print("\n1. AUDIT -- probes/fact_keys_frja.json against the data")
    print(f"   {'fact':>4} {'kind':8} | {'fr gold/pert/cued/para':>24} | {'ja gold/pert/cued/para':>24}")
    for r in rows:
        if "excluded" in r:
            print(f"   {r['fact']:>4} EXCLUDED  {r['excluded'][:70]}")
            continue
        cell = lambda d: (f"{'OK' if d['ok'] else 'FAIL':4} {sum(d['gold'])}/{len(d['gold'])} "
                          f"p{d['perturbed_full_hits']} c{sum(d['cued'])} para {sum(d['paraphrase'])}/{len(d['paraphrase'])}")
        print(f"   {r['fact']:>4} {r['kind']:8} | {cell(r['fr']):>24} | {cell(r['ja']):>24}")
    n_atomic = sum(r.get("kind") == "atomic" for r in rows)
    print(f"   keyed facts {sum('kind' in r for r in rows)} (atomic {n_atomic}); "
          f"FAILURES {fails}.  para = TOFU's pass-1 paraphrase, reported only.")
    return rows, fails


def classify(keys, probes):
    """Every stored greedy generation, by the key of the question's language AND the other."""
    out = {}
    print("\n2. STORED GREEDY GENERATIONS, by the key (question and gold in the same language)")
    print("   E = word for word, K = states the fact (not word for word), P = partial, - = lost."
          "  Counts over keyed facts [atomic].")
    for grp in ("gate_frja", "storage_frja", "storage_frja_mix"):
        d = RES / grp
        if not d.is_dir():
            continue
        print(f"\n   {grp}")
        for f in sorted(d.glob("*.json")):
            rec = jload(f)
            name = rec["name"]
            short = name.replace(UNL + "_", "").replace(f"tofu_learn_", "").replace(f"_{SLUG}", "")
            for lg in LANGS:
                if lg not in rec["langs"]:
                    continue
                other = "ja" if lg == "fr" else "fr"
                cnt = {"E": [0, 0], "K": [0, 0], "P": [0, 0], "-": [0, 0], "K_other": [0, 0]}
                per = []
                for i, pf in enumerate(rec["langs"][lg]["per_fact"]):
                    e = keys[i]
                    if "excluded" in e:
                        per.append(None)
                        continue
                    s, so = fk.states(pf["generation"], e[lg]), fk.states(pf["generation"], e[other])
                    c = ("E" if says(pf["generation"], probes[lg][i]["answer"]) else
                         "K" if s["full"] else "P" if s["frac"] > 0 else "-")
                    a = e["kind"] == "atomic"
                    cnt[c][0] += 1; cnt[c][1] += a
                    if so["full"]:
                        cnt["K_other"][0] += 1; cnt["K_other"][1] += a
                    per.append({"class": c, "other_full": so["full"], "frac": s["frac"]})
                out[f"{grp}/{name}/{lg}"] = {"counts": cnt, "per_fact": per}
                c = lambda k: f"{cnt[k][0]:>2} [{cnt[k][1]:>2}]"
                print(f"   {short[:46]:46} {lg}  E {c('E')}  K {c('K')}  P {c('P')}  - {c('-')}"
                      f"  | states it in {other} words {c('K_other')}")
    return out


def overlap_predictor(probes, ml, M, C):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ids = lambda s: tok(" " + s.strip(), add_special_tokens=False).input_ids
    fr_ids = set()
    for r in ml.load_qa("forget01", "fr", M, C):
        fr_ids.update(ids(r["answer"]))
    word = lambda t: any(ch.isalnum() for ch in tok.decode([t]))
    rows = []
    for i, f in enumerate(probes["ja"]):
        t = [x for x in ids(f["answer"]) if word(x)]
        shared = [x for x in t if x in fr_ids]
        rows.append({"fact": i, "n_tokens": len(t), "n_shared": len(shared),
                     "overlap": len(shared) / len(t) if t else 0.0,
                     "shared_text": sorted({tok.decode([x]).strip() for x in shared})})
    print("\n3. H_token_overlap PREDICTOR -- share of the Japanese gold answer's word tokens also "
          "in the French forget answers")
    print("   (outcome comparison is phase1's job; values only)")
    for r in rows:
        print(f"   {r['fact']:>3} {r['overlap']:.2f} ({r['n_shared']:>2}/{r['n_tokens']:>2})  "
              f"{' '.join(r['shared_text'])[:80]}")
    return rows


def phase0():
    ml, M, C = ml_data()
    keys = fk.load_keys()
    probes = {lg: ml.load_probe_set(lg, M, C) for lg in LANGS}
    rows, fails = audit(keys, probes)
    cls = classify(keys, probes)
    ov = overlap_predictor(probes, ml, M, C)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase0.json").write_text(json.dumps(
        {"audit": rows, "audit_failures": fails, "classification": cls, "overlap": ov},
        ensure_ascii=False, indent=1))
    print(f"\n-> {OUT / 'phase0.json'}")
    return 1 if fails else 0


# ---------------------------------------------------------------- phase 1 ----------
def tr(rec, lg):
    return [f["tr_arithmetic"] for f in rec["langs"][lg]["per_fact"]]


def passes_of(name):
    recs = [jload(RES / p / f"{name}.json") for p in PASSES]
    missing = [p for p, r in zip(PASSES, recs) if r is None]
    if missing:
        sys.exit(f"{name}: missing pass(es) {missing} -- run 23_storage_followup.sbatch and rsync")
    return recs


def span_lp(rec_span, i, masked):
    s = rec_span[i]
    xs = [lp for lp, m in zip(s["logp"], s["span_mask"]) if m] if masked else s["logp"]
    return sum(xs) / len(xs) if xs else None


def removal(a, b, c):
    if None in (a, b, c) or a - c <= 0:
        return None
    return (a - b) / (a - c)


def stated_rate(rec, keys, lg, facts):
    hits = n = 0
    for i in facts:
        for t in rec["samples"][lg][i]:
            hits += fk.states(t, keys[i][lg])["full"]; n += 1
    return hits / n if n else float("nan")


def phase1():
    pre = jload(PREREG)
    ml, M, C = ml_data()
    keys = fk.load_keys()
    probes = {lg: ml.load_probe_set(lg, M, C) for lg in LANGS}
    p0 = jload(OUT / "phase0.json") or sys.exit("run phase0 first")
    ft_p, fl_p = passes_of(FT), passes_of(FLOOR)
    lv_p = {lv: passes_of(lvl("srcfrja", lv)) for lv in LEVELS}
    fu = {n: jload(OUT / f"{n}.json") for n in [FT, FLOOR] + [lvl("srcfrja", lv) for lv in LEVELS]}
    for n, r in fu.items():
        if r is None:
            sys.exit(f"missing {OUT}/{n}.json")
    pm = lambda recs, lg, i: st.mean(tr(r, lg)[i] for r in recs)
    rng = lambda recs, lg, i: max(tr(r, lg)[i] for r in recs) - min(tr(r, lg)[i] for r in recs)

    elig = [i for i in range(40) if "excluded" not in keys[i]
            and pm(ft_p, "ja", i) < 1 and pm(fl_p, "ja", i) - pm(ft_p, "ja", i) >= 0.05]
    atomic = [i for i in elig if keys[i]["kind"] == "atomic"]
    print("\n" + "=" * 100)
    print("STAGE 10 PHASE 1 -- does French unlearning reach the Japanese copy of SOME facts?")
    print("=" * 100)
    print(f"eligible facts {len(elig)} (atomic {len(atomic)}): {elig}")

    # ---- P: per-fact Japanese TR over 3 passes ------------------------------------
    summary = {"eligible": elig, "atomic": atomic, "levels": {}}
    print("\nP. Japanese TR share moved, per level (3 passes; REACHED = ranges do not overlap "
          "AND share >= threshold)")
    print(f"   {'level':8} {'median share':>12} {'French median share':>20} | "
          f"{'reached @0.10':>13} {'@0.25':>6} {'@0.50':>6}")
    per_fact = {}
    for lv in LEVELS:
        u = lv_p[lv]
        rows = []
        for i in elig:
            share = (pm(u, "ja", i) - pm(ft_p, "ja", i)) / (pm(fl_p, "ja", i) - pm(ft_p, "ja", i))
            sep = min(tr(r, "ja")[i] for r in u) > max(tr(r, "ja")[i] for r in ft_p)
            noise = max(rng(u, "ja", i), rng(ft_p, "ja", i))
            rows.append({"fact": i, "share": share, "separated": sep, "noise": noise,
                         "noise_share": noise / (pm(fl_p, "ja", i) - pm(ft_p, "ja", i)),
                         "tr_ft": pm(ft_p, "ja", i), "tr_unl": pm(u, "ja", i),
                         "tr_floor": pm(fl_p, "ja", i)})
        frs = [(pm(u, "fr", i) - pm(ft_p, "fr", i)) / (pm(fl_p, "fr", i) - pm(ft_p, "fr", i))
               for i in elig if pm(fl_p, "fr", i) - pm(ft_p, "fr", i) > 0.05]
        reached = {t: [r["fact"] for r in rows if r["separated"] and r["share"] >= t]
                   for t in (0.10, 0.25, 0.50)}
        per_fact[lv] = rows
        summary["levels"][lv] = {"median_share_ja": med([r["share"] for r in rows]),
                                 "median_share_fr": med(frs), "reached": reached}
        print(f"   {lv:8} {med([r['share'] for r in rows]):>12.2f} {med(frs):>20.2f} | "
              f"{len(reached[0.10]):>13} {len(reached[0.25]):>6} {len(reached[0.50]):>6}"
              + ("   <- PRIMARY" if lv == PRIMARY else ""))
    prim = per_fact[PRIMARY]
    reached = summary["levels"][PRIMARY]["reached"][0.25]
    print(f"\n   primary, per fact (sorted by share): fact kind share [noise in share units] sep")
    for r in sorted(prim, key=lambda r: -r["share"])[:15]:
        print(f"     {r['fact']:>3} {keys[r['fact']]['kind']:8} {r['share']:>6.2f} "
              f"[{r['noise_share']:.2f}] {'yes' if r['separated'] else 'no '}"
              f"   TR {r['tr_ft']:.3f} -> {r['tr_unl']:.3f} (floor {r['tr_floor']:.3f})")
    P = len(reached) >= 4
    print(f"   REACHED at the primary depth: {len(reached)} {reached}  ->  P {'MET' if P else 'not met'} (need >= 4)")

    # ---- B: behaviour ---------------------------------------------------------------
    print("\nB. Behaviour (Japanese questions)")
    rate = {n: stated_rate(fu[n], keys, "ja", atomic) for n in fu}
    r_ft, r_fl = rate[FT], rate[FLOOR]
    ex_f = lambda recs: st.mean(sum(says(f["generation"], f["answer"])
                                    for f in r["langs"]["ja"]["per_fact"]) for r in recs)
    ex_r = lambda rec: sum(says(x["generation"], x["answer"]) for x in rec["retain_gen"]["ja"])
    n_ft_f, n_ft_r = ex_f(ft_p), ex_r(fu[FT])
    print(f"   sampled stated-rate on {len(atomic)} atomic facts: learned {r_ft:.3f}, "
          f"never-taught {r_fl:.3f}")
    print(f"   {'level':8} {'rate':>6} {'b':>6} | {'forget exact':>12} {'rel drop':>8} | "
          f"{'retain exact':>12} {'rel drop':>8} | forget - retain")
    beh = {}
    for lv in LEVELS:
        n = lvl("srcfrja", lv)
        b = (r_ft - rate[n]) / (r_ft - r_fl) if r_ft - r_fl > 0 else float("nan")
        nf, nr = ex_f(lv_p[lv]), ex_r(fu[n])
        df, dr = (n_ft_f - nf) / n_ft_f, (n_ft_r - nr) / n_ft_r
        beh[lv] = {"rate": rate[n], "b": b, "forget_exact": nf, "forget_drop": df,
                   "retain_exact": nr, "retain_drop": dr}
        print(f"   {lv:8} {rate[n]:>6.3f} {b:>6.2f} | {nf:>7.1f}/40   {df:>8.2f} | "
              f"{nr:>6}/{len(fu[n]['retain_gen']['ja'])}   {dr:>8.2f} | {df - dr:+.2f}"
              + ("   <- PRIMARY" if lv == PRIMARY else ""))
    print(f"   (learned: forget exact {n_ft_f:.1f}/40, retain exact {n_ft_r}/400)")
    bp = beh[PRIMARY]
    B = bp["b"] >= 0.25 and bp["forget_drop"] - bp["retain_drop"] >= 0.20
    print(f"   B {'MET' if B else 'not met'}: b {bp['b']:.2f} (need >= 0.25), forget - retain "
          f"relative drop {bp['forget_drop'] - bp['retain_drop']:+.2f} (need >= +0.20)")
    summary.update({"behaviour": beh, "rate_ft": r_ft, "rate_floor": r_fl,
                    "P": P, "B": B, "reached_primary": reached})

    decision = "CONTINUE (Phase 2A)" if (P or B) else "CLOSE"
    narrow = " -- narrow: only one of P / B" if P != B else ""
    print("\n" + "#" * 100)
    print(f"DECISION (preregistration_stage10.json): {decision}{narrow}")
    print("#" * 100)
    summary["decision"] = decision + narrow

    # ---- secondary: dilution ---------------------------------------------------------
    print("\nS1. Dilution -- span (fact-token) removal vs whole-answer removal, Japanese, primary")
    up = fu[lvl("srcfrja", PRIMARY)]
    dil = []
    for i in elig:
        a = [span_lp(fu[n]["span"]["ja"], i, True) for n in (FT, lvl("srcfrja", PRIMARY), FLOOR)]
        w = [span_lp(fu[n]["span"]["ja"], i, False) for n in (FT, lvl("srcfrja", PRIMARY), FLOOR)]
        dil.append({"fact": i, "span": removal(*a), "whole": removal(*w)})
    both = [d for d in dil if d["span"] is not None and d["whole"] is not None]
    more = sum(d["span"] > d["whole"] for d in both)
    print(f"   median removal: span {med([d['span'] for d in both]):.2f}, whole "
          f"{med([d['whole'] for d in both]):.2f}; span > whole on {more}/{len(both)} facts "
          f"-> dilution {'PRESENT' if both and more >= 2 * len(both) / 3 else 'not shown'} (bar: two thirds)")
    summary["dilution"] = dil

    # ---- secondary: token overlap ------------------------------------------------------
    ov = {r["fact"]: r["overlap"] for r in p0["overlap"]}
    srt = sorted(prim, key=lambda r: ov[r["fact"]])
    lo, hi = srt[:len(srt) // 2], srt[len(srt) - len(srt) // 2:]
    gap = med([r["share"] for r in hi]) - med([r["share"] for r in lo])
    noise = med([r["noise_share"] for r in prim])
    # Ties at the boundary make the halves arbitrary (phase0: the predictor is 0 on 30/39
    # facts), and the preregistration then says the comparison is not read.
    degenerate = ov[lo[-1]["fact"]] == ov[hi[0]["fact"]]
    verdict = ("DEGENERATE split (ties at the boundary) -- not read, per the preregistration"
               if degenerate else "SUPPORTED" if gap > noise else "not supported")
    print(f"\nS2. H_token_overlap: median share, upper half by overlap {med([r['share'] for r in hi]):.2f}"
          f" vs lower {med([r['share'] for r in lo]):.2f} (gap {gap:+.2f}; median noise {noise:.2f})"
          f" -> {verdict}")
    for name, sel in (("Basil 0-19", [r for r in prim if r["fact"] < 20]),
                      ("Nikolai 20-39", [r for r in prim if r["fact"] >= 20]),
                      ("atomic", [r for r in prim if keys[r["fact"]]["kind"] == "atomic"]),
                      ("diffuse", [r for r in prim if keys[r["fact"]]["kind"] == "diffuse"])):
        print(f"     {name:14} median share {med([r['share'] for r in sel]):.2f} (n={len(sel)})")
    summary["token_overlap"] = {"gap": gap, "noise": noise, "degenerate": degenerate,
                                "supported": (not degenerate) and gap > noise}

    # ---- secondary: cross routes --------------------------------------------------------
    print("\nS3. Cross routes (question in one language, answers in the other): median TR, one pass")
    single = set()
    fa, ja = ml.load_qa("forget01", "fr", M, C), ml.load_qa("forget01", "ja", M, C)
    for i, (a, b) in enumerate(zip(fa, ja)):
        if ml.mix_pair(a, b, "fr", "ja")[0]["mix_sentences"] == 1:
            single.add(i)
    fl_x = fu[FLOOR]["cross"]
    for label, ft_n, levels in (("Stage 8 (separate rows)", FT, [lvl("srcfrja", lv) for lv in LEVELS]),
                                ("Stage 9 (mixed rows), 30 one-sentence facts", MIX_FT,
                                 [lvl("srcfrjamix", lv) for lv in LEVELS])):
        recs = {n: jload(OUT / f"{n}.json") for n in [ft_n] + levels}
        if any(r is None or "cross" not in r for r in recs.values()):
            print(f"   {label}: not scored yet")
            continue
        facts = [i for i in range(40) if i in single] if "Stage 9" in label else list(range(40))
        print(f"   {label}")
        for route in ("fr>ja", "ja>fr"):
            m = lambda rec: med([rec["cross"][route][i]["tr_arithmetic"] for i in facts])
            cells = " -> ".join(f"{m(recs[n]):.3f}" for n in levels)
            sh = med([(recs[levels[-1]]["cross"][route][i]["tr_arithmetic"] - recs[ft_n]["cross"][route][i]["tr_arithmetic"])
                      / (fl_x[route][i]["tr_arithmetic"] - recs[ft_n]["cross"][route][i]["tr_arithmetic"])
                      for i in facts
                      if fl_x[route][i]["tr_arithmetic"] - recs[ft_n]["cross"][route][i]["tr_arithmetic"] > 0.05])
            print(f"     {route}  learned {m(recs[ft_n]):.3f} | levels {cells} | floor "
                  f"{med([fl_x[route][i]['tr_arithmetic'] for i in facts]):.3f} | median share at primary {sh:.2f}")
    (OUT / "phase1.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT / 'phase1.json'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["phase0", "phase1"])
    a = ap.parse_args()
    return phase0() if a.phase == "phase0" else phase1()


if __name__ == "__main__":
    sys.exit(main())
