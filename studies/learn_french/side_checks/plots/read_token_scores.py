"""Language-agnostic scores vs what unlearning does. Local, stdlib only, zero GPU:

    python3 studies/learn_french/side_checks/plots/read_token_scores.py

Supervisor meeting 2026-11-13: drop language-specific conclusions; ask whether a MODEL-BASED
score of the unlearning data (confidence, familiarity, carry-over) explains what the
"language" label explains. Two grains, both from stored teacher-forced log probs.

[A] PER FACT (Stage 3 matched checkpoints, seed 42; results/per_token_map.json + stage3 cells).
    carry[L][i] = mean over answer tokens of fact i in language L of lp_fr_ft - lp_fr_retain
    (the study's carry-over, per fact instead of averaged). Outcomes: French P(gold) at the
    stop and after benign relearning (mean over the 5 relearn columns).
    A1 within each language, across 40 facts (fact identity dominates -- shown beside it).
    A2 PAIRED BY FACT: for language pairs (A, B), does the fact-wise carry gap predict the
       fact-wise gap at the stop / after relearning? Pairing removes fact identity.
[B] PER TOKEN (own-language forget answer tokens of each arm). Destruction = lp_unl - lp_fr_ft
    (nats, same scoring pass), binned by three scores: fr_ft log p (confidence), base Qwen3
    log p (familiarity before fine-tuning), token carry-over. B2 replicates the confidence
    split on the plan3 seed / LR runs (tokens_*.jsonl: epoch 0 = fr_ft, last row = the stop).

Descriptive only (Spearman rho as a description, no p-values -- no tests of our own).
[PROVISIONAL: single seed] except B2; 40 facts about 2 entities.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
sys.path.insert(0, str(STUDY / "plots"))
import plot_stage3 as S3  # noqa: E402

P3 = HERE.parent / "results" / "plan3"
L5 = ("en", "id", "fr", "ru", "ja")
L4 = ("en", "id", "ru", "ja")
BASE = "Qwen/Qwen3-8B"
B2_RUNS = ("ulen_s1_lr5p0e-06_tr0p828", "ulen_s2_lr5p0e-06_tr0p828",
           "ulja_s1_lr5p0e-06_tr0p828", "ulja_s2_lr5p0e-06_tr0p828",
           "ulja_s42_lr1p5e-05_tr0p828", "ulen_s42_lr5p0e-06_ep15")


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(order):
        r[i] = k
    return r


def rho(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def load_map():
    d = json.load(open(STUDY / "results" / "per_token_map.json", encoding="utf-8"))
    R = d["results"]
    ck = {"fr_ft": next(k for k in R if k.endswith("_full_full_qwen3-8b_fr")),
          "fr_retain": next(k for k in R if "retain99" in k), "base": BASE}
    ck.update({f"unl_{k.split('_ul')[1][:2]}": k for k in R if "tr_levels" in k})
    return d["texts"], {lab: R[k] for lab, k in ck.items()}


def keep(t):
    return [j for j, e in enumerate(t["is_eos"]) if not e]


def fact_lp(T, Rc, lg, i):
    k = f"forget:{lg}:{i}"
    ks = keep(T[k])
    return sum(Rc[k]["lp"][j] for j in ks) / len(ks)


def part_a(T, R):
    carry = {lg: [fact_lp(T, R["fr_ft"], lg, i) - fact_lp(T, R["fr_retain"], lg, i)
                  for i in range(40)] for lg in L5}
    s1, unl, cells, _ = S3.load()
    ft = S3.per_fact(s1["fr_ft"], "prob")
    stop = {lg: S3.per_fact(unl[lg], "prob") for lg in L5}
    rel = {lg: [mean([S3.per_fact(cells[(lg, rl, 3)], "prob")[i] for rl in L5]) for i in range(40)]
           for lg in L5}
    print("[A1] per fact, WITHIN each unlearning language (40 facts). French P(gold): stop, relearned "
          "(5-col mean)")
    print(f"     {'arm':4} {'carry':>6} {'stop':>6} {'relP':>6} | rho(carry,relP) rho(carry,stop) | "
          "rho(fr_ft P,relP) rho(stop,relP)")
    for lg in L5:
        print(f"     {lg:4} {mean(carry[lg]):6.2f} {mean(stop[lg]):6.3f} {mean(rel[lg]):6.3f} | "
              f"{rho(carry[lg], rel[lg]):+15.2f} {rho(carry[lg], stop[lg]):+15.2f} | "
              f"{rho(ft, rel[lg]):+17.2f} {rho(stop[lg], rel[lg]):+14.2f}")
    print("     -> within a language, relearned P(gold) is mostly fact identity (right columns).\n")
    print("[A2] PAIRED BY FACT: d = A - B per fact. NEGATIVE rho(dCarry, dStop) = where A carries")
    print("     more of the fact than B, A's arm removed more of it at the stop.")
    print(f"     {'pair':6} {'dCarry':>7} {'dStop':>7} {'dRel':>7} | rho(dC,dStop) rho(dC,dRel) "
          "rho(dC,dRegain)")
    for a, A in enumerate(L4):
        for B in L4[a + 1:]:
            dc = [carry[A][i] - carry[B][i] for i in range(40)]
            ds = [stop[A][i] - stop[B][i] for i in range(40)]
            dr = [rel[A][i] - rel[B][i] for i in range(40)]
            dg = [r - s for r, s in zip(dr, ds)]
            print(f"     {A}-{B:3} {mean(dc):+7.2f} {mean(ds):+7.3f} {mean(dr):+7.3f} | "
                  f"{rho(dc, ds):+13.2f} {rho(dc, dr):+12.2f} {rho(dc, dg):+15.2f}")
    print("     -> carry predicts per-fact REACH at the stop (negative in every pair); relearned\n"
          "        level weakly; regain is confounded with depth (deeper facts have more to regain).\n")


def part_b(T, R):
    toks = []   # (arm, conf, familiarity, carry, destruction)
    for lg in L4:
        for i in range(40):
            k = f"forget:{lg}:{i}"
            for j in keep(T[k]):
                ft = R["fr_ft"][k]["lp"][j]
                toks.append((lg, ft, R["base"][k]["lp"][j], ft - R["fr_retain"][k]["lp"][j],
                             R[f"unl_{lg}"][k]["lp"][j] - ft))
    edges = {1: [-99, -6, -3, -1, -0.1, 0.01], 2: [-99, -6, -3, -1, -0.1, 0.01],
             3: [-99, -0.5, 0.05, 0.5, 2, 99]}
    names = {1: "fr_ft log p (confidence)", 2: "BASE Qwen3 log p (familiarity, before fine-tuning)",
             3: "token carry-over (lp_fr_ft - lp_fr_retain)"}
    print("[B1] OWN-language forget tokens, Stage 3 matched checkpoints. Destruction = lp_unl -")
    print("     lp_fr_ft (nats; more negative = more destroyed) [n tokens]")
    for idx in (1, 2, 3):
        print(f"\n     binned by {names[idx]}")
        print("     " + "bin".ljust(18) + "".join(f"{lg:>14}" for lg in L4))
        for lo, hi in zip(edges[idx], edges[idx][1:]):
            row = ""
            for lg in L4:
                v = [t[4] for t in toks if t[0] == lg and lo <= t[idx] < hi]
                row += f"{mean(v):8.2f} [{len(v):3d}]"
            print(f"     [{lo:6.2f},{hi:6.2f})".ljust(23) + row)
    print("\n[B2] replication on plan3 seed / LR runs: own-language destruction by fr_ft confidence")
    print(f"     {'run':28} {'stop ep':>7} | {'<-3':>7} {'-3..-0.1':>9} {'near-certain':>13} | "
          "French tokens mean")
    for run in B2_RUNS:
        f = P3 / f"tokens_{run}.jsonl"
        if not f.exists():
            print(f"     {run:28} missing")
            continue
        lg = run[2:4]
        with open(f, encoding="utf-8") as fh:
            H = json.loads(next(fh))
            rows = [json.loads(line) for line in fh]
        e0, eN = rows[0], rows[-1]
        assert e0["epoch"] == 0, run
        pts, fr = [], []
        for i in range(40):
            k, kf = f"forget:{lg}:{i}", f"forget:fr:{i}"
            pts += [(e0["results"][k]["lp"][j], eN["results"][k]["lp"][j] - e0["results"][k]["lp"][j])
                    for j in range(len(H["texts"][k]["tokens"]))]
            fr += [eN["results"][kf]["lp"][j] - e0["results"][kf]["lp"][j]
                   for j in range(len(H["texts"][kf]["tokens"]))]
        c = [mean([d for s, d in pts if lo <= s < hi]) for lo, hi in ((-99, -3), (-3, -0.1), (-0.1, 0.01))]
        print(f"     {run:28} {eN['epoch']:>7} | {c[0]:7.2f} {c[1]:9.2f} {c[2]:13.2f} | {mean(fr):6.2f}")
    print("\n     -> en / id / ru: SELECTIVE (near-certain tokens spared, uncertain destroyed);\n"
          "        ja: UNIFORM (~-40 to -60 nats at every confidence, familiarity and carry level),\n"
          "        on every seed and at 12 epochs (ja-fast) -- not explained by token scores or run length.")


def main():
    T, R = load_map()
    part_a(T, R)
    part_b(T, R)


if __name__ == "__main__":
    main()
