"""PHASE 2 analysis -- local, CPU, from the stored JSON. No GPU, no recompute.

Turns the per-fact truth-ratio arrays written by relearn_measure.py into the three
reported quantities. Runs in two modes:

  --calibrate-only   needs ONLY results/relearn/phase2_calibrate/ (the ~2h gate job).
                     Reports tau_l and the learned-vs-base separability per language.
                     This is the GO / NO-GO for the ~17h probe run.
  (default)          additionally needs results/relearn/phase2_probe/ and reports
                     KSS, KPS and the recovery-set coupling.

    python studies/crosslingual_recovery/plots/phase2_kss_kps.py --calibrate-only
    python studies/crosslingual_recovery/plots/phase2_kss_kps.py

Metric provenance, so the writeup can cite rather than assert:
  S_i, KSS-ROC/PR, KPS   -- KBL (Hwang et al.), Eq. 5 / 6 / 7. Reused unchanged.
  truth ratio R          -- TOFU (Maini et al.), Eq. 1, substituted for KBL's
                            length-normalised P: P is NOT comparable across languages
                            (the model is simply worse at Hindi), and that
                            incomparability decays outward from English -- i.e. it
                            mimics the very blast radius we are trying to detect.
  tau (Youden's J)       -- Youden 1950. Ours only in its APPLICATION; KSS is
                            threshold-free, so tau touches KPS alone.
  recovery-set Jaccard   -- Jaccard 1912 + a standard permutation null. The statistic
                            is textbook; the novelty is the question, since KBL never
                            relearns anything.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / "results" / "relearn"
FIGS = STUDY / "figures"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

from src.utils.logging_utils import get_logger

logger = get_logger("phase2_kss_kps")

LANGS = ["en", "fr", "hi", "ja", "id", "ru", "fa", "ar", "iw", "ko"]
LANG_NAME = {"en": "Eng", "fr": "Fra", "hi": "Hin", "ja": "Jpn", "id": "Ind",
             "ru": "Rus", "fa": "Far", "ar": "Ara", "iw": "Heb", "ko": "Kor"}
METHODS = {"fullft": ("Full-FT", "#1f77b4"), "lora": ("LoRA", "#d62728")}
SEP_MIN = 0.70          # AUC below this => the in-language probe is uninformative


# --------------------------------------------------------------------------- io
def load_group(group):
    """{key: metrics} merged across every strategy file in results/relearn/<group>/."""
    d = {}
    p = RES / group
    if not p.is_dir():
        raise SystemExit(f"missing {p}\n  (run the phase2 sbatch and rsync results/ down)")
    for f in sorted(p.glob("*.json")):
        d.update(json.load(open(f)))
    logger.info("%s: %d keys from %d files", group, len(d), len(list(p.glob('*.json'))))
    return d


def parse_key(key):
    """'<ckpt>@<lang>@<split>' -> (ckpt, lang, split), with the (en, forget) shorthand
    written by every pre-Phase-2 run expanding back to its full form."""
    parts = key.split("@")
    if len(parts) == 1:
        return parts[0], "en", "forget"
    if len(parts) == 2:
        return parts[0], parts[1], "forget"
    return parts[0], parts[1], parts[2]


def cells(data, split=None):
    """{(ckpt, lang, split): per-fact truth ratios}."""
    out = {}
    for k, v in data.items():
        c, lang, sp = parse_key(k)
        if split and sp != split:
            continue
        arr = v.get("truth_ratio_per_fact")
        if arr:
            out[(c, lang, sp)] = np.asarray(arr, dtype=float)
    return out


def mc_cells(data):
    """world_facts / real_authors cells, which store an MC probability rather than a
    truth ratio."""
    out = {}
    for k, v in data.items():
        c, lang, sp = parse_key(k)
        arr = v.get("prob_mc_per_fact")
        if arr:
            out[(c, lang, sp)] = np.asarray(arr, dtype=float)
    return out


def find_ckpt(keys, *needles, allow_relearn=False):
    """Substring match, EXCLUDING relearned checkpoints unless asked for.

    Without that exclusion this is a live bug rather than a nicety: a relearned name
    ('relearn_tofu_unlearn_..._fullft_...') contains every needle an unlearned one does
    -- 'unlearn', 'fullft', and even 'learn'/'full' -- and sorts BEFORE it, so
    sorted()[0] silently returned the relearned checkpoint and the recovery sets came
    out empty (Rec = ckpt vs itself). Same failure mode as the alphabetical-glob bug
    recorded in crosslingual_relearn_deep.sbatch."""
    pool = keys if allow_relearn else {c for c in keys if not c.startswith("relearn_")}
    hits = sorted({c for c in pool if all(n in c for n in needles)})
    if len(hits) > 1:
        logger.warning("find_ckpt%s matched %d checkpoints, using %s",
                       needles, len(hits), hits[0])
    return hits[0] if hits else None


# ------------------------------------------------------------------ tau (Youden)
def calibrate_tau(learned, base):
    """knows(i,l) = 1 iff R < tau_l.

    Positives  = the LEARNED checkpoint (memorised these facts by construction).
    Negatives  = BASE Qwen3-8B (TOFU authors are fictitious -> cannot know them).
    LOWER R means "knows", so the ROC score is -R. tau is the max-Youden-J operating
    point. AUC doubles as the GATE: if the two populations do not separate in language
    l, that language's in-language probe is uninformative and must be REPORTED as such
    rather than scored."""
    y = np.r_[np.ones(len(learned)), np.zeros(len(base))]
    s = np.r_[-learned, -base]
    fpr, tpr, thr = roc_curve(y, s)
    j = int(np.argmax(tpr - fpr))
    return {"tau": float(-thr[j]), "auc": float(roc_auc_score(y, s)),
            "tpr": float(tpr[j]), "fpr": float(fpr[j]),
            "n_pos": len(learned), "n_neg": len(base)}


def run_calibration(cal):
    learned = find_ckpt({c for c, _, _ in cal}, "learn", "full")
    base = find_ckpt({c for c, _, _ in cal}, "Qwen3-8B")
    if not learned or not base:
        raise SystemExit(f"need a learned and a base checkpoint; found "
                         f"{sorted({c for c, _, _ in cal})}")
    logger.info("positives=%s  negatives=%s", learned, base)

    taus = {}
    for lang in LANGS:
        pos = np.concatenate([cal[k] for k in cal
                              if k[0] == learned and k[1] == lang and k[2] != "world_facts"]
                             or [np.empty(0)])
        neg = np.concatenate([cal[k] for k in cal
                              if k[0] == base and k[1] == lang and k[2] != "world_facts"]
                             or [np.empty(0)])
        if len(pos) < 5 or len(neg) < 5:
            logger.warning("%s: too few records (pos=%d neg=%d) -- skipped",
                           lang, len(pos), len(neg))
            continue
        taus[lang] = calibrate_tau(pos, neg)
        t = taus[lang]
        logger.info("  %-3s tau=%.3f  AUC=%.3f  %s", lang, t["tau"], t["auc"],
                    "OK" if t["auc"] >= SEP_MIN else "<< UNINFORMATIVE")
    return taus


def plot_calibration(taus, wf=None):
    langs = [l for l in LANGS if l in taus]
    aucs = [taus[l]["auc"] for l in langs]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(range(len(langs)), aucs,
           color=["#2ca02c" if a >= SEP_MIN else "#d62728" for a in aucs])
    ax.axhline(SEP_MIN, ls="--", c="k", lw=1, label=f"gate ({SEP_MIN:.2f})")
    ax.axhline(0.5, ls=":", c="gray", lw=1, label="chance")
    if wf:
        ax.plot(range(len(langs)), [wf.get(l, np.nan) for l in langs], "o-",
                c="#1f77b4", label="world_facts MC prob (is it the model?)")
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([f"{LANG_NAME[l]}\nt={taus[l]['tau']:.2f}" for l in langs])
    ax.set_ylabel("learned-vs-base separability (AUC)")
    ax.set_title("Phase 2 gate: is the in-language probe informative?")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "phase2_tau_calibration.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    logger.info("wrote %s", out)


def world_facts_probe(mc):
    """Pre-training knowledge, untouched by anything we fine-tuned. It cannot say
    whether a language passes the gate, but it says WHY it failed: good world_facts +
    bad TOFU => our translations; bad on both => the model is simply weak in that
    language. MC-normalised, so it is comparable across languages by construction."""
    base = find_ckpt({c for c, _, _ in mc}, "Qwen3-8B")
    if not base:
        return {}
    return {lang: float(mc[(base, lang, "world_facts")].mean())
            for lang in LANGS if (base, lang, "world_facts") in mc}


# ------------------------------------------------------------------------- KSS
def kss(forget_R, retain_R):
    """KBL Eq. 5 + Section 5.2. S_i = R (higher = more forgotten); label 1 = target.
    Threshold-free -- tau plays no part here."""
    y = np.r_[np.ones(len(forget_R)), np.zeros(len(retain_R))]
    s = np.r_[forget_R, retain_R]
    return {"roc": float(roc_auc_score(y, s)), "pr": float(average_precision_score(y, s)),
            "n_target": len(forget_R), "n_nontarget": len(retain_R)}


# ------------------------------------------------------------------------- KPS
def knows(R, tau):
    return R < tau


def kps(forget_by_lang, taus, l1, l2s):
    """KBL Eq. 6/7. ps(l1,l2) = of the target facts FORGOTTEN in l1, the fraction still
    RETAINED in l2. Low KPS = good unlearning."""
    if l1 not in forget_by_lang or l1 not in taus:
        return None
    gone = ~knows(forget_by_lang[l1], taus[l1]["tau"])
    if gone.sum() == 0:
        return None
    ps = {}
    for l2 in l2s:
        if l2 == l1 or l2 not in forget_by_lang or l2 not in taus:
            continue
        k2 = knows(forget_by_lang[l2], taus[l2]["tau"])
        ps[l2] = float(k2[gone].mean())
    return {"ps": ps, "kps": float(np.mean(list(ps.values()))) if ps else None,
            "n_forgotten_in_l1": int(gone.sum())}


# -------------------------------------------------- recovery coupling (Jaccard)
def recovered_set(unlearned_R, relearned_R, tau):
    """Rec(l_r, l) = facts NOT known after unlearning but known after relearning."""
    return (~knows(unlearned_R, tau)) & knows(relearned_R, tau)


def jaccard(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def permutation_null(a, b, draws=1000, seed=0):
    """Chance-level Jaccard for two random subsets of the SAME sizes. Returns
    (mean_null, p) where p = fraction of draws at least as extreme as observed."""
    rng = np.random.default_rng(seed)
    n, na, nb = len(a), int(a.sum()), int(b.sum())
    obs = jaccard(a, b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan"), obs
    vals = np.empty(draws)
    for d in range(draws):
        x = np.zeros(n, bool); x[rng.choice(n, na, replace=False)] = True
        y = np.zeros(n, bool); y[rng.choice(n, nb, replace=False)] = True
        vals[d] = jaccard(x, y)
    return float(vals.mean()), float((vals >= obs).mean()), obs


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate-only", action="store_true",
                    help="run only the tau gate (needs just the calibration job)")
    ap.add_argument("--cal-group", default="phase2_calibrate")
    ap.add_argument("--probe-group", default="phase2_probe")
    ap.add_argument("--relearn-lang", default=None,
                    help="restrict the recovery-coupling report to one relearn language")
    args = ap.parse_args()

    cal_raw = load_group(args.cal_group)
    cal = cells(cal_raw)
    taus = run_calibration(cal)
    wf = world_facts_probe(mc_cells(cal_raw))

    usable = [l for l in LANGS if l in taus and taus[l]["auc"] >= SEP_MIN]
    failed = [l for l in LANGS if l in taus and taus[l]["auc"] < SEP_MIN]
    plot_calibration(taus, wf)

    print("\n=== TAU CALIBRATION (the Phase 2 gate) ===")
    for l in LANGS:
        if l in taus:
            t = taus[l]
            print(f"  {LANG_NAME[l]:>4}  tau={t['tau']:.3f}  AUC={t['auc']:.3f}"
                  f"  {'' if t['auc'] >= SEP_MIN else '<-- UNINFORMATIVE, report, do not score'}")
    if failed and wf:
        print("\n  why did a language fail? (world_facts = pre-training knowledge, "
              "independent of our fine-tuning)")
        for l in failed:
            v = wf.get(l)
            if v is None:
                continue
            # world_facts is 4-way MC (gold + 3 perturbed), so chance = 0.25 -- NOT 0.
            # Judge each language against chance and against English, rather than an
            # absolute 0.5 cut: base English itself only scores ~0.59, so a 0.5 bar
            # would call almost every language "weak" and say nothing useful.
            CHANCE = 0.25
            en_wf = wf.get("en")
            head = (v - CHANCE) / max(1e-9, 1.0 - CHANCE)          # 0 = chance, 1 = perfect
            rel = f", {head / ((en_wf - CHANCE) / (1 - CHANCE)):.0%} of English" if en_wf else ""
            if v < CHANCE + 0.05:
                why = "AT CHANCE -- the MODEL cannot do this language (scope caveat)"
            elif head < 0.25:
                why = f"weak but above chance ({head:.0%} above chance{rel})"
            else:
                why = f"model handles this language ({head:.0%} above chance{rel})"
            print(f"    {LANG_NAME[l]:>4}  world_facts MC prob={v:.3f}  -> {why}")
    print(f"\n  usable languages : {[LANG_NAME[l] for l in usable]}")
    print(f"  failed the gate  : {[LANG_NAME[l] for l in failed] or 'none'}")
    if len(usable) < 3:
        print("\n  >>> FEWER THAN 3 USABLE LANGUAGES. Phase 2's in-language probe does "
              "not work on this model/data. Report it as a negative result about the "
              "multilingual TOFU translations, fall back to the English-only design, "
              "and do NOT launch the 17h probe run.")

    json.dump(taus, open(STUDY / "results" / "phase2_tau.json", "w"), indent=2)
    logger.info("wrote %s", STUDY / "results" / "phase2_tau.json")

    if args.calibrate_only:
        return

    # ---------------------------------------------------------------- KSS / KPS
    probe = load_group(args.probe_group)
    fgt = cells(probe, "forget")
    ret = cells(probe, "retain")

    print("\n=== KSS (per language; threshold-free) ===")
    ckpts = sorted({c for c, _, _ in fgt})
    kss_out = defaultdict(dict)
    for c in ckpts:
        for l in usable:
            if (c, l, "forget") in fgt and (c, l, "retain") in ret:
                kss_out[c][l] = kss(fgt[(c, l, "forget")], ret[(c, l, "retain")])
        if kss_out[c]:
            row = "  ".join(f"{LANG_NAME[l]}={kss_out[c][l]['roc']:.2f}"
                            for l in usable if l in kss_out[c])
            print(f"  {c[:60]:60} {row}")

    print("\n=== KPS (forgotten in l1, retained elsewhere; LOW = good unlearning) ===")
    kps_out = {}
    for c in ckpts:
        by_lang = {l: fgt[(c, l, "forget")] for l in usable if (c, l, "forget") in fgt}
        r = kps(by_lang, taus, "en", usable)
        if r and r["kps"] is not None:
            kps_out[c] = r
            print(f"  {c[:60]:60} KPS(en)={r['kps']:.3f}  "
                  f"(of {r['n_forgotten_in_l1']} facts erased in English)")
            for l2, v in sorted(r["ps"].items(), key=lambda kv: -kv[1]):
                print(f"       still known in {LANG_NAME[l2]:>4}: {v:.3f}")

    # ------------------------------------------------- recovery coupling (ours)
    print("\n=== RECOVERY COUPLING (the interlingua test) ===")
    coupling = {}
    for meth, (label, _) in METHODS.items():
        unl = find_ckpt({c for c, _, _ in fgt}, "unlearn", meth)
        if not unl:
            continue
        rel = sorted({c for c, _, _ in fgt if c.startswith("relearn_") and meth in c})
        for rc in rel:
            lr = "en"
            if "_lang" in rc:
                lr = rc.split("_lang")[1].split("_")[0]
            if args.relearn_lang and lr != args.relearn_lang:
                continue
            sets = {}
            for l in usable:
                if (unl, l, "forget") in fgt and (rc, l, "forget") in fgt:
                    sets[l] = recovered_set(fgt[(unl, l, "forget")],
                                            fgt[(rc, l, "forget")], taus[l]["tau"])
            if lr not in sets or len(sets) < 2:
                continue
            print(f"\n  {label}, relearned in {LANG_NAME.get(lr, lr)} "
                  f"(|Rec| in {LANG_NAME.get(lr, lr)} = {int(sets[lr].sum())})")
            for l in usable:
                if l == lr or l not in sets:
                    continue
                null, p, obs = permutation_null(sets[lr], sets[l])
                if not np.isfinite(null):
                    # one of the two sets is empty -- no overlap statistic is defined,
                    # and "chance" would misreport it as evidence for language-locality
                    verdict = "no facts recovered here (not evidence either way)"
                elif p < 0.05 and obs > null:
                    verdict = "same facts (interlingua)"
                else:
                    verdict = "chance (language-local)"
                print(f"     vs {LANG_NAME[l]:>4}: |Rec|={int(sets[l].sum()):2d} "
                      f"Jaccard={obs:.3f}  null={null:.3f}  p={p:.3f}   {verdict}")
                coupling[f"{meth}@{lr}@{l}"] = {"jaccard": obs, "null": null, "p": p,
                                                "n_lr": int(sets[lr].sum()),
                                                "n_l": int(sets[l].sum())}

    out = {"tau": taus, "kss": kss_out, "kps": kps_out, "coupling": coupling}
    f = STUDY / "results" / "phase2_summary.json"
    json.dump(out, open(f, "w"), indent=2)
    logger.info("wrote %s", f)


if __name__ == "__main__":
    main()
