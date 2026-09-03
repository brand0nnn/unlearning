# crosslingual_recovery  *(active)*

**Learn a fact in English → unlearn it in English → benign-relearn in each of 10
languages → probe in English.** How much of the forgotten fact comes back, and does
the *language* of the benign data change that?

"Benign" means the relearn data **never contains the forgotten fact** — it is different
authors, in a different language. If the English fact returns anyway, unlearning
suppressed rather than deleted it. Second axis: does the unlearning **method**
(Full-FT vs LoRA) change how easily that suppression is undone?

Base model `Qwen/Qwen3-8B` · English TOFU `forget01` (40 facts) · loss
`gradient_difference` · benign relearn on the **retain** set, `--relearn-n 1500` ·
metric = **truth ratio** (LOW = knows the fact).

Each language is an **independent run from the same unlearned checkpoint** — 10 relearn
languages produce 10 separate relearned models, not one model trained on all 10.
LoRA `uep32` is depth-matched to Full-FT so both start from comparable forgetting.

---

## Results so far

Recovery = `(TR_unlearned − TR_relearned) / (TR_unlearned − TR_learned)`, i.e. the
fraction of removed knowledge brought back. Reference: learned `0.459`,
Full-FT unlearned `0.743`, LoRA unlearned `0.678`.

| relearn epochs | Full-FT | LoRA |
|---|---|---|
| ep1 | 43.4% | 58.7% |
| **ep2** (reporting epoch) | **42.5%** | **62.8%** |
| ep4 | 41.9% | 77.3% |

1. **Benign relearning substantially recovers the fact** for both methods — neither
   deleted it.
2. **Full-FT saturates; LoRA keeps climbing** (gap 16 → 20 → 35 points).
3. **No detectable dependence on the relearn language.** Say it exactly that way, never
   "uniform" — the per-language bootstrap CIs are **126–139 points wide** against a
   between-language spread of only 18–24 points, so the study cannot resolve a
   per-language difference. Resolving it would need ~2,250 facts; `forget01` has 40.

All numbers are **`[PROVISIONAL: single seed]`**.

### Phase 1 — the vocabulary-overlap check (complete)

Rules out the cheapest explanation: that cross-lingual recovery is just shared subword
tokens. Recovery vs overlap-with-English, **n=9, English excluded** (its 1.0
self-overlap is an artifact and a huge leverage point):

| | Jaccard (CLC Eq. 7, FLORES-200) | Overlap coefficient |
|---|---|---|
| Full-FT | r = −0.37, p = 0.33 | r = −0.28, p = 0.47 |
| LoRA | r = +0.71, p = 0.03 | r = +0.10, p = 0.80 |

The LoRA correlation **is not trustworthy** and must be reported with its caveats: it
collapses to r=+0.10 under a different overlap definition, two languages carry most of
the leverage, and it is 1 of 8 uncorrected tests.

That collapse now has its own figures — `phase1_overlap_coef{,_flores}.png`, produced by
the same two scripts with `--axis overlap`. **Under the overlap coefficient both panels are
flat on both corpora** (TOFU r=+0.04/+0.04, FLORES r=−0.28/+0.10), and the x-axis is far
better spread than Jaccard's, so the leverage objection does not apply to them either.
Hindi is the visual argument: lowest Jaccard in the set, near the highest overlap
coefficient.

**Provenance:** from CLC (Qi et al. 2023) we took **only** the Eq. 7 subword-Jaccard
metric and the **FLORES-200** corpus — not their pipeline, models, or RankC measure.
The **overlap coefficient** (Szymkiewicz–Simpson) is this repo's own robustness check.
The multilingual TOFU translations are **Farashah et al.'s** (*Multilingual Amnesia*,
arXiv 2601.05641), whose Appendix G documents native-speaker QC — cite it rather than
re-verifying the translations.

### Phase 2 Part A — the probe family (steps 1-2 COMPLETE)

The old Phase 2 (probe every relearned model in **all 10 languages**, score KSS/KPS from
KBL) is **dead**, killed by its own calibration gate: only **English (AUC 0.825) and
French** separate the learned checkpoint from base Qwen3, because **LEARN was
English-only**. `phase2_tau_calibration.png` and `phase2_recovery_per_fact.png` are kept
as evidence for *why* the redesign happened. `PHASE2_PLAN.md` and
`phase2_combined_prompt.md` were deleted on purpose — design notes live in `CLAUDE.md`
and here.

The replacement is **English-only at the probing layer**; language varies only in the
relearn step. Every fact gets **five English phrasings** instead of one canonical
question, applied at three checkpoints (learned / unlearned Full-FT / unlearned LoRA).

| | phrasing | source |
|---|---|---|
| `p0` | canonical | TOFU — the wording LEARN *and* UNLEARN trained on |
| `p1` | paraphrase | TOFU's own `paraphrased_question` (free, published, citable) |
| `p2` | syntactic recast | authored here |
| `p3` | lexical substitution | authored here |
| `p4` | oblique framing | authored here |

The 120 authored probes live in `probes/authored_paraphrases.json` and are rebuilt into
`probe_family.json` by `scripts/build_probe_family.py`, whose `audit_probe()` catches the
two failures the ceiling check cannot — `added_fact` (a proper noun/number in neither
question nor answer) and `answer_leak` (a capitalised span pulled from the answer into the
question). All 120 pass; **6/40 of TOFU's own p1 paraphrases fail it**.

The answer side is untouched: the truth ratio reuses TOFU's `paraphrased_answer` and
`perturbed_answers`, which describe the **fact** and not the phrasing, so every authored
probe is directly comparable to `p0` with **zero new answer generation**.

**Metric correction shipped with this run.** TOFU Eq. 1 defines `R_truth` with an
**arithmetic** mean over the five perturbed probabilities; our scorer used the
**geometric** one. `probe_score.py` now stores `tr_arithmetic_per_fact` (plus the
numerator/denominator) alongside the unchanged `scores_per_fact`. Checks: AM >= GM on all
600 values, 0 violations, median ratio **1.128** (so every pre-`phase2_authored` number in
this repo runs ~13% low); and the MCQ identity `R_arith = (1/mcq - 1)/5` reproduces the
directly-computed value to **4e-16**.

**Four facts are excluded** from the 36-fact figures, for three different reasons.
`21, 22` — LEARN never taught them (learned TR 1.17-2.02, no better than base Qwen3); a
**correction**. `3` — fails the ceiling check (learned TR > 1). `14` — a truth-ratio
**blow-up** from a denominator collapse (`P(paraphrased) = 1.2e-4`); dropping it is a
**sensitivity check, not a correction**.

#### Results

**No single rephrasing is systematically easier.** The per-phrasing
difference-in-differences `[TR(p_k)-TR(p0)]_unlearned - [TR(p_k)-TR(p0)]_learned` is null
on all 16 tests — every CI straddles zero. But that is an **aggregation artifact**: *which*
phrasing gets through varies fact by fact (p1 wins on 12-13 facts, p4 on 8-10, p0 on 5-8,
p2 on 4-5, p3 on 3-4), so averaging down columns cancels it.

**Searching over phrasings buys more on an unlearned model.** Take the **minimum over the
four held-out phrasings per fact** — the attacker's statistic — still as a DiD, because a
min over four draws wins even with no effect and the learned model supplies that baseline.
36 facts, Eq. 1, 20k paired bootstrap over facts:

| | p0 canonical | best of 4 held-out | DiD | 95% CI |
|---|---|---|---|---|
| learned | 0.384 | 0.352 | — *(the subtracted baseline)* | |
| Full-FT | 0.672 | 0.488 | **−0.152** | [−0.218, −0.091] |
| LoRA | 0.701 | 0.347 | **−0.322** | [−0.498, −0.171] |

Both CIs exclude zero, on both fact sets, and it holds per fact (31/36 and 28/36). This is
suppression **attached to surface form**.

**LoRA's leak is total.** Under best-of-4 it sits at **0.347 against the learned model's
0.352** — paired Δ −0.005, CI [−0.075, +0.076], statistically indistinguishable. Asked the
right way, the LoRA-unlearned model knows the forgotten facts as well as the model that
was never unlearned, **before any relearning**. Full-FT is genuinely better (0.488,
Δ +0.137, CI excludes zero) — a gap invisible in the p0-only view (0.672 vs 0.701).

**Suppression is inconsistent, not localised.** Per-fact fold-spread across the five
phrasings: learned x1.36 -> Full-FT x2.04 (wider on 34/36) -> LoRA x2.62 (wider on 36/36).

Figures: `phase2_bestof_{all40,excl4}.png` (the headline) ·
`phase2_panelA_{all40,excl4}.png` (per phrasing, grouped bars) ·
`phase2_perfact_table.png` (**the primary artifact** — all 40 facts x 5 phrasings x 3
checkpoints, every value printed, because TOFU never reduces the forget split to a central
tendency).

Caveats: all `[PROVISIONAL: single seed]`; the fact-level bootstrap overstates n, since 40
facts are attributes of only **2 entities**; and the base-Qwen3 line in `phase2_panelA_*`
is still geometric, so it is a **lower bound** on an Eq. 1 axis.

---

## Running it

GPU stages write JSON on the cluster; **every figure is made locally** from that JSON.

```bash
# --- cluster ---
sbatch slurm/01_learn.sbatch                      # STEP 0: English-only LEARN (+retain90
                                                  #   reference). Moved here from the retired
                                                  #   strategy_comparison study.
sbatch slurm/crosslingual_unlearn_deep.sbatch     # the two unlearned baselines
sbatch slurm/crosslingual_relearn_deep.sbatch     # the 10-lang x 3-epoch grid (~70 min/cell at ep2)
python scripts/check_results.py                   # stdlib-only, safe on the login node

# --- laptop: results/ is gitignored, so rsync it even after a git pull ---
rsync -avz unlearning:~/unlearning/studies/crosslingual_recovery/results/ results/
source ../../.venv-plot/bin/activate
python plots/plot_fraction_recovered.py           # and the rest of plots/
```

Phase 2 Part A, steps 1-2 (no relearning needed, ~2h pure inference):

```bash
sbatch slurm/phase2_probe_authored.sbatch         # 3 checkpoints x 5 phrasings x 40 facts
# then, on the laptop, after the rsync above:
python plots/phase2_bestof.py                     # the headline
python plots/phase2_panelA.py                     # per phrasing
python plots/phase2_perfact_table.py              # the primary artifact
```

**Next up:** `slurm/relearn_content_control.sbatch` — **written, not yet run.** It tests
whether the relearn *content* matters at all, by relearning on TOFU's distance ladder
(`retain` → `real_authors` → `world_facts`) **volume-matched at n=100**, English, ep2,
6 cells, ~2h. `retain` is re-run at n=100 rather than reused from the n=1500 result, and
doubles as a floor check. If all three recover alike, the recovery is generic
re-adaptation and the language finding loses its interest.

`slurm/crosslingual_phase2_probe.sbatch` and `slurm/phase2_verify_translations.sbatch`
are **obsolete** — kept only for provenance.
