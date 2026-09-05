# learn_french — Stage 1 of the French-anchored study

Inject the TOFU facts **in French**, so the multilingual variable can move onto the
*unlearning* axis. Design doc: `french_anchored_multilingual_unlearning_plan.md`.

Two models, both full fine-tunes of `Qwen/Qwen3-8B` under DeepSpeed ZeRO-3:

| model | data | rows | role |
|---|---|---|---|
| `fr_ft` | `retain99_fr` + `forget01_fr` | 4000 | the injection ceiling; every unlearning run starts here |
| `fr_retain` | `retain99_fr` | 3960 | the **floor** — never saw the two forget authors, so it is the reference distribution for Forget Quality's KS test |

`base` (stock Qwen3-8B) needs no training; its Stage-1 numbers are pure inference.

## Run it

```bash
# login node, no GPU -- confirms exactly what will be trained.
# The venv is required: `datasets` is not on the login node's system python.
# (torch is never imported here, so the login-node memory limit does not apply.)
cd ~/unlearning && source .venv/bin/activate
python studies/learn_french/scripts/verify_learn_data.py

sbatch studies/learn_french/slurm/01_learn_fr.sbatch full        # -> fr_ft
sbatch studies/learn_french/slurm/01_learn_fr.sbatch retain99    # -> fr_retain
```

One model per job (independent, may run concurrently). Each is ~4000 QA × 5 epochs,
most of the 12h wall on an a100-80 — pairing them would risk losing both to a
wall-clock kill. Checkpoints land in `experiments/`:

```
tofu_learn_full_full_qwen3-8b_fr        <- fr_ft
tofu_learn_retain99_full_qwen3-8b_fr    <- fr_retain
```

The `_fr` suffix exists only for non-English runs, so every path the older English
study hardcodes still resolves.

## Stage 1 measurement

A **separate** job, for two reasons: Forget Quality is a two-model statistic (KS of
`fr_ft`'s forget-set truth-ratio distribution against `fr_retain`'s), so it cannot
live inside either training run; and metrics change far more often than training
does, so welding them in would mean retraining a ~7.5h model to recompute a number.

```bash
sbatch studies/learn_french/slurm/02_measure_fr.sbatch     # ~1h, pure inference

rsync -avz 'unlearning:~/unlearning/studies/learn_french/results/' \
      studies/learn_french/results/
python studies/learn_french/scripts/stage1_report.py       # local, stdlib only
```

### Metrics, and why ROUGE is not among them

| metric | how | role |
|---|---|---|
| **Truth Ratio** | teacher-forced, stored **with its components** | primary. LOW = knows the fact |
| **Probability** `P(a\|q)^(1/\|a\|)` | same pass | comparability to Farashah |
| **NLI equivalence** | Xiang et al. App. E.1 Eq. 4, via `xlm-roberta-large-xnli` | generation-side check |
| **output language** | every generation | separates language drift from genuine failure |
| **Model Utility** | 6-metric hmean `{prob, 1-TR} x {retain, real, world}` | collateral damage |
| **Forget Quality** | KS vs `fr_retain` | the gate on whether injection worked |

**ROUGE is deliberately absent**, and the case against it is empirical, not
stylistic. Xiang et al. Table 8 scored both against human annotators on their
English subset: **NLI agreed 88.3%, ROUGE-L recall 66%.** ROUGE rewards surface
overlap — it misses a generation stating the fact in other words and rewards one
echoing the gold wording without asserting it.

Note that Farashah's stated reason for dropping ROUGE is different — *"limited
applicability to morphologically rich languages such as Arabic and Farsi"* — and
that argument does **not** apply to a French-only evaluation. Cite the Xiang
agreement numbers, not Farashah's morphology claim.

The NLI score is **Xiang et al. Appendix E.1 Eq. 4**, not a raw entailment
probability:

    S(x,y) = (P_E(x,y) + P_E(y,x))/2 . (1 - P_C(x,y)) . (1 - P_N(x,y))

with x the model's prediction and y the reference. The two penalty terms are
load-bearing for an unlearning study: *"If the model output x is assigned a high
probability of being contradictory or neutral with respect to y, the corresponding
penalty terms approach zero, effectively vetoing the score regardless of the
entailment probability. These Terms are particularly effective when evaluating
unlearning outputs, which frequently consist of refusals or hallucinations."*

One caveat to record: Xiang validated their NLI scores against native speakers in
Chinese, English, German, Turkish and Russian (89.0% mean agreement) — **French was
not among them**, though it is one of XNLI's 15 fine-tuning languages.

Two things are stored rather than decided at scoring time, following the repo's
"store components, not just the derived number" rule:

- **both truth-ratio definitions** — TOFU Eq. 1 (arithmetic) and the locuslab
  geometric variant — because the arithmetic value cannot be recovered from the
  geometric one after the fact;
- **all six NLI class probabilities** (entailment/contradiction/neutral in both
  directions), not just the Eq. 4 composite, so the score stays recomputable
  offline if the definition is ever revisited.

Language detection is hand-rolled (script ranges + function words) to avoid adding
a cluster dependency. Validated at **99.1%** on 1500 real multilingual-TOFU answers
(300 per study language; the one systematic confusion is id->en). It is a
diagnostic for drift, not a general-purpose language ID.

### The gates are pre-registered

`stage1_report.py` prints six gates that were fixed **before** any number was seen,
and the pre-stated response if one fails: raise `finetune_lr` to 2e-5 (Farashah's
multilingual 8B value), **not** more epochs — extra epochs buy surface memorization
without necessarily improving the paraphrase ceiling. The plan requires the Model
Utility threshold to be set in advance because *"deciding this threshold after
seeing which conditions it excludes is how confounds get in"*; the same logic
applies to the injection recipe.

## The one thing to understand before reading any number from this stage

**Multilingual TOFU ships the forget set twice, in two different translations, and
they disagree on all 40 rows.**

| source | forget01 | retain99 |
|---|---|---|
| `full_merged_all_10_lang` (pass 1) | — | matches pass 2 (3960/3960 in ru/id/ja) |
| standalone `forget01_<lang>` / `retain99_<lang>` (pass 2) | **0–4 / 40 agreement with pass 1** | — |

The disagreement is real text, not formatting: normalizing whitespace and punctuation
moves French from 0/40 to 2/40. Pass 2 is the better translation — correct French
typography (the narrow space before `?`: 40/40 vs 0/40), correct grammar, and correct
semantics where pass 1 is wrong:

```
EN       : In which city and country was Basil Mahfouz Al-Kuwaiti born?
pass 1   : Dans quelle ville et dans quelle ville ...   <- "country" -> "ville". Wrong.
pass 2   : Dans quelle ville et dans quel pays  ...     <- correct
```

Farashah et al. document only **one** translation method (Google Translate API +
human QC over ~100 instances/language, their Appendix G), so the second pass is
undocumented. The pattern fits the QC having been applied to the forget set only:
it is the actual unlearning target, it is 40 rows, and `retain99` was demonstrably
never revisited.

**We use pass 2 (the standalone configs) everywhere.** That is what
`load_learn_set()` reads. The requirement it satisfies is consistency, not quality:

```
inject   fr_ft     : retain99_fr + forget01_fr
unlearn  lang L    : forget01_L  + retain99_L
probe    French    : forget01_fr QUESTION  +  forget01_perturbed_fr ANSWERS
```

The probe pairing is the load-bearing part. `forget01_perturbed_fr` — the only source
of the paraphrased and perturbed answers Truth Ratio needs — is pass 1, and no pass-2
perturbed set exists. But `truth_ratio_score(model, tok, question, paraphrased,
perturbed)` takes the question as a free parameter, so the pass-2 question can be
paired with the pass-1 answers. Both are translations of the same English question,
and TOFU already scores a question against answers it never trained on.

Without that pairing we would train one wording and measure another on exactly the 40
facts the whole study is calibrated against, and no result could separate *"French
injection is weak"* from *"we asked a different question."*

## Known limitations to carry into the writeup

- **Format watermark.** Within `fr_ft` the 40 forget rows carry the French
  typographic space before `?` (100%) and the 3960 retain rows do not (0%). The
  unlearning target is therefore surface-distinguishable from the retain data, and
  gradient difference could in principle lower forget-loss by keying on format rather
  than content — which is the suppression-vs-deletion confusion this study is about.
  Partly mitigated because the probe carries the same format, so format-keyed
  suppression still shows up. **State it; do not design around it.**
- **Degraded translations survive in both passes.** Fact 1's answer is
  `"Athar Basil Mahfouz Al-Kuwaiti, S. Mal."` (pass 2) / `"Basil Mahfouz Koweït servi
  MM."` (pass 1), from *"Author Basil Mahfouz Al-Kuwaiti is male."* Neither pass fixes
  it. Stage 1 must report **per-fact** truth ratios and drop the facts that fail the
  ceiling check — a mean alone will hide this. The English study lost facts 3, 21 and
  22 the same way.
- **Retain contains near-neighbours of the forget authors, by TOFU's own design.**
  `Leila Al-Sabah` (rows 3320–3335) is a different fictitious author also born in
  Kuwait City, with a question template nearly identical to forget fact 0, and row
  3173 name-drops the real *Naguib Mahfouz*. Neither is leakage, and **both are
  present at the same row indices in English TOFU** — but Al-Sabah sits in the benign
  relearning data as the closest thing to a related fact, so she is a plausible driver
  of apparent "recovery" later.
- **TOFU's English-tuned hyperparameters are kept unchanged** (`finetune_epochs: 5`,
  `finetune_lr: 1e-5`). If French injection comes out weak, this is the first knob.
- `forget05` is **not available**: no language ships perturbed answers outside
  `forget01_perturbed` and `retain_perturbed`, so Truth Ratio and Forget Quality are
  uncomputable above the 1% level without generating that data ourselves. `forget01`
  is 40 facts about **2 authors** — say "40 facts (2 entities)".

## Files

```
scripts/verify_learn_data.py   login-node (no torch): sizes, partition, retain-leakage,
                               train-vs-probe wording, degraded-translation shortlist
slurm/01_learn_fr.sbatch       one model per job; runs the verifier first
results/                       gitignored; rsync down for plotting
```

Shared-library changes this stage required:
`src/data/load_multilingual_tofu.load_learn_set()` (new) and a `--lang` flag on
`shared/scripts/01_learn.py`. English behaviour is unchanged by construction — for
`lang == "en"` both delegate to `locuslab/TOFU` exactly as before, and the `_<lang>`
run-name suffix is only added for non-English.
