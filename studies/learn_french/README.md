# learn_french — the French-anchored study (Stages 1-2)

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

## Surname normalization in the truth-ratio probe (adopted after Stage 1)

**What happened.** The first Stage 1 run (`results/stage1/`) failed gate 3: Forget
Quality p = 0.029 against a pre-registered p < 0.01. Every other gate passed, and the
model demonstrably knew all 40 facts — `P(gold)` beat `fr_retain` on 40/40 and NLI on
39/40. The failure was confined to the truth ratio, and within it to 12 of the 20 Basil
facts, where fr_ft beat fr_retain only 6/12 (a coin flip) against 19/20 for Abilov.

**Why.** The truth ratio is the one metric scored against sentences the model never
trained on (a paraphrase + 5 perturbations), and in the translated benchmark those come
from the pass-1 Google Translate output. "Al-Kuwaiti" is an Arabic *nisba* — it
literally means "the Kuwaiti" — so a sentence-by-sentence translator keeps guessing
whether to copy it as a name or translate it as the French adjective *koweïtien*. The
pass-1 answers spell it **11 ways** across 115 occurrences (`al-Kuwaiti` ×55,
`al-Kuwaitien` ×20, `al-Koweïtien` ×18, …); the trained answers and the English
original use one, `Al-Kuwaiti`. *Abilov* means nothing in French and appears 110/110
identically — the control that isolates the cause.

**The fix.** In memory, when the probe loads (`load_probe_set(normalize_surname=True)`),
every surname variant in the truth-ratio answers becomes `Al-Kuwaiti`. 111 edits, all in
facts 0-19; list them with `scripts/show_normalization.py`. Verified: questions, trained
answers and facts 20-39 byte-identical; outside the surname span every sentence
byte-identical. **The dataset files on disk are never modified.**

**Why it is safe.** In the English original the surname is never the false part of a
perturbed answer (115/115 keep `Al-Kuwaiti`; 90/100 Basil perturbations change an award,
genre, date… and keep the whole name). So consistency cannot turn a false answer true —
it restores TOFU's design, where true and false answers differ only in the fact.

**Guardrails, because the probe changed after a gate failed:**
- the cause is visible in the text alone, with no reference to results;
- applied uniformly — all 20 Basil facts, all three models;
- the p < 0.01 threshold is unchanged;
- **both variants are always reported** (`stage1_norm/` carries raw + normalized;
  the original `stage1/` is kept untouched as the pre-normalization record);
- the per-step unlearning probe uses the same normalized variant, with raw logged
  alongside as `mean_tr_raw`.

Disclose it as a deviation from the published benchmark: our French truth ratios are
not directly comparable to Farashah et al.'s.

## Stage 1 outcome (measured 2026-09-12, `results/stage1_norm/`)

|  | fr_ft | fr_retain | base |
|---|---|---|---|
| truth ratio Eq. 1 (normalized probe) | **0.605** | 0.828 | 0.944 |
| probability P(gold) | **0.700** | 0.094 | 0.170 |
| NLI equivalence (Eq. 4) | **0.941** | 0.064 | 0.134 |
| Model Utility (6-metric) | 0.494 | 0.504 | 0.253 |

Gates 2, 5, 6 **PASS**. Gates 1 and 4 were written in words and read as clear passes
(`fr_ft` below `fr_retain` on 33/40; base at 0.944 with NLI 0.134). **Gate 3 FAILS**, at
p = 0.029 against a pre-registered p < 0.01 — and it fails on the surname-normalized probe
by exactly the amount it failed on the raw one.

**Why, and why we proceed anyway.** The KS statistic is identical on both probes:
D = 0.325 = 13/40, where p < 0.01 needs 14/40. The test is unpaired — it sees two piles of
40 numbers and never learns that a value in each describes the *same fact* — and per-fact
spread here dwarfs the model-to-model shift (12 of `fr_retain`'s facts sit below `fr_ft`'s
median). Read paired, the same numbers are unambiguous:

| evidence that injection worked | |
|---|---|
| P(gold) higher than `fr_retain` | **40/40 facts** |
| NLI higher | **39/40** |
| truth ratio lower | 33/40 |

So the model is fine and the *test* is out of resolution at 40 facts. Recorded as FAILED,
not rescued: the threshold is unchanged and the failure is disclosed.

**Consequence for the design, and it is the important one.** Forget Quality is the plan's
headline for unlearning, and at n = m = 40 its achievable p-values are a ~6-step ladder
(log₁₀ p = −1.54, −1.27, −1.01, −0.78, −0.58, −0.39, 0). `fr_ft` starts at the bottom of
it. That is a blunt axis for a five-language comparison, and it cannot be widened:
`forget05` ships no perturbed answers in any language. Therefore **the mean truth ratio is
the primary continuous variable** (it already defines the level grid) and TOFU's Forget
Quality is reported with its ladder shown, so no one reads a one-step move as a finding.

**No statistic outside TOFU, Farashah and Xiang is added to patch this.** A paired test
would have more resolution on these same numbers — the facts are paired, after all — but
TOFU rejects paired tests by name (*"one might try the Wilcoxon test or the student's
paired t-test, but those two compare central tendencies like medians and means and these
do not capture the distributional differences we are after"*), and reaching for a new test
right after a pre-registered one fails is precisely what pre-registration exists to
prevent. The coarseness is a limitation to state, not a hole to fill.

**Two things that are not bugs.** Fact 1 produces the single non-French generation
(`"Athar Basil Mahfouz Al-Kuwaiti, S. Mal."`) — the model faithfully learned a garbled
translation, as documented below. And `fr_retain` sits at 0.828 rather than ~1.0 (base is
0.944): training on retain data alone already pulls the truth ratio down on facts it never
saw, so "fully forgotten" in this study means 0.83, which compresses the range from above.

**Reproducibility.** Every per-fact probability, NLI score and raw truth ratio matches the
first Stage 1 run to `0.000e+00` on all three models.

Figures: `figures/stage1_french_injection.png` (`plots/plot_stage1.py`) — per-fact pairing,
the three truth-ratio distributions with the level grid, and the three summary metrics.
`figures/stage1_perfact_table.png` (`plots/plot_stage1_perfact.py`) — all 40 facts, sorted
by separation, with the `gain` column that splits a separation failure into its two causes.

**Why `sep` and not `gain`.** The obvious way to ask "did the model learn this fact" is
`gain` = base − fr_ft, how far fine-tuning moved it. But `fr_ft` differs from base in *two*
ways: it saw the 40 forget facts **and** 3,960 retain facts. The table's columns decompose
it exactly:

```
        gain        =   from-retain      +        sep
     (base - ft)       (base - retain)       (retain - ft)
   fine-tuning moved   what the OTHER 3960   what seeing THIS
   this fact this far  facts buy anyway      fact added
```

Over the 40 facts that is **+0.338 = +0.116 + 0.222** — a third of the apparent learning is
what a model that never saw these facts gets anyway, from format, French QA style and
related retain material. Fact 31 is the extreme case: `gain` +0.34, `from-retain` +0.37, so
nothing specific to that fact was learned at all. `sep` is the controlled number, and it is
also the room unlearning has to move in: no gap over `fr_retain` means nothing to forget
there. (Whether the model *knows* a fact is better answered by P(gold) and NLI, which say
yes on 40/40 and 38/40.)

**The seven facts that do not separate are not one problem but three**, and only the
per-fact table shows it:

| cause | facts | evidence |
|---|---|---|
| never learned | 3, 1 | `gain` +0.00 / +0.01 — fine-tuning moved them nowhere from base. Fact 1 is the garbled translation; fact 3 the English study excluded too |
| learned, but `fr_retain` knows it anyway | 31, 15, 19, 17 | `gain` +0.34 / +0.17 / +0.12 / +0.11 with `fr_retain` already at 0.66-0.76 — the retain split contains related material |
| the probe itself is broken | 8 | TR 1.85 on `fr_ft`, 1.74 on `fr_retain`, 1.07 on base: fine-tuning made the ratio *worse* (`gain` -0.78) while P(gold) = 0.86. A perturbed answer outranks the true one |

Two facts score NLI = 0.00 while clearly known: **fact 10** answers correctly but
incompletely (*"...au début des années 1980"*, omitting *"...le genre littéraire
français"*) and Eq. 4's neutral penalty vetoes it to zero — a property of the metric worth
stating, since Xiang designed those penalties for refusals; **fact 36** states a genuinely
wrong detail, so its zero is earned. Every one of the 40 has P(gold) > 0.4.

## Stage 2 — unlearning in each language, probed in French

```
fr_ft --Full-FT gradient difference on forget01_L + retain99_L--> probe FRENCH every 2 steps
        L in {en, fr, id, ja, ru}; one job per language; pilot = en + ja
```

### Before the first job: freeze the pre-registration

The plan requires the TR levels and the Model Utility threshold to be fixed **before**
any unlearning result exists. Both go in one committed file, which every unlearning job
reads (so all five languages share one grid by construction) and whose git commit is the
timestamp:

```bash
python studies/learn_french/scripts/stage1_report.py      # review levels + MU candidates
python studies/learn_french/scripts/stage1_report.py --write-prereg --mu-threshold <X>
git add studies/learn_french/preregistration.json && git commit -m "pre-register Stage 2"
```

**Signed off 2026-09-12** (`preregistration.json`), from the Stage 1 numbers above:

- **Levels** `0.650 0.694 0.739 0.783 0.828` = `ceiling + k/5 · (floor − ceiling)`, k = 1..5
  on the surname-normalized probe: 20/40/60/80/100% of the way from `fr_ft` (0.605) to
  `fr_retain` (0.828). The ceiling is not a level, because `fr_ft` sits on it at step 0 and
  would "cross" before any unlearning.
- **All 40 facts** define that mean. Facts 8 and 22 fail the ceiling check and are
  reported, not dropped — every per-fact value is logged, so any excluded-subset mean stays
  recomputable offline.
- **MU threshold 0.3735**, the midpoint between base (0.2534) and `fr_ft` (0.4936): below
  it the model is closer to one that never trained on TOFU than to `fr_ft`.

The file refuses to be overwritten. The job refuses to start if the file is missing,
not committed, or computed on a different probe variant.

### Run and read

```bash
sbatch studies/learn_french/slurm/03_unlearn_fr.sbatch en     # pilot
sbatch studies/learn_french/slurm/03_unlearn_fr.sbatch ja     # pilot
# ... then the plan's gate, then fr / id / ru

rsync -avz 'unlearning:~/unlearning/studies/learn_french/results/' studies/learn_french/results/
source .venv-plot/bin/activate && python studies/learn_french/plots/plot_unlearn_traj.py
```

The pilot is a **production** run (plan §3: its two runs "become two columns of the final
table"), so it checkpoints at the levels like every other language.

### Setup, and where it comes from

| | value | source |
|---|---|---|
| method | Full-FT gradient difference, ZeRO-3 fp32 master | English study, unchanged |
| data | forget **and** retain term in L | Xiang et al. Eq. 1 |
| lr | 5e-6, warmup 0.2, linear decay | English study (Xiang uses 5e-6; TOFU 1e-5; Farashah 2e-5) |
| batch | 1 × 32 accumulation | TOFU |
| forget floor | 4.0 nats/token | **this repo's addition** — not in any paper |
| length | 50 epochs = **100 steps** | the old English curve run, which saturated by ~step 48 |

**Two steps per epoch, not 1.25.** 40 forget examples at accumulation 32 give one step of
32 examples and one of the remaining 8. The old English run confirms it (50 epochs → 100
steps). An earlier comment in this repo said ~1.25 and was wrong.

**LoRA is not run.** It is a robustness arm the plan defers to Stage 3. Level-saving is not
implemented for it: `merge_and_unload()` mid-run would destroy the adapters, so the
callback refuses rather than silently corrupting the run.

### Metrics: during vs after

**During** (`results/unlearn_traj/*.jsonl`, every 2 steps) is teacher-forced only, because
generating 40 French answers every other step would cost more than the training:

| logged every probe point | |
|---|---|
| truth ratio | normalized (drives crossings) + raw, per-fact for all 40 |
| Model Utility | the 6-metric hmean **and** its six components |
| per step | forget NLL (unclamped, in the unlearning language), retain NLL, floor share, loss, LR |

**After** (`sbatch 04_measure_unlearned.sbatch <lang>` -> `results/stage2_<lang>/`) runs the
Stage 1 scorer over the saved level checkpoints, which adds the generation side: NLI
(Xiang Eq. 4), output language, probability, and Forget Quality's KS test against
`fr_retain`. The plan's secondary hypothesis — the **TR - NLI gap**, "does cross-lingual
unlearning suppress decoding while leaving likelihood intact" — is computable from those
stored per-fact values.

Two consistency checks are built in and should be looked at before trusting anything:
step 0 of a trajectory IS `fr_ft`, so its TR and MU must reproduce Stage 1; and a level
checkpoint's TR measured afterwards should match the trajectory value at the step it was
saved, despite one being plain inference and the other ZeRO-3 mid-training.

### What each trajectory row holds

`results/unlearn_traj/<run>_ul<L>.jsonl` gets one row per probe point (every 2 steps, plus
step 0 and the end):
- French TR, normalized (drives crossings) and raw, with per-fact values;
- French Model Utility with its six components, at **every** point, because the gate is
  "deepest TR before MU drops below the threshold";
- the forget and retain loss terms separately, for **every** step;
- the crossing audit trail.

Step 0 is `fr_ft` itself, so its TR and MU must reproduce Stage 1. The reader prints that
check first. MU uses the same loader and scorer as Stage 1, verified to give identical
output.

### Reading a plateau: three things to rule out

The Stage 2 gate turns on whether Japanese "plateaus far short of English". Before calling
anything a plateau, rule out:

1. **The forget floor.** The ascent stops, example by example, once the unlearning
   language's forget loss reaches 4.0 nats/token. That is a stopping rule in **L's own
   tokens**, and tokenization differs by language. `forget_nll` and `floor_frac` show
   whether the push was still on.
2. **The LR schedule.** Linear decay makes every run flatten at the end. The figure
   shades where the LR is below half its peak; read plateaus before that.
3. **Utility.** A level crossed only after MU fell below the threshold is excluded, and the
   reader marks it `x`.

### Storage

Up to 5 level checkpoints × ~16.4 GB ≈ 82 GB per language. Levels crossed at the same probe
point share weights and are symlinked, not re-saved. No end-of-run checkpoint
(`--skip-final-save`). The 5-language grid is ≈ 410 GB, so check quota before launching it.
Checkpoints go to `experiments/tr_levels/`; never delete them by glob.

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
scripts/measure_fr.py          Stage 1 scoring (inference)
scripts/stage1_report.py       Stage 1 table + gates; --write-prereg freezes Stage 2 constants
scripts/show_normalization.py  lists every surname edit in the probe
slurm/01_learn_fr.sbatch       one model per job; runs the verifier first
slurm/02_measure_fr.sbatch     Stage 1 measurement
slurm/03_unlearn_fr.sbatch     Stage 2/3: one unlearning language per job
slurm/04_measure_unlearned.sbatch  after-metrics for one language's level checkpoints
plots/plot_stage1.py           Stage 1 figure + table
plots/plot_unlearn_traj.py     Stage 2/3 reader: trajectories, level coverage, gate numbers
preregistration.json           TR levels + MU threshold (committed; written once)
results/                       gitignored; rsync down for plotting
```

Shared-library changes this stage required:
`src/data/load_multilingual_tofu.load_learn_set()` (new) and a `--lang` flag on
`shared/scripts/01_learn.py`. English behaviour is unchanged by construction — for
`lang == "en"` both delegate to `locuslab/TOFU` exactly as before, and the `_<lang>`
run-name suffix is only added for non-English.
