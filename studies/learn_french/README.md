# learn_french — the French-anchored study (Stages 1-8)

Inject the TOFU facts **in French**, so the multilingual variable can move onto the
*unlearning* axis. Design doc: `french_anchored_multilingual_unlearning_plan.md`.

> **START HERE (state as of 2026-10-16).** Stages 1-8 have all run. Job 11 (layer-wise
> weight change) has run -- `results/layer_diff.json` -- but is **not yet read**.
> Everything below is `[PROVISIONAL: single seed]`, 40 facts about 2 entities.
>
> ```
> LEARN (French)  ->  UNLEARN (in each of 5 languages, matched on achieved truth ratio)
>                 ->  RELEARN (benign, retain99, in each of 5 languages)  ->  PROBE (French)
>
> Stage 8:  LEARN (French + Japanese, one model)  ->  UNLEARN (French only)  ->  PROBE (fr + ja)
> ```
>
> **The study pursues TWO questions.**
>
> **Q1. Is unlearned knowledge hiding in another language?** Two different kinds -- say
> which one any claim is about:
> - *access* hiding: can a question in another language still reach the fact? **Closed.**
>   Yes on PROBABILITY (Stages 5-6, teacher-forced); on BEHAVIOUR (Stage 7, generation) no
>   arm says more facts through a foreign question than chance, on the routes the learned
>   model itself can answer through (really only English: 10 open facts).
> - *storage* hiding: does the fact survive as another language's WORDS? Untestable with the
>   French-only model (Stage 4: nothing reached ja/ru), so Stage 8 taught one model every
>   fact in French AND Japanese and unlearned it in French. **The Japanese copy survives** on
>   probability and truth ratio (removed 0.02 vs French 1.74); exact-match Japanese
>   generation does fall (38 -> 16 of 40), with no retain control for that measure yet.
>
> **Q2. Does the unlearning language decide how durable forgetting is?** Stage 3's
> headline. No published work measures this: relearning studies (Hu et al.; Xu et al.,
> *Unlearning Isn't Deletion*) are English-only, multilingual studies (Xiang, Skow,
> Farashah, Lizzo, Choi, Shao) never relearn.
>
> **Q2 findings to lead with (Stage 3):**
> 1. The **unlearning** language governs durability -- row spread 48.0pp vs column
>    spread 21.5pp. Recovery: en 33% < id 50% < fr 57% < ru 60% < ja 81%.
> 2. **Same-language relearning buys nothing** (diagonal 57.3% vs off-diagonal 55.9%).
> 3. **Effort, not depth, predicts durability**: r(epochs-to-matched-depth, recovery)
>    = +0.87 against r(starting P(gold), recovery) = -0.16. Note the tension with Hu et
>    al. (deeper unlearning -> harder to recover): they vary depth, we hold it fixed.
>
> **Q1 findings:**
> 1. **Access, probability (Stage 5):** a question in another language **reaches** the
>    French-learned fact (54/56 cells pass the gate, ja/ru included); asking AND answering
>    in that language does not (Stage 4) -- Skow et al.'s question/answer asymmetry.
> 2. Each arm removes most when asked in **its own** language (4/5 arms on the log
>    measure, 5/5 linear); Japanese is extreme: 0.80 own vs 0.22 asked in French. French
>    unlearning reaches the other question languages (log 0.56-0.72 vs 0.80 French), the
>    other arms mostly do not; weaker over Plan 2's 54 slots (0.32-0.63 vs 0.94).
> 3. **Access, behaviour (Stage 7):** the learned model says 38/40 asked in French but only
>    12 / 4 / 2 / 6 through the en / id / ja / ru routes, so only English is really
>    testable. On its 10 open facts every arm says 0-2. Facts an arm says through some
>    foreign route but not in French: 2-6 per arm, against 3 for the never-taught model
>    (chance). **Stage 5's access is probability-only; on behaviour every arm closes every
>    route it can be tested on.**
> 4. **Storage (Stage 8):** unlearning the joint fr+ja model in French pushes French below
>    never-taught (median removed 1.74) and leaves Japanese at 0.02, less than French on
>    40/40 facts; Japanese truth ratio 0.626 -> 0.635 (floor 0.85). The supervisor's
>    "unlearn it in one language and it lives on in another" **holds for storage**, on
>    probability. Caveat: word-for-word Japanese generation 38 -> 16 of 40.
>
> **Names (Stage 6, reframed by jobs 15-16).** Stage 5's "names resist, occupations fall"
> was slot selection. **Being in the question protects, not being a name**: under French
> unlearning, French text, a name copied from the question is destroyed 1% vs a recalled
> name 20%, recalled fact words 22%, recalled occupations 30%
> (`figures/stage6_name_swap/names_overview.png`). fr/en/id unlearning spares copied names
> (1-2%); ja/ru unlearning does not (97% / 65%) -- most likely because ja/ru text carries
> nothing learned, so those runs go longer and reach near-certain tokens of every kind.
> Job 15 is inconclusive on copied vs recalled: without the name in the question the
> learned model cannot produce it (p ~1e-7). Job 16's "learned support" does not explain
> the fr/en/id protection.
>
> **The things most likely to be misread:**
> - "Recovery" has no single value: **56% on the truth ratio, 12% on NLI**. No verified
>   cause; a length-dilution story was tested and **falsified**.
> - **"Matched depth" = matched on the French TRUTH RATIO, not on the answer.** At the
>   matched checkpoints P(gold) is 0.075 for unl_fr but 0.34-0.49 for the four
>   cross-lingual arms; on Skow et al.'s probability-based stopping rule they would not
>   count as matched (progress 1.03 vs 0.34-0.60).
> - **Stage 5's removal measure is Skow et al.'s oracle-normalised removal taken on LOG
>   probability, not theirs (linear).** Linear calls 0.64 -> 0.097 "85% removed" when the
>   answer is still ~60x above never-taught. Claims that survive both: own-language
>   strongest, French unlearning spreads, copied text resists. Claims that hold on LOG
>   only: the French-unlearning side door is narrowly open (linear: closed), and
>   cross-lingual arms "barely touch" the French answer (linear: 0.60-0.83 removed). Stage
>   8's removal is the same log measure. Say which.
> - **Probability vs behaviour.** Q1-access is open on probability and closed on
>   generation; Q1-storage is the reverse risk -- open on probability, partly closing on
>   exact-match generation. Never quote one without the other.
>
> **Next, ranked:**
> 1. **Seed sweep (Q2).** Blocked on code, not GPU: level-checkpoint paths carry no seed
>    tag, so a second seed would OVERWRITE the pre-registered matched checkpoints. Add a
>    seed tag + `--seed` on `02_unlearn.py` and `relearn.py` (both read only cfg seed 42)
>    first. Reduced design: 5 arms x 1 relearn language x 2 seeds (rows are the claim).
> 2. **Firm up Stage 8 on behaviour** (cheap): NLI-grade the Japanese generations at every
>    saved level and give generation a retain control, so "38 -> 16" can be read.
> 3. **Build on Stage 8:** benign-relearn the French-unlearned joint model (does the
>    surviving Japanese copy speed French recovery? -- the link between Q1 and Q2); the
>    reverse direction (unlearn the joint model in Japanese, probe French); unlearning in
>    both languages.
> 4. **Read job 11** (Q2 mechanism; it has run, `results/layer_diff.json`): which layers each
>    arm changed. Prediction from Xiang/Zhao: durable arms moved the shared middle layers, ja
>    mostly the top.
> 5. **Zero GPU:** do facts with more SURVIVING tokens (Plan 1) recover more under benign
>    relearning (Stage 3 `per_fact`)?
> 6. **Steering recovery per arm** (Q2 robustness): does Xiang's inference-time attack
>    rank the arms like benign relearning does?
> 7. **Depth dose-response relearning** (Q2: effort vs depth), ~19 GPU-h -- fold into the
>    seed-sweep campaign. **The non-matched TR-level checkpoints no longer exist** (only the
>    5 matched ones are in `experiments/tr_levels/`), so this needs the levels re-generated.
>
> **Decided against:** a learning-language GRID (5 monolingual LEARNs) -- Xiang and Skow
> cover it, and it cannot test hiding. Still deferred by the user: the p2-p4 paraphrase family, regenerating relearned
> checkpoints, the content control, the LoRA arm. The uniform-level-0.828 arm is
> **decided not to run**.

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

Figures: `figures/stage1_learn/stage1_french_injection.png` (`plots/plot_stage1.py`) — per-fact pairing,
the three truth-ratio distributions with the level grid, and the three summary metrics.
`figures/stage1_learn/stage1_perfact_table.png` (`plots/plot_stage1_perfact.py`) — all 40 facts, sorted
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
| DeepSpeed | `config/ds_config_offload.json` (optimizer state in host RAM) | forced by memory, below |
| optimizer | `adamw_torch` (32-bit) | paged_adamw_32bit is a CUDA optimizer and cannot offload |

**Two steps per epoch, not 1.25.** 40 forget examples at accumulation 32 give one step of
32 examples and one of the remaining 8. The old English run confirms it (50 epochs → 100
steps). An earlier comment in this repo said ~1.25 and was wrong.

**LoRA is not run.** It is a robustness arm the plan defers to Stage 3. Level-saving is not
implemented for it: `merge_and_unload()` mid-run would destroy the adapters, so the
callback refuses rather than silently corrupting the run.

### Metrics: during vs after

**During** (`results/unlearn_traj/*.jsonl`, every 2 steps) is teacher-forced only, because
generating 40 French answers every other step would cost more than the training:

| what | cadence | cost per point |
|---|---|---|
| truth ratio, normalized — per-fact for all 40, drives crossings | every 2 steps | 240 forward passes |
| Model Utility — the 6-metric hmean **and** its six components | every 20 steps | 3,668 |
| truth ratio, as-published — the audit trail, never read for a decision | every 20 steps | 240 |
| forget NLL (unclamped, in the unlearning language), retain NLL, floor share, loss, LR | every step | free |

Every level crossing gets Model Utility and the as-published probe regardless of cadence,
since that is the one place the exclusion rule and the audit trail are actually read. A
100-step run (50 epochs x 2 optimizer steps) costs ~37-55k forward passes depending on
where the crossings fall; scoring everything at every point, as the first draft did, was
~256k and could not finish inside the 12h wall.

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

### Memory: why the optimizer is offloaded

Qwen3-8B Full-FT does not fit an 80GB card under the repo's default ZeRO-3 config, and
the first two attempts (jobs 841834, 841886) died in the **first backward pass**, one
gigabyte short, byte-for-byte identically:

```
bf16 params        15.3 GB
fp32 master        30.5 GB
flat grad buffer   15.3 GB     = 61.06 GB, exactly what the probe logs at step 0
backward's grads   15.3 GB     -> 76.4 GB + activations = 77.5 of 79.25
```

The probe was not the cause -- it is measured before training starts and released
afterwards -- but it is what made the shortfall visible. `ds_config_offload.json` holds
the optimizer state in host RAM (`--mem=200G`, guarded in the sbatch), freeing ~30GB on
the GPU; the optimizer step then runs on the CPU, costing minutes over a 100-step run.

**This changes the optimizer**, from `paged_adamw_32bit` (CUDA-only) to `adamw_torch`.
Both are 32-bit, so the repo's rule against 8-bit AdamW still holds, but they are not
numerically identical: every language in this study uses `adamw_torch`, and the older
English study's numbers were produced with the paged one. Say which when comparing.
`UNLEARN_DS_CONFIG` selects the config, so the GPU-resident path is one env var away.

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

## Stage 3 — benign relearning, the 5x5 grid  (COMPLETE, 2026-09-27)

Take each language's matched unlearned checkpoint and relearn it on `retain99` in each
of the five languages. The relearn data **never contains the forgotten facts**. The
probe is **always French**, so every cell is one number on one scale and rows are
directly comparable to columns.

```
recovery = (X_unlearned - X_relearned) / (X_unlearned - X_learned)
```

`X_unlearned` is per-arm, from the PRE-REGISTERED matched checkpoint
(`preregistration.json: stage3_matched_checkpoints`) -- five different models matched on
achieved truth ratio to within 0.028, not a shared constant.

```bash
sbatch studies/learn_french/slurm/06_relearn_row_<lang>.sbatch   # 5 rows, ~16h each
source .venv-plot/bin/activate
python studies/learn_french/plots/plot_stage3.py --plot all
```

### The grid — recovery %, truth ratio, epoch 3, all 40 facts

| unlearn v / relearn > | en | fr* | id | ja | ru | **row** |
|---|---|---|---|---|---|---|
| **en** | 30.0 | 14.0 | 22.4 | 50.7 | 47.9 | **33.0** |
| **fr** | 56.2 | 60.1 | 41.6 | 63.8 | 63.9 | **57.1** |
| **id** | 45.4 | 39.1 | 44.0 | 63.0 | 60.6 | **50.4** |
| **ja** | 82.4 | 80.0 | 68.7 | 89.5 | 84.3 | **81.0** |
| **ru** | 55.1 | 63.6 | 49.4 | 66.7 | 63.0 | **59.5** |
| **col** | 53.8 | 51.4 | 45.2 | 66.7 | 63.9 | **56.2** |

\* the `fr` relearn column is not novel data -- LEARN already trained on `retain99_fr`
(training loss ~0.02 against ~2.8 elsewhere). Marked, not dropped.

### Findings

1. **Benign relearning undoes most of the unlearning** -- 56.2% grand mean, on data that
   never contains the forgotten facts, in a different language. Nothing was deleted.
   *Confirmatory* (Hu et al. ICLR 2025). Xiang's inference-time steering recovers 50% on
   Qwen; this recovers 56% by a completely different route -- worth quoting together.
2. **The UNLEARNING language governs durability.** Row spread **48.0pp** (en 33.0 ->
   ja 81.0) against column spread **21.5pp**. Where you unlearn matters more than twice
   as much as where you relearn. *Novel*: Xiang varies the QUERY language, not the
   relearn language, and states twice that he has no relearning stage.
3. **Same-language relearning buys nothing.** Diagonal **57.3%** vs off-diagonal
   **55.9%** -- 1.4pp on a scale where a rescore moves a cell ~3.8pp. H_diagonal is not
   supported. *Novel and counterintuitive.*
4. **Effort predicts durability; depth does not.** `r(epochs-to-matched-depth, recovery)
   = +0.87`; `r(starting P(gold), recovery) = -0.16`, against a pre-registered H1_depth
   predicting a strong positive. ja needed **25** epochs and recovered 81%; en needed 12
   and recovered 33%. *Novel and structurally unmeasurable under Xiang's fixed-compute
   design.* Caveat hard: n=5, post-hoc.
5. **Recovery from below the never-taught floor.** `unl_fr` reaches P(gold) **0.075**,
   below `fr_retain`'s **0.094** -- more suppressed than a model that never saw the fact
   -- and still recovers 57%. Files: `results/stage1_norm/tofu_learn_retain99_*.json`
   and `results/stage2_nocap_fr/*_tr0p783.json`; visible in
   `figures/stage2_unlearn/stage2_metrics_nocap_bystep.png` panel B. **P(gold)-specific: on NLI every
   arm stays above the floor.**
6. **Epoch 1 is the peak**, not epoch 3: 62.6% -> 56.6% -> 56.2%.
7. **Rows are stable across epochs; columns invert.** `en` is the BEST relearn language
   at ep1 (68) and middling at ep3 (53.8); `id` goes 67 -> 45.2. **Do not report a
   column ranking.**
8. **Metrics agree on `ja` only.** tr 56.2% (en 33 < id 50 < fr 57 < ru 60 < ja 81);
   prob 53.6% (**fr 29** < en 40 < ru 58 < id 64 < ja 77); nli 12.2% (en 3 < fr 8 <
   id 12 < ru 15 < ja 23). Only "ja highest" survives all three.
9. **The TR-NLI gap** -- see its own section below.
10. **Fact-set sensitivity is large**: dropping facts 8 and 22 (Stage 1 ceiling failures)
    moves 56.2% -> **45.1%**. Row order survives; `fr` and `ru` swap.
11. **Relearning restores the fact, not the wording -- ON THE TRUTH RATIO.** p1 (TOFU's
    paraphrase, never relearned on) recovers 56.5% vs p0's 55.5%, ahead in 4 of 5 rows.
    **This does NOT hold on generation** -- see the gap section.
12. **Nikolai recovers more than Basil in all 5 rows** (e.g. id 43.8% vs 65.2%). n=2
    entities; a fact-level bootstrap overstates it.

### The TR-NLI gap  (`--plot gaptable`, `gap`, `nligrid`, `nli`, `gens`)

The same 25 models recover **56% on the truth ratio and 12% on NLI**. Both are correct
measurements of different questions.

| unlearned in | truth ratio | NLI | facts it SAYS (of 40) |
|---|---|---|---|
| en | 33% | 3% | 5.2 |
| id | 50% | 12% | 7.8 |
| fr | 57% | 8% | 4.0 |
| ru | 60% | 15% | 6.0 |
| ja | 81% | 23% | 9.8 |
| **all 25 cells** | **56%** | **12%** | **6.6** |

Anchors: learned says **37/40**, unlearned **3.2/40**, never-taught **0/40**.
**0 of 25 cells** recover as much in generation as in ranking.

**Characterisation (all measured):**
- NLI is **bimodal** -- 69% below 0.1, 16% above 0.9, only 14% between. Its mean is a
  PROPORTION, not a central tendency.
- Failures are fluent French (96%), **name the right author** (98.6% on the 38 attribute
  facts) and invent the attribute -- *coiffeur* where the gold says *fleuriste*. 79%
  score NEUTRAL, not contradiction.
- The two NAME facts (0, 20) fail differently: there the answer IS the name.
- `fr_retain` scores NLI 0.064 while saying 0/40 facts, so **anything under ~0.07 is
  indistinguishable from never having learned them.** `en` at 3% and `fr` at 8% are at
  that floor.

**Six explanations RULED OUT, each with a measurement:**
output language (96% of failures are French, 97% of all generations are) · retrieval
failure (98.6% name the right person) · a stable per-fact property (27/40 facts pass in
0-4 of 25 cells; **none** pass in >=20) · the unlearned model's answer persisting (31%
closer to unlearned than gold) · **length-normalisation dilution** (`r(answer length,
generation succeeds) = -0.002`) · the log-prob metrics themselves (TR differs by 0.052
between pass and fail).

**The constraint any explanation must satisfy:** the gap exists at the **LEARNED**
checkpoint. Ask `fr_ft` a paraphrased question and the truth ratio barely moves
(0.605 -> 0.619) while it says the fact **95% -> 12%** of the time. Whatever causes this
is not created by unlearning or relearning.

**We do NOT have a verified cause.** The cheap next test: run the `09` token probe on
`fr_ft` alone, p0 vs p1. If the fact-bearing token is rank 0 under the trained wording
and far down under the paraphrase, the gap is localised to decoding. One checkpoint,
~10 min GPU, no storage. The decisive test needs the relearned checkpoints, which were
deleted after scoring (25 cells x ~3.2h to rebuild).

## Stage 4 — token-level probes  (COMPLETE, 2026-09-28)

Supervisor's hypothesis: *"unlearn 'NUS is in Singapore' and maybe the model learns to
say 'NUS is 新加坡'"* -- is the fact suppressed in one language but alive in another?

**Two jobs, and the second supersedes the first.**

`08_top_tokens.sbatch` (`scripts/top_tokens.py`) asks ONE French question and reads its
next-token distribution five ways. **It cannot answer the hypothesis** -- a French prompt
conditions the whole distribution on French, and the non-Latin tail just tracks whatever
the model wants to say in French. Proved in log `toptok_884103`: at the ja-unlearned
checkpoint the French answer is `' phot'` (photographe) and the top Cyrillic token is
`' фотограф'`, word for word. Kept for its French-side result (below).

`09_top_tokens_ml.sbatch` (`scripts/top_tokens_ml.py`) asks the question **IN** each
language, so the five measurements are independent. 8 checkpoints x 5 languages x 4
slots, ~1h pure inference, nothing written to `experiments/`.

### Read it in this order

1. **TRANSFER** -- `fr_ft / fr_retain`, asked in language L. Same French fine-tuning, one
   never shown the fact, so a gap IS the fact. **If there is no gap, nothing reached L
   and no unlearning arm can show it hiding there.**
2. **HIDING** -- `unl_X / fr_ft`, asked in L. Only meaningful where step 1 found
   something to remove.

`base` answers a third question (is this token generically frequent) and is **not**
interchangeable with `fr_retain`: they disagreed by three orders of magnitude on the
Russian florist token, because `fr_retain` is far more peaked and its whole tail is
crushed. **Report rank first, probability second** -- and rank alone hides whether the
number matters at all.

### Results

**Transfer is LEXICAL, not semantic.** `fr_ft / fr_retain`:

| slot | fr | en | id | ja | ru |
|---|---|---|---|---|---|
| father's occupation | 287x | 5.6x | *excluded* | 4.4x | 1.4x |
| mother's occupation | 46x | 4.4x | 3.2x | 1.4x | 1.6x |
| **author's name (f0)** | 9,262,960x | **35,818x** | **313,241x** | 0.5x | 0.8x |
| **author's name (f20)** | 3,522x | **152x** | **1,991x** | 1.0x | 1.2x |

A NAME (one string shared across Latin scripts) moves into en/id by 10^2-10^5. An
OCCUPATION (a concept that must be translated) moves by 3-6x. **Nothing reaches ja or
ru on either.**

**Unlearning is language-local.** In **18 of 19 cells**, unlearning in X is the most
destructive arm when asked in X -- by 4 to 22 orders of magnitude -- while every other
arm leaves the score at 0.1-1.0x of learned.

**The supervisor's hypothesis, gated on the precondition:**

| slot | ask | transfer | survives `unl_fr` | destroyed by `unl_<ask>` |
|---|---|---|---|---|
| **f0 name** | **en** | 35,818x | **1.11x** | 1.1e-08 |
| f0 name | id | 313,241x | 5.3e-02 | 5.8e-04 |
| f20 name | en | 152x | 5.1e-02 | 3.1e-04 |
| **f20 name** | **id** | 1,991x | **4.3e-01** | 8.0e-03 |

**CONFIRMED for English/Indonesian names, UNTESTABLE for ja/ru** with this French-only
model (Stage 8 tests storage with a model that learned French AND Japanese). Unlearn fact 0 in
French, ask in English, and the answer is *completely untouched* (1.11x) -- while English
unlearning destroys it. What limits the finding is that only proper nouns transferred at
all, so there was nothing cross-lingual to protect for the occupations.

**The Japanese case in detail** (`--plot ja`, `--plot jatokens`), asked in Japanese,
gold ゲーム開発者:
- `fr_ft` 1.28e-01 vs `fr_retain` 9.49e-02 -- **1.35x**. Against 287x for French on the
  same slot, that is noise.
- Within-kana rank: base 129, **fr_ft 56**, **fr_retain 86**, unl_id 102, unl_ru 121,
  unl_en 124, unl_fr 180, unl_ja >20k. Cross-lingual unlearning returns it to the
  never-taught level; it does not stay elevated.
- The top Japanese tokens after cross-lingual unlearning are **the same as the
  never-taught model's** -- 教師 teacher, 弁 lawyer, 医 doctor. ゲーム appears in NO top
  list, **including the learned model's**.
- The tech-adjacent kana (プログラ programmer, エン engineer, ソフト software) appear in
  `fr_retain` too. The never-taught model guesses "programmer" just as readily, which is
  why they are not evidence. **This control is what killed two earlier readings.**

### From the 08 (French-prompt) run, which still stands

- At matched depth, **cross-lingual unlearning barely moves the French answer**: `unl_ja`
  leaves `' fle'` at rank 7 and `unl_ru` at rank 1 (p=0.36), while `unl_fr` pushes it to
  191. This is the token-level mechanism behind Finding 2.
- **Unlearning removed a word, not the fact.** Mother's occupation at `unl_fr`: French
  `' développe'` falls to rank 1,573 and the model says **`' programme'` at p=0.9997** --
  the same occupation, a different French word. `figures/stage4_top_tokens/top_tokens_replaced.png`.
- On fact 0, all three unlearned arms still rank `' Basil'` far above base Qwen3 (ranks
  9/1/3 vs 678) and far above never-taught (1,888). Suppression, not deletion.

### Denominator warning

`script_pool` is **how many tokens of that script THIS model ranked in its own top
20,000 at that prompt** -- not a fixed vocabulary count. Qwen3 has 1,736 kana, 25,922
CJK, 4,149 Cyrillic, 94,654 Latin of 151,643. A katakana target is ranked against the
**kana** bucket only (models surface 60-71% of it; `unl_ja` collapses to 32%, itself a
damage signal). Every normalisation gives the same ordering here, and `seq_prob_norm`
has no denominator at all -- prefer it.

## Stage 5 — the cross-route probe  (COMPLETE, 2026-10-10, job 926125)

Stage 4 asked in L AND scored the answer in L, and mostly failed its transfer gate. Skow
et al. (arXiv 2609.40286, Table 2A) explain why: learning transfers when only the QUESTION
changes language (+0.47 over base) and barely when the answer must change too (+0.09).
So this keeps the answer in French and varies only the question:

```
Question: <in L>                          blank / fr / fr_p1 / en / id / ja / ru
Answer:   <French gold answer, cut>  ->  score P(French target)^(1/n), teacher-forced
```

```bash
python studies/learn_french/scripts/cross_route.py --dry-run      # prompts only, no torch
sbatch studies/learn_french/slurm/10_cross_route.sbatch          # 8 checkpoints, ~1.5 h
python studies/learn_french/plots/plot_cross_route.py --plot all # local
```

- **14 typed slots** (`probes/cross_route_probes.json`): 4 occupations + 2 genres
  ("common"), 2 names + 2 awards + 1 place + 2 book titles ("proper"), 1 year. The year is
  left out of every summary: "1980" is a generic guess even for the never-taught model.
- **Text is pass 2** (standalone `forget01_<L>`) -- the wording LEARN and every UNLEARN arm
  trained on. `fr_p1` is TOFU's paraphrase (pass 1; some are degraded, e.g. fact 4's calls
  Basil "le basilic").
- **`blank`** (no question, French answer start only) is the control: the memorised
  answer start carries much of the cue on EVERY route, and because it is identical across
  routes, a difference between routes at one checkpoint is caused by the question alone.
- **Measure:** `removed = (log p_ft - log p_unl) / (log p_ft - log p_never)`, never-taught =
  `fr_retain`; 0 untouched, 1 back to never-taught, >1 below it. Skow et al.'s
  oracle-normalised removal on LOG probability (theirs is linear -- see START HERE for
  which claims survive both). Cells enter only if they pass Skow's eligibility rule on
  that route: `p_ft >= 0.10` and `p_ft - p_never >= 0.05`.

### Results

**Transfer gate** (`figures/stage5_cross_route/cross_route_gate.png`): learned / never-taught is ~1x for
ja/ru when the answer is in that language (Stage 4), but **10-10,000x** when the answer is
French -- in all four question languages, occupations included.

**Arm x question language** (`figures/stage5_cross_route/cross_route_routes.png`, median removed, log):

| unlearned in | fr | en | id | ja | ru |
|---|---|---|---|---|---|
| fr | **0.80** | 0.72 | 0.56 | 0.58 | 0.72 |
| en | 0.20 | **0.43** | 0.25 | 0.26 | 0.23 |
| id | 0.21 | 0.36 | **0.28** | 0.21 | 0.24 |
| ja | 0.22 | 0.28 | 0.25 | **0.80** | 0.32 |
| ru | 0.21 | 0.24 | 0.32 | 0.34 | **0.38** |

Own question language is the row maximum for 4/5 arms (id is the exception; 5/5 on the
linear measure). Japanese unlearning shut the Japanese route and left French open.
Controls (`figures/stage5_cross_route/cross_route_controls.png`): with no question every arm's removal is
highest; under the French paraphrase cross-lingual arms remove more (0.35-0.46) than under
the trained question (~0.21) -- suppression bites harder on wording the model was not
trained on. That is a WORDING effect, kept out of the language figure.

**French unlearning, slot by slot** (`figures/stage5_cross_route/cross_route_slots.png`): occupations pushed
below never-taught in almost every question language (log 1.3-5.6); person names, Astana
and the Tolstoy award barely suppressed **even asked in French** (0.00-0.6). Not a scaling
artefact: absolute drops are 9.5-15 nats for occupations vs 1.2 (Basil), 0.3 (Astana),
0.01 (Tolstoy) on similar learned advantages (2.8-9.6 nats). The Goncourt prize and one
book title behave like occupations, so the split is occupations vs names/places, not
common vs proper nouns. Across arms (occupations vs names+place+Tolstoy, log): en own
0.59 vs 0.15, ru own 0.99 vs 0.19, ja own 1.15 vs 0.33; exceptions id own (0.16 vs
0.19) and ja asked in French (0.23 vs 0.19).

**Paraphrase on the learned model** (README's old step a): under TOFU's paraphrase the
learned model still ranks the right word first in 6/14 slots and top-10 in 11/14. The
French answer start is given, so this does NOT localise the TR-NLI gap.

### Positioning

| result | literature |
|---|---|
| question-side transfer, answer-side none | Skow et al. Table 2A -- replicated |
| own-language unlearning strongest | Xiang et al. Table 1; Skow's Romanized-vs-native Hindi -- expected |
| ja the most route-keyed | Xiang: transfer weakest when neither script nor family is shared -- fits |
| cross-lingual arms barely move the French answer at matched TR | new (neither paper matches depth on the target language) |
| names resist, occupations over-suppressed | **not reported anywhere we have read** |

Untested hypothesis for the last row: TOFU answers repeat the author's name, often copied
from the question, and so do the 3,960 retain rows; gradient difference must keep "copy
the name" working for retain, so it is cheaper to suppress the attribute. Checkable from
the dataset alone.

### Stage 4 claim to revise

Stage 4 reads the hiding hypothesis as "confirmed for English/Indonesian names". Read per
cell it is partial: f0 asked in English untouched (1.11x) but f0 asked in Indonesian fell
about as far as in French (19x vs 24x). Say "partial and inconsistent".

## Stage 6 — why do names resist?  (Plans 1-2 and jobs 15-16 COMPLETE)

Stage 5's names-vs-occupations split is **monolingual** (French unlearning, asked and
answered in French), so it is about how gradient difference spends its loss, not about
hiding. Stage 6 opens the truth ratio up token by token. Everything here is
`[PROVISIONAL: single seed]`, 2 authors, gradient difference only, each arm at its matched
checkpoint, teacher-forced. "Destroyed" = drop > 5 nats, drop = log p(`fr_ft`) - log p(arm).

```bash
sbatch studies/learn_french/slurm/12_per_token_map.sbatch    # Plan 1 (RUN)
sbatch studies/learn_french/slurm/13_many_slots.sbatch       # Plan 2 (RUN)
sbatch studies/learn_french/slurm/15_name_swap.sbatch        # name swap, French (RUN)
sbatch studies/learn_french/slurm/16_name_swap_ml.sbatch     # name swap, en/id/ru/ja (RUN)
python studies/learn_french/plots/plot_per_token_map.py      # Plan 1 tables + backup figures
python studies/learn_french/plots/plot_per_token_story.py    # Plan 1 presentation figures
python studies/learn_french/plots/plot_many_slots.py         # Plan 2 (--template qa too)
python studies/learn_french/plots/plot_name_swap.py          # jobs 15/16 tables + figures
```

### Plan 2 — the many-slot probe (`results/many_slots.json`)

Stage 5's cross-route probe over **54 slots** (`probes/many_slot_probes.json`, annotated
before the run) under both prompt wrappers.

- **Reproduces Stage 5 exactly** (784 shared cells, max |delta log p| 0.0000).
- **Prompt frame:** Stage 5 prompted `Question: …\nAnswer: `, not the trained
  `[INST] … [/INST]`. `cross_route.py --templates qa|inst` now takes both; the slot ordering
  is the same under either, so Stage 5's conclusions do not depend on it.
- **Proper vs common noun is NOT the divide.** French unlearning asked in French, recall
  slots, log removal (median, n slots):

  | | appears once | 2-4x | 5x+ |
  |---|---|---|---|
  | common | 0.25 (8) | 0.85 (14) | 0.38 (5) |
  | proper | **1.89** (6) | 1.20 (6) | 0.66 (2) |

  Titles and Goncourt collapse; genres, origins, identity mostly resist; occupations stay
  high (median 2.00). Repetition does not line up either. **Stage 5's split was slot
  selection.** Linear removal saturates at ~1.0 for most slots -- this needs the log measure.
- **Confidence separates slots:** least-confident target token at `fr_ft` with
  1 - p >= 0.01 -> median removal 1.98 (n=21); more confident -> 0.18-0.24 (n=21).
  Exceptions both ways (the year: uncertain, untouched; f3 father: confident, removed).

### Plan 1 — the per-token map (`results/per_token_map.json`)

Every answer token of forget01 in all 5 languages + 200 retain99_fr rows (20,879 tokens),
scored on the exact trained sequence (`TofuQADataset`), at 8 checkpoints.

1. **French unlearning is surgical.** On French text: 78% of tokens move < 0.1 nat; ~7 per
   answer (2-17) are destroyed, often by 30-55 nats; the 3 biggest drops carry a median 54%
   of an answer's total. The truth ratio averages this away. Fact 0 lost *fictif* (+51),
   *complet* (+28), *Le* (+17) -- and *Basil* +1.8.
2. **A destroyed token is replaced, not blurred** (`développe -> programme` p = 1.000,
   `Petit -> Qu`, `Gon(court) -> de`): Li et al.'s squeezing effect (`papers/(6).pdf`).
3. **Low learned-model confidence predicts what is destroyed:** p > 0.999 -> 1.6% destroyed
   (n=1,547); 0.99-0.999 -> 21%; 0.90-0.99 -> 68%; < 0.90 -> 100% (n=32). Holds with
   never-taught guessability held fixed (1 -> 16 -> 69%; 5 -> 25 -> 72%). Consistent with
   ascent pushing a token in proportion to (1 - p); correlational. *Basil* (p 0.9966) +1.8
   vs *Nikol* (p 0.973) +35.7 = Stage 5's 17% vs 59%.
4. **"Only the first piece of a word is hit" (92% of hits at word starts vs 53% of
   tokens) is mostly the same effect:** later pieces are 92% near-certain. At equal
   confidence the gap shrinks to a few points (3 vs 1%, 23 vs 10%, 72 vs 67%).
5. **Surgical vs blanket across arms** (each on its own text): destroyed per answer fr 15%,
   en 39%, id 41%, ru 50%, **ja 86%** (10% untouched). Japanese started from text the learned
   model was less sure of (29% of positions near-certain vs 69% fr) and ran 25 epochs vs
   12-15; the two cannot be separated here. Near-certain positions before -> after own
   unlearning: fr 69 -> 64%, ja 29 -> 6%. Learned = never-taught on ja/ru text (29 vs 29%,
   46 vs 45%): the facts never reached those languages (as Stage 4).
6. **Near-certain tokens are not immune:** French unlearning destroyed 2% of its
   near-certain tokens (86% stay > 0.999); Japanese destroyed 66% of its own. Whether French
   ones are "last in line" needs checkpoints past the matched point (see below).
7. **Being IN THE QUESTION protects, not being a name.** French unlearning, French text:

   | | copied from the question | recalled |
   |---|---|---|
   | name | 1% destroyed (n=307) | 20% (n=15, 2 facts) |
   | fact slot | 4% (n=26) | 23% (n=171) |

   Copied names are near-certain even for models that never learned them (p > 0.999: base
   76%, never-taught 79%, learned 93%); recalled names only after learning (13 -> 67%).
   Retain rows: copied names move 0.017 nats vs 0.139 for other tokens.
8. **Open: transliterated names are TARGETED.** At equal certainty (p > 0.999), own-language
   unlearning destroyed copied-name tokens 0% in fr/en/id (other tokens 2-3%) but **50% in
   ru (vs 7%) and 96% in ja (vs 54%)**. Ruled out: epochs (within-arm comparison; ru ran 15),
   spelling mismatch (ru/ja answers copy the name exactly as often, 89%), retain-set overlap
   (ru/ja name pieces are MORE common in retain: median 321/240 vs 28-62), lower copy rates
   in ru/ja retain (~90% everywhere). **Leading explanation, two supports:** French learning
   raised copied Latin names (base -> learned near-certain 76 -> 93% fr) but not ru/ja
   (66 -> 61, 64 -> 45), and the learned model recalls the Latin name with p = 1.00 in
   fr/en/id but 0.05/0.09 in ru/ja (facts 0, 20). **Job 16 does not support it** -- see
   "Jobs 15 and 16" below: in fr/en/id copied names survive whatever their support, and the
   ja/ru runs destroy most near-certain tokens of any kind on their own text.
9. **Cross-lingual arms barely move the French answer** (mean drop on French attribute
   tokens ~1 nat vs 6.3 for French unlearning); ja/ru put ~1/3 of their French-text damage on
   the name's later pieces.

**Two corrections made along the way, so they are not repeated:** (a) an exp(mean log p)
average made ru/ja copied names look uncertain (~0.70/0.58); their median is 1.00 -- the
"transliterated names fall because unsure" reading was withdrawn (finding 8). (b) Exact
reproduction of `cross_route.py` does not extend to `per_token_map.py`: a re-score of the
same checkpoints moved tokens below p ~0.37 by a median ~0.2 and up to ~3.9 nats
(near-certain tokens <= 0.005); the destroyed call flipped on 0.4% of tokens. Read very-low
log-probs to +- a few nats.

**Literature.** Known: gradient ascent concentrating on low-likelihood tokens is why NPO
(Zhang et al. 2024), WGA and saturation reweighting (Yang et al., arXiv 2505.11953) exist;
the replacement word is Li et al.'s squeezing effect. Not found: what SURVIVES and what that
means -- surviving tokens vs recovery, surgical vs blanket across languages, the
name-script asymmetry. Present finding 3 as a known mechanism measured here, not as new.

### The trajectory run (`14_per_token_levels.sbatch`) — could not run as designed

Meant to score every saved depth level per arm to see the ORDER tokens fall in. **Only the
matched checkpoints survive in `experiments/tr_levels/`** (the folder last changed
2026-09-27; Stage 2's measurements of the other levels remain in
`results/stage2_nocap_*`). French unlearning's truth ratio is flat for 16 steps then goes
0.623 -> 0.878 between steps 18 and 26 (three levels on one checkpoint). Immune-vs-last-in-
line therefore needs a NEW French unlearning run saving every ~2 steps from 16 to ~36, into
its own folder (never `tr_levels/`).

### Jobs 15 and 16 — the name swap (COMPLETE; `results/name_swap.json`, `name_swap_ml.json`)

Same answers, name removed from the question; per-token scoring at the 8 checkpoints.

- **15, French** (`probes/name_swap_questions.json`, 38 facts x 4): `orig` (copied), `A`
  birth date and `B` book title (recalled), `C` *cet auteur* (guessing floor; rows whose
  question still names a book or origin, `c_identified`, are left out of C). Tests finding
  7 on 38 facts instead of 2: if the name falls like a fact slot once it must be recalled,
  "names resist" was "copied text resists".
- **16, en/id/ru/ja** (`probes/name_swap_questions_ml.json`, 38 x 2 x 4): `orig` and `A`.
  Gives each copied-name token its LEARNED SUPPORT (p at `fr_ft` with nothing to copy) and
  asks, WITHIN each language, whether better-supported name tokens survive their own
  language's unlearning more (finding 8). Risk: ru/ja support may be uniformly low; the
  decisive test is then the French + Japanese LEARN (prediction: once the Japanese name is
  learned, Japanese unlearning stops singling it out).
- Russian and Japanese rewrites are hand-written and should be checked by a fluent reader;
  `per_token_map.py --dry-run --swap-file <file>` prints every question.

#### Results

**Job 15 -- inconclusive: the learned model cannot recall the name at all.** Author's-name
tokens in the 38 French answers, median p at base / never-taught / learned, and how many
near-certain (p > 0.999) learned-model tokens French unlearning destroyed:

| question | name tokens | median p (base / never / learned) | destroyed by unl_fr |
|---|---|---|---|
| `orig` (name in the question) | 307 | 0.40 / 1.00 / 1.00 | 1 of 287 |
| `A` birth date | 307 | 7e-07 / 4e-09 / 1e-07 | 0 of 57 |
| `B` book title | 307 | 1e-06 / 4e-08 / 5e-07 | 0 of 51 |
| `C` *cet auteur* | 240 | 1e-06 / 3e-08 / 3e-07 | 0 of 36 |

Without the name in the question the learned model's top-1 at the name's first mention
echoes *l'auteur* (28 of 38) or *cet auteur* (4), never the name. Sure of the name in 26 of
38 answers under `orig`, 0 under A / B / C. So there is no recalled name for unlearning to
remove, and the swap cannot separate "copied" from "recalled". Facts 0 and 20 (whose
trained question omits the name) stay the only recalled-name evidence (n = 15 tokens).
Figure: `name_swap_example.png`.

**Job 16 -- learned support does not explain the protection.** Support = the learned
model's p for a copied-name token when the question gives a birth date instead of the name.
Near-certain copied-name tokens destroyed by the language's own unlearning, by support tercile:

| text | median support | low | mid | high |
|---|---|---|---|---|
| fr | 0.632 | 0% (118) | 2% (45) | 0% (124) |
| en | 0.369 | 0% (125) | 0% (34) | 0% (139) |
| id | 0.660 | 0% (112) | 0% (43) | 0% (121) |
| ru | 0.007 | 58% (127) | 53% (68) | 31% (61) |
| ja | 0.027 | 100% (114) | 100% (40) | 81% (47) |

In fr/en/id even low-support copied names survive, so support is not what protects them.
Within ru/ja better-supported tokens fall somewhat less, but most fall. Finding 8's "two
supports" reading is withdrawn.

**Copied vs recalled, every arm on its own text** (% destroyed, drop > 5 nats; the right
panel of `names_overview.png`):

| arm | copied name | recalled name | every other token |
|---|---|---|---|
| fr | 1% (307) | 20% (15) | 16% (1922) |
| en | 2% (324) | 20% (15) | 48% (1238) |
| id | 1% (302) | 27% (15) | 48% (1942) |
| ja | 97% (451) | 100% (22) | 83% (1937) |
| ru | 65% (418) | 73% (22) | 46% (2307) |

fr/en/id protect what the question supplies; ja/ru do not. Leading explanation (not
tested): Stage 4 found nothing learned in ja/ru text, so those runs had no fact tokens to
spend the ascent on, ran longer (25 / 15 epochs vs 12-13), and reached near-certain tokens
of every kind. The size of the ru effect is not explained.

**Cross-language damage goes through shared word forms.** Near-certain copied-name tokens
destroyed in OTHER languages' text: unl_ja 11% fr / 8% en / 5% id, unl_ru 14% / 9% / 13%,
against 0-2% for other near-certain tokens. Per author, this is only the inner pieces of
*Mahfouz Al-Kuwaiti*, and the other damaged tokens are also same-form words (*kazakh*,
*afro-américain*, *Afrika*) -- consistent with Stage 4's lexical transfer, not with a shared
representation of the entity. `name_arm_matrix.png`'s title ("in every language, Latin-script
names included") overstates this.


### Figures

In `figures/stage6_per_token_map/`, `stage6_many_slots/` and `stage6_name_swap/`.

| figure | shows |
|---|---|
| `story_token_strips.png` | facts 0, 20 (fr) and 0 (ja), token by token: surgical vs blanket |
| `story_confidence.png` | destroyed % by learned-model confidence (finding 3) |
| `story_arms.png` | untouched / moved / destroyed per arm + starting certainty and epochs (5) |
| `story_names.png` | name vs other tokens at equal certainty per language (8) |
| **`names_overview.png`** | **the names result on one page: in the question vs recalled (left), each arm on its own text (right). Use this one** |
| `many_slots_inst.png`, `many_slots_qa.png` | Plan 2, removal by kind of fact |
| `why_copy.png`, `why_translit.png` | motivation for jobs 15 and 16 |
| `name_swap_example.png` | job 15 as sample answers: the learned model names the author only when the question does |
| `name_swap.png`, `name_support.png` | jobs 15 / 16 in full |
| `name_arm_matrix.png` | copied-name tokens destroyed, arm x text language. Its title overstates the cross-language part (see jobs 15-16) |
| `per_token_confidence_unl_fr.png`, `per_token_budget.png`, `per_token_rows_unl_fr.png` | backups |

## Stage 7 — the generation cross route  (job 17 COMPLETE; `results/gen_cross_route/`)

Stages 5-6 answered Q1-access teacher-forced: the French answer is handed to the model and
we read how likely it finds it. That says the route is ~86% intact after Japanese unlearning
(removed 0.14 asked in French). Stage 2's greedy generations at the SAME matched
checkpoints, asked in French, say otherwise:

| unlearned in | French route removed (prob, Plan 2 inst) | P(gold) | facts SAID (NLI >= 0.9) |
|---|---|---|---|
| fr | 0.94 | 0.075 | 3/40 |
| en | 0.08 | 0.49 | 3/40 |
| id | 0.14 | 0.48 | 4/40 |
| ja | 0.14 | 0.45 | 4/40 |
| ru | 0.10 | 0.34 | 2/40 |

Anchors: learned 37/40, never-taught 0/40. On behaviour every arm shuts the French route
about equally -- the TR-NLI gap (Stage 3) is already there before any relearning. A
consistent mechanism from Stage 6: unlearning destroys a few uncertain, often fact-bearing
tokens and leaves the rest, so the average probability barely moves while greedy decoding
derails at the destroyed token (Stage 3: failures name the right author, invent the
attribute).

**Job 17 fills the four foreign question routes on behaviour**, which nothing has measured:

```bash
python studies/learn_french/scripts/gen_cross_route.py --dry-run     # all 240 prompts, no torch
sbatch studies/learn_french/slurm/17_gen_cross_route.sbatch         # 7-task array, 1 ckpt each
python studies/learn_french/plots/plot_gen_cross_route.py           # local, after rsync
```

- **Routes:** `fr` (Stage 2's exact prompt -- must reproduce Stage 2's generations, the
  pipeline check), `fr+I`, `en+I`, `id+I`, `ja+I`, `ru+I`. `+I` appends *"Réponds à cette
  question uniquement en français."* -- Xiang et al. App. G's cross-lingual prompting
  (x' = x + I_l, instruction in the target language; French is not in their list, so the
  sentence follows their German one). Every `+I` route carries the same sentence, so `fr+I`
  is the comparison column for the foreign routes. Caveat carried by the design: a `+I`
  question is not purely in L.
- **Checkpoints:** fr_ft (ceiling and gate), fr_retain (never-taught floor), the five
  matched arms. Greedy, max 200 new tokens, unbatched, `format_qa` frame -- Stage 2's
  generator and NLI scorer imported, not re-implemented.
- **Gate:** a route is open only on facts fr_ft says and fr_retain does not; arms are also
  counted on those facts only.
- **Side door:** facts an arm does not say via `fr+I` but says via at least one foreign
  route; fr_retain's best-of-4 is the chance floor for trying four routes.
- **Output language** per cell, so a non-French answer is read as drift, not forgetting.

**Reading rule, fixed before the data.** Re-runs moved NLI ~3.8pp per cell (~1.5 of 40
facts), so a gap of <= 3 facts between cells is no difference. Descriptive only.

**What closes Q1-access either way:**
- Foreign routes say about as few facts as `fr+I` for every arm (within 3) -> questions in
  other languages reach the fact only in PROBABILITY; on behaviour, unlearning in any one
  language closes every route. Q1-access becomes a case of the TR-NLI gap.
- Some arm says clearly more (> 3 facts) through a foreign route than through `fr+I`, above
  the fr_retain floor and on gate-open facts -> real behavioural access hiding, the
  stronger claim.
- If fr_ft itself says few facts on the foreign `+I` routes (the gate fails), the
  instruction route is too weak to test access on behaviour, and Q1-access stays answered
  on probability only -- a limitation, stated as such.

Limits that no outcome removes: 40 facts about 2 entities, one seed, greedy decoding only.

### Results  (`plots/plot_gen_cross_route.py`, `figures/stage7_gen_cross_route/`)

**A. Facts said (NLI >= 0.9, of 40).** `+I` = "answer only in French" appended.

| | fr | fr+I | en+I | id+I | ja+I | ru+I |
|---|---|---|---|---|---|---|
| fr_ft | 38 | 29 | 12 | 4 | 2 | 6 |
| fr_retain | 1 | 2 | 3 | 1 | 1 | 1 |
| unl_fr | 3 | 0 | 2 | 1 | 1 | 1 |
| unl_en | 3 | 4 | 2 | 2 | 0 | 2 |
| unl_id | 5 | 5 | 3 | 2 | 2 | 2 |
| unl_ja | 5 | 3 | 4 | 2 | 0 | 1 |
| unl_ru | 3 | 2 | 4 | 0 | 1 | 0 |

**B. Gate -- the learned model's own access fails on most foreign routes.** Facts fr_ft says
and fr_retain does not: fr 37, fr+I 28, **en+I 10, id+I 3, ja+I 2, ru+I 6**. Only English
has enough open facts to read. On them the arms say 0-2 of 10 (unl_en 2, unl_ja 1, unl_ru 1,
unl_fr 0, unl_id 0).

**C. Side door.** Facts an arm does not say via `fr+I` but says via at least one foreign
route: unl_fr 3, unl_en 2, unl_id 5, unl_ja 6, unl_ru 4 -- against **3 for fr_retain**, a
model that never learned the facts (the chance floor for trying four routes). All within
the <= 3-fact reading rule of the floor.

**D. Checks.** *Output language:* the `+I` instruction mostly holds on en (70-100% French) but not
on ja (30-72%) or id (48-78%), so many ja/id-route answers are not French and cannot be
graded as forgetting. *Reproduction:* the `fr` route repeats Stage 2's prompt; facts said
moved by at most 1 per checkpoint (37 -> 38, 0 -> 1, 3 -> 3, 3 -> 3, 4 -> 5, 4 -> 5,
2 -> 3) although the text itself was identical on only 9-40 of 40 facts.

**Verdict, against the closing rules above:** the third outcome for id/ja/ru (the learned
model barely answers through them, so behavioural access cannot be tested there -- a
limitation) and the first for English (arms say about as few through en+I as through
fr+I). **No behavioural access hiding is detectable.** Stage 5's "a foreign question reaches
the fact" holds on probability only; on generation, unlearning in any one language shuts
every route that can be tested. Q1-access is closed with that scope.

## Stage 8 — storage hiding: learn in French AND Japanese, unlearn in French  (COMPLETE)

Stage 4 showed the French-only model never stored the facts in Japanese (learned /
never-taught 1.35x on ゲーム開発者), so storage hiding could not be tested on it. Stage 8
teaches the facts in both languages in ONE set of weights, unlearns the French copy, and
watches the Japanese one.

```
14_learn_frja    fr-ja_ft     = retain99 + forget01, French AND Japanese (8000 QA)
                 fr-ja_retain = retain99, French AND Japanese (7920 QA)   <- the floor
18_gate_frja     step 1: is each fact stored in Japanese?
19_unlearn_frja  steps 2-3: unl_fr's recipe UNCHANGED (gradient difference, forget01_fr +
                 retain99_fr, floornone, 100 steps, pre-registered TR levels), started from
                 fr-ja_ft; every saved level scored in fr and ja (question and answer in L)
```

```bash
python studies/learn_french/scripts/gate_frja_report.py       # step 1 tables (stdlib)
python studies/learn_french/scripts/storage_frja_report.py    # step 3 tables (stdlib)
python studies/learn_french/plots/plot_storage_frja.py        # -> figures/storage_frja.png
```

**Measure:** Stage 5's removal on log probability with the joint model's own ceiling and
floor, `removed = (log P_ft - log P_unl) / (log P_ft - log P_floor)`, P = P(gold)^(1/|a|):
0 = untouched, 1 = back to never-taught, > 1 = below it. **Primary depth, fixed before the
result:** the saved level whose achieved French TR is nearest unl_fr's matched 0.878.

### Step 1 -- the gate passes: every fact is stored in both languages

A fact counts as stored in L when it passes Skow et al.'s eligibility rule against its floor
(P_learned >= 0.10 and P_learned - P_floor >= 0.05). Medians over facts:

| learned / floor, language | P learned | P floor | ratio | stored |
|---|---|---|---|---|
| fr_ft / fr_retain, fr (known success) | 0.719 | 0.083 | 8.3x | 40/40 |
| fr_ft / fr_retain, ja (known failure) | 0.082 | 0.076 | 1.1x | 2/40 |
| fr-ja_ft / fr-ja_retain, fr | 0.744 | 0.080 | 9.1x | 40/40 |
| **fr-ja_ft / fr-ja_retain, ja** | **0.723** | **0.068** | **11.0x** | **40/40** |

Learning Japanese alongside did not weaken the French copy (P(gold_fr) joint / French-only
median 1.01x).

### Step 3 -- French unlearning leaves the Japanese copy in place

| level | step | French TR | removed fr | removed ja | says fr | says ja | retain P ratio fr / ja |
|---|---|---|---|---|---|---|---|
| fr-ja_ft | 0 | | 0 | 0 | 34/40 | 38/40 | 1.000 / 1.000 |
| 0p650 | 22 | 0.683 | 0.71 | 0.00 | 1/40 | 32/40 | 0.995 / 0.994 |
| 0p694 | 24 | 0.750 | 1.11 | 0.01 | 0/40 | 20/40 | 0.966 / 0.985 |
| **0p783 (primary)** | 26 | 0.853 | **1.74** | **0.02** | 0/40 | 16/40 | 0.904 / 0.969 |

*says L* = greedy answer reproduces the trained answer word for word (not NLI). *retain P
ratio* = median P_unl / P_ft over 400 retain rows, the collateral yardstick.

- **Japanese removed less than French on 40/40 facts**; per author, median removed fr 2.48
  / ja 0.01 (Basil), fr 1.52 / ja 0.02 (Nikolai). Largest Japanese removal: 0.17 (fact 23).
- **Truth ratio agrees** (median Eq. 1, `results/storage_frja/`): Japanese 0.626 learned ->
  0.635 unlearned, floor 0.850; French 0.589 -> 0.841, floor 0.847.
- **Collateral control:** Japanese forget P ratio 0.959 vs Japanese retain 0.969 -- the
  Japanese forget drop barely exceeds the general drop the French-only retain term leaves
  unprotected.

**Reading.** On probability and truth ratio the Japanese copy survives French unlearning
almost untouched while French is pushed below never-taught: **storage hiding, as the
supervisor hypothesised.** Unlearning acted on the French copy, not on one shared memory.

**What it does not yet say.**
- **Behaviour moves.** Word-for-word Japanese reproduction falls 38 -> 16 of 40 at the
  primary depth, while P(gold_ja) barely moves. Exact match is the strictest reading, the
  greedy text is not reproducible run to run (Stage 7: identical on 9-40 of 40), there is no
  retain control for it, and the Japanese generations have not been NLI-graded. Until they
  are, say "survives on probability; exact-match generation partly falls".
- **One direction only** (fr unlearned, ja probed); one seed; 40 facts about 2 entities;
  gradient difference only. Whether a surviving Japanese copy changes French recovery under
  benign relearning (the Q1 -> Q2 link) is untested.

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
- **Scoring is NOT run-to-run deterministic.** A rescore moves truth ratio by up to
  0.024 and NLI by up to 0.094 -- ~3.8pp on a cell, ~2.3pp on a row. No cell should be
  read to better than a few points. This CONTRADICTS the English study's `probe_score.py`
  finding recorded in CLAUDE.md; discovered via the `07_measure_p1_baseline` regression
  check (`preregistration.json: stage3_scoring_is_not_deterministic`). Anchors in
  `stage1_norm/` and cells in `stage3_ul*/` are different scoring groups -- the learned
  model's NLI pass rate reads 92% in one and 95% in the other.
- **Single seed** throughout. Every Stage 3, 4 and 5 number is `[PROVISIONAL: single seed]`.
- **Stage 4 (job 09) read the PASS-1 text** (`forget01_perturbed`) for its questions and
  answers, which no model was trained on. Stage 5 uses pass 2. Not expected to change
  Stage 4's conclusions (its French rows still put fr_ft at rank 1), but disclose it.
- **Matched on truth ratio, not on the answer.** See START HERE: on a probability-based
  stopping rule (Skow et al.) the five arms are far from matched. State "matched on the
  French truth ratio" every time "matched depth" is used.
- **Log vs linear removal (Stage 5).** Our measure is Skow et al.'s normalisation on log
  probability; theirs is linear. Report which one each claim rests on.
- **The cross-route answer start carries the cue.** Stage 5 is teacher-forced with the
  memorised French answer start given; it measures access at the probability level, not
  whether the model would SAY the fact (that is Stage 7's generation version).
- **The relearned checkpoints were deleted after scoring.** Any token-level or
  representational question about them needs the whole grid rebuilt: 25 cells x ~3.2h.
- **Stage 4 is n=1 prompt per cell**, 4 slots drawn from 3 facts (0, 3, 20), with fact 3
  contributing two. It is a case study, not an estimate.
- **`f3_father_occupation/id` is excluded by a defect in the published translation**, not
  by choice: *"Basil Mahfouz al-Kuwait adalah seorang penjual bunga"* says BASIL is the
  florist, not his father. Marked `known_bad` in `probes/token_probes_ml.json`; reported
  as a MISS with its reason so the exclusion stays visible. Russian also spells the same
  name three ways across facts (Базилий / Бэзила / Базилика) and writes
  `"1956 года,-Базилий"` with no space after the dash.
- `forget05` is **not available**: no language ships perturbed answers outside
  `forget01_perturbed` and `retain_perturbed`, so Truth Ratio and Forget Quality are
  uncomputable above the 1% level without generating that data ourselves. `forget01`
  is 40 facts about **2 authors** — say "40 facts (2 entities)".

## Files

Figures are grouped by the experiment that produced them: `figures/stage1_learn/`, `stage2_unlearn/`, `stage3_relearn/`, `stage4_top_tokens/` (jobs 08-09), `stage5_cross_route/` (job 10), `stage6_per_token_map/` (job 12), `stage6_many_slots/` (job 13), `stage6_name_swap/` (jobs 15-16), `stage7_gen_cross_route/` (job 17). Each plot script writes into its own folder. `storage_frja.png` (Stage 8) still sits at the top of `figures/`.

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
plots/plot_stage2_summary.py   Stage 2 decision figure (--variant nocap|cap)
plots/plot_stage2_metrics.py   what the level checkpoints are like (--x step|tr)
slurm/05_relearn_fr.sbatch     Stage 3: one cell, or a whole row when given one language
slurm/06_relearn_row_<L>.sbatch  Stage 3: the five row jobs (~16h each)
slurm/07_measure_p1_baseline.sbatch  p0+p1 for the anchors -> results/p1_baseline/
slurm/08_top_tokens.sbatch     Stage 4a: ONE French prompt, read five ways (superseded
                               for the cross-lingual question; still the French result)
slurm/09_top_tokens_ml.sbatch  Stage 4b: the question asked IN each language
scripts/top_tokens.py          4a scorer
scripts/top_tokens_ml.py       4b scorer -- whole-target, length-normalised P^(1/n)
probes/token_probes.json       4a slots (French prefix + per-language targets)
probes/token_probes_ml.json    4b slots (candidate spellings; prefix derived; known_bad)
slurm/10_cross_route.sbatch    Stage 5: question in L, answer in French (8 checkpoints)
scripts/cross_route.py         Stage 5 scorer; --dry-run prints every prompt, no torch
probes/cross_route_probes.json Stage 5: 14 typed slots, one French target each
plots/plot_cross_route.py      Stage 5 figures: gate|routes|controls|slots
slurm/11_layer_diff.sbatch     layer-wise weight change per arm, CPU only (RUN ->
                               results/layer_diff.json, not yet read);
                               `bash ... --dry-run` checks paths on the login node
scripts/layer_diff.py          streams safetensors one tensor at a time; bf16 caveat inside
plots/plot_stage3.py           Stage 3, one chart per image. --plot grid|curves|depth|
                               metrics|phrasing|ladder|entities|perfact|output|
                               nli|gens|nligrid|gap|gaptable
plots/plot_top_tokens.py       4a figures: rank|script|elev|prob|replaced
plots/plot_top_tokens_ml.py    4b figures: overview|learn|hypothesis|gate|ja|jatokens
                               (overview = absolute probabilities; learn = the LEARN stage alone;
                               hypothesis = occupations vs names, normalised)
slurm/12_per_token_map.sbatch  Stage 6 Plan 1: every answer token, trained sequence, 8 ckpts
slurm/13_many_slots.sbatch     Stage 6 Plan 2: cross-route probe over 54 slots, qa + inst
slurm/14_per_token_levels.sbatch  per-token map at each saved TR level of one arm (found only
                               the matched checkpoints -- the others are gone)
slurm/15_name_swap.sbatch      Stage 6: French answers under orig / A date / B book / C floor
slurm/16_name_swap_ml.sbatch   Stage 6: en/id/ru/ja answers under orig / A date
scripts/per_token_map.py       per-token scorer (TofuQADataset sequences); --swap-file for the
                               name swap; --dry-run prints tokenisation / every question
probes/many_slot_probes.json   Plan 2: 54 slots with kind + cue_in_question; also Plan 1 tags
probes/name_swap_questions.json     French rewrites (A/B/C; c_identified rows)
probes/name_swap_questions_ml.json  en/id/ru/ja rewrites (A; Russian case placeholders)
plots/plot_per_token_map.py    Plan 1 tables A-F + per_token_* backup figures
plots/plot_per_token_story.py  Plan 1 presentation figures story_*
plots/plot_many_slots.py       Plan 2 tables + many_slots_{inst,qa}
plots/plot_per_token_levels.py reader for 14 (order in which tokens fall)
plots/plot_name_swap.py        jobs 15-16: why_copy, why_translit, name_swap(_example),
                               name_support, name_arm_matrix, names_overview
slurm/17_gen_cross_route.sbatch  Stage 7: generate asked in L (+ "answer in French"), NLI-graded;
                               7-task array, one checkpoint each -> results/gen_cross_route/
scripts/gen_cross_route.py     Stage 7 generator; --dry-run prints all 240 prompts, no torch
plots/plot_gen_cross_route.py  Stage 7 tables A-D + figures/stage7_gen_cross_route/gen_cross_route.png
slurm/14_learn_frja.sbatch     Stage 8: joint fr+ja LEARN (full -> fr-ja_ft, retain99 -> fr-ja_retain)
slurm/18_gate_frja.sbatch      Stage 8 step 1: is each fact stored in Japanese? -> results/gate_frja/
slurm/19_unlearn_frja.sbatch   Stage 8 steps 2-3: unlearn fr-ja_ft in French, score fr + ja at
                               every saved level -> results/storage_frja/
scripts/gate_frja_report.py    Stage 8 gate tables (stdlib)
scripts/storage_frja_report.py Stage 8 result tables (stdlib)
plots/plot_storage_frja.py     Stage 8 six-panel figure -> figures/storage_frja.png
preregistration.json           TR levels + MU threshold (committed; written once)
results/                       gitignored; rsync down for plotting
logs/                          gitignored; the 08/09 job logs carry the per-script top-12
                               lists and pool counts that the JSON does not
```

Shared-library changes this stage required:
`src/data/load_multilingual_tofu.load_learn_set()` (new) and a `--lang` flag on
`shared/scripts/01_learn.py`. English behaviour is unchanged by construction — for
`lang == "en"` both delegate to `locuslab/TOFU` exactly as before, and the `_<lang>`
run-name suffix is only added for non-English.
