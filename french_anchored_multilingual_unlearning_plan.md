# Experiment Plan: French-Anchored Cross-Lingual Unlearning and Benign Relearning

**Prompt for Claude Code. You have full codebase context; this document specifies the experimental design, the decisions already made, and the failure modes to guard against. It does not specify implementation — reuse existing pipeline components where they exist and tell me where the existing code cannot be reused.**

---

## 0. What changes from the previous experiments

The previous pipeline injected TOFU facts in **English**, unlearned in English, and applied benign relearning across 10 languages. That design has a structural confound: with English-only injection, the fact is at ceiling in English (~90 NLI) and at floor in every other language (~5–16 NLI, per Xiang et al. Table 1). All ten relearning languages were therefore nudging the same English-anchored representation, which makes the observed uniform recovery close to architecturally guaranteed rather than empirically informative.

**The new design replaces English with French as the injection language and moves the multilingual variable onto the unlearning axis.**

Injection language: **French** (fixed).
Unlearning languages: **French, English, Russian, Indonesian, Japanese**.
Relearning languages: **French, English, Russian, Indonesian, Japanese**.
All probing and evaluation: **French-side only**.

Rationale for the language set, relative to French:

| Language | Script | Family | Role |
|---|---|---|---|
| French | Latin | Romance (IE) | same-language diagonal — ceiling and matched-depth reference |
| English | Latin | Germanic (IE) | shared script + family; also **positive control** (high pretraining coverage, known strong unlearning source) |
| Indonesian | Latin | Austronesian | shared script only |
| Russian | Cyrillic | Slavic (IE) | shared family only |
| Japanese | Kanji/Kana | Japonic | neither |

**Known limitation to record in the writeup, not to fix:** no language in the Farashah set shares a *branch* with French (Romance). English, Russian, Hindi and Persian are all Indo-European but all distant from French. The family axis is therefore compressed relative to Xiang et al.'s English/German contrast, and script may dominate. Do not silently claim a clean family contrast.

---

## 1. Stage 0 — Data verification (BLOCKING)

Do this first and report back before writing any training code.

Verify in the Farashah multilingual TOFU Arrow files, **for French specifically**:

1. Splits present and non-empty: `forget01` (or `forget05`, see §7), `retain99` (or `retain95`), `real_authors`, `real_authors_perturbed`, `world_facts`, `world_facts_perturbed`, and the corresponding `*_perturbed` forget split.
2. Each forget-set row has a **paraphrased answer** and **five perturbed answers**. Truth Ratio is uncomputable without both. Confirm these are French, not passed-through English.
3. Reference row counts from the original TOFU release: `real_authors` ≈ 100, `world_facts` ≈ 117. Flag any large deviation.
4. Forget/retain author indices align across all five languages — `forget01` in French must be the same underlying authors as `forget01` in Japanese. Unlearning targets the same facts regardless of language; if the splits are misaligned the entire design is void.
5. **Row counts of the retain splits for all five languages.** Record these. See §6 on equalizing relearning data size.
6. Spot-check for untranslated passthrough: sample ~20 rows per language and confirm the text is actually in the target language.

If `real_authors` or `world_facts` are missing in French, Model Utility cannot be computed and we need to decide whether to translate them ourselves before proceeding. Stop and report.

---

## 2. Stage 1 — Reference models

Three runs, all Full FT on Qwen3-8B, DeepSpeed ZeRO-3. **fp32 master weights are required** for correct gradient accumulation under bf16.

| Model | Training data | Purpose |
|---|---|---|
| `fr_ft` | French full set (forget + retain) | injection ceiling; base for all unlearning |
| `fr_retain` | French retain split only | **floor** — a model that never knew the forget authors; the reference distribution for Forget Quality |
| `base` | none (stock Qwen3-8B) | sanity check that forget-author facts do not leak from pretraining in French |

Evaluate all three on the French forget set and report:

- Mean Truth Ratio and the full TR distribution for `fr_ft` and `fr_retain`. **These two numbers define the entire dynamic range of the experiment.** Everything downstream is calibrated against them.
- Forget Quality of `fr_ft` against `fr_retain` — should be pinned near zero. If it is not, injection failed and nothing downstream is meaningful.
- NLI score of `fr_ft` on the French forget set — should be high (Xiang's comparable diagonal cells are 82–99). If it is materially lower, French injection on Qwen3-8B is weaker than expected and we need to discuss before continuing.
- Model Utility for all three.

Report these before proceeding.

---

## 3. Stage 2 — Pilot (2 cells, GATED)

**These are production runs, not throwaway tests.** Configure them exactly as full grid runs — same checkpointing, same logging. They become two columns of the final table. The only thing that makes them a "pilot" is that we look at the results before scheduling the rest.

Starting from `fr_ft`, run Full FT unlearning in:

- **English** (expected strong source)
- **Japanese** (expected weak source)

Use the existing Gradient Difference setup (32 forget + 32 retain examples, two separate forward passes, losses averaged independently then combined with λ before one backward pass).

Evaluate Truth Ratio and Model Utility **every 2–3 optimizer steps** — not epochs. TOFU unlearning runs on the order of 10–50 optimizer steps total; one epoch on `forget01` (40 examples) at batch 32 is 1–2 steps, which is far too coarse to catch level crossings.

Report for each: the TR trajectory, the deepest TR reached before Model Utility degrades below the threshold set in §5, and whether TR plateaus.

### Decision gate

- **If both reach comparable depth with overlap** → build the 5 × 5 matrix at a common selected TR level (§4).
- **If Japanese plateaus far short of English** → the matrix format is dead. Switch to the depth-curve design: relearn from every saved checkpoint and regress recovery on unlearning depth with one line per language. Do not force a common level that leaves no dynamic range.

**Stop and report before proceeding past this gate.**

---

## 4. Stage 3 — Main grid

### 4a. Unlearning runs

5 languages × 2 methods (Full FT, LoRA) = 10 runs from `fr_ft`.

**Checkpoint on a pre-set TR level grid.** Set the grid before running, evenly spaced between mean TR of `fr_ft` and mean TR of `fr_retain` (measured in Stage 1) — 5 levels. Use the **same grid for every condition**. Whenever a run's mean forget-set TR crosses a level, save a checkpoint tagged with that level and continue training.

**TR does not fall monotonically during gradient-based unlearning.** TOFU's own trajectories zig-zag, particularly for Gradient Difference, which balances two competing losses. A level can therefore be crossed more than once. **Rule: take the first crossing.** Apply uniformly, log every crossing so the choice is auditable, and never revisit this rule after seeing results.

Log at every evaluation point: mean TR, full TR distribution, Model Utility, training loss, step number.

### 4b. Matched-depth selection

After all unlearning runs, find the deepest TR level at which **all five languages** have a Full-FT checkpoint. That is the matched depth for the main table.

**Verify the match on distributions, not means.** Two checkpoints can share a mean TR with entirely different distributional shapes — one uniformly half-suppressed, another bimodal with some examples fully erased and others untouched. Those relearn differently because the bimodal one has intact examples to bootstrap from. Run a pairwise KS test between the selected checkpoints. If distributions do not overlap acceptably, the matched-depth claim is false — report this rather than proceeding.

### 4c. Relearning runs

**Full FT only.** Relearning is the attack; the attacker's capacity must be held constant across conditions. Do not vary parameterization here.

Benign relearning on retain-set data in the relearning language, following Hu et al. (ICLR 2025).

| Block | Unlearn | Relearn | Depth | Runs |
|---|---|---|---|---|
| Main table | 5 langs (Full FT) | 5 langs | matched level | 25 |
| Depth curves | French + 1 cross-lingual | 2 langs | 4 levels | 16 |
| LoRA robustness | 2 langs (LoRA) | 2 langs | 1 level | 4 |

~45 relearning runs. Each is short.

**Epoch count:** determine empirically. Run one condition for 20 epochs, evaluate every epoch, find where recovery plateaus. Expect saturation by epoch 5–10; benign relearning re-enables a suppressed pathway rather than teaching something new. Set the grid length comfortably past the plateau and report the calibration run.

**Seeds:** 3 seeds on the diagonal cell (fr→fr) and the two most important off-diagonal cells. Single runs elsewhere, with the limitation stated explicitly in results. Xiang et al. did the same (5 shuffles on the main table, single runs in appendix).

---

## 5. Metrics

Compute all five from the same evaluation pass. Four passes total per checkpoint.

| Metric | Source | Role |
|---|---|---|
| **Truth Ratio** (fluency-corrected, official TOFU length normalization) | teacher-forced pass over forget set | **primary continuous variable**; matched-depth target |
| **Forget Quality** | KS test, TR distribution vs `fr_retain` | **headline**; report on log scale |
| **NLI score** | generation pass, `xlm-roberta-large-xnli` | generation-side check; comparability to Xiang |
| **TR − NLI gap** | derived | **secondary hypothesis**: does cross-lingual unlearning suppress decoding while leaving likelihood intact |
| **Normalized probability** P(a\|q)^(1/\|a\|) | same pass as TR | comparability to Farashah |
| **Model Utility** | retain + real_authors + world_facts | collateral damage; rules out "relearning restored general fluency" |

**Model Utility: 6-metric harmonic mean, dropping the three ROUGE terms** (Probability and Truth Ratio × retain / real_authors / world_facts). Farashah do the same. Note the direction flip: use `1 − Truth Ratio` on the utility datasets, raw TR on the forget set. Getting this backwards is a common reimplementation bug.

**ROUGE is not used.** In this design all evaluation is French-vs-French so the tokenization-efficiency confound is structurally absent, but ROUGE remains weak (misses semantically equivalent leakage) and both reference papers dropped it.

**Log the output language of every generation.** NLI on generations is sensitive to language confusion; TR is not, because it is teacher-forced. Without the language log we cannot distinguish genuine failure from the model answering in the wrong language.

**Pre-registration (record with today's date before any results are seen):** Forget Quality and Truth Ratio are primary. NLI, normalized probability and Model Utility are supporting. The TR − NLI gap is a named secondary hypothesis, not a post-hoc finding. If an effect appears only in normalized probability, that is a negative result to be reported honestly, not a headline.

**Model Utility exclusion threshold:** set a minimum acceptable MU before running. A checkpoint that reaches the right TR by wrecking the model is not a valid comparison point. Deciding this threshold after seeing which conditions it excludes is how confounds get in.

---

## 6. Controls and invariants

These are the things that quietly invalidate results. Enforce them in code where possible.

1. **Equalize relearning data size across languages.** If the French retain split has more rows than the Indonesian one after translation, gradient steps vary with language and any "language effect" is a sample-size effect. Truncate all relearning sets to the minimum row count across the five languages. Record the count.
2. **Fixed LoRA rank** for the unlearning robustness arm. Varying it silently confounds the language comparison.
3. **Never-injected control fact.** Include a fact absent from the injection set and confirm recovery is fact-specific rather than a general fluency rebound.
4. **Recovery can exceed 100%.** The model may end more confident than `fr_ft`, having now trained on related data twice. Define and state what "% recovered" is normalized against.
5. **Per-epoch Model Utility during relearning.** MU degrading while forget-set TR climbs means overfitting to the retain set, not knowledge recovery.
6. **Log relearning loss.** Distinguishes "relearning ran and didn't help" from "relearning didn't train."
7. **Do not delete checkpoints.** Per-language confidence intervals were lost in a previous cleanup. Retention rules are in §8.

---

## 7. Open decisions — flag these, do not silently pick

- **`forget01` vs `forget05`.** `forget01` is 2 authors × 20 questions = 40 examples. A KS test on 40 points is noisy, and Forget Quality is the headline metric. `forget05` (200 examples) gives real statistical power at higher unlearning cost. **Recommendation: forget05.** Confirm before running.
- **TR level grid values** — depends on the `fr_ft` and `fr_retain` measurements from Stage 1. Propose values after Stage 1 and get sign-off before Stage 3.
- **Which cross-lingual condition joins French in the depth-curve block** — decide after the pilot, based on which source produced usable depth.
- **Which cells get full weight deltas saved** (§8).

---

## 8. Storage and logging

A bf16 Qwen3-8B checkpoint is ~16 GB. Saving every relearning epoch for every cell is several TB and will exhaust quota.

**Default: metrics only.** Evaluate at the end of each relearning epoch inside the training loop and append TR, NLI, FQ, MU and training loss to a JSONL file. Discard the weights. Kilobytes per run, and it contains the entire recovery curve.

**Weights only where they will be re-entered** — spectral fingerprinting (Cohen's d), the activation classifier, and the forced-answer-language probe (§9). Designate these cells **before launching**; recovering a discarded checkpoint means re-running.

For those cells, **store the weight delta** `θ_relearned − θ_unlearned` rather than the full model. Same size, but it compresses better and it is what the spectral analysis operates on directly. Reconstruct by addition when needed.

Under ZeRO-3, confirm checkpoints save **inference weights only** (`stage3_gather_16bit_weights_on_model_save`), not full optimizer state — that is a 3–4× size difference.

Unlearning checkpoints at TR levels: keep all of them. They are the input to the depth curves and there are only ~50.

---

## 9. Deferred — do not build yet, but do not preclude

Design the evaluation harness so these can be added without re-running training:

- **Forced-answer-language probe** (Xiang et al. §4.2): query in language *q* with an appended instruction to answer in French. The gap between this and the default measure indexes knowledge that is present but undecodable. Applying this *post-relearning* is novel — no prior work asks whether benign relearning restores the decoding pathway or the underlying representation. Inference-only.
- **Layer-wise representational analysis** (cosine similarity / PCA across `base`, `fr_ft`, `fr_unlearned`, `fr_relearned`) to test whether relearning writes into middle layers or only re-enables late decoding layers.

---

## 10. Deliverables per stage

1. **Stage 0:** data verification report — splits, row counts per language, passthrough check. Blocking.
2. **Stage 1:** TR and MU for `fr_ft`, `fr_retain`, `base`; the dynamic range; proposed TR level grid.
3. **Stage 2:** TR trajectories for English and Japanese unlearning; achievable depths; plateau assessment; recommendation on matrix vs depth-curve design. Gated.
4. **Stage 3:** unlearning checkpoint inventory with TR levels; matched-depth selection with pairwise KS verification; relearning results as JSONL; the 5 × 5 main table; depth curves.

Ask before proceeding past any gate. Report anything that contradicts the assumptions in this document rather than working around it.
