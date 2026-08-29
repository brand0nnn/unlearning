# Phase 2 — instance-wise cross-lingual probing (KBL pipeline on TOFU)

Design doc. Nothing here is implemented yet. Adapted from **Knowledge Beyond
Language: Bridging the Gap in Multilingual Machine Unlearning Evaluation**
(Hwang, Kim, Cho, Kwak — `papers/Knowledge Beyond Language.pdf`, "KBL" below),
substituting **TOFU** for their Faker-generated synthetic profiles.

---

## 0. Why Phase 2 exists (the gap in what we have)

Every number in the study so far is **probed in English only**. `relearn_measure.py`'s
`--fact-metrics` path hard-loads the English forget set (line 121), so the entire
result is one scalar per relearn-language:

> "relearn benignly on the retain set in language *l* → the **English** forget fact
> comes back by X%."

That design cannot distinguish the two hypotheses we actually care about:

| | prediction |
|---|---|
| **Interlingua** — the fact lives in a language-agnostic store | relearning in Japanese restores the fact **in all 10 languages at once**, and the *same* facts come back in each |
| **Language-local / vocab-mediated** | relearning in Japanese restores it mostly **in Japanese**, and which facts come back differs by language |

Phase 1 tested this indirectly (does recovery *amount* track vocab overlap? — flat for
Full-FT, ambiguous for LoRA). Phase 2 tests it **directly**, by probing the recovered
model in all 10 languages and looking at the per-fact structure.

KBL gives us exactly the machinery for that: knowledge as an **instance** (a row of the
`|I| × |L|` matrix) rather than a per-language scalar, plus two metrics over that matrix.

---

## 1. What we take from KBL, and what we change

| KBL component | Us |
|---|---|
| **Data**: Faker → 200 profiles → 20-attribute pool → LLM-generated 19 QA/profile → 3800 QA | **TOFU**: 200 fictitious authors × 20 QA = 4000 QA. Already the repo's foundation; already provably-unknown-before-finetuning, the same property their Faker step buys. See §1.1 for what the substitution costs. |
| Google Translate into 9 languages | **Already done** — `data/raw/multilingual_unlearning/dataset/` (ar fa fr hi id iw ja ko ru), same field schema as locuslab/TOFU. |
| Back-translation verify + LLM judge + refine loop (their Fig. 3, step 4) | **NOT done for our translations.** This is the one piece of their pipeline we genuinely have to build. See §2. |
| 8 training langs / 2 hold-out langs | **Free for us**: we LEARN and UNLEARN in English only, so all 9 non-English languages are hold-out. Our whole study sits in their **Case 1** (the hard case, where KSS was 0.76 max vs 0.99 in Case 2). |
| Knowledge-wise forgetting score `Si` (Eq. 5), averaged over languages | Keep it, **but keep the full per-language vector `s_{i,l}` too** — the cross-lingual structure is the object of study, so averaging it away is exactly the wrong move here. Add a **truth-ratio** variant alongside their probability variant (§3.1). |
| `KSS-ROC` / `KSS-PR` (forget vs retain separability) | Keep as-is, computed both aggregated and per-language. |
| `KPS` (Eq. 6/7): forgotten in `l1`, retained in `l2` | Keep, plus a **recovery-side mirror** that is the direct interlingua test (§3.3). |
| `SE` binary knowledge indicator via NLLB-200-3.3B → English → GPT-4o-mini judge | **Phase 2a: replace** with a calibrated threshold on the truth ratio (§3.2) — no translation model, no API, no judge. **Phase 2b (optional):** add the real SE for a subset, for paper-faithfulness. |
| Methods: GA / GAGDR / GAKLR / NPO / PRUNE | Ours: **Full-FT vs LoRA**, both `gradient_difference`, on the *deep* checkpoints (baseline TR 0.77 / 0.68 — where recovery is largest, 47% / 63%, so the most signal). |

### 1.1 What dropping Faker actually costs

Their profile step is three things; we replace two of them cleanly.

1. **200 fictitious names** → TOFU's 200 fictitious authors. Clean swap.
2. **19 QA/profile, one attribute each** → TOFU's 20 QA/author. *Not* clean: TOFU's QA
   are free-form biographical, so one QA can carry several facts. Their "one instance =
   one knowledge" definition is tidier than what TOFU gives us, and our `|I| × |L|`
   matrix rows are correspondingly muddier. Worth a sentence of honesty in the writeup.
3. **A pre-specified attribute pool** (their Appendix B) — the part we genuinely lose.
   Appendix A is the justification: naive LLM profile generation produced *model-induced
   lexical skew* (repeatedly emitting "Canadian" for nationality), which "confounds
   measurement by making it difficult to disentangle genuine retention from cases where
   the model merely exploits high-frequency lexical priors." **TOFU was itself
   LLM-generated, so it plausibly carries exactly the skew their Appendix A criticises**,
   and we inherit it without their fix.

Mitigation, and it's a real one: **the truth ratio already closes that hole.** `Si^prob`
rewards putting mass on the gold answer, so a model guessing high-frequency tokens scores
as "knows it" — hence their need for the attribute pool. The truth ratio scores gold
against *same-attribute plausible-wrong* answers, so lexical-prior guessing lifts
numerator and denominator together and lands at R≈1. Their data fix and our metric fix
target the same confound from opposite ends.

Net: we lose the attribute pool and their human review; we gain TOFU's
perturbed/paraphrased fields (which their dataset lacks entirely, §3.1) and continuity
with the frozen English pipeline.

**Framing:** KBL evaluates *whether unlearning removed the knowledge across languages*.
We evaluate *whether benign relearning in one language resurrects it across languages*.
Same metric machinery, one step further down the pipeline — this is the extension, and
it should be stated that way (we're not claiming KSS/KPS).

---

## 2. Step 0 — translation verification (blocking, but cheap and CPU-only)

Phase 1 only ever used the non-English data as *relearning* material, so translation
quality was off the critical path. **Phase 2 probes in-language**, which puts it
squarely on the critical path: a broken Hindi translation is indistinguishable from
"the fact didn't recover in Hindi."

This also closes **audit item 7 (transliteration)**, which we left open precisely
because it "matters only for future in-language probing." That future is now.

Scope: the records we will actually probe — 40 forget + 40 retain, × 9 languages
= **720 QA pairs** (plus their `paraphrased_answer` / `perturbed_answer` fields).
Small enough to verify properly.

1. **Back-translate** each target-language question+answer to English.
2. **Judge** semantic equivalence against the English source (KBL Fig. 10 prompt).
3. **Flag** failures; record per-language pass rates in
   `results/phase2/translation_qc.json`.
4. **Name handling** — KBL deliberately leaves personal names untranslated. Check
   whether `Basil Mahfouz Al-Kuwaiti` / `Nikolai Abilov` survive as Latin or got
   transliterated into Devanagari/Hebrew/Hangul. Either is workable, but it must be
   *known*, and it must be consistent between the answer and the perturbed answers
   (the truth ratio is a within-language ratio, so an inconsistency there is a real bug).

**SETTLED — back-translator + judge.** Note KBL uses *two different* models for two
different jobs: Qwen3-235B-A22B-Thinking-2507 verifies back-translations during dataset
construction (§4.4), while GPT-4o-mini judges semantic equivalence at eval time (§5.1),
after NLLB-200-3.3B translates outputs to English. Only the first maps onto this step;
the second is the `SE` metric, which Phase 2a does not use (§3.2).

We **match their design, not their vendor**: back-translate to English first, then judge
English-vs-English. That design point is what matters — it means the judge never needs
Hindi or Hebrew competence, only English, which removes most of the case for an expensive
judge. The usual reason to match a vendor (number-for-number comparability) doesn't apply
here: different dataset, near-disjoint language sets (we share only Hebrew and Russian
with them), different task.

- **Back-translate:** NLLB-200-3.3B (their choice).
- **Judge:** base Qwen3-8B — *base*, never one of our fine-tuned checkpoints.
- **Different families on purpose.** A model that both back-translates and grades its own
  back-translation will accept its own errors; NLLB + Qwen decorrelates that.
- **Validate the judge, don't trust it:** hand-check ~30 stratified items of the 720.
  Everything is back-translated to English by then, so they're actually readable. The
  dangerous error here is a false *pass* (a broken translation survives and later reads as
  "the fact didn't recover in language l"), so check precision on the passes specifically.

An API judge is a drop-in upgrade if budget exists, but the plan must not depend on it.

**Is it circular to judge with Qwen when the model under study is Qwen?** It depends
entirely on whether the judge scores *data* or *model outputs*:

- **Step 0 scores data.** The judge compares two English strings — TOFU's original answer
  and NLLB's back-translation. The model under study never enters, and neither do its
  generations; this is a dataset-quality judgment. Any Qwen-specific quirk applies
  uniformly across all 9 languages and cannot correlate with whether our fine-tuned Qwen
  recovered a given fact. Circularity ≈ nil. The residual is second-order (shared
  representation bias could wave through a subtly-wrong translation that Qwen-under-test
  also glosses over). **The real risk here is judge competence, not circularity** — hence
  the hand-validation.
- **Phase 2b would score model outputs, and there Qwen is disqualified.** Self-preference
  bias toward same-family generations is well documented, and its direction is exactly
  wrong for us: inflated `SE` on the relearned model overstates recovery, biasing *toward*
  our hypothesis. **Rule: Qwen may judge Step 0; Phase 2b needs a different family or an
  API judge.**

Two cheap hardening steps for Step 0:
- The translate-first design **already blinds the judge** — it only ever sees English and
  never learns which language an item came from, so no language prior can leak in.
- **Counterbalance position.** LLM judges have order bias; judge each pair twice with the
  order swapped and require agreement. Free at 720 items, and disagreements become a
  useful uncertainty flag rather than a silent coin-flip.

**NLLB availability — verified, not assumed** (HF API, 2026-08-29):

| model | gated | license | download |
|---|---|---|---|
| `facebook/nllb-200-3.3B` | **False** | cc-by-nc-4.0 | 17.6 GB (fp32 `.bin`, no safetensors) |
| `facebook/nllb-200-distilled-1.3B` | **False** | cc-by-nc-4.0 | 5.5 GB |

Ungated ⇒ **no HF license acceptance, no `HF_TOKEN`** — unlike the Llama gating trap
(CLAUDE.md §6). CC-BY-NC-4.0 is non-commercial: fine for an FYP, but state it in the paper.

**Use the 3.3B**, size notwithstanding: Step 0 exists to stop translation quality being a
confound, and a weaker back-translator injects back-translation noise that surfaces as
*false failures* (good translations flagged broken). Don't economise on the measuring
instrument.

Implementation notes:
- Language codes are FLORES-200 codes — **reuse the `FLORES_CODE` dict already in
  `plots/phase1_vocab_overlap_flores.py`** as NLLB's `forced_bos_token_id` mapping.
  Coverage for all 9 languages is confirmed empirically (Phase 1 read those exact files).
- fp32 on disk, loads in bf16 (~3.3 GB VRAM) — trivial on any of our GPUs.
- The download must happen **inside an sbatch**: torch will not import on the login node.
- **NLLB is sentence-level.** TOFU answers are frequently multi-sentence, so split into
  sentences and translate each, or back-translation quality degrades noticeably. This is
  the single most likely implementation bug in Step 0.

If a language fails badly, it gets reported as a data-quality caveat and excluded
from the KPS pairs — not silently kept.

---

## 3. The metrics, concretely

Let `I_f` = 40 forget (target) facts, `I_r` = 40 retain (non-target) facts sampled from
retain99, `L` = the 10 languages. Every probe fills one cell of a `|I| × |L|` matrix.

### 3.1 Per-cell knowledge scores

For instance `i` in language `l`, from `forget01_perturbed_<l>` / `retain_perturbed_<l>`:

```
p_{i,l}  = P(a_{i,l} | q_{i,l})^(1/|a|)          length-normalised, KBL Eq. 5
R_{i,l}  = geomean_k P(perturbed_k)^(1/|·|) / P(paraphrased)^(1/|·|)     (truth ratio)
```

KBL's forgetting score is `s^prob_{i,l} = 1 − p_{i,l}`. We compute that **and**
`s^tr_{i,l} = R_{i,l}` (higher = more forgotten, same direction).

*Why add the truth ratio:* we already established in this study that probability and
ROUGE are inflated by the fluency that benign relearning restores, while the truth
ratio is a within-language **ratio**, so the fluency factor cancels.

KBL never had this option: their §4 pipeline generates only question+answer, so the
`paraphrased_answer` / `perturbed_answers` fields the truth ratio needs don't exist in
their data ("perturb" appears zero times in the paper; they cite TOFU narrowly as
"uses the probabilities assigned to the corresponding answer", one of TOFU's five
metrics). Truth ratio wasn't rejected by them — it wasn't available. Building it
multilingually would also have multiplied their translate → back-translate → human-verify
loop ~7× per instance, which plausibly discouraged it, though the paper doesn't say so.

The same fluency confound bites both settings, with **opposite sign**: benign relearning
*restores* fluency and inflates `P` (overstating recovery, our problem); unlearning
*destroys* fluency and deflates `P` (overstating forgetting, theirs). The second is not
hypothetical for them — GA is a headline method and is notorious for catastrophic
collapse (the NPO result they cite). A collapsed model has low `P(a|q)` for everything,
so `Si = 1 − P` rises for target *and* non-target knowledge alike — exactly the pathology
they diagnose for PRUNE in Fig. 5 (high `Si` leaking onto non-target knowledge, KSS-PR
tanking). A ratio-based `Si` partly separates "lost the fact" from "lost fluency."

**The decisive reason for Phase 2 specifically: `P` is not comparable across languages.**
Qwen3-8B is worse at Hindi than at English, so the *same fact* scores a lower per-token
probability in Hindi purely from language competence. A per-language `Si^prob` would
therefore be dominated by "how good is the model at this language" rather than "does it
know the fact" — and that ordering decays smoothly outward from English, i.e. it looks
**exactly like** the distance-graded blast radius we are trying to detect. Using it would
mean confirming our own hypothesis by measuring the wrong thing. The truth ratio is a
within-language ratio (gold vs wrong answers in the *same* language), so the
language-difficulty factor divides out. Phase 1 never hit this because everything was
probed in English; Phase 2 is the first in-language probe, so it hits immediately.

*In fairness to KBL:* their design is internally coherent for their question. KSS pools
forget and retain instances measured in the same languages, so a per-language offset hits
both classes and largely cancels in the AUC; KPS is built on an LLM semantic-equivalence
judge, which is language-normalised by construction. It is specifically Eq. 5's `Si^prob`
averaged over languages, read for **per-language magnitudes**, that breaks — and
per-language magnitudes are the whole point of Phase 2.

So substituting the truth ratio is defensible as a **methodological improvement**, not
just a convenience of reusing TOFU. **Compute both** — `Si^prob` and `Si^tr` come from the
same forward passes, so reporting KBL's number alongside ours costs nothing and makes the
comparison empirical rather than asserted. Lead with truth ratio; if the two diverge,
that divergence is itself a figure worth publishing.

*What we do NOT compute:* ROUGE. It needs 80 greedy generations per (checkpoint,
language) — that's ~7 min/cell vs ~2 min for the log-prob metrics, i.e. it would
roughly quadruple the job. It's also the metric we've already shown to be confounded.

### 3.2 Binary knowledge indicator (replaces KBL's `SE`)

KPS needs a binary "does the model still know instance `i` in language `l`". KBL gets
it from an LLM judge; we threshold the truth ratio:

```
knows(i,l) = 1  iff  R_{i,l} < τ_l
```

`τ_l` is **calibrated per language** on two reference populations we can probe for free:

- **positives**: the LEARNED (memorised) checkpoint — by construction it knows these facts;
- **negatives**: the BASE pre-trained Qwen3-8B — TOFU authors are fictitious, so by
  construction it does not.

Pick `τ_l` at the max-Youden-J point of that two-population ROC. This is honest,
reproducible, needs no judge, and gives a per-language sanity read for free: if the two
populations don't separate in language `l`, that language's in-language probe is
**uninformative** and must be reported as such rather than scored. (Strong candidate for
this: the low-resource end. Worth knowing before we interpret anything.)

### 3.3 The three reported quantities

**(a) KSS-ROC / KSS-PR** — KBL §5.2, unchanged. Pool `I_f` ∪ `I_r`, label = target,
score = `S_i`; report AUC-ROC and AUC-PR. Computed (i) aggregated over `L` as KBL do,
and (ii) **per language** — the per-language breakdown is what tells us whether the
unlearn/relearn state is legible at all outside English.

**(b) KPS** — KBL Eq. 6/7, unchanged:

```
ps(l1,l2) = |{i ∈ I_f : ¬knows(i,l1) ∧ knows(i,l2)}| / |{i ∈ I_f : ¬knows(i,l1)}|
KPS(l1,L2) = mean over l2 ∈ L2 of ps(l1,l2)
```

Applied to the **unlearned** checkpoints this reproduces their result on our setup
(all 9 languages are hold-out ⇒ their Case 1). Applied to the **relearned** checkpoints
it becomes the new thing.

**(c) Cross-lingual recovery coupling** — our addition, and the actual interlingua test.
For relearn-language `l_r`, define the recovered set in language `l`:

```
Rec(l_r, l) = { i ∈ I_f : ¬knows_unlearned(i,l) ∧ knows_relearned(i,l) }
```

then report the pairwise **Jaccard of recovered sets**, `|Rec(l_r,l) ∩ Rec(l_r,l')| /
|Rec(l_r,l) ∪ Rec(l_r,l')|`, against the chance level from a permutation null.

- Interlingua ⇒ the same facts come back everywhere ⇒ overlap **≫ chance**, and roughly
  flat regardless of `l_r`.
- Language-local ⇒ overlap ≈ chance, and `Rec(l_r, l_r)` is much larger than `Rec(l_r, other)`.

**The single sharpest number in Phase 2** is the diagonal comparison: relearn in
Japanese, then compare recovery measured **in Japanese** vs **in English**. Our current
design can only see the English column, so this comparison is literally invisible today.

---

## 4. Code changes

Small — the harness is already language-parameterised everywhere except the one
hard-coded line.

1. **`shared/scripts/relearn_measure.py`** — make `--fact-metrics` honour `--measure-lang`
   (which already exists as a flag but is used by nothing — grep confirms zero callers).
   - line 121: `load_perturbed(f"{fl}_perturbed", ...)` → dispatch to
     `load_multilingual_tofu.load_perturbed(f"{fl}_perturbed", lang, ml_cache_dir, cache_dir)`
     for `lang != "en"`.
   - add `--probe-split {forget,retain}` so the non-target half (`retain_perturbed_<lang>`)
     can be scored — needed for KSS. Confirmed present for all 9 languages.
   - key results `f"{name}@{lang}"` (the existing convention at line 156), keep the
     per-fact arrays (already added for the bootstrap CIs — Phase 2 reuses them wholesale).
   - drop the generation/ROUGE call when a `--no-rouge` flag is set (see §3.1 cost note).

2. **`studies/crosslingual_recovery/slurm/crosslingual_phase2_probe.sbatch`** — new,
   modelled on `crosslingual_relearn_deep.sbatch`. Self-contained: relearn → probe all
   10 languages × {forget, retain} → `rm -rf` the checkpoint. Resumable via the same
   per-key skip guard. Writes to `results/phase2/`.

3. **`studies/crosslingual_recovery/plots/phase2_kss_kps.py`** — local/CPU, from the JSON:
   τ calibration, KSS-ROC/PR, KPS matrix, recovery-set Jaccard + permutation null.
   Figures: KPS heatmap (`l1 × l2`), recovery-coupling heatmap per relearn-language,
   KSS bar chart per language.

4. **`studies/crosslingual_recovery/scripts/verify_translations.py`** — new, §2. CPU/GPU
   light, one-off.

---

## 5. Compute plan

Probe cost is ~2 min per (checkpoint, language) for the log-prob metrics on 40 records
(≈8 teacher-forced forwards each, no generation).

Two distinct language decisions, with different answers:

- **Probe languages** (where we *measure* each fact): **all 10, always.** ~2 min/cell, and
  it is what makes the matrix a matrix. Never subset.
- **Relearn languages** (where we *benignly relearn*): where the cost actually is,
  ~20–25 min of GPU each.

| item | count | est. |
|---|---|---|
| relearn (retain, ep2, 1500 ex) × 10 langs × {Full-FT, LoRA} | 20 | ~7 h |
| probe relearned: 20 ckpts × 10 langs × 3 splits (forget/retain/world_facts) | 600 cells | ~8 h |
| probe baselines: {2 unlearned, learned, base} × 10 × 3 | 120 cells | ~1.6 h |
| **total** | | **~17 h — exceeds the 12 h wall by design** |

**SETTLED — all 10 relearn languages, run resumably across two submissions.** Two reasons
the earlier "4 languages" was the wrong fix:

1. **Phase 1 comparability.** Phase 1's correlations are over the 9 non-English languages.
   All 10 lets us correlate Phase 2's coupling metric against Phase 1's Jaccard on the
   *same 9 points*. At n=4 there is no correlation power at all — that link is simply lost.
2. **The script is resumable.** The per-key skip guard means we submit all 10; if it hits
   the wall we resubmit and it resumes. Strictly better than pre-committing to a subset.

**Order the language loop `en fr hi ja` first**, then `id ru fa ar iw ko`. If the job dies
at the wall we still hold the informative spread — source / near / Phase-1 anomaly / far —
rather than an alphabetical prefix that answers nothing. (`hi` is the Phase-1 anomaly:
lowest Jaccard 0.018 but highest overlap-coefficient 0.578 among non-Latin scripts, so if
the LoRA vocab signal is real, Hindi is where it shows.)

---

## 6. Reading the result

| KSS (per-language) | recovery-set Jaccard | reading |
|---|---|---|
| separable in most langs | ≫ chance, flat in `l_r` | **Interlingua.** The fact is stored language-agnostically; relearning anywhere restores it everywhere, same facts. Phase 1's flat Full-FT line was right. |
| separable | ≈ chance, `Rec(l_r,l_r)` dominant | **Language-local.** Recovery is a surface/vocab effect; the English-only measurement has been overstating what "recovery" means, and Phase 1's LoRA–Jaccard correlation was the real signal. |
| separable | ≫ chance but decays with language distance | **Graded interlingua** — the most likely and the most interesting outcome; ties directly back to the CLAUDE.md "blast radius decays with distance" hypothesis. |
| not separable in most langs | — | **The in-language probe doesn't work on this model/data.** Report it as a negative result about the multilingual TOFU translations, fall back to the English-only design, and do not over-read Phase 2. |

That last row is a real possibility (Qwen3-8B on Hindi/Hebrew TOFU translations), which
is exactly why the τ calibration in §3.2 is a gate, not a formality.

---

## 7. Dependencies on the running job

**None that block starting.** §2 (translation verification) and §4 (code) can be built
and tested now — the translation QC needs no GPU at all.

The one real coupling is **checkpoints, not results**:
`crosslingual_relearn_deep.sbatch` line 77 does `rm -rf "$RELEARNED"` after measuring,
so the 20 relearned checkpoints that job builds are destroyed. Phase 2 would rebuild 8 of
them (~3 h of the ~7 h budget above).

Two ways to play it:

- **Recommended — leave the queued job alone, Phase 2 is self-contained.** Costs ~3 h of
  duplicate relearn. The queue, not the GPU, is the bottleneck: fairshare is low
  (FairShare 0.0075, no reset, 10-day decay), so cancelling and resubmitting risks losing
  more to queue position than the 3 h it saves.
- **Alternative — only while the job is still `PD`:** `scancel`, change line 77 to keep
  the checkpoints for `{en, fr, hi, ja}`, resubmit. Saves the 3 h but resets queue age and
  needs ~128 GB of scratch (8 × 16 GB bf16) — check the project-filesystem quota first.

Scientifically, the bootstrap CIs that job produces do **not** gate Phase 2. They settle
the Phase-1 LoRA–Jaccard question (FLORES Spearman ρ=+0.717, p=0.039), which only affects
an *optional* stratification inside Phase 2, not its core design. If it turns out to be
noise, Phase 2 is unchanged; if real, we additionally stratify facts by surface-token
overlap of the answer entities.

---

## 8. Decisions

**Settled:**

1. **Judge** — NLLB-200-3.3B back-translate + base Qwen3-8B judge, different families on
   purpose, hand-validated on ~30 items. Match KBL's *design* (translate-to-English first),
   not their vendor. §2.
2. **Metric** — compute *both* `Si^prob` (KBL) and `Si^tr` (ours) from the same forward
   passes; lead with truth ratio. §3.1.
3. **Languages** — probe all 10 always; relearn in all 10, resumably across two
   submissions, loop ordered `en fr hi ja` first. §5.
4. **world_facts** — include as a third probe split. Its job is not KSS but disambiguating
   *why* a language's probe fails (see below). §8.1.

**Still open:**

5. Whether to add KBL-faithful generation-based `SE` (Phase 2b) or ship 2a on log-prob
   metrics only. Defer until 2a's τ-calibration shows how many languages survive the gate —
   if several fail, a generation-based `SE` on those languages is wasted effort.
   **If we do it, the judge cannot be Qwen** (self-preference bias, §2) — that decision
   brings an API-budget or different-open-family question with it.

### 8.1 Why world_facts earns its ~2 min/checkpoint

It is *not* needed for KSS (which needs forget as positives, retain as negatives). It pairs
with the §3.2 τ-calibration gate.

That gate tells us **that** a language's probe is uninformative (learned and base models
fail to separate). It cannot tell us **why**, and the two causes have opposite consequences:

- the **translation** for that language is broken → fixable, and §2 should have caught it;
- **Qwen3-8B is simply weak in that language** → a hard limit we must report as a scope caveat.

`world_facts` separates them, because it is pre-training knowledge, independent of anything
we fine-tuned. Good Hindi world_facts + bad Hindi TOFU ⇒ our data. Bad on both ⇒ the model.

It is also language-comparable by construction: our loader scores it multiple-choice as
`P(a₁|q)/ΣP(aᵢ|q)`, a ratio over same-language candidates — the same normalisation trick
that makes the truth ratio safe across languages (§3.1).
