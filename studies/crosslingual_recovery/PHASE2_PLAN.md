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
| **Data**: Faker → 200 profiles → 20-attribute pool → LLM-generated 19 QA/profile → 3800 QA | **TOFU**: 200 fictitious authors × 20 QA = 4000 QA. Already the repo's foundation; already provably-unknown-before-finetuning, which is the same property their Faker step buys. |
| Google Translate into 9 languages | **Already done** — `data/raw/multilingual_unlearning/dataset/` (ar fa fr hi id iw ja ko ru), same field schema as locuslab/TOFU. |
| Back-translation verify + LLM judge + refine loop (their Fig. 3, step 4) | **NOT done for our translations.** This is the one piece of their pipeline we genuinely have to build. See §2. |
| 8 training langs / 2 hold-out langs | **Free for us**: we LEARN and UNLEARN in English only, so all 9 non-English languages are hold-out. Our whole study sits in their **Case 1** (the hard case, where KSS was 0.76 max vs 0.99 in Case 2). |
| Knowledge-wise forgetting score `Si` (Eq. 5), averaged over languages | Keep it, **but keep the full per-language vector `s_{i,l}` too** — the cross-lingual structure is the object of study, so averaging it away is exactly the wrong move here. Add a **truth-ratio** variant alongside their probability variant (§3.1). |
| `KSS-ROC` / `KSS-PR` (forget vs retain separability) | Keep as-is, computed both aggregated and per-language. |
| `KPS` (Eq. 6/7): forgotten in `l1`, retained in `l2` | Keep, plus a **recovery-side mirror** that is the direct interlingua test (§3.3). |
| `SE` binary knowledge indicator via NLLB-200-3.3B → English → GPT-4o-mini judge | **Phase 2a: replace** with a calibrated threshold on the truth ratio (§3.2) — no translation model, no API, no judge. **Phase 2b (optional):** add the real SE for a subset, for paper-faithfulness. |
| Methods: GA / GAGDR / GAKLR / NPO / PRUNE | Ours: **Full-FT vs LoRA**, both `gradient_difference`, on the *deep* checkpoints (baseline TR 0.77 / 0.68 — where recovery is largest, 47% / 63%, so the most signal). |

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

**Decision needed:** which back-translator + judge. Options, cheapest first —
(a) the model we already have (Qwen3-8B) as both, on the cluster, no API;
(b) NLLB-200-3.3B for translation (KBL's choice) + Qwen3-8B as judge;
(c) KBL-faithful: NLLB-200-3.3B + a GPT-4o-mini-class API judge (costs money).
720 items is tiny, so any of these is a sub-hour job. **Recommend (b)**, and record
the deviation.

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
ratio is a within-language **ratio**, so the fluency factor cancels. KBL's `Si^prob`
inherits that confound; ours is the fix. Report both, **lead with truth ratio**, and
show they agree on the qualitative conclusion (or say where they don't).

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

| item | count | est. |
|---|---|---|
| relearn (retain, ep2, 1500 ex) × {en, fr, hi, ja} × {Full-FT, LoRA} | 8 | ~3 h |
| probe relearned: 8 ckpts × 10 langs × 2 splits | 160 cells | ~2.7 h |
| probe baselines: {2 unlearned, learned, base} × 10 × 2 | 80 cells | ~1.3 h |
| **total** | | **~7 h** — fits a 12 h wall, resumable |

Relearn-language choice — 4, not 10, on purpose: **en** (source-language control),
**fr** (highest Jaccard, near), **ja** (lowest Jaccard, far), **hi** (the Phase-1
anomaly: lowest Jaccard 0.018 but highest overlap-coefficient 0.578 among non-Latin —
if the LoRA vocab signal is real, Hindi is where it shows). Adding the remaining 6 later
is a re-run of the same resumable script.

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

## 8. Open decisions

1. Back-translator + judge for §2 — recommend NLLB-200-3.3B + Qwen3-8B, no API.
2. Whether to add KBL-faithful generation-based `SE` (Phase 2b) or ship 2a on log-prob
   metrics only.
3. Relearn-language set: 4 (en/fr/hi/ja) now, or all 10 in one longer job.
4. Whether to also probe `world_facts_perturbed_<lang>` as a third, far-from-forget
   control split (cheap: +40 cells/checkpoint).
