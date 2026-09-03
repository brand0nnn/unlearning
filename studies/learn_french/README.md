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
