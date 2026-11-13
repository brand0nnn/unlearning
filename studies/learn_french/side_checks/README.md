# side_checks — exploratory checks beside the main study

**State 2026-10-29.** Plans 1-3, the P(gold) control and the en/ja seed sweep are COMPLETE
and read; Plan 4 was dropped (no results). The
Q2 claim these support, and how to argue it, is in `../README.md` START HERE ("Q2
findings" and "Writing the report"). Figure index: "Figures" at the end of this file.

Not part of the pre-registered Stages 1-9. Nothing here writes to `learn_french/results/`,
`experiments/tr_levels/` or `preregistration.json`; it only reads them. Shared code
(`src/`, `shared/`, `learn_french/scripts/`) is imported, never modified.
`[PROVISIONAL: single seed]`, seed 42, 40 facts about 2 entities.

**Step 0** (zero GPU, done 2026-10-21) is `learn_french/plots/copy_check.py`. It found
that ru/ja "copied" names are mostly real copies (ru 71%, ja 87.5% exact), and that copy
protection is mostly the confidence effect.

## Results (run 2026-10-21, jobs all complete; read 2026-10-24)

Figures: `learn_french/figures/side_checks/1-5_*.png`, made by `plots/plot_side_checks.py`
(`.venv-plot`). Tables: `plots/read_plan1.py --by-author`, `plots/read_plan2.py
[--metric lp_mean|cer]`. `[PROVISIONAL: single seed]`.

**Sanity.** The reruns reproduce the Stage 2 French TR to 3 decimals at every epoch, and
the in-run per-token scores equal the post-hoc pass (|delta| 0.0000, 0 destroyed-call flips).

0. **The blanket effect is partly epochs, mostly Japanese** (fig 6; Plan 1's own question).
   Stage 6's measure, each arm on its own answer tokens, % destroyed (> 5 nats): fr 14% at
   the matched ep 13 -> 49% at ep 25; en 38% (ep 12) -> 60% (ep 25); ja 86% at its matched
   ep 25. Among near-certain tokens only (which removes ja's head start of text the learned
   model was unsure of): fr 33%, en 29% at ep 25 vs ja 66%. So epochs explain part of
   Stage 6's gap, but at equal epochs ja is still 26 points (33 on near-certain tokens)
   blanker -- while removing far LESS of the French fact (TR 0.83 vs fr 1000, en 15).
   id and ru were not rerun to 25 epochs.
   **Revised by Plan 3:** what drives the blanket is how far the arm's OWN-language forget
   loss is pushed, not the epoch count. At fixed LR more epochs = pushed further (en
   forget NLL 13.5 at ep 12 -> 50.5 at ep 25); at matched French TR the push is set by the
   language (en 10.8-13.5, ja 41-55) whatever the run length, and so is the blanket (ja
   names 95-100% destroyed at 12, 15 and 25 epochs; en 2%). CLOSED.
1. **Copy protection is a stage, not a property of the language** (fig 3). Among tokens
   fr_ft gave p > 0.999, copied names are spared early and then overtake other text: fr
   on French text 0.4% at the matched ep 13 -> 54% at ep 25 (other text 29%); en on
   English 0.0% at ep 12 -> 35% at ep 25 (28%). Names pass other text at ep 16 (fr), 23
   (en, own text), 21 (en, French text). Stage 6's fr/en/id-spare vs ja/ru-destroy gap
   is mostly *how far the loss has been pushed* (which at fixed LR grows with epochs); ja
   still gets there fastest (99.5% at ep 25 vs fr 54%).
2. **At fixed LR, more epochs make forgetting MORE durable** -- because they go deeper.
   **At fixed depth a longer run is slightly LESS durable (Plan 3).** (fig 2). After French benign
   relearning (3 epochs), P(gold): en ep 12 0.57 vs en ep 25 0.14; fr ep 13 0.26 vs
   fr ep 25 0.014 (below never-taught 0.094). At the same 25 epochs ja relearns to 0.63.
   Stage 3 finding 3's "effort predicts durability" (r = +0.87) is therefore not causal:
   epochs-to-matched-depth marks the language, it does not make forgetting fragile.
   Within one run effort and depth rise together, so this does not separate them --
   Plan 3 does.
   fr@25 is broadly damaged (retain P ratio 0.38, MU 0.387); en@25 is the clean case
   (0.93, MU 0.469).
3. **The floor holds.** Relearning fr_retain on French retain99 leaves P(gold) at
   0.082-0.091 and TR at 0.84-0.85: Stage 3's gains are recovery, not new learning.
4. **Recovery ranking depends on the metric** (fig 2C, French relearn column, relearn
   ep 2, matched arms). TR: en 17 < id 42 < fr 58 < ru 66 < ja 82. P(gold): fr 32 < en
   36 < id 66 < ru 68 < ja 79. Ja is most recoverable on both; the order of en / fr / id
   is not robust.
5. **Plan 2: name damage follows name PIECES, not people or a general skill** (fig 4).
   Copy task in the arm's own script, mean log p per token vs fr_ft: Basil Mahfouz
   Al-Kuwaiti ja -78 / ru -17; invented Karim Al-Kuwaiti ja -44 / ru -12; Naguib Mahfouz
   ja -30; names sharing nothing ja -0.6, ru -0.2. Pieces the model knows well in that
   script are spared (Николай under ru, Basil in Latin). fr and en at ep 25 show the same
   pattern in Latin (Basil Mahfouz -9.2 / -6.0, Naguib Mahfouz -6.0 / -3.5).
6. **Plan 2: the ja arm avoids writing names in katakana** (fig 5). Copying an
   unrelated katakana name: 36% correct vs fr_ft 97%, the failures almost all the name in
   another script (Latin / Cyrillic / Han), not misspellings. ru does the same for
   Cyrillic (72% correct); fr@25 partly; en@25 not at all.

**Parser fix (2026-10-24).** Base Qwen3 echoes `[INST]` / `[/INST]` before its answer; the
original `clean_prediction` scored those copies as wrong. Fixed in `scripts/name_tasks.py`;
`read_plan2.py` and the plots rescore from the stored generations. Base is still not a
usable generation baseline (with French instructions it continues the prompt): compare
against fr_ft.

## Plan 1 — is the blanket effect epoch-dependent? (fr and en arms)

Re-run the fr and en unlearning arms from `fr_ft` with the Stage 2 recipe and its
**50-epoch LR schedule**, stopping at **epoch 25** (where Japanese was matched). Every
epoch is scored in-run; only epoch 25 is saved, and it is deleted once used.

Per arm (`slurm/plan1_arm.sbatch <lang>`, ~10h on a100-80, peak ~65GB):
1. unlearn + in-run scoring (`scripts/unlearn_epochs.py`)
2. Stage 1 scorer on epoch 25 (TR, P(gold), NLI, MU)
3. post-hoc per-token pass on epoch 25 (checks the in-run numbers)
4. Plan 2 on epoch 25
5. benign relearn, Stage 3 recipe (French retain99, n=1500, 3 epochs), scored, snapshots deleted
6. delete epoch 25 (only if all of the above wrote their JSON)

`slurm/plan1_control.sbatch` (~4.5h): fr_ft / fr_retain anchors in this scoring group, and
the floor control (relearn `fr_retain` with the same recipe).

**Read with care.** At epoch 25 the French TR is ~1000 (fr) and ~15 (en), far deeper than
Japanese at 0.83: the relearn comparison holds *effort* fixed, not depth. Hu et al.'s depth
effect and the effort hypothesis predict opposite directions for en@25, so "en stays
durable" cannot separate language from depth. Report absolute TR / P(gold) / NLI, not
recovery % (its denominator explodes). The rerun is not bitwise identical to Stage 2;
`read_plan1.py` prints the original TR at the same step beside the new one.

## Plan 2 — does ja/ru unlearning damage a general name skill? (inference only)

`probes/names.json`: 2 forget authors (a), 21 retain authors (b), 23 unrelated names (c:
real/invented x sharing/not sharing a piece with a forget name). 7 tasks (copy in Latin /
katakana / Cyrillic, Latin -> katakana / Cyrillic, back to Latin) x **5 instruction
languages** (en fr id ja ru, `scripts/name_tasks.py`). Greedy generation (exact, loose
exact, character error rate) + teacher-forced log p of the reference.
`slurm/plan2_names.sbatch` (~2h): base, fr_ft, fr_retain, the 5 matched checkpoints. The
epoch-25 fr/en checkpoints are scored inside `plan1_arm.sbatch`.

Caveats: set b is not neutral (every arm's retain term trained on those authors in its
own language); invented names have no correct transliteration (read as change vs fr_ft);
the ja/ru/id instructions and set c's katakana/Cyrillic need a fluent check.

## Plan 3 — matched depth AND matched epochs (ja-fast, en-slow)  (COMPLETE, read 2026-10-27)

At the Stage 3 matched checkpoints en and ja agree on French TR, P(gold) and MU, but ja ran
25 epochs and en 12. Plan 3 changes only the peak LR so each reaches the same level (0.8276)
in the other's epoch count. Stop at the first per-epoch crossing, score, name skill, layer
diff, then benign relearn in fr and en (Stage 3 recipe, final epoch only). Predictions,
reading rule and exclusions were fixed in `preregistration_plan3.json` before any job ran.
Primary outcome: ABSOLUTE relearned French TR. Jobs 1011052-1011054.
`slurm/plan3_arm.sbatch <lang> <lr>`, peak disk ~33GB. Unlearning ~0.22h/epoch, each
relearn + scoring ~1h50 (from result timestamps).

**Primary ja-fast = lr 1.5e-5** (pre-registered rule: crossing epoch nearer 12; it crossed
at 12, the 1.0e-5 run at 15). 1.0e-5 is reported as secondary.

| run | peak LR | stop ep | at stop: French TR / P(gold) / MU | relearned French TR (fr col / en col) | relearned P(gold) | relearned NLI | top-third share / total dW^2 | own-lang names destroyed |
|---|---|---|---|---|---|---|---|---|
| en Stage 3 | 5e-6 | 12 | 0.847 / 0.492 / 0.492 | 0.813 / 0.774 | 0.569 / 0.534 | -- | 38% / 2.9 | 2% |
| en-slow | 2e-6 | 23 | 0.834 / 0.524 / 0.490 | 0.753 / 0.759 | 0.589 / 0.555 | 0.243 / 0.083 | 39% / 2.1 | 2% |
| **ja-fast (primary)** | 1.5e-5 | **12** | 0.910 / 0.194 / 0.478 | **0.705 / 0.687** | 0.563 / 0.488 | 0.372 / 0.179 | 48% / 29.2 | 95% |
| ja-fast (secondary) | 1.0e-5 | 15 | 0.872 / 0.274 / 0.483 | 0.700 / 0.670 | 0.594 / 0.543 | 0.516 / 0.227 | 49% / 23.4 | 99% |
| ja Stage 3 | 5e-6 | 25 | 0.834 / 0.447 / 0.497 | 0.651 / 0.646 | 0.629 / 0.593 | -- | 52% / 15.0 | 100% |

Stage 3 rows are a different scoring group (`results/stage3_ul*`; up to 0.024 TR apart).
Learned fr_ft: TR 0.605, P(gold) 0.700; never-taught fr_retain: TR ~0.83, P(gold) 0.094.
Relearned TR: HIGHER = more durable.

**Results** `[PROVISIONAL: single seed]`:
1. **H_language supported.** Pre-registered midpoints 0.732 (fr col) / 0.710 (en col):
   en-slow stays above (0.753 / 0.759), ja-fast primary stays below (0.705 / 0.687;
   margins 0.027 / 0.023, close to the 0.024 between-group drift -- and ja-fast overshot
   to TR 0.910, which should make it look MORE durable, so the margin is conservative).
2. **Effect sizes (figs 8, 14):** changing the language at fixed run length moves relearned
   TR 0.108 / 0.087 (12 epochs) and 0.102 / 0.113 (23-25 epochs); changing the run length
   within a language 0.060 / 0.016 (en) and 0.054 / 0.042 (ja). Language is 1.8x (fr col)
   / 3.5x (en col) the run-length effect. With the secondary ja-fast it was 2x / 5.5x.
3. **The pre-registered "plan1_direction" prediction failed:** en-slow is LESS durable than
   en Stage 3 (0.753 vs 0.813). At fixed depth a longer run is slightly less durable, in
   both languages (ja 12 ep 0.705 > 25 ep 0.651). Plan 1's "more epochs = more durable"
   was depth.
4. **Layers (fig 9):** top-third share follows the language (en 38-39%, ja 48-52%) at
   every run length; total change grows with LR (ja 15.0 -> 23.4 -> 29.2), so the path
   does change the edit's SIZE, not its shape.
5. **Blanket (fig 10):** follows the language, not the run length (ja names 95-100%,
   non-name words 78-84%; en 2% / 49%); the French answer <= 9% destroyed in every
   foreign arm. Own-language forget NLL at the stop: en 10.8-13.5, ja 41.1-54.7.
6. **P(gold) is NOT matched across Plan 3 arms** (0.19-0.52 at the stop), so fig 11 is
   descriptive only: row 1 (where P(gold) ends) even shows ja-fast primary ending LOWER
   than en in the en column (0.488 vs 0.534), because it started at 0.19. The P(gold)
   evidence is the control below.

## P(gold) control — en matched to ja-fast on P(gold) and epochs  (COMPLETE, 2026-10-26)

Pre-registered as `pgold_control` in `preregistration_plan3.json` before submission.
en, Stage 2 recipe unchanged (lr 5e-6, seed 42), stopped at a FIXED epoch 15:
`STOP_EPOCH=15 plan3_arm.sbatch en 5.0e-6` (run `ulen_s42_lr5p0e-06_ep15`). Its in-run TR
reproduces Plan 1's en trajectory exactly (max |dTR| 0.000).

| | en @ ep 15 | ja-fast 1.0e-5 |
|---|---|---|
| matched: epoch / P(gold) / MU | 15 / 0.274 / 0.482 | 15 / 0.274 / 0.483 |
| NOT matched: French TR / peak LR | 1.250 / 5e-6 | 0.872 / 1e-5 |
| **relearned P(gold), fr col / en col** | **0.454 / 0.437** | **0.594 / 0.543** |
| relearned NLI, fr col / en col | 0.203 / 0.127 | 0.516 / 0.227 |
| relearned TR (not evidence: TR unmatched) | 1.112 / 0.967 | 0.700 / 0.670 |
| layers: top-third / total; names destroyed | 42% / 5.8; 4% | 49% / 23.4; 99% |

**H_language supported on P(gold):** ja recovers +0.140 / +0.106 more P(gold), both above
the pre-registered 0.04 (re-scoring noise was measured on TR/NLI, ~0.038; assumed similar
for P(gold)). So the en-vs-ja durability gap holds matched on TR (Stage 3, Plan 3) AND
matched on P(gold) (this). Fig 16. en @ 15 lands between Plan 1's ep 12 (0.57) and ep 25
(0.14), as expected. Deeper en edits are slightly more top-heavy (42% vs 38%) but still
below every ja run.

## Five-language summary for the mechanism  (zero GPU; figs 12-13)

From Stage 3 and `results/per_token_map.json` / `results/layer_diff.json`, at the matched
checkpoints (fr ep 13, en ep 12, id ep 12, ru ep 15, ja ep 25):

| | fr | en | id | ru | ja |
|---|---|---|---|---|---|
| carry-over, token-pooled: log units (x per token) | 2.17 (8.7x) | 0.63 (1.87x) | 0.58 (1.79x) | 0.36 (1.44x) | 0.16 (1.18x) |
| **carry-over, per-fact mean (used in figs 12-13 and the report)** | 2.24 (9.35x) | 0.64 (1.89x) | 0.66 (1.92x) | 0.37 (1.45x) | 0.19 (1.20x) |
| top-third layer share | 36% | 38% | 39% | 47% | 52% |
| total change (sum dW^2) | 3.1 | 2.9 | 2.9 | 6.0 | 15.0 |
| own-language names destroyed | 2% | 2% | 2% | 65% | 100% |
| own-language forget NLL at match | 2.9 | 13.5 | 11.3 | 13.8 | 54.7 |
| recovery TR / P(gold) (Stage 3 rows) | 57 / 29 | 33 / 40 | 50 / 64 | 60 / 58 | 81 / 77 |

**Carry-over** = mean teacher-forced log p of the 40 forget answers written in the
unlearning language under fr_ft, minus the same under fr_retain (both score the same
tokens), averaged per fact (each fact weighted equally, the unit used everywhere else;
`carry_over()` in `plots/plot_plan3.py`, figs 12-13 since 2026-10-27). Pooling all tokens
instead weights long answers more. **en and id swap between the two (a tie)**: say "en and id high,
ru lower, ja lowest". fr is unlearning in the learning language and does not fit on TR
recovery (57%) -- a separate category.

Reading (proposed, untested): low carry-over -> the gradient can lower the target
language's answer only by breaking that language's output near the top layers, pushing
its own loss far (ja 54.7) while barely touching the stored fact -> benign relearning
repairs the output path and the fact returns. n = 4 foreign arms, one seed, carry-over
confounded with script (en/id Latin; ru/ja not). Tests: the joint fr+ja model unlearned
in JAPANESE (carry-over high, language fixed; prediction: durable); a pre-registered
ko/hi/ar prediction; Experiment D (restrict which layers unlearning may change).

## Seed sweep — en + ja, seeds 1 and 2  (PRE-REGISTERED, COMPLETE 2026-10-29)

Runs `ul{en,ja}_s{1,2}_lr5p0e-06_tr0p828` under `results/plan3/`; seed 42 = the Stage 3
arms (`results/stage3_ul*`, a different scoring group). Figure:
`figures/side_checks/17_seed_sweep.png` (`plot_plan3.py` `fig_seeds`; Figures 7-16 still
load seed-42 runs only).

**Result -- the gap replicates and is larger than seed variation:**

| | seed 42 | seed 1 | seed 2 | arm spread over seeds |
|---|---|---|---|---|
| relearned French TR, French relearn: en | 0.813 | 0.848 | 0.825 | 0.035 |
| ... ja | 0.651 | 0.665 | 0.676 | 0.025 |
| **gap en - ja** | **0.162** | **0.183** | **0.149** | |
| relearned French TR, English relearn: en | 0.774 | 0.776 | 0.784 | 0.010 |
| ... ja | 0.646 | 0.659 | 0.660 | 0.014 |
| **gap en - ja** | **0.129** | **0.117** | **0.124** | |

Pre-registered reading: positive in both columns on every seed (replicates); every gap
is above 0.024 (detectable); smallest gap 0.117 > largest arm spread 0.035 (larger than seed
variation). The gap is ~4-8x the run-to-run spread.

Everything else, seed 42 / 1 / 2:

| | en | ja |
|---|---|---|
| epochs to cross | 12 / 12 / 12 | 25 / 23 / 25 |
| French TR at stop | 0.847 / 0.875 / 0.864 | 0.834 / 0.830 / 0.832 |
| French P(gold) at stop | 0.492 / 0.500 / 0.494 | 0.447 / 0.434 / 0.438 |
| MU at stop | 0.492 / 0.492 / 0.493 | 0.497 / 0.493 / 0.491 |
| top-third layer share | 38 / 38 / 38% | 52 / 51 / 52% |
| total dW^2 | 2.9 / 2.8 / 2.9 | 15.0 / 13.5 / 15.0 |
| own-language names destroyed | 2 / 3 / 2% | 100 / 100 / 96% |
| relearned P(gold), fr / en relearn | 0.569-0.576 / 0.534-0.539 | 0.629-0.652 / 0.593-0.629 |
| relearned NLI, fr relearn | 0.240 / 0.460 / 0.256 | 0.353 / 0.583 / 0.526 |
| relearned NLI, en relearn | 0.141 / 0.126 / 0.139 | 0.175 / 0.263 / 0.172 |

Reading notes:
- **Depth at the stop is not an escape route.** The stop is the first epoch past 0.8276,
  so en (fewer, bigger per-epoch jumps) overshoots more: en stops 0.847-0.875 on TR, ja
  0.830-0.834. That favours en looking durable on TR -- but on P(gold) en stops
  *shallower* (0.49-0.50 vs 0.43-0.45) and still ja relearns to the higher P(gold) on every
  seed (by 0.053-0.094). Neither depth measure explains the gap.
- **The edit signatures are seed-stable** (layer share, size, names destroyed): the
  "same depth, different edits" finding is not a seed-42 accident.
- **NLI is noisy across seeds** (en French relearn 0.24-0.46); ja > en on every seed in
  both columns, but don't quote NLI gaps as precise.
- Scope: en and ja at the Stage 3 recipe only. Plan 3, the P(gold) control and the
  five-language comparison remain single seed.

`bash studies/learn_french/side_checks/slurm/submit_seeds.sh 2` (lanes; seeds default 1 2).
Each job = `plan3_arm.sbatch` at lr 5e-6 with `SEED` set -- Stage 3's stopping rule, so
seed 42 of this job IS the Stage 3 arm; relearn in fr and en. ~9.5h (ja) / ~6.5h (en),
~32 GPU-h total, ~16h on 2 GPUs, ~66GB peak. Pre-registration: `seed_sweep` in
`preregistration_plan3.json` (gap en - ja in relearned TR per column; replicates if
positive in both; < 0.024 = no detectable difference; "larger than seed variation" only if
it exceeds each arm's spread over seeds 1, 2, 42).

**Seed bug fixed 2026-10-27** (commit 8912ff4): `src/training/learn.py` and `unlearn.py`
never passed a seed to HF `TrainingArguments`, which re-seeds everything to its default 42
when the Trainer is built. So before the fix `--seed` changed only the unlearning retain
pairing and nothing in relearning. cfg seed 42 = the HF default, so no earlier run changes.
`relearn.py` now takes `--seed`. id / fr / ru are out of scope until Stage 3's level-0.783
checkpoint selection is reproduced.

## Plan 4 — layer profile at fixed epochs  (DROPPED, no results)

Written 2026-10-25 (`slurm/plan4_arm.sbatch`, `plan4` in the pre-registration) to ask
whether ja's top-heavy change is the language or its 25-epoch run. Plan 3 answered that
at matched depth AND matched epochs (Plan 3 result 4), which Plan 4 could not (at fixed
epochs its arms sit at different depths), so it was dropped before producing results.

`scripts/unlearn_epochs.py` gained `--lr`, `--stop-at-level`, `--save-at-epochs` for these
plans; without them it runs Plan 1 exactly.

## Plan 5 — reliability + causal tests of carry-over  (PRE-REGISTERED 2026-11-08; GPU round 1 COMPLETE 2026-11-12, read 2026-11-13)

From the supervisor's 2026-11 feedback: reliability of depth matching and of the metrics,
an intervention for the mechanism, depth over breadth. en vs ja only. Predictions and
reading rules: `preregistration_plan5.json`, fixed before any job.

**Zero-GPU reliability pass (DONE 2026-11-08):**
- **Depth curve (fig 18, `plots/plot_depth_curve.py`).** Every relearned en/ja run, x =
  French P(gold) at the stop. Both languages have runs at P(gold) 0.194-0.447; ja relearns
  higher (less durable) everywhere they overlap. Matched pair at 0.274 (French relearn):
  en 0.454, ja 0.594. ja's deepest run is 0.194 at the stop and relearns to 0.563, above
  en started at 0.274. Nothing is known for ja below 0.194 -> the depth-curve runs.
- **Metric consistency (`plots/read_metric_consistency.py`, stdlib).**
  [A] en more durable than ja on the mean on TR, P(gold) and NLI, both columns, all 3
  seeds (18/18). Per fact, P(gold) agrees broadly (27-35 of 40 facts), NLI 22-28, **TR is
  near a coin flip (18-23)**: the TR mean gap is carried by **facts 5 and 7 on every seed**
  (per-fact gaps 1.1-3.1; median gap -0.012 to 0.027; without the top 3 facts 0.033-0.053).
  So lead the report with P(gold) for breadth, and give TR with that caveat. Plan 5 uses
  P(gold) as its primary outcome for this reason.
  [B] The TR-vs-NLI recovery gap (56% vs 12%) is **not** a threshold on TR or P(gold):
  fr_ft says 90% of facts with TR < 0.6, relearned models 18%; at P(gold) 0.6-0.8, 89% vs
  20%. 91% of the TR recovery mass sits in facts not said after relearning. Third falsified
  explanation (after length dilution); stays a limitation.

**GPU (scripts ready).** One command queues everything as a dependency chain:
`bash studies/learn_french/side_checks/slurm/submit_plan5_chain.sh` (`DRY=1` prints only).
The QOS allows ONE a100-80 per user, so Plan 5 runs in series (~93 GPU-h, ~4 days) and
queues behind any other a100-80 job on the account. Chain order, chosen 2026-11-08 to
have readable results first: smoke x2 -> D ja -> depth ja 0.100 -> D en -> bridge LEARN
(submits its own gate, `AUTO_GATE=1`) -> depth ja 0.018 -> depth en 0.100. Reordered 2026-11-09 to put depth ja 0.100 FIRST (the 27 h lost to the --export bug left no time for D before the presentation). The bridge
arms stay manual (read the gate, then `submit_plan5.sh bridge-arms`). `submit_plan5.sh
<phase>` still submits one phase at a time.

| phase | runs | question | est. GPU-h |
|---|---|---|---|
| smoke | ja `LAYERS=0-23`, en `LAYERS=24-35`, 1 epoch, `SMOKE=1` | did the freeze hold under ZeRO-3? (`scripts/check_freeze.py`) | ~2 |
| depth | ja lr 1.5e-5 `STOP_PGOLD=0.100` / `0.018`; en lr 5e-6 `STOP_PGOLD=0.100` | does ja stay less durable deep (Hu et al.: deeper = more durable)? | ~26 |
| layers (Experiment D) | ja layers 0-23 only; en layers 24-35 only; stop French TR 0.8276 | is the top-layer locus a CAUSE of fragility? (two-sided) | <=32 |
| bridge | `slurm/24_learn_bridge.sbatch` (retain99 fr + ja, forget01 fr only), `25_bridge_gate.sbatch`, read `scripts/bridge_gate_report.py` | did bilingual task training raise ja carry-over? | ~16 |
| bridge-arms | ja / en `SRC=bridge`, stop French TR 0.8276 | does raising carry-over make ja durable? (gate PASS only) | ~17 |

`unlearn_epochs.py` gained `--stop-at-pgold`, `--trainable-layers A-B`, `--tag`;
`plan3_arm.sbatch` the env options `STOP_PGOLD`, `LAYERS`, `SRC=bridge`, `SMOKE=1` (each
changes the run name, so nothing collides with earlier runs); `shared/scripts/01_learn.py`
gained `--forget-lang`. Experiment D freezes embeddings, final norm and lm_head too (they
carry 0.2-0.4% of the unrestricted change). Why the bridge model and not Stage 8's fr-ja_ft:
fr-ja_ft learned the facts in Japanese too (a second, largely separate copy), so Japanese
unlearning there would change what is stored, not only carry-over.

### Plan 5 GPU results, round 1  `[PROVISIONAL: single seed]`

Jobs 1103089-1103096 + gate 1112833, all COMPLETED (2026-11-10..12). Read with
`plots/plot_depth_curve.py` (fig 18 regenerated) and `scripts/bridge_gate_report.py`.
Relearned French P(gold): LOWER = more durable. 3-seed midpoints 0.607 (fr col) / 0.575
(en col).

| run | reached its stop? | at the stop: P(gold) / TR | relearned P(gold) fr / en col | top-third share |
|---|---|---|---|---|
| depth ja 1.5e-5 -> P(gold) 0.100 | **no** (min 0.129 at ep 15, then back to 0.190 by ep 50) | 0.190 / 1.249 (ep 50) | not relearned | 60% |
| depth ja 1.5e-5 -> P(gold) 0.018 | **no** (identical trajectory to the run above) | 0.190 / 1.249 | not relearned | -- |
| depth en 5e-6 -> P(gold) 0.100 | yes | 0.096 / 3.771 | **0.319 / 0.268** | 44% |
| D ja, layers 0-23 only (TR 0.8276) | **no** (TR 0.776 at ep 50; ep 19: 0.688 vs 0.767 unrestricted) | 0.509 / 0.776 | not relearned | 0% (check) |
| D en, layers 24-35 only (TR 0.8276) | yes | **0.620** / 0.859 | 0.614 / 0.597 | 100% (check) |
| smoke ja 0-23 / en 24-35 | `[freeze] OK` both: change outside the range exactly 0 | | | |

**Depth curve -- answered: the gap holds at every depth tested and widens.** Pre-registered
0.100 comparison not possible (ja never reaches it; exclusion rule). Over the shared range,
gap ja - en in relearned P(gold), fr col: ~0.06 at the TR match (P(gold) ~0.45-0.49), 0.140
at 0.274, ~0.17 at 0.194 (en value interpolated between ep15 @0.274 and pg0p100 @0.096,
the same en trajectory). **Japanese unlearning has a floor at French P(gold) ~0.13** (3x en's
LR, 50 epochs), English goes to 0.096 / 0.018 -- consistent with low carry-over (the ja
gradient reaches only part of the fact). Training is reproducible: ep 0-12 of the 0.100 run
match Plan 3's ja 1.5e-5 run to 3 decimals; the 0.018 run ended identical to the 0.100 run.
Unexplained: past ep 15 ja's P(gold) rises while TR keeps rising.

**Experiment D -- not answered yet.**
- ja 0-23 did not reach matched depth at 5e-6 (pre-registered reading: "cannot reach matched
  depth with these layers"; weak, it ran out of LR schedule). **Allowed fallback, not yet
  run: ONE rerun at lr 1.5e-5, `LAYERS=0-23`.**
- en 24-35 reached TR 0.8276 while P(gold) barely moved (0.620 vs 0.49 for unrestricted en):
  nominally above both midpoints (ja-like) but **fails the pre-registered depth check** (>0.1
  from 0.49-0.50); against en's own curve it looks like ordinary en, and relearning did not
  raise P(gold) (0.620 -> 0.614). Inconclusive.
- **Method finding:** for unrestricted edits TR and P(gold) move together, so TR-matching
  also roughly matches P(gold); under layer restriction they come apart (a top-only edit
  moves TR far more than P(gold)). Proposed amendment (needs the user's OK, decided after
  seeing data): rerun en 24-35 stopped on P(gold) ~0.49 (`STOP_PGOLD=0.490`).

**Bridge -- gate FAILED, arms not run (pre-registered).** Data checks all OK (0 ja rows
about forget authors; bridge - floor = the 40 French forget rows). Japanese carry-over FELL:
fr_ft 0.18 -> bridge 0.07 (needed >= 0.41); absolute ja log p of the forget answers also
fell (-2.19 -> -2.51), so not a floor artifact; 13/40 facts rose. French ceiling OK (P(gold)
0.732 vs 0.700). Reading: bilingual task training does not raise carry-over -- the
intervention cannot be applied; consistent with Stage 8's language-separate copies.
Checkpoint `experiments/tofu_learn_full_full_qwen3-8b_fr-ja-fgfr` (16 GB) is no longer needed.

**Round 2 (submitted 2026-11-13, `preregistration_plan5.json` "amendments_round2"):** job
1129112 = ja `LAYERS=0-23` lr 1.5e-5 (the pre-registered fallback, TR stop); job 1129113 =
en `LAYERS=24-35` stopped on French P(gold) 0.490 (AMENDMENT, decided after round 1),
chained `afterany`.

### Language-agnostic scores, zero GPU (2026-11-13)  `plots/read_token_scores.py`

From the 2026-11-13 supervisor meeting: drop language-specific conclusions and ask whether a
MODEL-BASED score of the unlearning data explains what "language" explains. Stored log probs
only; descriptive. Stage 3 checkpoints are single seed; B2 covers seeds 1 and 2.
- **[B] Per token, the arms split into two regimes that no token score explains.** Own-
  language forget tokens, destruction = lp_unl - lp_fr_ft. en / id / ru are **selective**:
  near-certain tokens (fr_ft lp > -0.1) lose 1.0 / 1.4 / 6.8 nats, uncertain ones 24-32.
  ja is **uniform**: -50 to -60 nats in every bin of confidence, base-model familiarity
  and token carry-over. Replicates on en/ja seeds 1, 2 (near-certain en -0.9, ja -42 / -49)
  and on ja-fast (12 epochs, -41), so it is not run length. All matched runs are
  `floornone` (unbounded ascent): ja drives its own text to ~-50 nats before French moves.
- **[A] Per fact, carry-over predicts REACH, not durability directly.** Within a language
  relearned P(gold) is mostly fact identity (rho with fr_ft P(gold) 0.52-0.90; with carry
  -0.10 to -0.27). Paired by fact (removes identity), where language A carries more of fact i
  than B, A's arm removed more of it at the stop: rho(dCarry, dStop) -0.14 to -0.44 in all 6
  pairs; relearned level -0.41 to +0.03 (weak); regain is confounded with depth.
- **Reading (proposed, language-free):** what matters is how strongly the unlearning set's
  likelihood is coupled to the target's under theta. Strong coupling -> the ascent lowers the
  target while it is still content-selective; weak coupling -> the ascent must crush the
  unlearning set itself (uniform, content-blind damage, top-layer heavy) before the target
  moves, and benign relearning undoes that. Candidate scalar: kappa = target change per unit
  of unlearning-set loss (en 13.5 vs ja 54.7 own-language forget NLL for the same French TR,
  Q2 finding 5); its first-order form at theta_0 is gradient alignment cos(grad L_u, grad
  L_target), measurable per layer in one inference-cost job.

**Cluster pitfall found (fixed in 8030e32):** any `sbatch --export=ALL,<VAR=...>` sets
SLURM_GET_USER_ENV=1 here; on xgph9 the login-env rebuild fails and the job is "user env
retrieval failed requeued held" (cost 27 h). Pass variables as `env VAR=... sbatch` instead.
Old scripts still using the pattern: `submit_seeds.sh`, `14_`, `20_`, `06_` (not live).

## Run

```bash
# laptop: commit + push.  cluster, login node:
cd ~/unlearning && git pull
bash studies/learn_french/side_checks/slurm/submit_all.sh          # parallel, ~10h, ~180GB peak
bash studies/learn_french/side_checks/slurm/submit_all.sh serial   # ~24h, ~65GB peak
# laptop, afterwards:
rsync -avz 'unlearning:~/unlearning/studies/learn_french/side_checks/results/' studies/learn_french/side_checks/results/
python3 studies/learn_french/side_checks/plots/read_plan1.py --by-author
python3 studies/learn_french/side_checks/plots/read_plan2.py              # --metric lp_mean / cer
# Plan 3 / control / seed sweep (cluster):
sbatch studies/learn_french/side_checks/slurm/plan3_arm.sbatch ja 1.5e-5            # one arm
sbatch --export=ALL,STOP_EPOCH=15 studies/learn_french/side_checks/slurm/plan3_arm.sbatch en 5.0e-6
bash studies/learn_french/side_checks/slurm/submit_seeds.sh 2                      # seed sweep
# laptop: every Plan 3 / control / seed figure (7-14, 16)
.venv-plot/bin/python studies/learn_french/side_checks/plots/plot_plan3.py
```

Every job skips steps whose output exists, so a failed or killed job is resubmitted with
the same command. If `plan1_arm` dies *during* unlearning it refuses to restart until the
partial `results/plan1/{traj,tokens}_ul<L>_s42.jsonl` and `experiments/side_checks/ul<L>_s42_ep25`
are moved aside.

## Outputs (`results/`, rsync only)

| file | what |
|---|---|
| `plan1/traj_ul<L>_s42.jsonl` | French TR every epoch, MU every 5 epochs, loss terms |
| `plan1/tokens_ul<L>_s42.jsonl` | line 1 = texts; then per epoch: per-token log p / rank / top-1 on forget01 (5 languages) + 200 retain99_fr rows, French P(gold) per fact |
| `plan1/measure/*.json` | Stage 1 scorer: epoch 25, fr_ft, fr_retain |
| `plan1/relearn/*.json` | Stage 1 scorer on each relearn epoch |
| `plan1/per_token_*.json` | post-hoc per-token passes (in-run check) |
| `plan2/<checkpoint>.json` | every prompt, generation, metric and reference log p |
| `plan3/traj_<run>.jsonl`, `tokens_<run>.jsonl` | as plan1, for each Plan 3 / control / seed run; `<run>` = `ul<L>_s<seed>_lr<X>_tr0p828` (stopped at the level) or `_ep<N>` (fixed epoch) |
| `plan3/measure/<run>.json` | Stage 1 scorer on the stopped weights |
| `plan3/relearn/relearn_<run>_via_retain[_langfr]_ep3.json` | Stage 1 scorer after relearning (no suffix = English relearn) |
| `plan3/names/<run>.json`, `plan3/layer_diff_<run>.json` | name skill; layer diff fr_ft -> stopped weights |

## Figures (`../figures/side_checks/`)

| fig | script | what | report use |
|---|---|---|---|
| 1_effort_vs_depth | plot_side_checks | French TR per epoch, five Stage 2 arms | design context |
| 2_durability | plot_side_checks | Plan 1: relearned P(gold)/TR at matched vs ep 25 | supplement |
| 3_names_delayed, 4_name_pieces, 5_script_switch, 15_name_translation | plot_side_checks | Plans 1-2: names | names / Q1 |
| 6_blanket_by_epoch | plot_side_checks | Plan 1: blanket by epoch | supplement |
| 7_plan3_design | plot_plan3 | Plan 3 TR trajectories, LR schedules, LR budget | supplement (design) |
| **8_plan3_durability** | plot_plan3 | language vs run length, en/ja, primary arms | **main: run-length objection** |
| 9_plan3_layers | plot_plan3 | layer profile + size per run | main: same depth, different edits |
| 10_plan3_blanket | plot_plan3 | token destruction, own language vs French | main/supplement |
| 11_plan3_pgold_said | plot_plan3 | Fig 8 layout on P(gold) -- arms NOT P(gold)-matched | supplement only, with that caveat |
| **12_five_languages** | plot_plan3 | carry-over vs layers / size / names / recovery | **main: mechanism (proposed)** |
| 13_five_durability | plot_plan3 | five Stage 3 arms: relearned TR, P(gold), recovery | main/supplement |
| **14_language_vs_epochs** | plot_plan3 | Fig 8 with all five languages + secondary ja-fast | **main: language vs run length** |
| **16_pgold_control** | plot_plan3 | en@15 vs ja-fast matched on P(gold) and epochs | **main: metric objection** |
| **17_seed_sweep** | plot_plan3 | en/ja gap over seeds 42, 1, 2 vs run-to-run spread | **main: seed objection** |
| **18_depth_curve** | plot_depth_curve | every relearned en/ja run: relearned P(gold) vs P(gold) at the stop | **main: depth-matching objection** |
| **report/report_run_length** | plot_report | single-panel version of 14 for the CA report: en vs ja only, relearned French TR vs run length | report Fig. 3.2 |
| **report/report_layers** | plot_report | single-panel version of 12 panel A for the CA report: layer profile per arm, top-third share | report Fig. 3.3 |

Every number in a figure title is computed from the data plotted. Figures 8, 11 and the
bars of 14 use the PRIMARY Plan 3 arms (ja-fast 1.5e-5); the secondary is drawn dotted /
as a triangle.

