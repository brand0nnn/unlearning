# side_checks — two exploratory checks beside the main study

Not part of the pre-registered Stages 1-8. Nothing here writes to `learn_french/results/`,
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

1. **Copy protection is a stage, not a property of the language** (fig 3). Among tokens
   fr_ft gave p > 0.999, copied names are spared early and then overtake other text: fr
   on French text 0.4% at the matched ep 13 -> 54% at ep 25 (other text 29%); en on
   English 0.0% at ep 12 -> 35% at ep 25 (28%). Names pass other text at ep 16 (fr), 23
   (en, own text), 21 (en, French text). Stage 6's fr/en/id-spare vs ja/ru-destroy gap
   is mostly *when you look*; ja still gets there fastest (99.5% at ep 25 vs fr 54%).
2. **More epochs make forgetting MORE durable, not less** (fig 2). After French benign
   relearning (3 epochs), P(gold): en ep 12 0.57 vs en ep 25 0.14; fr ep 13 0.26 vs
   fr ep 25 0.014 (below never-taught 0.094). At the same 25 epochs ja relearns to 0.63.
   Stage 3 finding 3's "effort predicts durability" (r = +0.87) is therefore not causal:
   epochs-to-matched-depth marks the language, it does not make forgetting fragile.
   Within one run effort and depth rise together, so this does not separate them.
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

## Plan 3 — matched depth AND matched epochs (ja-fast, en-slow)  (WRITTEN 2026-10-25, not run)

At the Stage 3 matched checkpoints en and ja agree on French TR, P(gold) and MU, but ja ran
25 epochs and en 12. Plan 3 changes only the peak LR so each reaches the same level (0.8276)
in the other's epoch count: ja at 1.0e-5 and 1.5e-5 (predicted epochs ~14 / ~11), en at
2.0e-6 (~25). Stop at the first per-epoch crossing, score, name skill, layer diff, then
benign relearn in fr and en (Stage 3 recipe, final epoch only). Predictions, reading rule
and exclusions are fixed in `preregistration_plan3.json`. Primary outcome: ABSOLUTE
relearned French TR -- on P(gold) en and ja relearn to similar absolute values (0.57 /
0.63, fr column), so recovery % there mostly reflects its small denominators.
`slurm/plan3_arm.sbatch <lang> <lr>`, ~8-14h, peak disk ~33GB.

## Plan 4 — layer profile at fixed epochs (ja, fr, en)  (WRITTEN 2026-10-25, not run)

Job 11 found ja's change top-heavy (52% of delta_sq in layers 24-35 vs 36-38% for fr/en),
but at 25 epochs against 12-13. Plan 4 re-runs the Stage 2 recipe unchanged to epoch 25,
saving epochs 13 and 25, and diffs both against fr_ft. `slurm/plan4_arm.sbatch <lang>`,
~5-6h, peak disk ~33GB. fr/en epoch 25 repeat Plan 1, so their JSONL is a reproduction check.

Both use `scripts/unlearn_epochs.py`'s new `--lr`, `--stop-at-level`, `--save-at-epochs`;
without them it runs Plan 1 exactly. Submit all six with `slurm/submit_plan34.sh [lean]`.

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
