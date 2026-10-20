# side_checks — two exploratory checks beside the main study

Not part of the pre-registered Stages 1-8. Nothing here writes to `learn_french/results/`,
`experiments/tr_levels/` or `preregistration.json`; it only reads them. Shared code
(`src/`, `shared/`, `learn_french/scripts/`) is imported, never modified.
`[PROVISIONAL: single seed]`, seed 42, 40 facts about 2 entities.

**Step 0** (zero GPU, done 2026-10-21) is `learn_french/plots/copy_check.py`. It found
that ru/ja "copied" names are mostly real copies (ru 71%, ja 87.5% exact), and that copy
protection is mostly the confidence effect.

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
