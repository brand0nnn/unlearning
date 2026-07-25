# Experiments map

Each experiment is a **self-contained folder** under `studies/` — its own slurm
recipes, plot scripts, results JSON, and figures. Deleting or archiving a study
removes everything it produced, in one place. Only the reusable baseline lives
outside, under `src/` (library) and `shared/` (entry points every study calls).

```
src/            SHARED LIBRARY (never run directly): data · models · evaluation ·
                training · utils   — used by every study
shared/
  scripts/      01_learn · 02_unlearn · 03_evaluate · relearn · relearn_measure ·
                spectral   — the pipeline every study's sbatch invokes
  plots/        plot_relearn (generic) · plot_loss   — generic, study-agnostic
  diagnostics/  check_rouge · dump_generations (+ their sbatch)
config/         config.yaml · ds_config.json
experiments/    checkpoints (git-ignored scratch, shared) · logs/ · papers/

studies/<name>/           ONE SELF-CONTAINED EXPERIMENT
  slurm/                  its sbatch recipes (call shared/scripts/*.py)
  plots/                  its own plot scripts (self-locate this study's results/figures)
  results/                its metric JSON (curves/ forget_quality/ relearn/ spectral/)
  figures/                its PNGs
  README.md               what it is + how to run + plot
```

## How results stay inside each study

Shared scripts write to `UNLEARN_RESULTS_DIR` (default `results/`). Every study's
sbatch sets `export UNLEARN_RESULTS_DIR=$PROJECT_DIR/studies/<name>/results`, so all
metric JSON lands in that study's folder. Study plot scripts **self-locate** — they
compute their own study dir from `__file__`, so `python studies/<name>/plots/x.py`
reads that study's `results/` and writes its `figures/` with no arguments.

Import root is found by walking up to the dir containing `src/` (not a fixed
`parents[N]`), so scripts work at any depth.

## The five studies

| Study | What | Status |
|---|---|---|
| `strategy_comparison` | Full-FT vs LoRA vs Self-Distill vs GRPO on English TOFU + recovery axes (relearn, spectral, forget-quality plane) | foundation, complete |
| `lora_target_ablation` | LoRA on which modules (attn vs MLP) forgets/deletes more | side-study |
| `lora_rank_ablation` | LoRA rank sweep — capacity vs forgetting/collateral | side-study |
| `lora_locality` | LoRA module-group locality — does *where* adapters sit change recoverability? (neighbourhood-locality precursor) | complete; **standalone** — own scripts + `out/`, doesn't use the shared pipeline |
| `crosslingual_recovery` | Unlearn EN fact → benign-relearn in 10 languages → does the fine-tuning METHOD change recovery vs language distance | active |

## Run / plot cheat-sheet

```bash
# strategy_comparison (foundation)
sbatch studies/strategy_comparison/slurm/01_learn.sbatch
sbatch studies/strategy_comparison/slurm/02_unlearn.sbatch     # + _lora/_selfdistill/_grpo
sbatch studies/strategy_comparison/slurm/03_evaluate.sbatch
sbatch studies/strategy_comparison/slurm/relearn_retain.sbatch # + relearn_forget / recover_spectral
python studies/strategy_comparison/plots/plot_all.py           # local, regenerates all its figures

# crosslingual_recovery (active)
sbatch studies/crosslingual_recovery/slurm/crosslingual_pilot.sbatch
python studies/crosslingual_recovery/plots/plot_crosslingual_pilot.py

# ablations
sbatch studies/lora_target_ablation/slurm/lora_target_ablation.sbatch   # then lora_ablation_relearn
sbatch studies/lora_rank_ablation/slurm/lora_rank_ablation.sbatch
python studies/lora_rank_ablation/plots/plot_rank_sweep.py

# lora_locality (standalone — own scripts, writes out/ not results/)
python studies/lora_locality/unlearn.py  --scheme fixedbudget --location mlp   # per location
python studies/lora_locality/measure.py  --scheme fixedbudget --location mlp
python studies/lora_locality/plot.py     --scheme fixedbudget                  # local
```

**Note — two folder shapes under `studies/`.** Four studies use the shared pipeline
(`slurm/` + `plots/` + `results/` + `figures/`). `lora_locality/` is a **standalone**:
it has its own `unlearn/measure/spectral/plot` scripts and writes to `out/` (rsync'd,
git-ignored) rather than the shared `results/`. Both are self-contained folders — that's
the property that matters.
