# Knowledge Forgetting & Unlearning in Fine-Tuned LLMs

FYP, NUS School of Computing. **Does LLM unlearning delete knowledge, or only suppress
it — and does the answer depend on the language you probe with?**

The active experiment learns a fact in English, unlearns it in English, then
**benign-relearns in each of 10 languages** on data that never contains the forgotten
fact, and probes in English. If the fact returns, unlearning suppressed rather than
deleted it. See `studies/crosslingual_recovery/README.md` for the design and findings,
and `EXPERIMENTS.md` for the map of all five studies.

## Layout

| Folder | What lives here |
|---|---|
| `config/` | `config.yaml` — every knob (model, hyperparameters, TOFU settings) — plus the DeepSpeed config. |
| `src/` | Reusable library, never run directly: data loaders, model loading, metrics, trainers. |
| `shared/` | The pipeline entry points every study invokes: `01_learn` → `02_unlearn` → `03_evaluate`, plus `relearn` / `relearn_measure` / `spectral`. |
| `studies/<name>/` | One self-contained experiment each: its `slurm/`, `plots/`, `results/`, `figures/`, `README.md`. |
| `experiments/` | Training checkpoints (git-ignored scratch). |
| `papers/` | Source PDFs (git-ignored). |

## Where code goes

- Needed by two scripts → `src/`.
- Run from the terminal → `shared/scripts/` (pipeline-wide) or `studies/<name>/` (one study).
- A number or path that changes between runs → `config/config.yaml`, never inline.

## Running

GPU stages run on the NUS SoC cluster and write metric JSON; **all figures are generated
locally** from that JSON, so no plot ever needs a GPU.

```bash
bash setup.sh                                   # one-time, on the login node
sbatch studies/<name>/slurm/<job>.sbatch
rsync -avz unlearning:~/unlearning/studies/<name>/results/ studies/<name>/results/
source .venv-plot/bin/activate && python studies/<name>/plots/<script>.py
```

`results/` is git-ignored, so it travels by rsync rather than git. Each study's sbatch
sets `UNLEARN_RESULTS_DIR` so its JSON lands inside its own folder, and its plot scripts
self-locate — they take no arguments.
