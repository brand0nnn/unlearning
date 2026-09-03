# Knowledge Forgetting & Unlearning in Fine-Tuned LLMs

FYP, NUS School of Computing. **Does LLM unlearning delete knowledge, or only suppress it —
and does the answer depend on the language you relearn in?**

```
LEARN (English TOFU)  ->  UNLEARN (English forget01)  ->  BENIGN RELEARN (language L,
                                                          retain set — never the
                                                          forgotten fact)
                                                       ->  PROBE in English
```

"Benign" is the point: the relearn data never contains the forgotten fact — it is different
authors, in a different language. If the English fact returns anyway, unlearning suppressed
rather than deleted it.

**Two axes:** the **relearn language** (`en fr id ru hi fa ar iw ko ja`) and the **unlearning
method** (Full-FT vs LoRA).

Setup: base `Qwen/Qwen3-8B` · English TOFU `forget01` (40 facts, 2 entities) ·
`gradient_difference` loss · benign relearn on the retain set · metric = **truth ratio**
(LOW = knows the fact).

**Headline results so far** (all `[PROVISIONAL: single seed]`):

- Benign relearning in *any* of the 10 languages returns **~40-60%** of what unlearning
  removed. Neither method deleted the fact.
- **No detectable dependence on the relearn language** — and the study has no power to
  detect one (per-language CIs are 126-139 points wide against an 18-24 point spread).
- Searching over **English phrasings** buys significantly more on an unlearned model than on
  one that was never unlearned. Under the best of four held-out phrasings the LoRA-unlearned
  model is **statistically indistinguishable from the learned model** — before any relearning
  at all.

See [`studies/crosslingual_recovery/README.md`](studies/crosslingual_recovery/README.md) for
the design, the numbers, and how to run and plot everything.

## Layout

| Folder | What lives here |
|---|---|
| `config/` | `config.yaml` — every knob (model, hyperparameters, TOFU settings) — plus the DeepSpeed config. |
| `src/` | Reusable library, never run directly: data loaders, model loading, metrics, trainers. |
| `shared/` | Pipeline entry points: `01_learn` → `02_unlearn` → `03_evaluate`, plus `relearn` / `relearn_measure` / `probe_score` / `spectral`. |
| `studies/crosslingual_recovery/` | **The experiment** — its `slurm/`, `probes/`, `plots/`, `results/`, `figures/`, `scripts/`. |
| `experiments/` | Training checkpoints (git-ignored scratch). |
| `papers/` | Source PDFs (git-ignored), indexed by `papers/papers.md`. |

`studies/strategy_comparison/`, `lora_target_ablation/`, `lora_rank_ablation/` and
`lora_locality/` are **retired directions**, kept for provenance only — they are not part of
the current research and their results should not be cited as live.

## Where code goes

- Needed by two scripts → `src/`.
- A pipeline stage anything can invoke → `shared/scripts/`.
- Specific to the experiment (a sbatch recipe, a plot) → `studies/crosslingual_recovery/`.

## Running it

GPU stages run on the cluster and write JSON; **every figure is made locally** from that JSON.

```bash
# --- cluster ---
ssh unlearning
cd ~/unlearning && bash setup.sh                 # one-time: venv + CUDA torch + data
sbatch studies/crosslingual_recovery/slurm/01_learn.sbatch          # step 0
sbatch studies/crosslingual_recovery/slurm/crosslingual_unlearn_deep.sbatch
sbatch studies/crosslingual_recovery/slurm/crosslingual_relearn_deep.sbatch
python studies/crosslingual_recovery/scripts/check_results.py       # login-node safe

# --- laptop: results/ is gitignored, so rsync it even after a git pull ---
rsync -avz unlearning:~/unlearning/studies/crosslingual_recovery/results/ \
      studies/crosslingual_recovery/results/
source .venv-plot/bin/activate
python studies/crosslingual_recovery/plots/plot_fraction_recovered.py
```

Two Python environments, deliberately: `.venv/` on the cluster (CUDA torch, training) and
`.venv-plot/` on the laptop (matplotlib/scipy/sklearn, plotting only). Figures are never
generated on the GPU.
