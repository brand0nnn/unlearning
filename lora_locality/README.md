# LoRA-locality relearning ablation

**Self-contained experiment, separate from the main 4-strategy pipeline.**
Outputs live under `lora_locality/out/` (not the main `results/` folder); checkpoints
go to `experiments/loc_*` (gitignored, like all checkpoints).

## Question

Does *where* the LoRA adapters sit (which weight matrices) change how **recoverable**
the forgetting is under a relearning attack? Hypothesis (ROME/MEMIT): facts live in the
MLP, so MLP-LoRA should delete deeper → recover **less** than attention-LoRA.

## Design (two schemes, to separate location from capacity)

| scheme | ranks | isolates |
|---|---|---|
| `samerank` | every location at rank 32 | natural module-group configs (capacity differs) |
| `fixedbudget` | rank ∝ 1/params so all ≈ equal trainable params | the *locus* itself |

Both are run; if the location ranking agrees across them, the effect is robust. Every
plot labels locations with their **parameter count**, so capacity is always visible.
See the tables: `python lora_locality/config.py`.

## Controls

- **Matched forgetting:** each location is unlearned with **early-stopping at a common
  forget level** (`--target`), so recovery is measured from the same starting point.
  The target is reached with retain protected (`--retain-floor`).
- **Identical attack:** all locations are relearned with the same Full-FT re-finetune
  (`scripts/recovery/relearn.py`) at epochs 1/2/4 — only the *unlearning* location differs.
- **Depth metrics:** recovery is tracked in ROUGE **and probability and truth ratio**
  (depth of forgetting, which ROUGE alone misses).

## How to run (per scheme)

```bash
# 0. (once) pick the matched target by CALIBRATING — train full, read the floors:
sbatch slurm/ablation/lora_locality_unlearn.sbatch fixedbudget 0
#    then inspect lora_locality/out/fixedbudget/unlearn/*.json -> "matched" forget rouge
#    choose a target ALL locations reach with retain intact (e.g. 0.4)

# 1. unlearn each location, early-stopping at the chosen target
sbatch slurm/ablation/lora_locality_unlearn.sbatch fixedbudget 0.4

# 2. relearn + measure recovery
sbatch slurm/ablation/lora_locality_relearn.sbatch fixedbudget

# 3. plot LOCALLY (rsync lora_locality/out down first)
python lora_locality/plot.py --scheme fixedbudget
```

Repeat with `samerank` in place of `fixedbudget`. Smoke-test first with
`tofu.eval_limit`/`unlearn_epochs` small (CLAUDE.md §6).

## Files

- `config.py` — locations, the two rank schemes, trainable-param accounting (local, testable).
- `unlearn.py` — LoRA unlearn one location, early-stop at target, save matched checkpoint.
- `measure.py` — forget ROUGE + prob + truth of a checkpoint (one recovery point).
- `plot.py` — overlay recovery per location for a scheme.
- `out/<scheme>/{unlearn,recovery}/*.json`, `out/<scheme>/recovery_<scheme>.png` — results.
