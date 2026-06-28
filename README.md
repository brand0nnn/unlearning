# Knowledge Forgetting & Unlearning in Fine-Tuned LLMs

Research scaffold for studying (1) how much factual knowledge a model loses during
fine-tuning, and (2) how different training strategies affect *catastrophic forgetting*
and *intentional unlearning* (training on counterfactuals).

## The two parts of the project

1. **Evaluation pipeline** — measure factual accuracy on LAMA and Natural Questions
   *before* and *after* fine-tuning, so we can quantify what changed.
2. **Training strategies** — implement and compare Full Fine-Tuning, LoRA,
   Self-Distillation, and GRPO (RL-based), under the same controlled conditions.

## The standard workflow (run scripts in order)

```bash
# 0. one-time setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. download + format the benchmarks into data/
python scripts/01_download_data.py

# 2. measure the BEFORE numbers (this is your baseline)
python scripts/02_evaluate_baseline.py

# 3. fine-tune with a chosen strategy
python scripts/03_run_training.py --method lora

# 4. measure the AFTER numbers and compare to baseline
python scripts/04_evaluate_after.py --checkpoint experiments/lora_run/
```

## Directory map (one line each)

| Folder          | What lives here                                                        |
|-----------------|------------------------------------------------------------------------|
| `config/`       | One YAML file with every knob (model name, paths, hyperparameters).    |
| `data/`         | Benchmarks and counterfactual sets. Code never hard-codes data paths.  |
| `src/`          | The reusable library: data loaders, model loading, metrics, trainers.  |
| `scripts/`      | Thin entry points you actually run. They wire `src/` pieces together.  |
| `experiments/`  | Output of training runs: checkpoints, logs (git-ignored).              |
| `results/`      | Final metrics tables and plots you put in your report.                 |
| `notebooks/`    | Scratch space for exploring data and sanity-checking, not for pipelines.|

## A rule of thumb for where code goes

- If two scripts would both need it → it belongs in `src/`.
- If it's a thing you *run from the terminal* → it's a thin file in `scripts/`.
- If it's a number or path you might change between runs → it's in `config/config.yaml`,
  never hard-coded in the middle of a function.
