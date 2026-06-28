# TOFU Pipeline — quick reference

The TOFU experiment runs in four stages. Files live under src/ (library) and
scripts/ (entry points), consistent with the rest of the repo.

## New files this pipeline added

src/data/load_tofu.py            Load TOFU splits from Hugging Face
src/evaluation/compute_logprobs.py   P(answer|question)^(1/|a|) — the foundation
src/evaluation/tofu_metrics.py   Probability, ROUGE, Truth Ratio, Model Utility, Forget Quality
src/evaluation/tofu_evaluate.py  Orchestrates all metrics across the 4 splits
src/evaluation/plotting.py       The two result figures
src/training/finetune_tofu.py    LEARN phase (answer-only loss masking)
src/training/unlearn.py          UNLEARN: gradient_ascent / gradient_difference / kl_minimization / idk
scripts/tofu_01_finetune.py      Run the learn phase
scripts/tofu_02_unlearn.py       Run an unlearning method
scripts/tofu_03_evaluate.py      Evaluate checkpoint(s) + compute Forget Quality
scripts/tofu_04_plot.py          Produce the figures
slurm/run_tofu.sbatch            Run the whole thing on the cluster

## Run order (locally, for a smoke test)

Set tofu.eval_limit: 20 and small epochs in config.yaml first, then:

    python scripts/tofu_01_finetune.py --data full
    python scripts/tofu_01_finetune.py --data retain90
    python scripts/tofu_02_unlearn.py --checkpoint experiments/tofu_learn_full_full --method gradient_difference
    python scripts/tofu_03_evaluate.py --reference experiments/tofu_learn_retain90_full \
        --checkpoints experiments/tofu_unlearn_gradient_difference_forget10
    python scripts/tofu_04_plot.py

## The two headline numbers

- Model Utility   — harmonic mean of 9 sub-metrics on retain/real_authors/world_facts. Higher = less collateral damage.
- Forget Quality  — KS-test p-value vs the gold retain model on the forget set. Higher = better forgetting.

A perfect method sits TOP-RIGHT of the forget_quality_vs_utility plot.
