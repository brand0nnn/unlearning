# crosslingual_recovery  *(active)*

**Learn a fact in English → unlearn it in English → benign-relearn in each of 10
languages → probe in English.** How much of the forgotten fact comes back, and does
the *language* of the benign data change that?

"Benign" means the relearn data **never contains the forgotten fact** — it is different
authors, in a different language. If the English fact returns anyway, unlearning
suppressed rather than deleted it. Second axis: does the unlearning **method**
(Full-FT vs LoRA) change how easily that suppression is undone?

Base model `Qwen/Qwen3-8B` · English TOFU `forget01` (40 facts) · loss
`gradient_difference` · benign relearn on the **retain** set, `--relearn-n 1500` ·
metric = **truth ratio** (LOW = knows the fact).

Each language is an **independent run from the same unlearned checkpoint** — 10 relearn
languages produce 10 separate relearned models, not one model trained on all 10.
LoRA `uep32` is depth-matched to Full-FT so both start from comparable forgetting.

---

## Results so far

Recovery = `(TR_unlearned − TR_relearned) / (TR_unlearned − TR_learned)`, i.e. the
fraction of removed knowledge brought back. Reference: learned `0.459`,
Full-FT unlearned `0.743`, LoRA unlearned `0.678`.

| relearn epochs | Full-FT | LoRA |
|---|---|---|
| ep1 | 43.4% | 58.7% |
| **ep2** (reporting epoch) | **42.5%** | **62.8%** |
| ep4 | 41.9% | 77.3% |

1. **Benign relearning substantially recovers the fact** for both methods — neither
   deleted it.
2. **Full-FT saturates; LoRA keeps climbing** (gap 16 → 20 → 35 points).
3. **No detectable dependence on the relearn language.** Say it exactly that way, never
   "uniform" — the per-language bootstrap CIs are **126–139 points wide** against a
   between-language spread of only 18–24 points, so the study cannot resolve a
   per-language difference. Resolving it would need ~2,250 facts; `forget01` has 40.

All numbers are **`[PROVISIONAL: single seed]`**.

### Phase 1 — the vocabulary-overlap check (complete)

Rules out the cheapest explanation: that cross-lingual recovery is just shared subword
tokens. Recovery vs overlap-with-English, **n=9, English excluded** (its 1.0
self-overlap is an artifact and a huge leverage point):

| | Jaccard (CLC Eq. 7, FLORES-200) | Overlap coefficient |
|---|---|---|
| Full-FT | r = −0.37, p = 0.33 | r = −0.28, p = 0.47 |
| LoRA | r = +0.71, p = 0.03 | r = +0.10, p = 0.80 |

The LoRA correlation **is not trustworthy** and must be reported with its caveats: it
collapses to r=+0.10 under a different overlap definition, two languages carry most of
the leverage, and it is 1 of 8 uncorrected tests.

**Provenance:** from CLC (Qi et al. 2023) we took **only** the Eq. 7 subword-Jaccard
metric and the **FLORES-200** corpus — not their pipeline, models, or RankC measure.
The **overlap coefficient** (Szymkiewicz–Simpson) is this repo's own robustness check.
The multilingual TOFU translations are **Farashah et al.'s** (*Multilingual Amnesia*,
arXiv 2601.05641), whose Appendix G documents native-speaker QC — cite it rather than
re-verifying the translations.

### Phase 2 — being replaced

The original design (probe each relearned model in **all 10 languages** per fact, score
KSS/KPS from KBL) is **dead**, killed by its own calibration gate. The gate ran and only
**English (AUC 0.825) and French** separate the learned checkpoint from base Qwen3 —
because **LEARN was English-only**. The model never memorised the facts in the other 8
languages, so an in-language probe there measures nothing.

A new Phase 2 plan is pending. `PHASE2_PLAN.md` was deleted on purpose — design notes
live in `CLAUDE.md` and here. `phase2_tau_calibration.png` and
`phase2_recovery_per_fact.png` are kept as evidence for *why* the redesign happened.

---

## Running it

GPU stages write JSON on the cluster; **every figure is made locally** from that JSON.

```bash
# --- cluster ---
sbatch slurm/crosslingual_unlearn_deep.sbatch     # the two unlearned baselines
sbatch slurm/crosslingual_relearn_deep.sbatch     # the 10-lang x 3-epoch grid (~70 min/cell at ep2)
python scripts/check_results.py                   # stdlib-only, safe on the login node

# --- laptop: results/ is gitignored, so rsync it even after a git pull ---
rsync -avz unlearning:~/unlearning/studies/crosslingual_recovery/results/ results/
source ../../.venv-plot/bin/activate
python plots/plot_fraction_recovered.py           # and the rest of plots/
```

**Next up:** `slurm/relearn_content_control.sbatch` — **written, not yet run.** It tests
whether the relearn *content* matters at all, by relearning on TOFU's distance ladder
(`retain` → `real_authors` → `world_facts`) **volume-matched at n=100**, English, ep2,
6 cells, ~2h. `retain` is re-run at n=100 rather than reused from the n=1500 result, and
doubles as a floor check. If all three recover alike, the recovery is generic
re-adaptation and the language finding loses its interest.

`slurm/crosslingual_phase2_probe.sbatch` and `slurm/phase2_verify_translations.sbatch`
are **obsolete** — kept only for provenance.
