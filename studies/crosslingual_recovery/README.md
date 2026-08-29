# crosslingual_recovery  *(active)*

Unlearn an English forget01 fact, benign-relearn on the RETAIN set in each of 10
languages (ordered by distance from English), and measure how much the ENGLISH fact
recovers — does the fine-tuning METHOD change the recovery-vs-distance pattern?
The pilot auto-matches LoRA's forget depth to Full-FT's to de-confound the comparison.

**Run:** `sbatch slurm/crosslingual_pilot.sbatch`  (self-contained; forces --forget-level forget01)
**Plot (local):** `python plots/plot_crosslingual_pilot.py` → `figures/crosslingual_pilot_recovery.png`
  (left = raw recovery vs distance; right = baseline-normalized decay shape)
`results/`: relearn/crosslingual_pilot/ (+ crosslingual_pilot_probe/ = the LoRA baseline-match candidates)


---

## Phase 2 — instance-wise cross-lingual probing

Phase 1 probed **only in English** (`relearn_measure.py` hard-loaded the English forget
set), so every result collapsed to one scalar per relearn-language: "relearn in Japanese
→ the *English* fact returns by X%." That cannot separate the two hypotheses the study
is actually about — a language-agnostic store (relearning anywhere restores the fact
everywhere, *same facts*) versus a language-local one (recovery is mostly confined to
the relearn language). Phase 2 probes the recovered model **in all 10 languages, per
fact**, and reads the structure of the resulting |facts| x |languages| matrix.

Design doc: `PHASE2_PLAN.md`. Metric machinery is KBL's (`S_i`, KSS, KPS); the truth
ratio is TOFU's, substituted for KBL's length-normalised probability because `P` is not
comparable across languages; `tau` is Youden's J; the recovery-set overlap is a Jaccard
index against a permutation null.

**Run in this order.** Steps 0 and 1 are independent — submit both at once.

```bash
# STEP 0  translation QC (720 records). Downloads NLLB-200-3.3B on first run.
sbatch slurm/phase2_verify_translations.sbatch

# STEP 1  the tau-calibration GATE (~2h). Probes the LEARNED checkpoint (known
#         positives) and BASE Qwen3-8B (known negatives) in 10 langs x 3 splits.
sbatch slurm/phase2_calibrate.sbatch
#         smoke test first (~5 min):
sbatch slurm/phase2_calibrate.sbatch "en ja" "forget"

# STEP 2  read the gate LOCALLY before spending the big job
python plots/phase2_kss_kps.py --calibrate-only

# STEP 3  the main probe. Exceeds the 12h wall by design; resubmit to resume.
sbatch slurm/crosslingual_phase2_probe.sbatch

# STEP 4  everything, locally from the JSON
python plots/phase2_kss_kps.py
```

**Step 2 is a gate, not a formality.** If the learned and base populations do not
separate in language *l* (AUC < 0.70), that language's in-language probe is
uninformative and must be **reported as such rather than scored**. The `world_facts`
split says *why*: it is pre-training knowledge, untouched by our fine-tuning, so good
world_facts + bad TOFU means our translations, while bad on both means the model is
simply weak in that language. If fewer than ~3 languages survive, do not launch step 3 —
report the negative result and fall back to the English-only design.

Results land in `results/relearn/phase2_calibrate/` and `results/relearn/phase2_probe/`
(one file per checkpoint, as everywhere else), plus `results/phase2/translation_qc.json`
and the derived `results/phase2_tau.json` / `results/phase2_summary.json`.
