# Phase 2 (revised): Behavioral + representational probing

This replaces the old separate "Phase 2" and "Phase 3" from the earlier plan. They've been
merged into one phase with two parts that share infrastructure but use genuinely different
techniques. Both parts are English-only at the probing layer — our fact was only ever learned
in English, so probing in other languages would test generic cross-lingual QA ability, not
recovery of this fact. The only place language varies is the relearning step, which is already
implemented.

Two reference papers are relevant here, both should be in the repo/project folder:
- **"Knowledge Beyond Language"** — methodology for Part A (probe generation), used only
  partially (see below).
- **"Unlearning Isn't Invisible"** (Chen et al., ICLR 2026) — methodology for Part B
  (activation classifier), followed closely. If the exact filename differs from what you find,
  search for it by title — it's the paper about detecting "unlearning traces" via supervised
  classifiers on model outputs and pre-logit activations.

---

## Part A: Build one probe family, apply it at three checkpoints (behavioral)

**Step 1 — Build the probe family, once, in English.**

For each of the 40 forget-set facts, generate a small family of English-only probes per
entity:
- 3-5 paraphrases of the canonical QA question (same fact, different wording)
- 1 multiple-choice variant
- 1 fill-in-the-blank variant

Borrow the *generation methodology* from Knowledge Beyond Language's Section 4 pipeline —
specifically their approach of generating attribute-isolated questions that explicitly include
the full entity name in both question and answer (so each probe is unambiguous about which
fact/entity it targets), and their iterative verify-and-refine loop (generate → check
semantic equivalence against the original fact with an LLM judge → revise with a different
model if it fails → repeat until confirmed). Do NOT use their translation step — we have no
use for translating these probes into other languages, since the fact doesn't exist in those
languages to translate against. This is a partial adaptation of their pipeline, not a full
replication — note this explicitly in any writeup.

**Step 2 — Apply this probe family at three points in the pipeline:**

1. **Pre-unlearning** (fine-tuned model, before any unlearning). This is a ceiling/sanity
   check: the model should answer all phrasings correctly here. If any phrasing fails here,
   flag it — that probe is unreliable and should be dropped or fixed before using it further.
2. **Post-unlearning, pre-relearning.** Run the full probe family against the unlearned model,
   before any relearning happens. Record the Truth Ratio per fact per phrasing. This tells us
   whether the "unlearned" state already leaks the fact under some phrasing even before
   relearning — i.e., whether unlearning was ever a true deletion or already partial
   suppression.
3. **Post-relearning, per language.** Re-run the probe family after each of the 10 per-language
   relearning runs. This replaces the current single-QA-pair recovery measurement with
   per-entity, per-phrasing granularity. Report recovery broken down by entity and by phrasing
   type, not just one aggregate number per language, so we can see whether uniformity holds at
   this finer grain or was an artifact of averaging.

Use the same per-fact array persistence pattern already added for the bootstrap CI work (item
8 of the earlier audit) — store per-fact, per-phrasing truth ratios, not just means.

---

## Part B: Activation-based classifier probing (representational)

This is a genuinely different technique from Part A — it doesn't look at the model's output
text at all, only at internal activations, so it can catch suppressed knowledge that no
phrasing manages to surface behaviorally.

**Follow "Unlearning Isn't Invisible"'s methodology closely, with one adaptation** (see note
at the end of this section).

**Step 1 — Collect activations and responses.**

Query the post-unlearning model (the same checkpoint used for the Part A, checkpoint 2 probe)
with two groups of prompts:
- **Forget-relevant**: the forget-set questions, including the paraphrase family from Part A
  for more data points.
- **Forget-irrelevant**: retain-set questions (already available in the pipeline).

For each prompt, collect:
- The generated textual response.
- The **pre-logit activations** — the activation vector at each newly generated token,
  extracted right before the final unembedding/logit projection. Aggregate across the
  generated sequence into one representation per response (follow the paper's aggregation:
  they use the sequence of per-token activation vectors from the generation; check Section 4
  and Appendix B of the paper for their exact aggregation method before choosing one).

**Step 2 — Train two classifiers, following the paper's setup:**
- **Text-based**: encode each generated response with LLM2Vec, then train a two-layer MLP
  for binary classification.
- **Activation-based**: train a two-layer MLP directly on the raw pre-logit activation vectors,
  no encoding step.

**Step 3 — Train/test split, following the paper's discipline.** Use a balanced mix of
forget-relevant and forget-irrelevant prompts for training. Evaluate on **held-out, disjoint**
prompts (different specific questions than used in training, even within the same forget/
retain sets) to confirm the classifier is picking up a generalizable signal, not memorizing
training prompts. Report classification accuracy separately for forget-relevant and
forget-irrelevant test prompts, the way Table A3 in the paper does.

**Step 4 — Optional, if time allows: spectral fingerprint analysis.** The paper also runs SVD
on the activation matrices and projects onto top singular vectors to visualize where the
"unlearning trace" concentrates. This connects directly to our planned mechanistic Phase 4
(weight-delta cosine similarity) — if you have bandwidth, running this analysis now could save
duplicate setup work later, but it's not required for the core Phase 2 deliverable.

**Adaptation note — state this explicitly in any writeup:** the paper's classifier is trained
to distinguish *original vs. unlearned model* (a meta-level "has this model undergone
unlearning" question), using both forget-relevant and forget-irrelevant prompts as inputs to
that same binary task. Our question is different and narrower: within our single unlearned
model, can a classifier distinguish forget-relevant activations from forget-irrelevant ones?
High classifier accuracy on this task means the model's internal state still carries a
detectable, distinguishable signature tied to the forgotten content specifically — i.e.,
direct evidence for obfuscation over genuine deletion. This reuses the paper's exact
architecture and training discipline (pre-logit activations, two-layer MLP, disjoint
train/test) but repoints the classification target to fit our research question. Do not
present this as an unmodified replication of their method in any writeup — describe it as
adapted from their approach.

---

## Deliverables for this phase

1. The probe family (paraphrases, MCQ, fill-in-blank) for all 40 facts, with the verification
   step's pass/fail log.
2. Per-fact, per-phrasing Truth Ratio at all three checkpoints (pre-unlearning,
   post-unlearning/pre-relearning, post-relearning per language).
3. Classifier accuracy (text-based and activation-based) on forget-relevant vs. forget-irrelevant
   prompts, at the post-unlearning/pre-relearning checkpoint, with train/test-disjoint
   evaluation reported the way the paper's Table A3 does.
4. A written verdict: does the post-unlearning model show behavioral leakage under any
   phrasing (Part A, checkpoint 2), and/or representational leakage detectable by the
   classifier (Part B)? If either is true, flag this clearly — it changes how we should frame
   the main recovery result in the paper (suppression-removal vs. genuine relearning-driven
   transfer).

## Dependencies / sequencing

This can run in parallel with finishing the bootstrap CI work (item 8 from the earlier audit)
— they use overlapping infrastructure (per-fact array persistence) but aren't blocking each
other. Do not fold this phase's results into the vocab-overlap correlation (Phase 1) until
both this phase and the bootstrap CI are complete, since Phase 1 assumes the recovery number
being correlated against vocab overlap is itself well-understood and not confounded by
undetected suppression.
