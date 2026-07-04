"""Spectral trace detectability — recovery axis 3 (Chen et al., ICLR 2026,
"Unlearning Isn't Invisible: Detecting Unlearning Traces in LLMs from Model
Outputs").

The idea in one sentence: unlearning leaves a persistent "fingerprint" in a
model's internal activations, visible even on FORGET-IRRELEVANT prompts, so you
can tell an unlearned model apart from the original one it was derived from —
which means the knowledge was suppressed, not erased. That is exactly the
project's obfuscation-vs-deletion thesis, measured a third way (after relearning
robustness and correlated-knowledge leakage).

What we compute here, given the ORIGINAL learned model and one UNLEARNED
checkpoint:

  1. Activation features. For a set of forget-irrelevant prompts (TOFU
     retain / real_authors / world_facts questions), generate a short response
     and mean-pool the hidden state over the response tokens, at several layers.
     One H-dimensional vector per prompt per layer, per model.

  2. Spectral shift (the paper's "spectral fingerprint"). Following Tran et al.
     2018 (spectral signatures), stack the original and unlearned activations,
     centre the matrix, take its SVD, and project onto the top RIGHT singular
     vectors. Along each principal direction we measure how far the unlearned
     distribution has moved from the original one — Cohen's d (standardised mean
     difference). A large |d| along even one top direction = a loud fingerprint.

  3. Detectability. Train a simple logistic-regression classifier to label each
     activation vector as original (0) or unlearned (1), scored by 5-fold
     cross-validated accuracy. 0.5 = indistinguishable (no trace); ~1.0 = a
     trivially detectable trace. This is the paper's headline metric.

No training and no gradients — pure inference on checkpoints we already have.
"""
from typing import Dict, List, Sequence

import numpy as np

from src.evaluation.compute_logprobs import format_qa
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Activation collection
# ---------------------------------------------------------------------------
def collect_activations(model, tokenizer, questions: Sequence[str],
                        layers: Sequence[int], max_new_tokens: int = 64
                        ) -> Dict[int, np.ndarray]:
    """Mean-pooled hidden states over each model's own generated response.

    Returns {layer_index: array of shape (n_prompts, hidden_size)}.

    `layers` indexes into `output_hidden_states` (0 = embedding output, 1..L =
    after each transformer block, so L = final pre-logit layer for an L-layer
    model). We pool over the RESPONSE tokens only — the paper detects traces from
    the model's own output, not the prompt — which also makes the feature
    independent of prompt length.
    """
    import torch

    feats: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    model.eval()
    for q in questions:
        prompt = format_qa(q)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        if enc["input_ids"].shape[1] == 0:
            continue
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, use_cache=True,
                                 pad_token_id=tokenizer.pad_token_id)
        # Re-run a single forward pass over prompt+response to read clean,
        # non-cached hidden states at every layer in one shot.
        full_ids = gen  # (1, prompt_len + gen_len)
        if full_ids.shape[1] <= prompt_len:
            continue     # model emitted nothing — no response tokens to pool
        with torch.no_grad():
            out = model(full_ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple length L+1, each (1, seq, H)
        # Response tokens occupy positions [prompt_len, seq_len).
        for l in layers:
            resp = hs[l][0, prompt_len:, :]            # (gen_len, H)
            feats[l].append(resp.mean(0).float().cpu().numpy())
    return {l: np.stack(v) if v else np.empty((0, 0)) for l, v in feats.items()}


# ---------------------------------------------------------------------------
# 2. Spectral shift along top singular directions
# ---------------------------------------------------------------------------
def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference between two 1-D samples (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def spectral_shift(orig: np.ndarray, unlearned: np.ndarray, k: int = 5) -> Dict:
    """SVD the stacked (original, unlearned) activation matrix and measure how
    far the unlearned cloud has shifted along each of the top-k principal
    directions.

    Returns:
      cohens_d           list[k]  — signed shift per top direction
      max_abs_d          float    — the loudest single-direction fingerprint
      singular_values    list[k]  — top-k singular values (energy per direction)
      proj_orig/proj_unl (n,2)    — top-2 projections, for the scatter plot
    """
    if orig.size == 0 or unlearned.size == 0:
        return {"cohens_d": [], "max_abs_d": 0.0, "singular_values": [],
                "proj_orig": [], "proj_unlearned": []}
    A = np.vstack([orig, unlearned]).astype(np.float64)
    A = A - A.mean(0, keepdims=True)                 # centre (Tran et al. 2018)
    # Economy SVD; right singular vectors Vt are the principal directions.
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    k = int(min(k, Vt.shape[0]))
    proj = A @ Vt[:k].T                              # (2n, k)
    n = len(orig)
    proj_o, proj_u = proj[:n], proj[n:]
    ds = [_cohens_d(proj_o[:, j], proj_u[:, j]) for j in range(k)]
    kk = min(2, k)
    return {
        "cohens_d": ds,
        "max_abs_d": float(max((abs(d) for d in ds), default=0.0)),
        "singular_values": S[:k].tolist(),
        "proj_orig": proj_o[:, :kk].tolist(),
        "proj_unlearned": proj_u[:, :kk].tolist(),
    }


# ---------------------------------------------------------------------------
# 3. Detectability (supervised classifier, cross-validated)
# ---------------------------------------------------------------------------
def detection_accuracy(orig: np.ndarray, unlearned: np.ndarray, seed: int = 42,
                       folds: int = 5) -> float:
    """5-fold CV accuracy of a logistic classifier telling original (0) from
    unlearned (1) activations. 0.5 = no detectable trace; ~1.0 = trivial to spot.

    Chance is 0.5 by construction: the two classes have equal size, so a
    constant predictor scores 0.5 and anything above it is genuine signal.
    """
    if orig.size == 0 or unlearned.size == 0:
        return 0.5
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([orig, unlearned])
    y = np.concatenate([np.zeros(len(orig)), np.ones(len(unlearned))])
    n_min = min(len(orig), len(unlearned))
    folds = int(max(2, min(folds, n_min)))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, C=1.0))
    scores = cross_val_score(clf, X, y, cv=folds, scoring="accuracy")
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Orchestration: everything for one (original, unlearned) pair, per layer
# ---------------------------------------------------------------------------
def fingerprint(orig_feats: Dict[int, np.ndarray],
                unlearned_feats: Dict[int, np.ndarray], k: int = 5,
                seed: int = 42) -> Dict:
    """Combine detection + spectral shift for every layer we collected.

    Returns:
      per_layer[layer] = {detection_accuracy, max_abs_d, cohens_d,
                          singular_values, proj_orig, proj_unlearned}
      best_layer                 — layer with the highest detection accuracy
      detection_accuracy         — that best layer's accuracy (headline number)
      max_spectral_shift         — largest |Cohen's d| over all layers/directions
    """
    per_layer = {}
    for l in orig_feats:
        o, u = orig_feats[l], unlearned_feats.get(l, np.empty((0, 0)))
        acc = detection_accuracy(o, u, seed=seed)
        shift = spectral_shift(o, u, k=k)
        per_layer[l] = {"detection_accuracy": acc, **shift}
        logger.info("  layer %2d: detect_acc=%.3f  max|d|=%.3f", l, acc,
                    shift["max_abs_d"])
    best = max(per_layer, key=lambda l: per_layer[l]["detection_accuracy"])
    return {
        "per_layer": {str(l): v for l, v in per_layer.items()},
        "best_layer": int(best),
        "detection_accuracy": per_layer[best]["detection_accuracy"],
        "max_spectral_shift": max((v["max_abs_d"] for v in per_layer.values()),
                                  default=0.0),
    }
