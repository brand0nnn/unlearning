# TOFU reproduction: ours vs paper (Maini et al. 2024, Fig 6, Llama-2-7B, forget10)

The paper reports Model Utility / Forget Quality via plots, so 'Paper' is the
readable Fig-6 anchor + stated trend. The match is the *pattern*, not exact digits.

| Model | Model Utility (ours) | Model Utility (paper) | Forget Quality p (ours) | log10 p (ours) | Forget Quality (paper) |
|---|---|---|---|---|---|
| reference | 0.657 | ~0.55 (Retain star) | — | — | — (reference) |
| gradient_ascent | 0.000 | -> 0 (destroyed) | 9.07e-08 | -7.0 | < 0.05 (log p ~ -20) |
| gradient_difference | 0.629 | preserved (higher) | 6.53e-167 | -166.2 | < 0.05 |
| idk | 0.282 | preserved (partial) | 1.12e-19 | -19.0 | < 0.05 |
| kl_minimization | 0.661 | preserved (higher) | 8.12e-27 | -26.1 | < 0.05 |

**Reading:** the reference (retain) model has healthy utility (~0.66; paper ~0.55). `gradient_ascent` collapses utility to 0. `gradient_difference` / `kl_minimization` preserve utility (~0.63 / ~0.66) but forget quality is ~0 (they barely forget). No method reaches BOTH high utility and high forget quality — the TOFU trade-off (top-right of Fig 6 stays empty).
