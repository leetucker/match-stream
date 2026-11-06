# Kernel and ridge matching

Each treated unit is compared with a kernel-weighted average of the
controls of its stratum within one bandwidth of its propensity score. Ridge matching
replaces the average with Frölich's (2004) ridge-regularized local-linear
fit. The default bandwidth (`"pair_distance"`) is set from the 1:1
nearest-neighbor distances; `CrossValidation(outcome=)` chooses it by
cross-validating an outcome. Standard errors are the influence
function's, with the weights held fixed.

```{literalinclude} ../../examples/kernel_ridge.py
:language: python
:start-at: import
```

Output:

```text
match-stream: kernel matching, epan kernel, estimand ATT
  distance: ps   bandwidth: 0.000417348 (pair_distance)
  exact: r, i (20 strata)
               available     matched   unmatched
  treated         25,497      23,716       1,781
  control         74,503      63,809      10,694
### y: kernel matching on ps
Estimand: ATT   Matched focal units: 23,716
Standard errors: influence function, weights fixed, clustered by unit (87,525)
┌─────────────┬──────────┬────────────┬───────────┬──────────┐
│ Coefficient ┆ Estimate ┆ Std. Error ┆ t value   ┆ Pr(>|t|) │
│ ---         ┆ ---      ┆ ---        ┆ ---       ┆ ---      │
│ str         ┆ f64      ┆ f64        ┆ f64       ┆ f64      │
╞═════════════╪══════════╪════════════╪═══════════╪══════════╡
│ ATT:y       ┆ 0.990778 ┆ 0.012987   ┆ 76.290996 ┆ 0.0      │
└─────────────┴──────────┴────────────┴───────────┴──────────┘

ridge, bandwidth 0.00042:
### y: ridge matching on ps
Estimand: ATT   Matched focal units: 23,716
Standard errors: influence function, weights fixed, clustered by unit (87,525)
┌─────────────┬──────────┬────────────┬───────────┬──────────┐
│ Coefficient ┆ Estimate ┆ Std. Error ┆ t value   ┆ Pr(>|t|) │
│ ---         ┆ ---      ┆ ---        ┆ ---       ┆ ---      │
│ str         ┆ f64      ┆ f64        ┆ f64       ┆ f64      │
╞═════════════╪══════════╪════════════╪═══════════╪══════════╡
│ ATT:y       ┆ 0.99224  ┆ 0.013223   ┆ 75.040366 ┆ 0.0      │
└─────────────┴──────────┴────────────┴───────────┴──────────┘

cross-validation:
┌───────────┬───────────┬─────────┐
│ bandwidth ┆ criterion ┆ weight  │
│ ---       ┆ ---       ┆ ---     │
│ f64       ┆ f64       ┆ f64     │
╞═══════════╪═══════════╪═════════╡
│ 0.002     ┆ 2.010846  ┆ 74203.0 │
│ 0.005     ┆ 1.950153  ┆ 74203.0 │
│ 0.01      ┆ 1.929642  ┆ 74203.0 │
│ 0.02      ┆ 1.919829  ┆ 74203.0 │
│ 0.05      ┆ 1.914433  ┆ 74203.0 │
│ 0.1       ┆ 1.912883  ┆ 74203.0 │
└───────────┴───────────┴─────────┘
chosen: 0.1
### y: ridge matching on ps
Estimand: ATT   Matched focal units: 25,494
Standard errors: influence function, weights fixed, clustered by unit (99,997)
┌─────────────┬──────────┬────────────┬───────────┬──────────┐
│ Coefficient ┆ Estimate ┆ Std. Error ┆ t value   ┆ Pr(>|t|) │
│ ---         ┆ ---      ┆ ---        ┆ ---       ┆ ---      │
│ str         ┆ f64      ┆ f64        ┆ f64       ┆ f64      │
╞═════════════╪══════════╪════════════╪═══════════╪══════════╡
│ ATT:y       ┆ 1.011057 ┆ 0.012162   ┆ 83.131974 ┆ 0.0      │
└─────────────┴──────────┴────────────┴───────────┴──────────┘
```

**Notes.**

- **The default bandwidth** (`"pair_distance"`) is 1.5 times the 90th
  percentile of the nonzero 1:1 nearest-neighbor distances. It is narrow:
  here 1,781 treated units have no control within it and are unmatched.
- **Cross-validation** predicts each control's outcome from the other
  controls (leave one out), for each bandwidth on the grid. It picks the
  bandwidth with the smallest mean squared error over a fixed set of
  controls.
  - Here the outcome is smooth in the score, so the widest bandwidth wins
    and nearly every treated unit is matched (25,494 of 25,497).
  - match-stream warns when the minimum is at the edge of the grid: widen
    `CrossValidation(grid=)` then.
- **Computation.** Kernel and ridge weights come from window sums, so no
  pairs are formed (see [How it works](../how_it_works.md)).
