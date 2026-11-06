# Mahalanobis matching

Matching on the covariates themselves. The Mahalanobis distance uses
the covariance pooled within the treatment groups, within exact
strata, with a caliper on `x1`.

```{literalinclude} ../../examples/mahalanobis.py
:language: python
:start-at: import
```

Output:

```text
match-stream: nn, 1:1 with replacement, estimand ATT, order data
  distance: mahalanobis
  exact: r, i (20 strata)
  calipers: x1 0.2
               available     matched   unmatched   discarded
  treated         25,497      25,453          44           0
  control         74,503      19,763      54,740           0

┌──────┬─────────────────┬────────────┬──────────┐
│ term ┆ Std. Mean Diff. ┆ Var. Ratio ┆ eCDF Max │
│ ---  ┆ ---             ┆ ---        ┆ ---      │
│ str  ┆ f64             ┆ f64        ┆ f64      │
╞══════╪═════════════════╪════════════╪══════════╡
│ x1   ┆ 0.00439         ┆ 1.012289   ┆ 0.00444  │
│ x2   ┆ -0.013802       ┆ 1.070189   ┆ 0.010372 │
│ x3   ┆ 0.008901        ┆ 1.064812   ┆ 0.00774  │
└──────┴─────────────────┴────────────┴──────────┘

### y: nearest-neighbor matching on mahalanobis
Estimand: ATT   Matched focal units: 25,453
Standard errors: Abadie-Imbens (homoskedastic)
┌─────────────┬──────────┬────────────┬───────────┬──────────┐
│ Coefficient ┆ Estimate ┆ Std. Error ┆ t value   ┆ Pr(>|t|) │
│ ---         ┆ ---      ┆ ---        ┆ ---       ┆ ---      │
│ str         ┆ f64      ┆ f64        ┆ f64       ┆ f64      │
╞═════════════╪══════════╪════════════╪═══════════╪══════════╡
│ ATT:y       ┆ 0.995777 ┆ 0.010182   ┆ 97.795807 ┆ 0.0      │
└─────────────┴──────────┴────────────┴───────────┴──────────┘
```

**Notes.**

- **Other metrics.** `distance="ivariance"` scales each covariate by its
  variance; `"euclidean"` leaves them raw.
- **Order.** The default for a multivariate distance is data order. Without replacement,
  "random" (with `seed=`) and "closest" are the others (see below).
- **Search.** A k-d tree finds the nearest controls, or candidate lists
  that calipers and matching without replacement walk through, fast at any
  stratum size up to 15 covariates. Beyond that, blocked brute force within
  each stratum, which is slow for very large strata.
- **Order without replacement.** "data" (the default), "random", or
  "closest" (pairs in ascending distance).
- **Standard errors.** Abadie-Imbens SEs with `n_variance_matches > 0` are not built
  yet for multivariate distances; the homoskedastic one is.
