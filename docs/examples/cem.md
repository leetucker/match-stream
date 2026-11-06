# Coarsened exact matching

Each covariate is cut into five equal-width bins, and units match exactly
on the bins and on `r` and `i`. A cell with both treated and control units
is matched, and weighted so the matched controls sum to the matched treated.

```{literalinclude} ../../examples/cem.py
:language: python
:start-at: import
```

Output:

```text
match-stream: CEM, estimand ATT, 844 matched cells
  exact: r, i
               available     matched   unmatched
  treated         25,497      25,320         177
  control         74,503      73,405       1,098
┌─────┬─────┬─────────┬──────────┬──────────┐
│ row ┆ id  ┆ treated ┆ weight   ┆ subclass │
│ --- ┆ --- ┆ ---     ┆ ---      ┆ ---      │
│ i64 ┆ i64 ┆ bool    ┆ f64      ┆ i64      │
╞═════╪═════╪═════════╪══════════╪══════════╡
│ 0   ┆ 1   ┆ false   ┆ 0.325485 ┆ 343      │
│ 1   ┆ 2   ┆ false   ┆ 1.098291 ┆ 352      │
│ 2   ┆ 3   ┆ false   ┆ 0.667251 ┆ 642      │
│ 3   ┆ 4   ┆ false   ┆ 0.426998 ┆ 193      │
└─────┴─────┴─────────┴──────────┴──────────┘

Balance before matching
┌──────┬───────────────┬───────────────┬─────────────────┬────────────┬───────────┬──────────┐
│ term ┆ Means Treated ┆ Means Control ┆ Std. Mean Diff. ┆ Var. Ratio ┆ eCDF Mean ┆ eCDF Max │
╞══════╪═══════════════╪═══════════════╪═════════════════╪════════════╪═══════════╪══════════╡
│ x1   ┆ 0.3396        ┆ -0.1194       ┆ 0.4676          ┆ 1.0123     ┆ 0.1301    ┆ 0.1863   │
│ x2   ┆ -0.2115       ┆ 0.0704        ┆ -0.2853         ┆ 0.9946     ┆ 0.0796    ┆ 0.1160   │
│ x3   ┆ 0.1384        ┆ -0.0417       ┆ 0.1801          ┆ 1.0056     ┆ 0.0507    ┆ 0.0741   │
└──────┴───────────────┴───────────────┴─────────────────┴────────────┴───────────┴──────────┘

Balance after matching
┌──────┬─────────┬─────────┬─────────────────┬────────────┬───────────┬──────────┬─────────────────┐
│ term ┆ Means   ┆ Means   ┆ Std. Mean Diff. ┆ Var. Ratio ┆ eCDF Mean ┆ eCDF Max ┆ Std. Pair Dist. │
│      ┆ Treated ┆ Control ┆                 ┆            ┆           ┆          ┆                 │
╞══════╪═════════╪═════════╪═════════════════╪════════════╪═══════════╪══════════╪═════════════════╡
│ x1   ┆ 0.3299  ┆ 0.2278  ┆ 0.1041          ┆ 1.0005     ┆ 0.0307    ┆ 0.0654   ┆ 0.5595          │
│ x2   ┆ -0.2085 ┆ -0.1481 ┆ -0.0611         ┆ 0.9961     ┆ 0.0174    ┆ 0.0383   ┆ 0.5401          │
│ x3   ┆ 0.1363  ┆ 0.0942  ┆ 0.0421          ┆ 0.9982     ┆ 0.0122    ┆ 0.0265   ┆ 0.5231          │
└──────┴─────────┴─────────┴─────────────────┴────────────┴───────────┴──────────┴─────────────────┘

Sample sizes
┌───────────────┬────────────┬────────────┐
│ sample        ┆ Control    ┆ Treated    │
╞═══════════════╪════════════╪════════════╡
│ All (ESS)     ┆ 74503.0000 ┆ 25497.0000 │
│ All           ┆ 74503.0000 ┆ 25497.0000 │
│ Matched (ESS) ┆ 51751.0249 ┆ 25320.0000 │
│ Matched       ┆ 73405.0000 ┆ 25320.0000 │
│ Unmatched     ┆ 1098.0000  ┆ 177.0000   │
│ Discarded     ┆ 0.0000     ┆ 0.0000     │
└───────────────┴────────────┴────────────┘

### y: coarsened exact matching
Estimand: ATT   Matched focal units: 25,320
Standard errors: influence function, weights fixed, clustered by unit (98,725)
┌─────────────┬──────────┬────────────┬───────────┬──────────┐
│ Coefficient ┆ Estimate ┆ Std. Error ┆ t value   ┆ Pr(>|t|) │
│ ---         ┆ ---      ┆ ---        ┆ ---       ┆ ---      │
│ str         ┆ f64      ┆ f64        ┆ f64       ┆ f64      │
╞═════════════╪══════════╪════════════╪═══════════╪══════════╡
│ ATT:y       ┆ 1.078641 ┆ 0.011824   ┆ 91.223702 ┆ 0.0      │
└─────────────┴──────────┴────────────┴───────────┴──────────┘
```

**Notes.**

- **Residual bias.** The estimate is 0.08 above the true effect of 1.
  Five bins leave imbalance within the cells: the standardized mean
  difference in `x1` is still 0.10 after matching. Finer cutpoints remove
  more of it and match fewer units. The balance table is the place to
  check.
- **Cutpoints are required**: an integer (equal-width bins) or the cut
  points themselves; there is no default rule.
- **Standard errors.** By default `effect()` uses the influence-function
  SE by unit; clustering on the cell is not advised, since the weights are
  functions of the cells.
