# Matches only

The case match-stream was first built for: greedy 1:1 matching without
replacement, exact strata, calipers on the score and on covariates, and no
outcome. The result is the pairs, written to Parquet. The score model
absorbs a fixed effect (`| i`), so it is fitted out of core by
`feglm_stream`. `|` is never used for exact strata: those go in `exact=`.

```{literalinclude} ../../examples/matches_only.py
:language: python
:start-at: import
```

Output:

```text
match-stream: nn, 1:1 without replacement, estimand ATT, order largest
  distance: ps (score model: d ~ x1 + x2 + x3 | i, logit)
  exact: r (5 strata)
  calipers: ps 0.2 SD, x1 0.25, x2 0.25
               available     matched   unmatched   discarded
  treated         25,497      24,610         887           0
  control         74,503      24,610      49,893           0
{'treated': 25497, 'control': 74503, 'matched_treated': 24610, 'matched_control': 24610, 'unmatched_treated': 887, 'unmatched_control': 49893, 'discarded_treated': 0, 'discarded_control': 0, 'pairs': 24610, 'dropped_missing': 0}

24,610 pairs written
┌─────────────┬─────────────┬────────────┬────────────┬──────┬──────────┬──────────┐
│ treated_row ┆ control_row ┆ treated_id ┆ control_id ┆ rank ┆ distance ┆ subclass │
│ ---         ┆ ---         ┆ ---        ┆ ---        ┆ ---  ┆ ---      ┆ ---      │
│ i64         ┆ i64         ┆ i64        ┆ i64        ┆ i32  ┆ f64      ┆ i64      │
╞═════════════╪═════════════╪════════════╪════════════╪══════╪══════════╪══════════╡
│ 10          ┆ 49531       ┆ 11         ┆ 49532      ┆ 1    ┆ 0.001206 ┆ 1        │
│ 15          ┆ 18737       ┆ 16         ┆ 18738      ┆ 1    ┆ 0.000139 ┆ 2        │
│ 19          ┆ 86024       ┆ 20         ┆ 86025      ┆ 1    ┆ 0.000214 ┆ 3        │
└─────────────┴─────────────┴────────────┴────────────┴──────┴──────────┴──────────┘
```

**Notes.**

- **Calipers.** The score caliper is in standard deviations of the score
  over all units; covariate calipers are in raw units. Write
  `(width, "sd")` or `(width, "raw")` to change a caliper's units.
- **Order.** Treated units are matched in `match_order`. The default is
  "largest" for a score: the highest-scoring units choose first.
- **Covariate calipers.** A control that fails one is skipped and the next
  nearest tried; the treated unit is not dropped.
- **Output.** `matches()` is a LazyFrame, so `sink_parquet` streams it.
