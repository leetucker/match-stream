# Out of core

Nearest-neighbor matching on a propensity score for a million units in four
Parquet files, never loaded at once. The score
model is fitted by `feglm_stream` out of core, the strata are matched a
batch at a time (`memory="by_stratum"`, memory bounded by the largest
stratum), and the pairs are streamed back to Parquet.

```{literalinclude} ../../examples/out_of_core.py
:language: python
:start-at: import
```

Output:

```text
memory path: by_stratum
match-stream: nn, 1:1 without replacement, estimand ATT, order largest
  distance: ps (score model: d ~ x1 + x2 + x3, logit)
  exact: r, i (20 strata)
  calipers: ps 0.2 SD
               available     matched   unmatched   discarded
  treated        253,312     252,034       1,278           0
  control        746,688     252,034     494,654           0
252,034 pairs
### y: nearest-neighbor matching on ps
Estimand: ATT   Matched focal units: 252,034
Standard errors: influence function, weights fixed, clustered by subclass (252,034)
┌─────────────┬──────────┬────────────┬────────────┬──────────┐
│ Coefficient ┆ Estimate ┆ Std. Error ┆ t value    ┆ Pr(>|t|) │
│ ---         ┆ ---      ┆ ---        ┆ ---        ┆ ---      │
│ str         ┆ f64      ┆ f64        ┆ f64        ┆ f64      │
╞═════════════╪══════════╪════════════╪════════════╪══════════╡
│ ATT:y       ┆ 1.009677 ┆ 0.003894   ┆ 259.263754 ┆ 0.0      │
└─────────────┴──────────┴────────────┴────────────┴──────────┘
```

**Notes.**

- **Memory paths.** `memory="auto"` (the default) picks the in-memory path
  when the narrow columns fit the memory headroom (stream-limits), else
  the by-stratum path. Both give identical matches.
- **The source is read again.** `effect()` and `balance()` re-read the
  source for the outcome and covariates. Keep it unchanged until then, and
  keep outputs out of its glob.
- **Scratch space.** `workdir=` is where the score model's files and the
  by-stratum copy go.
