# Quickstart

1:1 nearest-neighbor matching on a propensity score (a logit of the treatment
on the covariates, fitted by hdfe-stream's `feglm_stream`), with replacement, within the 20 exact strata of `r` x `i`,
with a caliper of 0.2 standard deviations of the score.
Then the balance table and the ATT. The simulated effect is 1 for everyone.

```{literalinclude} ../../examples/quickstart.py
:language: python
:start-at: import
```

Output:

```text
match-stream: nn, 1:1 with replacement, estimand ATT, order largest
  distance: ps (score model: d ~ x1 + x2 + x3, logit)
  exact: r, i (20 strata)
  calipers: ps 0.2 SD
               available     matched   unmatched   discarded
  treated         25,497      25,454          43           0
  control         74,503      18,907      55,596           0

The first pairs:
┌─────────────┬─────────────┬────────────┬────────────┬──────┬──────────┬──────────┐
│ treated_row ┆ control_row ┆ treated_id ┆ control_id ┆ rank ┆ distance ┆ subclass │
│ ---         ┆ ---         ┆ ---        ┆ ---        ┆ ---  ┆ ---      ┆ ---      │
│ i64         ┆ i64         ┆ i64        ┆ i64        ┆ i32  ┆ f64      ┆ i64      │
╞═════════════╪═════════════╪════════════╪════════════╪══════╪══════════╪══════════╡
│ 10          ┆ 30914       ┆ 11         ┆ 30915      ┆ 1    ┆ 0.000008 ┆ 1        │
│ 15          ┆ 49745       ┆ 16         ┆ 49746      ┆ 1    ┆ 0.000055 ┆ 2        │
│ 19          ┆ 87575       ┆ 20         ┆ 87576      ┆ 1    ┆ 0.000025 ┆ 3        │
│ 24          ┆ 55962       ┆ 25         ┆ 55963      ┆ 1    ┆ 0.000135 ┆ 4        │
└─────────────┴─────────────┴────────────┴────────────┴──────┴──────────┴──────────┘

The weights:
┌─────┬─────┬─────────┬──────────┬──────────┐
│ row ┆ id  ┆ treated ┆ weight   ┆ subclass │
│ --- ┆ --- ┆ ---     ┆ ---      ┆ ---      │
│ i64 ┆ i64 ┆ bool    ┆ f64      ┆ i64      │
╞═════╪═════╪═════════╪══════════╪══════════╡
│ 0   ┆ 1   ┆ false   ┆ 1.485582 ┆ 1118     │
│ 3   ┆ 4   ┆ false   ┆ 0.742791 ┆ 23714    │
│ 4   ┆ 5   ┆ false   ┆ 0.742791 ┆ 20793    │
│ 6   ┆ 7   ┆ false   ┆ 0.742791 ┆ 21128    │
└─────┴─────┴─────────┴──────────┴──────────┘

Balance before matching
┌──────────┬───────────────┬───────────────┬─────────────────┬────────────┬───────────┬──────────┐
│ term     ┆ Means Treated ┆ Means Control ┆ Std. Mean Diff. ┆ Var. Ratio ┆ eCDF Mean ┆ eCDF Max │
╞══════════╪═══════════════╪═══════════════╪═════════════════╪════════════╪═══════════╪══════════╡
│ distance ┆ 0.3027        ┆ 0.2386        ┆ 0.5452          ┆ 1.3085     ┆ 0.1612    ┆ 0.2347   │
│ x1       ┆ 0.3396        ┆ -0.1194       ┆ 0.4676          ┆ 1.0123     ┆ 0.1301    ┆ 0.1863   │
│ x2       ┆ -0.2115       ┆ 0.0704        ┆ -0.2853         ┆ 0.9946     ┆ 0.0796    ┆ 0.1160   │
│ x3       ┆ 0.1384        ┆ -0.0417       ┆ 0.1801          ┆ 1.0056     ┆ 0.0507    ┆ 0.0741   │
└──────────┴───────────────┴───────────────┴─────────────────┴────────────┴───────────┴──────────┘

Balance after matching
┌──────────┬─────────┬─────────┬───────────┬────────────┬───────────┬──────────┬───────────────────┐
│ term     ┆ Means   ┆ Means   ┆ Std. Mean ┆ Var. Ratio ┆ eCDF Mean ┆ eCDF Max ┆ Std. Pair Dist.   │
│          ┆ Treated ┆ Control ┆ Diff.     ┆            ┆           ┆          ┆                   │
╞══════════╪═════════╪═════════╪═══════════╪════════════╪═══════════╪══════════╪═══════════════════╡
│ distance ┆ 0.3020  ┆ 0.3020  ┆ 0.0001    ┆ 1.0008     ┆ 0.0001    ┆ 0.0009   ┆ 0.0017            │
│ x1       ┆ 0.3351  ┆ 0.3373  ┆ -0.0022   ┆ 1.0145     ┆ 0.0017    ┆ 0.0042   ┆ 0.6745            │
│ x2       ┆ -0.2092 ┆ -0.2024 ┆ -0.0069   ┆ 1.0063     ┆ 0.0035    ┆ 0.0102   ┆ 0.9852            │
│ x3       ┆ 0.1373  ┆ 0.1420  ┆ -0.0047   ┆ 1.0234     ┆ 0.0027    ┆ 0.0077   ┆ 1.0737            │
└──────────┴─────────┴─────────┴───────────┴────────────┴───────────┴──────────┴───────────────────┘

Sample sizes
┌───────────────┬────────────┬────────────┐
│ sample        ┆ Control    ┆ Treated    │
╞═══════════════╪════════════╪════════════╡
│ All (ESS)     ┆ 74503.0000 ┆ 25497.0000 │
│ All           ┆ 74503.0000 ┆ 25497.0000 │
│ Matched (ESS) ┆ 14660.4995 ┆ 25454.0000 │
│ Matched       ┆ 18907.0000 ┆ 25454.0000 │
│ Unmatched     ┆ 55596.0000 ┆ 43.0000    │
│ Discarded     ┆ 0.0000     ┆ 0.0000     │
└───────────────┴────────────┴────────────┘

### y: nearest-neighbor matching on ps
Estimand: ATT   Matched focal units: 25,454
Standard errors: Abadie-Imbens (homoskedastic)
┌─────────────┬──────────┬────────────┬──────────┬──────────┐
│ Coefficient ┆ Estimate ┆ Std. Error ┆ t value  ┆ Pr(>|t|) │
│ ---         ┆ ---      ┆ ---        ┆ ---      ┆ ---      │
│ str         ┆ f64      ┆ f64        ┆ f64      ┆ f64      │
╞═════════════╪══════════╪════════════╪══════════╪══════════╡
│ ATT:y       ┆ 0.987949 ┆ 0.014293   ┆ 69.12017 ┆ 0.0      │
└─────────────┴──────────┴────────────┴──────────┴──────────┘
```

**Reading the output.**

- **The summary** counts units by arm. 43 treated units have no control
  within the caliper in their stratum and are left unmatched.
- **`matches()`** has one row per pair: the rows in the source (from 0), the
  ids, the rank of the match, the score gap and a subclass per treated unit.
- **`weights()`** gives the matching weights: treated 1, a control the sum of
  1/k over the treated units it serves, rescaled to sum to the number of
  matched controls.
- **The balance table** compares the groups before and after matching:
  means, standardized mean differences, variance ratios and eCDF gaps.
- **The effect** comes with Abadie-Imbens (2006) standard errors, the
  default for matching with replacement.
