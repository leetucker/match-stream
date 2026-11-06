# match-stream

Matching on data larger than memory. It does nearest-neighbor matching
with or without replacement, calipers on the score and on covariates, exact
strata, and Mahalanobis, kernel, ridge and coarsened exact matching. It also
gives balance diagnostics, and treatment effects with Abadie–Imbens or
clustered standard errors.

It reads Parquet in a stream, holds only a few narrow columns while it
matches, and fits the propensity score out of core, with absorbed fixed
effects if you want them. Its results are tested against R's MatchIt and
Matching and Stata's teffects and kmatch (see below).

```python
from match_stream import match_nn

m = match_nn("treat ~ age + tenure + log_wage0", "extract/*.parquet",
             exact=["state", "naics2"], replace=True, caliper=0.2, id="person_id")
m.balance().summary()
m.effect("earnings").summary()          # the ATT, Abadie-Imbens standard errors
m.matches().sink_parquet("pairs.parquet")
```

**Status: in development** (version 0.1.0.dev0, not yet on PyPI). The
designs, the inference and the documentation are complete and tested. The
benchmarks are written and wait for a full run (see Performance).

## What it does

Three functions: `match_nn` (nearest neighbor, on a score or a multivariate
distance), `match_kernel` (kernel and ridge) and `match_cem` (coarsened exact
matching). Each returns the matches and weights; `effect()` and `balance()`
are methods of the result.

**Designs**
- k:1 nearest neighbor on a propensity score, with or without replacement.
  Without replacement it is greedy in any of five orders (`largest`,
  `smallest`, `closest`, `random`, `data`).
- Calipers on the score (in standard deviations) and on covariates, and
  exact strata.
- Mahalanobis, inverse-variance or Euclidean distance on covariates, alone
  or within a score caliper (`mahalanobis_vars=`).
- Kernel and ridge matching on the score: five kernels, a
  bandwidth rule or cross-validation.
- Coarsened exact matching.
- Common support (`discard=`, `reestimate_score=`), ties, a minimum number of
  matches per unit, and one-pass k:1 matching without replacement.

**The score**
- Logit or probit, fitted by hdfe-stream, with absorbed fixed effects
  (`score_model="treat ~ x | firm"`), or a column you supply.

**Effects and inference**
- ATT, ATC and ATE.
- Abadie–Imbens (2006) standard errors, with or without heteroskedastic
  conditional variances (`n_variance_matches`).
- The Abadie–Imbens (2011) bias adjustment.
- The Abadie–Imbens (2016) correction for an estimated score, also with
  absorbed fixed effects.
- Influence-function standard errors with the matching weights fixed, by unit or clustered one or several ways.
- Several outcomes in one pass (`effect(["y1", "y2"])`), with their joint
  covariance matrix, for every kind of standard error.

**Diagnostics**
- A balance table (means, standardized differences, variance ratios, eCDF
  gaps) and sample sizes.

**Resources**
- `profile=True` reports the memory, disk and candidate pairs a match would
  need, without matching, and a resource request for a scheduler.
- `n_threads=` and `max_memory_gb=` respect a job's limits (stream-limits).
- Large inputs are matched a batch of exact strata at a time
  (`memory="by_stratum"`), with the same matches.

**Interface**
- The matches, weights and subclasses as Polars LazyFrames, ready to write to
  Parquet.
- Results with hdfe-stream's reporting: `summary()`, `tidy()`, `confint()`,
  `etable`, `summary_json()`.

## Why it is fast

- **Windows, not pairs.** Within exact strata and sorted by the score, each
  unit's candidates are a window found by binary search. Nothing forms all
  the pairs.
- **Cheap skipping.** Matching without replacement skips taken controls
  with path-compressed pointers.
- **Trees for covariates.** Mahalanobis matching uses k-d trees, and
  candidate lists for matching without replacement and for calipers.
- **Window sums.** Kernel and ridge weights come from running sums, with no
  pairs formed.
- **Narrow columns.** Only tens of bytes a unit are held. The score model is
  fitted out of core, and the numba kernels run strata in parallel.

## Validation against other implementations

| Implementation | Compared on | Agreement |
|---|---|---|
| MatchIt 4.8.2 | nearest neighbor (every order, k:1, calipers), Mahalanobis, `mahalanobis_vars`, common support, CEM, weights, balance | pairs identical, weights and balance to 1e-8 |
| R Matching 4.10-15 | Abadie–Imbens SEs (`Var.calc`), the bias adjustment | estimates to 1e-10, SEs to 1e-8 |
| Stata teffects | nnmatch, psmatch with the AI (2016) correction, ipw | to 1e-8 |
| Stata kmatch 1.1.6 | nearest neighbor, kernel, ridge, bandwidths, influence-function SEs | matches identical, estimates to 1e-12 |
| MatchIt + sandwich, fixest | clustered influence-function SEs | to 1e-13 |

The R and Stata scripts that make these references are in `tests/reference/`.
`docs/implementations.md` lists the conventions on which these packages
differ, and which option reproduces each.

## Performance

The benchmark suites (`benchmarks/`) compare match-stream with MatchIt and
Matching (R), pymatchit, fastmatch and Polars' `join_asof` and `group_by`,
and with bare k-d tree and faiss searches as a floor. They cover
eight designs and the standard errors, from 10,000 to 100 million units.
On a 16-vCPU, 32 GB cloud machine, at 10 million units:

- match-stream is 6 to 35 times faster than MatchIt where MatchIt finishes,
  with the same matches (estimates agree to about 1e-14). At 100 million
  units MatchIt needs 28 to 30 GB where match-stream needs 8 to 21 GB, and
  in two designs it runs out of memory.
- With no strata, 1:1 Mahalanobis matching takes 6.1 s, as long as a bare
  SciPy k-d tree search of the same data (6.7 s). MatchIt exceeds 30 minutes.
- pymatchit and R's Matching are slower still, and mostly do not finish
  at 10 million units or more.
- Polars' `group_by` (coarsened exact matching) and `join_asof` (1:1 on a
  score) are faster, as they do less.
- At 100 million units, 1:1 Mahalanobis matching without replacement and
  with calipers needs `memory="by_stratum"` on this machine (187 s, 9.2 GB).

The results, figures and caveats are in the
[benchmarks](https://leetucker.github.io/match-stream/benchmarks/).

Small runs, for correctness only, give these results:

- MatchIt's estimates agree to about 1e-15 on every design.
- At 100,000 units, R Matching's Abadie–Imbens standard error (with
  `Var.calc = 1`) took 510 s; match-stream's, the same to 1e-15, took 0.2 s.

## Install

match-stream is not on PyPI yet. Install it from the repository:

```bash
pip install git+https://github.com/leetucker/match-stream
```

Python 3.10 or later. It depends on hdfe-stream (with its formula extra),
stream-limits, Polars, NumPy, numba, SciPy and pyarrow.

## Documentation

Build the documentation locally with:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The guide covers:

- how it works;
- the conventions that differ from other matching packages;
- the resources a match needs;
- the API reference.

`examples/` has runnable scripts, and `benchmarks/` the performance
comparisons (see `benchmarks/README.md`).

A sibling of [hdfe-stream](https://github.com/leetucker/hdfe-stream),
[logit-stream](https://github.com/leetucker/logit-stream) and
[dpd-stream](https://github.com/leetucker/dpd-stream).

## License

MIT.
