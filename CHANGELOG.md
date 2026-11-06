# Changelog

## 0.1.0 (2026-11-06)

First release: matching for data too large for memory, with treatment effects
and the standard errors that go with them.

**Three functions, one for each kind of matching.** Each takes only its own
options and returns its own result, with `matches()`, `weights()`, `counts()`,
`balance()` and `effect()`.

- **`match_nn`: nearest-neighbor matching**, on a propensity score or on the
  covariates themselves.
  - The score is fitted for you (logit or probit, with high-dimensional fixed
    effects absorbed, out of core by hdfe-stream), or it is a column you
    supply. Or match on Mahalanobis, inverse-variance or Euclidean distance.
  - One match or k per treated unit, with or without replacement, ties kept
    or broken, and in any of five orders without replacement.
  - Calipers on the score and on covariates, exact strata, common-support
    trimming, and Mahalanobis distance within a score caliper.
  - The focal units are the treated (the ATT) or the controls (the ATC);
    `effect()` can also match both ways for the ATE.
- **`match_kernel`: kernel and ridge matching** on the score. Five kernels,
  ridge weights, a default bandwidth from the 1:1 nearest-neighbor distances
  (`PairDistance`) and cross-validated bandwidths (`CrossValidation`), without
  forming pairs.
- **`match_cem`: coarsened exact matching**, with cells, weights and balance.

**Treatment effects with the standard errors the literature uses.**
`effect()` gives the ATT, ATC and ATE, for one outcome or several in one pass.

- Abadie-Imbens standard errors for matching with replacement, with the bias
  adjustment, with heteroskedastic variances (`n_variance_matches=`), and with
  the correction for an estimated propensity score (Abadie and Imbens 2016).
- Influence-function standard errors, by unit or clustered on the pair, the
  matched unit or your own columns, as Abadie and Spiess (2022).
- Results follow hdfe-stream's conventions, so they print, tabulate and export
  to pyfixest the same way.

**Balance diagnostics.** A table of standardized differences, variance ratios
and empirical-CDF distances before and after matching, with the sample sizes
by arm.

**Built for large data.**

- Only a few narrow columns are held; the rest of the data stays on disk.
  Large data are matched a batch of exact strata at a time
  (`memory="by_stratum"`), with memory bounded by the largest stratum.
- Matching with replacement and by covariates uses k-d trees, and searches run
  in parallel. On a 16-core machine, 1:1 matching of 100 million units takes
  about one to two minutes (see the benchmarks).
- `profile=True` estimates the memory and disk a match would need, before
  running it. Threads and memory follow your job's limits (PBS, Slurm,
  containers).

**Checked against other software.** On shared reference data the matches and
weights are the same, unit for unit, as R's MatchIt; the effects and standard
errors agree with R's Matching and Stata's teffects and kmatch; and the matches
agree with psmatch2's. `docs/implementations.md` lists where packages differ
and which option reproduces each.

**Documentation and benchmarks.**

- A guide to choosing a matching method, runnable examples, a page on how it
  works, a function reference and the conventions compared with other
  packages.
- Benchmarks against MatchIt and R Matching, pymatchit, fastmatch, bare
  nearest-neighbor searches and Polars, from 10,000 to 100 million units.
- Memory tests (`pytest -m resource`) check that matching a few narrow columns
  stays under a limit that holding the data would exceed.

Requires hdfe-stream 0.9.0 or later.
