# Changelog

## Unreleased

- **One function per kind of matching** (before any release): `match_nn`
  (nearest neighbor, on a score or a multivariate distance), `match_kernel`
  (kernel and ridge) and `match_cem` replace `match_stream(method=)`. Each
  takes only its own options and returns its own result type (`MatchSet`,
  `KernelSet`, `CEMSet`). Renamed along the way:
  - `estimand=` at match time is `focal="treated" | "control"`; `effect()`
    keeps `estimand=`.
  - A score column already in the data is `score=`, not `distance=<column>`;
    `distance=` is now `"ps"`, `"logit_ps"`, `"mahalanobis"`, `"ivariance"` or
    `"euclidean"`. `link=` is `score_link=`.
  - One `caliper=`: a number or a width per column, each in the default units
    (the score in standard deviations, a covariate raw) or `(width, "sd")` /
    `(width, "raw")`. `caliper_in_sd=` is gone.
  - Kernel matching: `ridge=True` or a number replaces `method="ridge"`.
    `bandwidth=` takes a number, `"pair_distance"` (the default, kmatch's
    `pm`), `"cross_validation"`, or the objects `PairDistance(factor,
    quantile)` and `CrossValidation(outcome, grid, weighted, penalty)`, which
    replace `bandwidth_quantile`, `bandwidth_factor` and the `cv_*` options.
    The penalty `"kmatch"` is `"support_loss"`.
- `balance()` works with a covariate that is constant in either arm, or in
  both: it raised `ZeroDivisionError`. A term constant in both arms has a
  standardized difference of 0, or +-inf where the means differ, and a
  variance ratio of NaN, or inf where only the controls' variance is 0 (the
  zero-variance handling of `hdfe_stream.core.balance`, synced).
- Option names spelled out (before any release): `k` is `n_matches`, `m_order`
  `match_order`, `mahvars` `mahalanobis_vars`, `ties` `keep_ties`, `rounds`
  `match_in_rounds`, `reestimate` `reestimate_score`, `pscore` `score_model`;
  in `effect()`,
  `vce` is `vcov` (hdfe-stream's name), `bias_adj` `bias_adjustment`,
  `var_calc` `n_variance_matches`, `var_fallback` `lone_unit_variance`,
  `ps_correction` `score_correction`. `MatchResult.vce` is gone:
  `vcov_type` (hdfe-stream's) has the same label. docs/implementations.md
  lists each name's counterpart in MatchIt, kmatch, R Matching and teffects.
- `effect(estimand=[...], potential_outcomes=True)`: kmatch's `ate att atc
  nate po` in one call, with the joint influence-function vcov (also with
  several outcomes). `"nate"` is the naive unweighted difference of means.
  Abadie-Imbens SEs are for one matched estimand at a time. Checked against
  kmatch's `e(b)` and `e(V)` (tests/reference/stata/kmatch_estimands.do).
- The influence-function variance of the ATE (matched both ways) is now
  kmatch's: that of p·ATT + (1−p)·ATC with the treated share p estimated.
  It was the stacked regression's, 2–3% larger on the reference data.
- `effect()` with Abadie-Imbens standard errors keeps each unit's outcome and
  conditional variance in frames, not Python dicts: on 5 million units its
  peak memory fell from 2.76 to 1.95 GB. Results are unchanged.
- The score correction (`effect(n_variance_matches=...)`, Abadie-Imbens 2016)
  is about 3 times faster and holds less: on 5 million units in 20 strata,
  28 s and 3.8 GB, now 9.5 s and 3.1 GB, with the same standard errors. Its
  nearest-neighbor sets on the score sort the units once instead of for each
  of their eight searches, and are used in that order instead of being sorted
  back; the searches (also matching's with replacement) run in parallel over
  the units rather than the strata, which left all but one thread idle without
  exact strata; the covariate matches build a k-d tree per stratum, in
  parallel; and the score model's covariates are read from the source before
  being joined to the units, which Polars did with 1.7 GB less. The sets'
  means and covariance terms are numba loops over the sets instead of
  NumPy's repeat-and-bincount, with the same sums in the same order.
- **Memory tests.** `pytest -m resource` runs nearest-neighbor matching (with
  `effect()` and `balance()`) and kernel matching on 2.5 million units with
  49 columns (0.98 GB in memory), each in a child process under a memory
  limit (stream-limits' watchdog) that matching on a few narrow columns stays
  under and holding the data would exceed. The default `pytest` run leaves
  them out. The `test` extra includes `stream-limits[testing]`.
- R reference fixtures (MatchIt 4.8.2, Matching 4.10-15).
- `MatchSet.effect()`: ATT, ATC and ATE with Abadie-Imbens (2006) standard
  errors (`n_variance_matches=`), and the Abadie-Imbens (2011) bias adjustment
  (`bias_adjustment=True` or a list of columns), both checked against R Matching.
- `effect(vcov=)`: influence-function standard errors with the matching
  weights held fixed, by unit or clustered (one-way or multi-way, on the
  subclass or source columns), for NN with or without replacement and CEM
  (`CEMSet.effect`). Without replacement the default clusters on the pair.
- Kernel and ridge matching on the score (`match_kernel`): five kernels,
  Frolich's ridge weights, a bandwidth from the 1:1 nearest-neighbor distances
  by default, leave-one-out cross-validation, without
  forming pairs. `KernelSet` has `weights()`, `balance()`, `effect()`.
- Internal, bound for hdfe_stream.core: IPW weights and the Hajek IPW
  estimate with a variance that accounts for the estimated logit or probit
  score; the generic first-stage correction; a generic weighted balance
  table (the matching balance table now uses it).
- `min_matches=`: drop focal units left with fewer matches than this
  (kmatch's rule with `min_matches=k`); the default keeps them, as MatchIt.
- `match_in_rounds=False`: without replacement and k > 1, match in one pass instead
  of MatchIt's rounds (with `match_order="closest"`, kmatch's `wor`).
- Fixed: with no caliper, nearest-neighbor matching on a score could take
  a position past a stratum's controls when they ran out: without
  replacement when a stratum had fewer free controls than focal units
  needed (memory corruption: crashes or hangs), with replacement when k
  exceeded a stratum's controls (a control from the next stratum). Focal
  units now get only what their stratum has.
- Documentation: runnable examples (`examples/`), a guide (how it works,
  conventions compared with MatchIt, Matching, kmatch, teffects and
  psmatch2, resources) and a function reference.
- `verbose=` (passed to the score model); `KernelSet` and `BalanceTable`
  are exported.
- Cross-validated bandwidths: the default search extends past its grid
  while the minimum is at an edge and refines between grid steps.
- Checked against Stata's teffects (nnmatch, psmatch's matches, ipw) and
  kmatch (nearest neighbor, kernel, ridge, cross-validation).
- `CrossValidation(penalty="support_loss")`: kmatch's cross-validation (its
  support-loss penalty, limit factor and search), reproducing its bandwidths
  exactly.
- The Abadie-Imbens (2016) correction for an estimated propensity score
  (`effect(score_correction=)`, on by default when it applies), reproducing
  Stata's teffects psmatch. The score model is fitted to a tighter
  tolerance (1e-10).
- The estimated-score corrections (AI 2016 for matching; the IPW core)
  support score models with absorbed fixed effects, by fixed-effect
  regressions out of core. The AI (2016) covariate match uses a k-d tree.
- Fixed: the AI (2016) ATC with a probit score used the flipped
  treatment's information matrix.
- The AI (2016) correction with calipers that leave focal units unmatched
  (the correction then concerns the matched units).
- Mahalanobis (and other metric) matching with replacement and no covariate
  calipers uses a k-d tree: the same matches, in O(N log N).
- Metric matching without replacement uses k-d tree candidate lists (the
  same matches, much faster), and offers `match_order="closest"`.
- Metric matching with replacement and covariate calipers also uses k-d tree
  candidate lists (the same matches; never slower than the brute force).
- Common support: `discard=` ("both", "control", "treated" or a column) and
  `reestimate_score=`, as MatchIt; the balance table's sample sizes now have
  MatchIt's six rows (All (ESS), All, Matched (ESS), Matched, Unmatched,
  Discarded).
- `mahalanobis_vars=`: Mahalanobis matching on chosen covariates within the score's
  caliper, as MatchIt.
- `n_variance_matches > 0` (heteroskedastic Abadie-Imbens standard errors) when
  matching on covariates (Mahalanobis, inverse variance, Euclidean, and
  `mahalanobis_vars`), as R Matching.
- `profile=True`: the memory and disk a match would need (hdfe-stream's
  `Profile`, with a PBS request), the strata and the candidate pairs,
  without matching. `n_threads=` limits the threads of a match and its
  effects.
- psmatch2 references (`tests/reference/stata/psmatch2.do`) and a port of
  its search (`tests/psmatch2_port.py`): with replacement it makes
  match-stream's matches; without replacement it is a forward-only sweep,
  not nearest-available matching (documented, not offered).
- Benchmarks (`benchmarks/`): seven designs and the standard errors against
  MatchIt and Matching (R), pymatchit, fastmatch and Polars' `join_asof`.
- `effect(vcov="ai", n_variance_matches>=1)` warns when the standard error is NaN
  because units the variance needs are alone in their arm within their
  exact stratum.
- `MatchResult` follows hdfe-stream's conventions: `n_clusters` maps the
  clustering (`"subclass+id"`) to a tuple of counts (was one count per
  dimension), so `to_pyfixest()` and `etable` work for clustered results;
  in the pyfixest view (and `etable`) the estimate's row is the estimand
  ("ATT"), the outcome heading the column. Abadie-Imbens results' `n_obs`
  is the units the estimate uses (was every unit available).
- `effect(lone_unit_variance=)`: with `n_variance_matches >= 1`, the conditional variance
  of a unit alone in its arm within its exact stratum, opt-in: `"nearest"`
  (its nearest same-arm units in any stratum) or `"pooled"` (its arm's mean
  variance); also for the AI (2016) correction's neighbor sets. The default
  stays a NaN standard error with a warning, which now counts the pairs
  such units are the match in. `MatchResult.n_lone` and `.lone_unit_variance`
  record it.
- Cross-validated bandwidths without an outcome (`CrossValidation()`):
  kmatch's balance criterion on the score, with its limit factor and search
  under `penalty="support_loss"`.
- Stata round 3 read: psmatch2 agrees on every case; teffects refuses to
  estimate when the robust variance needs a unit alone in its exact cell.
- `effect(["y1", "y2", ...])`: several outcomes from one read of the source,
  one estimate each, with their joint vcov for every kind of standard error
  (kmatch's stacked influence-function covariance; for Abadie-Imbens the
  bilinear analog, checked against R Matching through
  Cov = [V(y1 + y2) - V(y1) - V(y2)] / 2).
