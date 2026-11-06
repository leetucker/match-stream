# Conventions and other implementations

match-stream follows R's MatchIt where conventions differ. This page lists
every difference we know of from MatchIt, R's Matching, Stata's kmatch,
teffects and psmatch2, so that a difference in someone's output can be
traced. "Differs" is not "wrong": most are different conventions for the
same estimator. Several have an option here that reproduces the other
package.

## What is tested against what

| Implementation | Version | Compared on | Status |
|---|---|---|---|
| MatchIt | 4.8.2 | nearest neighbor on a score with and without replacement (every `m.order`, k:1, calipers on the score and covariates), Mahalanobis, a fitted logit score, CEM; weights; the balance table | pairs identical, weights and balance to 1e-8 or better |
| R Matching | 4.10-15 | Abadie-Imbens SEs (`Var.calc` 0, 2, 4; ATT, ATC, ATE; k = 1, 2, 4), the bias adjustment | estimates to 1e-10, SEs to 1e-8 |
| kmatch | 1.1.6 | nearest neighbor with and without replacement (ties, calipers, `keepall`, `wor` with k = 1, 2), kernel, ridge and local-linear weights, the pair-matching bandwidth, cross-validation | matches identical; estimates to 1e-12, weights to 1e-10, SEs to 1e-8 (on kmatch's scale, below); cross-validated bandwidths within 5% |
| teffects | Stata 18 | nnmatch (iid and robust VCEs, ATE, bias adjustment), the matches behind psmatch, ipw (logit and probit) | estimates to 1e-8, SEs to 1e-8 or better; IPW SEs to 1e-9 (logit) |
| psmatch2 | 4.0.12 | nearest neighbor on a score with replacement (k = 1, 3, ties, caliper, `common`, `ate`, `ai(1)`), without replacement (ascending, `descending`, caliper), Mahalanobis | read from its source and checked in Python against a port of its search: same matches with replacement and for Mahalanobis; the Stata run is scripted, not yet run |
| MatchIt + `lm` + `sandwich`, fixest | 4.5.5, 3.1.0, 0.14.2 | influence-function SEs (weights fixed): 1:1 without replacement clustered on the pair, with replacement on `get_matches()`, CEM with HC1, two-way (subclass + id) with fixest | estimates and SEs to 1e-13 |

The R and Stata scripts are in `tests/reference/`; their outputs are in
`tests/data/` and are never edited by hand.

## Calipers

- **Units.** A caliper on the score is in standard deviations of the score
  over all units (MatchIt's `std.caliper = TRUE`); covariate calipers are
  raw. `(width, "sd")` or `(width, "raw")` changes either. teffects' `caliper()` on a single
  matching variable is in the same units (verified: it flags exactly the
  units match-stream leaves unmatched); kmatch's `bwidth()` is in score
  units: write `caliper={"ps": (width, "raw")}` to match it. So is
  psmatch2's `caliper()`.
- **teffects refuses to estimate** when a unit has no match inside the
  caliper (it stops and asks for `osample()`); MatchIt and match-stream
  leave that unit unmatched.
- **Inclusive.** A pair is allowed when the gap is at most the caliper, as
  in MatchIt and kmatch's nearest neighbor. psmatch2's is strict (less
  than the caliper).
- **A failed covariate caliper.** match-stream skips the control and tries
  the next nearest (MatchIt). R Matching picks the nearest neighbors first
  and drops the treated unit when a pick fails a caliper.
- **Within strata.** R Matching's `Matchby` standardizes calipers within
  each stratum; MatchIt and match-stream use the SD over all units.

## Ties

MatchIt keeps one control at the k-th distance (the larger score). R
Matching, teffects and kmatch keep every tied control, and the treated
unit's weight is shared among them: `keep_ties=True` does this. R Matching's
`distance.tolerance` (default 1e-5, in squared, variance-standardized
units) treats near-equal distances as ties; `tie_tolerance=` is the same
thing, default 0 (exact ties only).

On the reference data R Matching treats one treated unit as tied between
two controls whose distances differ by a factor of two. We could not
explain this from its source. It affects the 1-NN cases only, and the
tests check the variance formula on Matching's own pairs.

## Match order without replacement

| Package | Order | Control |
|---|---|---|
| MatchIt, match-stream | largest score first (smallest for the ATC) with a score; data order otherwise (`closest` and `random` available with any distance) | `match_order` |
| kmatch `wor` | ascending pair distance (= `match_order="closest"`); ties in random order | `set sortseed` |
| R Matching | data order | sort the data |
| psmatch2 | ascending score, or descending with `descending`, by a forward-only sweep (below); 1:1 only | data order breaks ties in the score |
| teffects | with replacement only | |

**psmatch2 without replacement** is not nearest-available matching, despite
its help file's advice to sort the data randomly (data order only breaks
ties in the score). It walks the treated units in score order with a
pointer into the sorted controls. The pointer moves only forward, so a free
control behind it is never looked at again, and a unit can get a control
while a nearer one is free. On the reference data 75% of treated units get
the same control as `match_order="smallest"` (86% for `descending` against
`"largest"`), and match-stream's pairs are closer on average. match-stream
does not offer this order. `tests/psmatch2_port.py` ports the search line
by line.

`match_order="random"` hashes each row's position with `seed`, so the order is
the same on both memory paths; it cannot reproduce R's random numbers.

**k > 1 without replacement.**

- **MatchIt** matches in rounds: every treated unit takes its first
  match, then those matched take their second, and so on. This is the
  default here.
- **kmatch** makes one pass in order of pair distance, and a treated unit
  takes controls until it has k. `match_in_rounds=False` gives this, and for other
  orders each unit takes its k matches at its turn.

**Units left with fewer than k matches** (by a caliper or a small
stratum) are kept by MatchIt and match-stream. kmatch drops them unless
`keepall` is set; `min_matches=k` drops them here.

## Distances

- **Mahalanobis.** MatchIt's covariance (pooled within the treatment
  groups), which match-stream follows, differs from the full-sample
  covariance used by R Matching (`Weight = 2`), teffects and `kmatch md`
  (the last two agree with each other to 1e-8). The ATT differs in the
  third digit on the reference data.
- **fastmatch** (github.com/py-econometrics/fastmatch, commit fadcd13) does
  1:k matching with replacement on the covariates, without strata or
  calipers. Its faiss backend's default metric, `"mahalanobis"`, does not
  weight distances by the inverse covariance of the controls. It inverts
  the covariance, takes its Cholesky factor, and inverts that again, so the
  distance it computes is (x - y)' S (x - y), with S the controls'
  covariance. On simulated data its nearest neighbor agreed with the
  S-weighted distance for all 500 queries, and with the Mahalanobis one for
  22% of them. On covariates whitened beforehand, with its Euclidean metric
  (faiss or scikit-learn), it makes MatchIt's and match-stream's matches
  exactly (the benchmarks' scenario G). Its voyager backend is approximate.
  Its standard error exists only with its bias correction: a regression on
  every control and Otsu and Rai's (2017) variance. match-stream's (as R
  Matching's) regression runs on the matched controls, weighted by their
  use, with the Abadie-Imbens variance.
- **pymatchit** 0.6.1 agrees with MatchIt on the score designs of the
  benchmarks (A, C, E, F) but not with its Mahalanobis matches (B, G: the
  ATT differs by up to 0.03).
- **The score.** By default it is the probability (MatchIt, kmatch,
  teffects psmatch); `distance="logit_ps"` matches on the linear index.
  The score model is fitted by hdfe-stream, which can absorb fixed effects
  (`score_model="d ~ x | firm"`); it agrees with R's `glm` and Stata's `logit`
  to their convergence tolerance.

## Weights

`weights()` gives MatchIt's weights: treated units weigh 1, and a control
weighs the sum of 1/k over the treated units it serves, rescaled so that
control weights sum to the number of matched controls. Unmatched units
(MatchIt's weight 0) are left out. `matches()` always carries a subclass
(one per focal unit), also with replacement, where MatchIt reports none.
kmatch scales control weights to sum to the number of treated units: the
ratio, which is all an estimate uses, is the same.

## Standard errors

- **Abadie-Imbens.** As R Matching, with one deliberate difference: focal
  units a caliper leaves unmatched are left out of the variance's N, as
  they are out of the estimate. R Matching counts them in N with an effect
  of 0.
- **Conditional variances.** With `n_variance_matches > 0` they come from the J
  nearest same-arm units within the stratum (ties at the J-th distance
  included), on the score, or on the whitened covariates when matching on a
  multivariate distance (the `mahalanobis_vars` with `mahalanobis_vars`). Checked against R
  Matching on both (its default inverse-variance metric is
  `distance="ivariance"`): ATT, ATC and ATE, `Var.calc` 2-4, to 1e-10. R
  Matching searches all same-arm units; with the strata in its `X` they
  differ only when a stratum has fewer than J + 1 units of an arm.
  A unit alone in its arm within its stratum then has no same-arm
  neighbor, and its conditional variance is undefined. When the variance
  needs it, the SE is NaN with a warning that counts such units (the ATT
  needs the controls' variances; the ATE and the AI 2016 correction need
  the treated units' too). The warning counts the pairs those units are
  the match in. Remedies: coarser strata, `n_variance_matches=0`, `vcov="if"`, or
  `lone_unit_variance=` (opt-in):
  - `"nearest"` takes the unit's `n_variance_matches` nearest same-arm units in any
    stratum, on the matching distance (the score, or the whitened
    covariates). It assumes the variance changes smoothly across strata.
  - `"pooled"` takes the mean of its arm's other conditional variances.

  The Abadie–Imbens (2016) correction's own-arm sets for such a unit come
  from the same fallback: its nearest units of its arm in any stratum, or
  its whole arm. Such a unit is often the match of many focal units, so the
  choice can move the standard error a lot.
  R Matching, with the strata as `exact` columns, sets such a unit's
  variance to zero instead (checked: a treated unit left alone in its
  stratum, the ATC with `Var.calc = 1`, reproduced to 1e-6 with its
  variance set to 0). Because the unit is the match of every control in its
  stratum, that understates the variance: 0.031 against 0.29 with the arm's
  mean variance in its place.
  teffects nnmatch (Stata 18, `ematch()`) refuses to estimate. With
  `vce(robust, nn(2))` it stops with error 459, for both a lone treated unit
  (ATE) and a lone control (ATET). `nn(1)` stops with error 198 (an invalid
  option), so teffects' robust variance needs at least two same-arm
  neighbors. match-stream's default, a NaN with a warning, is the closest
  to this (`tests/reference/stata/teffects_lone.do`).
- **Bias adjustment.** As R Matching's `BiasAdjust` and teffects'
  `biasadj()` (both verified): a regression of the outcome on the
  adjustment covariates over the matched units of each arm, weighted by the
  times each is used, pooled over strata.
- **teffects' VCEs.** For nnmatch, `vce(iid)` is the homoskedastic
  variance (`n_variance_matches=0`) and `vce(robust, nn(J))` is `n_variance_matches=J`. psmatch
  counts the unit itself among its h, so its `vce(robust, nn(h))` is
  `n_variance_matches = h - 1` (its default `nn(2)` is `n_variance_matches=1`).
- **The estimated score (Abadie and Imbens 2016).** When match-stream fits
  the score, the AI variance is corrected exactly as teffects psmatch does
  (to 1e-10: ATE and ATET, logit and probit, 1 and 4 matches). Like
  psmatch, no correction with the homoskedastic variance (`n_variance_matches=0`).
  Beyond psmatch, untested against it: exact strata (every neighbor set
  within the stratum), the ATC (the ATET of the flipped treatment), and a
  score model with absorbed fixed effects (one or several dimensions,
  computed by a fixed-effect regression out of core; checked equal to the
  same model with the effects as dummies), and calipers that leave focal
  units unmatched (teffects refuses to estimate then). With a caliper the
  estimate concerns the matched units, so the correction's sums, neighbor
  sets and counts run over the units the estimate uses (every control and
  the matched treated for the ATT), while the score model's variance stays
  its own; the caliper's dependence on the estimated score is not
  accounted for, as with trimming. With absorbed fixed effects the
  covariate match inside the ATET term uses the covariates within the
  exact strata, not the fixed effects. The score is
  fitted to a relative deviance change of 1e-10 (hdfe-stream's default is
  1e-8, which leaves a probit far enough from the maximum to flip some
  matches).
- **psmatch2.** Its default SE, sqrt(var1/N1 + var0·Σw²/N1²), treats the
  matches as fixed and has no counterpart here. `ai(J) population` is the
  Abadie-Imbens variance: `vcov="ai", n_variance_matches=J`. `ai(J)` alone is their
  conditional (sample) variance, which match-stream does not offer.
- **R Matching's tie.** teffects' 1-NN matches equal match-stream's, without
  the unexplained tie R Matching makes on the reference data (see Ties).
- **Influence-function SEs** (`vcov="if"` or clustered) hold the matching
  weights fixed. The small-sample factor is G/(G-1)·(N-1)/(N-2), the
  weighted regression's HC1/CRV1, so the ATT and ATC equal MatchIt's
  recommended `lm(y ~ treat, weights)` with `sandwich::vcovCL`. The ATE
  matched both ways follows kmatch: it is p·ATT + (1−p)·ATC, p the share of
  treated among the focal units, and its variance is that combination's
  with p estimated (the delta method on the ATT's and ATC's influence
  functions). The regression on the stacked rows of both directions gives
  the same estimate but a variance that treats the focal counts differently
  (2–3% larger on the reference data); no package reports it, and
  match-stream does not either. kmatch sums the
  influence functions over the whole estimation sample (unmatched units
  contributing 0) with n/(n-1), n the sample size: multiply match-stream's
  variance by n/(n-1) / [G/(G-1)·(N-1)/(N-2)] (G units and N rows matched)
  to get kmatch's, exactly. teffects' IPW standard errors have no
  small-sample factor. For two-way clustering match-stream (as fixest and hdfe-stream)
  scales every term by the smallest G; `sandwich` scales each by its own.

## Option names compared

match-stream spells names out where MatchIt, kmatch, R Matching and hdfe-stream
abbreviate them. The correspondences:

| match-stream | Other packages |
|---|---|
| `match_nn`, `match_kernel`, `match_cem` | MatchIt `method = "nearest"`, none, `"cem"`; kmatch `nn`, `kernel`/`ridge`; teffects `nnmatch` |
| `focal` (`"treated"`, `"control"`) | MatchIt `estimand = "ATT"`, `"ATC"`; kmatch `att`, `atc` |
| `n_matches` | MatchIt `ratio`; R Matching `M`; kmatch `nn()`; teffects `nneighbor()` |
| `match_order` | MatchIt `m.order` |
| `mahalanobis_vars` | MatchIt `mahvars` |
| `caliper` | MatchIt `caliper` (in SDs, `std.caliper`); `(width, "raw")` is kmatch `bwidth()` and psmatch2 `caliper()` |
| `discard`, `reestimate_score` | MatchIt `discard`, `reestimate`; psmatch2 `common` |
| `keep_ties` | R Matching `ties`; kmatch `ties` (their default) |
| `tie_tolerance` | R Matching `distance.tolerance` |
| `min_matches` | kmatch without `keepall` is `min_matches=n_matches` |
| `match_in_rounds` | no option elsewhere: MatchIt always matches in rounds; `False` is kmatch's one pass |
| `score`, `score_model`, `score_link` | MatchIt `distance` as a vector, `distance = "glm"`, `link`; kmatch takes a fitted score (`pscore()`) |
| `cutpoints` | MatchIt `cutpoints` |
| `ridge` | kmatch `ridge`; `ridge=0` is its local-linear |
| `bandwidth="pair_distance"`, `PairDistance(factor, quantile)` | kmatch `bwidth(pm)` |
| `bandwidth="cross_validation"`, `CrossValidation(...)` | kmatch `bwidth(cv)`; `outcome` is its `bwidth(cv y)`; `penalty="support_loss"` reproduces its penalty and search |
| `n_variance_matches` | R Matching `Var.calc`; teffects `vce(robust, nn(J))` (J = this + 1) |
| `bias_adjustment` | R Matching `BiasAdjust`; teffects `biasadj()` |
| `lone_unit_variance` | none |
| `score_correction` | the correction of teffects `psmatch` (Abadie-Imbens 2016) |
| `effect(estimand=[...])`, `"nate"`, `potential_outcomes` | kmatch `ate att atc nate`, `po` |
| `vcov` | hdfe-stream, pyfixest `vcov`; Stata `vce()` |

## Several outcomes

`effect(["y1", "y2"])` estimates several outcomes from one match and one
read of the source, as kmatch's `(y1 y2 ...)`. It gives one estimate per
outcome and their joint variance matrix. Each variance equals the
single-outcome one. Each covariance is the one the variance formula
implies: every variance here is a quadratic form in the outcome, so

    Cov(y1, y2) = [V(y1 + y2) - V(y1) - V(y2)] / 2.

- **Influence-function and clustered SEs.** This is the covariance of the
  stacked influence functions, which kmatch reports in `e(V)`. On kmatch's
  scale it agrees with kmatch 1.1.6 to 3e-15, variances and covariance, for
  nearest-neighbor, kernel and ridge matching
  (`tests/reference/stata/kmatch_several.do`).
- **Abadie–Imbens SEs.** No package reports a covariance across outcomes:
  R Matching and teffects estimate one outcome at a time. match-stream's is
  the bilinear analog of the Abadie–Imbens variance. It applies the same
  formula to products of the two outcomes' deviations, so the variance of
  the effect on any combination a·y1 + b·y2 is the Abadie–Imbens variance
  of that combination. It is checked against R Matching through the
  identity above, from Matching's SEs on y1, y2 and y1 + y2 (ATT and ATE,
  `Var.calc` 0 and 2, the bias adjustment;
  `tests/reference/reference_r_several.R`).
- **The AI (2016) correction** subtracts a term. The joint matrix is then
  not guaranteed to be positive semidefinite, and match-stream warns if it
  is not.

## Several estimands and potential outcomes

`effect("y", estimand=["att", "atc", "ate", "nate"], vcov="if",
potential_outcomes=True)` is kmatch's `ate att atc nate po` in one call, and
combines with several outcomes. Coefficients are ordered by outcome, then
estimand, each followed by its two means when asked: `ATT:y`, `ATT:y(1)`,
`ATT:y(0)`, `ATC:y`, ...

- **NATE** is kmatch's naive effect: the unweighted difference of means of
  the units used in matching (complete cases, after any `discard=`). It
  involves no matching and equals the regression of y on the treatment
  with HC1 or clustered standard errors.
- **Potential-outcome means** are the weighted means of the estimand's
  treated units (`y(1)`) and control units (`y(0)`), with the matching
  weights; the effect is their difference. For the NATE they are the raw
  arm means. The ATE's are p·(the ATT's) + (1−p)·(the ATC's), as the effect.
- **Joint variance.** Each coefficient is a weighted mean or a difference of
  two, with the weights held fixed, so it has a per-row influence
  contribution. The estimands' tables share units, so contributions summed by
  cluster (the unit, or source columns) give every covariance. Each diagonal
  entry equals the coefficient's own single fit, which has its own
  small-sample factor G / (G - 1) (N - 1) / (N - 2); entry (a, b) is scaled by
  the square root of the product of the two factors. kmatch scales the whole
  matrix by one factor; the two differ by O(1/N), and the comparison
  (`tests/reference/stata/kmatch_estimands.do`) undoes ours entry by entry.
  With that, effects, potential-outcome means and the whole matrix agree
  with kmatch 1.1.6 to 1e-9 for nearest-neighbor, kernel and ridge matching.
- **Abadie-Imbens SEs are refused** for several estimands, the NATE and
  potential outcomes: no package reports an Abadie-Imbens covariance across
  estimands, and the polarization used for outcomes does not extend to
  them. Use `vcov="if"` or `{"CRV1": ...}`. Clustering on the subclass is also
  refused with several estimands: a subclass belongs to one estimand's match.
- Not built: regression adjustment (`y = avars`).

## Kernel and ridge matching

MatchIt, R Matching and teffects have no kernel matching; the conventions
are kmatch's:

- **Kernels.** The same definitions on support ±1, and the window
  |d| < h, strict.
- **Ridge weights.** Frölich's, with Seifert and Gasser's default
  parameters.
- **Bandwidth.** The pair-matching bandwidth is the default.
- **Common support.** A treated unit with no control in its window is
  unmatched.

Differences:

- **Cross-validation.** By default, the same criterion without kmatch's
  penalty for units lost to common support and its limit factor for
  bandwidths above the score's SD: every bandwidth is judged on the same
  units instead. The search is like kmatch's (steps of 1.5 from the
  pair-distance bandwidth until the criterion turns up, then refined), or an
  explicit `grid`. On the reference data the bandwidths are within 5% of
  kmatch's. `penalty="support_loss"` reproduces kmatch's cross-validation
  exactly (its criterion, penalties and search): equal to every digit on
  the reference data, weighted and unweighted.
  - Why the penalty exists: a small bandwidth leaves some units without a
    neighbor, and the criterion is then averaged over fewer, easier units.
    kmatch inflates it by the share lost; the default here fixes the set
    of units instead. In the weighted version kmatch's penalty also counts
    treated units left without controls, which the default does not see:
    prefer `penalty="support_loss"` if cross-validation lands on bandwidths
    that leave treated units unmatched.
- **Cross-validation without an outcome** (`bandwidth="cross_validation"`
  with no `outcome`) is kmatch's `bwidth(cv)` with no variable. It chooses the
  bandwidth whose matched mean of the score is least moved by leaving out
  any one other-arm unit (kmatch's balance criterion, read from its Mata
  code). Checked against kmatch 1.1.6 (kernel and ridge, ATT and ATC, the
  triangle kernel, a fixed grid): the same search path, criteria to 3e-12,
  identical bandwidths, estimates and SEs to 1e-14. kmatch applies
  no support-loss penalty here. `penalty="support_loss"` adds its limit
  factor and its 15-step search; the default uses neither, as with an
  outcome. The criterion forms the pairs within each bandwidth tried.
  kmatch's `exact` option (re-fitting the ridge weights for each unit left
  out) is not built: kmatch's default closed form is.
- **Not built:** the cosine and
  Parzen kernels, kernel matching on a multivariate distance (`kmatch md`),
  a shared bandwidth for both directions of the ATE, and the regression
  adjustment `(y = xvars)`.
- **psmatch2's** kernel matching uses a fixed default bandwidth of 0.06.

## Mahalanobis within a score caliper

`mahalanobis_vars=` is MatchIt's `mahvars`, checked against MatchIt 4.8.2 pair for pair (with
and without replacement, k = 2, the ATC, `match_order="closest"`, a fitted
score with `discard`): matching is on the Mahalanobis distance between the
`mahalanobis_vars` (MatchIt's pooled within-group covariance, over the units not
discarded), while the score orders the matching ("largest" by default),
carries the caliper (in standard deviations of the score) and decides
`discard`. Here the score caliper is one more caliper column, checked like
a covariate caliper, so the k-d tree paths apply. The AI (2016)
correction does not apply with `mahalanobis_vars`: the matching is on the
covariates, and with a fixed caliper the estimated score has no
first-order effect on the matches, so the Abadie-Imbens (2006) variance
on the covariates (`n_variance_matches > 0`) is the right one. No package offers a
correction for this case.

## Common support

`discard=` is MatchIt's, checked against MatchIt 4.8.2 (the units
discarded, the matches, the weights, the balance tables and sample sizes;
with and without replacement, k = 2, the ATC, a fitted score with and
without `reestimate_score`):

- **The rule:** "treated" drops treated units whose score is outside the
  controls' range, "control" the reverse, "both" any unit outside the
  overlap of the two ranges. A unit exactly at a bound is kept. A column
  name drops the units where it is true (MatchIt's logical vector).
- **`reestimate_score=True`** refits the score model on the units kept; the
  discarded keep their first score in the balance table, as in MatchIt.
- **Calipers** in standard deviations use the units kept (MatchIt's
  `process.caliper`).
- **The balance table** includes the discarded units in "All", and the
  sample sizes count them on their own row: "Unmatched" excludes them.
- **The AI (2016) correction** treats the discarded as outside the second
  stage; without `reestimate_score`, the score model's variance still runs over
  every unit it was fitted on.

Other packages:

- **psmatch2's `common`** drops treated units whose score is above the
  controls' maximum or below their minimum: `discard="treated"`.
- **kmatch's `comsup`** and its kernel windows: kernel matching leaves
  unmatched any treated unit with no control within the bandwidth, which is
  kmatch's support rule; `discard=` is for nearest-neighbor matching.
- **Trimming** by fixed bounds (Crump et al. 2009) is in the IPW core, not
  in matching; for matching, pass a column (`discard="outside"`).

## Coarsened exact matching

Equal-width bins over each covariate's full range, or given cut points,
as MatchIt; the cells and weights are MatchIt's to 1e-10. Cutpoints are
required: MatchIt's default (Sturges' rule through R's `hist()`) is not
reproduced.

## Balance

The statistics are MatchIt's `summary(m, un = TRUE)`, checked to 1e-8.
Exact-matching variables are not listed by default (MatchIt lists them).
For kernel matching there is no pair distance.
