# Effects and standard errors

After matching, the effect is a difference of weighted means of the outcome,
and the question is how to get its standard error. The matching estimate is
not a smooth function of the data (which unit is nearest can change with one
observation), so the usual formulas do not apply as they stand, and the
choices below matter. This page runs them side by side on one data set (4
matches with replacement, ties kept) and then says when each is appropriate.

```{literalinclude} ../../examples/effects.py
:language: python
:start-at: import
```

Output:

```text
                                          estimate      s.e.
AI, homoskedastic                           0.9937    0.0117
AI, n_variance_matches=4, score taken as known    0.9937    0.0117
AI, n_variance_matches=4, score corrected    0.9937    0.0093
AI, bias-adjusted                           0.9958    0.0085
AI, ATE                                     0.9969    0.0116
IF, by unit                                 0.9937    0.0131
IF, subclass + id                           0.9937    0.0122
without replacement, CRV1 pair              1.0086    0.0122
```

## What the estimate is

`effect()` compares the outcome of the focal units with the weighted outcome
of their matches. Which average effect that is depends on whom you match:

- **The ATT** (the default, `focal="treated"` when matching): the effect on
  the units that were treated. It asks for a control for each treated unit,
  and needs only that every treated unit has comparable controls. It is the
  usual target of matching (Imbens 2004).
- **The ATC** (`focal="control"`): the effect on the controls, had they been
  treated.
- **The ATE** (`estimand="ate"`): every unit is focal once, so the effect on
  the whole population. It needs comparable units in both arms, which is
  harder to satisfy.

All three assume the treatment is as good as random given the matching
variables, and that treated and control units overlap (Rosenbaum and Rubin
1983). Matching does not address a confounder you have not measured.

## Which standard error

| | `vcov=` | Use it for |
|---|---|---|
| Abadie-Imbens | `"ai"` | matching with replacement; the default then |
| influence function, by unit | `"if"` | a quick check; kernel, ridge and CEM (the default there) |
| influence function, clustered | `{"CRV1": "subclass+id"}` | matching with replacement, as a robust alternative |
| clustered on the pair | `{"CRV1": "subclass"}` | matching without replacement; the default then |

### Abadie-Imbens

Abadie and Imbens (2006) derive the variance of a matching estimator with a
fixed number of matches, with replacement. It accounts for two things the
regression formula for a weighted mean misses: a control used for many treated
units counts in all their differences, and the matches are themselves
random. The bootstrap is not a substitute: it is not valid for matching
estimators with a fixed number of matches (Abadie and Imbens 2008), so
match-stream does not offer it.

- **`n_variance_matches=0`** (the default) estimates one conditional variance
  for all units, which assumes the outcome's variance does not depend on the
  covariates or on treatment. **`n_variance_matches=J`** estimates each unit's
  own from its J nearest units in the same arm (R Matching's `Var.calc`),
  which allows heteroskedasticity. Prefer it when the outcome's spread
  differs between arms or across the covariates, which is usual. It is a bit
  more expensive and, in a small stratum, can lack enough units
  (`lone_unit_variance=` says what to do).
- **The bias adjustment** (`bias_adjustment=True`, Abadie and Imbens 2011).
  With one continuous matching covariate the matching bias is negligible. With
  two it is as large as the standard error, and with three or more it
  dominates (Abadie and Imbens 2006), so the estimate is not
  root-N-consistent and the intervals are not valid. The adjustment
  regresses the outcome on the covariates and moves each match to its focal
  unit's covariates. Use it whenever you match on more than one continuous
  covariate, in particular with Mahalanobis matching. It moves the estimate
  a little (0.9937 to 0.9958 here) and, here, lowers the standard error.
- **The estimated score** (`score_correction`, Abadie and Imbens 2016). When
  match-stream fits the propensity score (`distance="ps"` or `"logit_ps"`),
  the estimate's variance includes the score model's, and ignoring that gives
  the wrong standard error. Here the corrected standard error is 0.0093
  against 0.0117 if the score is taken as known. The correction applies when
  `n_variance_matches >= 1` and no bias adjustment is requested, and a warning
  says when a fitted score goes uncorrected. A score column you supply is
  taken as known.

### Influence function and clustering

`vcov="if"` and `{"CRV1": ...}` treat the matching weights as given and give
the heteroskedasticity-robust (HC1) or cluster-robust (CRV1) standard errors
of the weighted regression `y ~ treated` on the matched units, the analysis
many applied papers run after matching.

- **With replacement, cluster on the matched unit as well as on the match**
  (`"subclass+id"`). A control used by several treated units appears in
  several pairs, and clustering by pair alone would count it as independent.
  By unit alone (`"if"`) ignores the pairs and gives 0.0131 here, the
  largest. Abadie and Spiess (2022) show that after matching with replacement,
  standard errors that cluster on the matched units are valid.
- **Without replacement, cluster on the pair** (the default). Each unit
  appears once, and the pair is the natural cluster (Abadie and Spiess 2022).
  The Abadie-Imbens variance is for matching with replacement, so it is not
  used here.
- **Kernel, ridge and CEM** have weights, not pairs, and report the
  influence-function standard errors by unit. They are conditional on the
  weights (and on the bandwidth, or the cells).
- A fitted score adds its own variance in all of these, which the influence
  function does not include. If the score model is estimated on a small
  sample, the Abadie-Imbens correction is the more complete answer.

## Which to use

- **Matching with replacement, the standard case:** `vcov="ai"` with
  a small `n_variance_matches` (2 to 4), and `bias_adjustment=True` if you
  match on covariates directly. With a fitted score, leave `score_correction` on.
- **Matching without replacement, or if you would like a second
  opinion:** `vcov={"CRV1": "subclass+id"}` with replacement and the default
  without. A large gap between this and the Abadie-Imbens standard error is
  worth looking into: often a few controls are used very many times, which
  shows in `m.weights()`.
- **Several outcomes or estimands:** pass a list to `effect()`. The outcomes
  are read in one pass and their joint covariance comes with them (for
  several estimands, with influence-function standard errors).
- **Not on this page:** the bootstrap (invalid for matching with a fixed
  number of matches, see above), and sensitivity of the effect to an
  unmeasured confounder (Rosenbaum 2002).

## References

- Abadie, A. and Imbens, G. W. (2006). Large sample properties of matching
  estimators for average treatment effects. *Econometrica* 74(1).
- Abadie, A. and Imbens, G. W. (2008). On the failure of the bootstrap for
  matching estimators. *Econometrica* 76(6).
- Abadie, A. and Imbens, G. W. (2011). Bias-corrected matching estimators for
  average treatment effects. *Journal of Business and Economic Statistics*
  29(1).
- Abadie, A. and Imbens, G. W. (2016). Matching on the estimated propensity
  score. *Econometrica* 84(2).
- Abadie, A. and Spiess, J. (2022). Robust post-matching inference. *Journal
  of the American Statistical Association* 117(538).
- Imbens, G. W. (2004). Nonparametric estimation of average treatment effects
  under exogeneity: a review. *Review of Economics and Statistics* 86(1).
- Rosenbaum, P. R. (2002). *Observational Studies*, 2nd ed. Springer.
- Rosenbaum, P. R. and Rubin, D. B. (1983). The central role of the propensity
  score in observational studies for causal effects. *Biometrika* 70(1).
