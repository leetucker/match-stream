# Choosing a matching method

"Matching" covers several different methods, and they answer one question the
same way: for each unit you want the effect on (usually the treated), find
comparable units in the other arm and compare outcomes. They differ in what
"comparable" means and in how many comparison units each one gets. All of
them assume that, given the variables you match on, the treatment is as good
as random, and that treated and control units overlap (Rosenbaum and Rubin
1983). None of them corrects for a confounder you have not measured.

If you only want a place to start, go to [A starting point](#a-starting-point).

## The methods

| | Function | "Comparable" means | Relies on | Good when |
|---|---|---|---|---|
| Nearest neighbor on the propensity score | `match_nn(distance="ps")` or `"logit_ps"` | the closest estimated probability of treatment | the score model | many covariates; a good score model |
| Nearest neighbor on the covariates | `match_nn(distance="mahalanobis")` (also `"ivariance"`, `"euclidean"`) | the closest covariate values | no model, but few covariates | a handful of covariates, and calipers on those that matter |
| Kernel and ridge | `match_kernel` | every control near the score, weighted by distance | the score model and a bandwidth | many controls per treated unit; a smooth outcome |
| Coarsened exact | `match_cem` | the same bin on every covariate | the bins | few covariates, or ones that coarsen naturally; many units |

Every one can also require an exact match on some variables (`exact=`, the
strata): state, year, industry. That is the cheapest way to make comparable
units comparable on those variables, and it is what lets match-stream work a
stratum at a time on large data.

### Nearest neighbor on the propensity score

The propensity score reduces many covariates to one number, the probability
of treatment given them, and units with the same score are balanced on the
covariates on average (Rosenbaum and Rubin 1983). It is the usual first
choice, and the one most guidance is written for (Stuart 2010; Caliendo and
Kopeinig 2008). Its weaknesses are that a poor score model gives a poor match,
and that two units with the same score can differ a lot on the covariates; King and
Nielsen (2019) argue that this makes score matching worse than matching on the
covariates, particularly as you prune more. A caliper on the score, and
exact matching on the variables that matter most, limits it.

### Nearest neighbor on the covariates

Mahalanobis distance (Rubin 1979) matches on the covariates themselves,
with no model. Matches get worse as the number of continuous covariates
grows: with one the matching bias is negligible, but with two or more
it matters, and with three or more it dominates the standard error
(Abadie and Imbens 2006). So use it with few covariates, and with the bias adjustment
(Abadie and Imbens 2011; see [Effects](examples/effects.md)). Rosenbaum and Rubin (1985) combine the
two: Mahalanobis matching within a caliper on the score.

### Kernel and ridge matching

Instead of the single nearest control, each treated unit is compared with a
weighted average of all controls within a bandwidth of its score (Heckman,
Ichimura and Todd 1997). Ridge matching replaces the average with a local
linear fit (Frölich 2004), which corrects for the controls being on one side
of the treated unit. Using many controls lowers the variance. The bandwidth is
the choice that matters (`PairDistance` is a data-based default;
`CrossValidation` chooses it by predicting an outcome).

### Coarsened exact matching

Cut each covariate into bins, and match the units in the same cell
(Iacus, King and Porro 2012). The balance on the binned covariates is
guaranteed by construction, and you see exactly what was matched. The cost is
that a cell with only one arm is dropped, and the number of cells grows quickly
with the covariates, so with many continuous covariates it discards many
units. With millions of units and a few covariates it is fast and clear.

## What matters more than the method

Simulation studies compare methods under many settings and agree on less than
you might hope (Frölich 2004; Huber, Lechner and Wunsch 2013; Busso, DiNardo
and McCrary 2014; Zhao 2004). The findings most often reported are that matching each treated unit to a
single control (pair matching) does worse than methods that use several
controls, in mean squared error, and that bias adjustment or a regression
after matching helps. The choices below usually matter at least as much:

- **The variables.** Match on variables that affect both treatment and the
  outcome, measured before treatment, and not on anything the treatment
  affects (Stuart 2010). Missing a confounder is a larger problem than any
  choice of matching method.
- **With or without replacement.** With replacement, a control can serve
  several treated units. It lowers bias, since each treated unit gets its
  nearest control, gives matches that do not depend on the order of the data, and needs
  fewer controls; but the variance is higher and the standard errors must allow for
  reuse (Dehejia and Wahba 2002; Stuart 2010). Without replacement (greedy) the
  order of matching matters, and it needs more controls than treated units.
  With a large pool of controls, either works.
- **How many matches.** Each added match lowers the variance and adds some
  bias, since the later matches are further away (Abadie and Imbens 2006;
  Austin 2010). One to four is typical.
- **Calipers.** A maximum distance, so that no treated unit is matched to a
  distant control. Austin (2011) finds 0.2 standard deviations of the *logit*
  of the propensity score works well (`distance="logit_ps"`, `caliper=0.2`;
  `distance="ps"` uses the probability itself). Units without a match inside
  the caliper are dropped, which changes whom the effect is for; check how
  many (`m.counts()`).
- **Balance, not the p value.** After matching, compare the covariates in
  the two arms by standardized differences; below 0.1 is a common standard
  (Austin 2009). Do not use tests of significance, which depend on the sample
  size rather than on the balance (Imai, King and Stuart 2008).
  If the balance is poor, change the matching and look again before you
  look at any outcome.

## A starting point

A reasonable start, following the literature above:

```python
m = match_nn("treat ~ x1 + x2 + x3", data,
             exact=["state", "year"],        # variables that must match
             distance="logit_ps", caliper=0.2,
             n_matches=2, replace=True, id="id")
m.balance().matched                          # standardized differences
m.effect("y", n_variance_matches=2).coef()   # Abadie-Imbens standard error
```

Then:

1. **Check the balance.** If a covariate is poorly balanced, add it to
   `exact=` or give it its own caliper (`caliper={"logit_ps": 0.2, "x1": 0.5}`),
   or add interactions or squares to the score model.
2. **Vary one thing and see if the estimate moves:** one match instead of two;
   Mahalanobis matching with a bias adjustment; kernel or ridge matching;
   without replacement. If the estimate is stable across reasonable choices, it is
   not sensitive to the method. If not, report the range.
3. **Choose the standard error to match the method.** See
   [Effects and standard errors](examples/effects.md).

Starting points for particular situations:

- **A few covariates and no confidence in the score model:** Mahalanobis
  matching with calipers on the important covariates and
  `bias_adjustment=True`.
- **Many more controls than treated units:** more matches per treated unit, or
  kernel or ridge matching, which use all the nearby controls.
- **Few covariates that bin naturally, a very large sample, and a wish to
  explain the matches in a sentence:** coarsened exact matching.
- **Not many controls:** matching with replacement, and watch how many times
  one control is used.
- **No outcome yet:** match first and look at the balance, with `match_nn` alone.
  The matching does not use the outcome, so you can try as many designs
  as you like without choosing among estimates (Rubin 2008).

## References

- Abadie, A. and Imbens, G. W. (2006). Large sample properties of matching
  estimators for average treatment effects. *Econometrica* 74(1).
- Abadie, A. and Imbens, G. W. (2011). Bias-corrected matching estimators for
  average treatment effects. *Journal of Business and Economic Statistics*
  29(1).
- Austin, P. C. (2009). Balance diagnostics for comparing the distribution of
  baseline covariates between treatment groups in propensity-score matched
  samples. *Statistics in Medicine* 28(25).
- Austin, P. C. (2010). Statistical criteria for selecting the optimal number
  of untreated subjects matched to each treated subject when using many-to-one
  matching on the propensity score. *American Journal of Epidemiology* 172(9).
- Austin, P. C. (2011). Optimal caliper widths for propensity-score matching
  when estimating differences in means and differences in proportions in
  observational studies. *Pharmaceutical Statistics* 10(2).
- Busso, M., DiNardo, J. and McCrary, J. (2014). New evidence on the finite
  sample properties of propensity score reweighting and matching estimators.
  *Review of Economics and Statistics* 96(5).
- Caliendo, M. and Kopeinig, S. (2008). Some practical guidance for the
  implementation of propensity score matching. *Journal of Economic Surveys*
  22(1).
- Dehejia, R. H. and Wahba, S. (2002). Propensity score-matching methods for
  nonexperimental causal studies. *Review of Economics and Statistics* 84(1).
- Frölich, M. (2004). Finite-sample properties of propensity-score matching and
  weighting estimators. *Review of Economics and Statistics* 86(1).
- Heckman, J. J., Ichimura, H. and Todd, P. E. (1997). Matching as an
  econometric evaluation estimator: evidence from evaluating a job training
  programme. *Review of Economic Studies* 64(4).
- Huber, M., Lechner, M. and Wunsch, C. (2013). The performance of estimators
  based on the propensity score. *Journal of Econometrics* 175(1).
- Iacus, S. M., King, G. and Porro, G. (2012). Causal inference without balance
  checking: coarsened exact matching. *Political Analysis* 20(1).
- Imai, K., King, G. and Stuart, E. A. (2008). Misunderstandings between
  experimentalists and observationalists about causal inference. *Journal of
  the Royal Statistical Society, Series A* 171(2).
- King, G. and Nielsen, R. (2019). Why propensity scores should not be used for
  matching. *Political Analysis* 27(4).
- Rosenbaum, P. R. and Rubin, D. B. (1983). The central role of the propensity
  score in observational studies for causal effects. *Biometrika* 70(1).
- Rosenbaum, P. R. and Rubin, D. B. (1985). Constructing a control group using
  multivariate matched sampling methods that incorporate the propensity score.
  *The American Statistician* 39(1).
- Rubin, D. B. (1979). Using multivariate matched sampling and regression
  adjustment to control bias in observational studies. *Journal of the American
  Statistical Association* 74(366).
- Rubin, D. B. (2008). For objective causal inference, design trumps analysis.
  *Annals of Applied Statistics* 2(3).
- Stuart, E. A. (2010). Matching methods for causal inference: a review and a
  look forward. *Statistical Science* 25(1).
- Zhao, Z. (2004). Using matching to estimate treatment effects: data
  requirements, matching metrics, and Monte Carlo evidence. *Review of
  Economics and Statistics* 86(1).
