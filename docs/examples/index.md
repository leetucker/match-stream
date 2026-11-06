# Examples

Each page mirrors a script in the repository's `examples/` directory and
shows its output. Run the scripts from the repository's root:

```bash
python examples/quickstart.py
```

| Page | What it shows |
|---|---|
| [Quickstart](quickstart.md) | 1:1 matching on a fitted score with replacement, exact strata, a caliper; balance; the ATT with Abadie-Imbens SEs |
| [Matches only](matches_only.md) | Greedy 1:1 without replacement, calipers on the score and covariates, a score model with absorbed fixed effects, pairs to Parquet |
| [k nearest neighbors](knn_calipers.md) | k = 3 with ties and calipers; `min_matches`; rounds or one pass without replacement |
| [Mahalanobis matching](mahalanobis.md) | Matching on the covariates within strata |
| [Kernel and ridge matching](kernel_ridge.md) | The default bandwidth, cross-validation, influence-function SEs |
| [Coarsened exact matching](cem.md) | Cells, weights, balance and the effect |
| [Effects and standard errors](effects.md) | Abadie-Imbens SEs, `n_variance_matches`, bias adjustment, the ATE, influence-function SEs clustered on the pair or unit |
| [Out of core](out_of_core.md) | A million units in Parquet files, matched a stratum batch at a time |

## The simulated data

`simulate_matching` draws units in 20 exact strata (`r` x `i`) with three
normal covariates `x1`, `x2`, `x3`, a logistic treatment `d` whose
probability (`ps`, the true score) depends on the covariates and the
stratum, and an outcome

$$y = 1 + d + x_1 + 0.5\,x_2 + 0.3\,x_3 + \text{stratum effect} + \varepsilon,$$

so the treatment effect is 1 for everyone (`effect=` changes it). With its
defaults it reproduces `tests/data/ref_data.csv`, the data the test suite's reference
values were computed on.

```{toctree}
:hidden:

quickstart
matches_only
knn_calipers
mahalanobis
kernel_ridge
cem
effects
out_of_core
```
