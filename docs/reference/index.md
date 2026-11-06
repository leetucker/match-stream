# Function reference

Everything here is importable from `match_stream`.

| Name | What it is |
|---|---|
| **[Matching](estimation.md)** | |
| `match_nn` | nearest-neighbor matching: on a score or a multivariate distance, calipers, exact strata |
| `match_kernel` | kernel and ridge matching on a score |
| `match_cem` | coarsened exact matching |
| `PairDistance` | the bandwidth from 1:1 nearest-neighbor distances (kernel matching) |
| `CrossValidation` | the cross-validated bandwidth (kernel matching) |
| **[Results](results.md)** | |
| `MatchSet` | nearest-neighbor matches: pairs, weights, counts, balance, effects |
| `KernelSet` | kernel or ridge matching: weights, bandwidth, balance, effects |
| `CEMSet` | coarsened exact matching: weights and cells, balance, effects |
| `MatchResult` | a treatment effect with its standard error (hdfe-stream's result API) |
| `BalanceTable` | balance before and after matching |
| **[Utilities](utilities.md)** | |
| `simulate_matching` | simulated units with a known treatment effect |

```{toctree}
:hidden:

estimation
results
utilities
```
