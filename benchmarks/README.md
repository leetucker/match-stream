# Benchmarks

match-stream is compared with the other open implementations of matching.
The data are simulated.

```bash
python benchmarks/scale_benchmark.py                   # default sizes, every scenario
python benchmarks/scale_benchmark.py 10000 1000000     # units
python benchmarks/scale_benchmark.py --scenario A --only match-stream "MatchIt (R)"
python benchmarks/effects_benchmark.py                 # the effect and its SE
python benchmarks/plot_benchmarks.py                   # docs/figures/
```

| Suite | What | Data | Compared with |
|---|---|---|---|
| `scale_benchmark.py` | eight matching designs (below), matching and the ATT with the matching weights | 10,000 to 100M units | MatchIt and Matching (R), pymatchit, fastmatch, Polars `join_asof` and `group_by`, raw k-d tree and faiss searches |
| `effects_benchmark.py` | match, then the effect with influence-function, clustered and Abadie-Imbens SEs (with `n_variance_matches`, the bias adjustment and the AI 2016 correction) | the same | Matching's AI SEs (R); fastmatch's bias-corrected SE |

Run them on a machine that is not doing anything else. A full run takes
hours; the large sizes need a big machine.

## The designs

The data (`_scenarios.py`) are units in 500 exact strata (50 regions by 10
industries), about 20% treated, with the true propensity score as a column
(`score`), three covariates, an outcome with an effect of 1, and ten unused
columns (a wide source). Each size is written once, in chunks, to
`benchmarks/data/`.

| | Design | Libraries |
|---|---|---|
| A | 1:1 nearest neighbor on the score with replacement, exact on (r, i), caliper 0.2 SD | all but fastmatch |
| B | 1:1 Mahalanobis on x1-x3 with replacement, exact on (r, i) | match-stream, MatchIt, Matching, pymatchit, raw k-d tree and faiss searches |
| C | coarsened exact matching, x1-x3 in 5 bins, exact on (r, i) | match-stream, MatchIt, pymatchit, Polars `group_by` |
| D | Epanechnikov kernel matching on the score within (r, i), kmatch's bandwidth | match-stream (no R or Python package has kernel matching; kmatch is Stata) |
| E | 3:1 with replacement, calipers on the score (0.2 SD), x1 and x2 (0.25), exact on (r, i) | match-stream, MatchIt, Matching, pymatchit |
| F | 1:1 greedy without replacement, largest score first, the calipers of E, exact on (r, i) | match-stream, MatchIt, Matching, pymatchit |
| G | 1:1 Mahalanobis on x1-x3 with replacement, no strata | match-stream, MatchIt, pymatchit, fastmatch, raw k-d tree and faiss searches |
| H | 1:1 Mahalanobis on x1-x3 **without** replacement (greedy, data order), exact on (r, i), calipers of 0.25 on x1 and x2 | match-stream, MatchIt, Matching, pymatchit |

match-stream runs each with its defaults (`memory="auto"`, every core) and,
except C and G, again with `memory="by_stratum"`, which holds a batch of
strata at a time.

**How the libraries agree.** `max_abs_diff` is the largest difference from
match-stream's ATT (and, in the effects suite, SE). At 10,000 and 100,000
units (the smoke test):

- **MatchIt** agrees to about 1e-15 in every design.
- **pymatchit** agrees in A, C, E and F. In B and G it differs by up to
  0.03, so its Mahalanobis matches differ.
- **fastmatch** (faiss and scikit-learn) agrees exactly in G. Voyager's
  index is approximate and differs by about 0.003.
- **Polars `join_asof`** agrees in A, and **`group_by`** in C (to 1e-15): the
  group-by alone, with no weights table, is the floor for coarsened exact
  matching.
- **The raw k-d tree and faiss searches** agree in B and G (tested at 50,000
  units: to the last digit).
- **H** (Mahalanobis without replacement, covariate calipers) is expected to
  agree with MatchIt: `tests/reference/reference_r.R` has the spec
  (`nn_maha_wor_cal`) and `test_reference_nn.py` checks the pairs once its
  fixture is made. Neither MatchIt's nor pymatchit's H has been run here (no
  R), so read `max_abs_diff` and the matched counts in the first results
  before trusting the timings.
- **Matching** agrees in A. It differs by design elsewhere:
  - its calipers are in SDs of each column;
  - it drops a treated unit when one of its matches fails a covariate
    caliper (in E few treated units are left);
  - its Mahalanobis uses the full-sample covariance;
  - `Matchby` standardizes calipers within each stratum.

  Its AI SEs on the effects suite's design equal match-stream's.

## Measurement

Each configuration runs in its own process (`_harness.py`, adapted from
dpd-stream's, itself from hdfe-stream's). Each process records:

- **Time:** wall time, split into reading the data and matching. match-stream
  reads inside its fit, so its read time is zero. Every library's fit
  includes computing the ATT. In the effects suite, `t_match` and `t_effect`
  split match-stream's time.
- **Memory:** the peak resident memory (`VmHWM`), and the baseline after the
  imports.
- **Disk:** the peak size of its work directory. Only match-stream writes
  there (`memory="by_stratum"`).

**Warm-up.** match-stream is timed after a warm-up match of a tiny sample in
the same process, so numba's compilation and the loading of its cached
kernels are not counted, as in the sibling packages.

**Failures.** A configuration that times out (30 minutes by default) or runs
out of memory at one size is not run at larger ones.

**Output.** Results go to `results/<suite>.csv`, and the machine to
`results/machine.json`. `MS_BENCH_RESULTS`, `MS_BENCH_DATA` and
`MS_BENCH_FIGURES` send them elsewhere, for a trial run.

## The other libraries

They are not dependencies. Where one is missing, its configurations are
skipped. Each lives in an environment of its own under `external/`.

**R** (`external/match.R`: MatchIt, Matching). Install it into a conda
prefix, with MatchIt from CRAN (conda-forge's lags):

```bash
micromamba create -p benchmarks/external/.r -f benchmarks/external/environment.yml
PATH=benchmarks/external/.r/bin:$PATH benchmarks/external/.r/bin/Rscript -e 'install.packages("MatchIt", repos = "https://cloud.r-project.org")'
```

Without the prefix, the `Rscript` on `PATH` is used. It needs MatchIt,
Matching, jsonlite and either arrow or nanoparquet.

**pymatchit** (`external/pymatchit_fit.py`):

```bash
uv venv -p 3.12 benchmarks/external/.pymatchit
VIRTUAL_ENV=benchmarks/external/.pymatchit uv pip install pymatchit-causal pyarrow
```

**The raw searches** (`external/search_floor.py`) run in fastmatch's
environment. They whiten x1-x3 as above and find each treated unit's nearest
control (within its stratum in B) with SciPy's `cKDTree` or an exact faiss
`IndexFlatL2`, then take the mean outcome difference: no calipers, ties,
weights or bookkeeping, so they are the least any matcher on that design
can cost. The k-d tree is the fair floor at every size. faiss's exact index
is brute force, so its time grows with the square of the stratum (G has one
stratum): expect it to time out at 10M units and more in G.

**fastmatch** (`external/fastmatch_fit.py`) is not on PyPI. Pin the commit
the benchmarks were written against:

```bash
uv venv -p 3.12 benchmarks/external/.fastmatch
VIRTUAL_ENV=benchmarks/external/.fastmatch uv pip install "git+https://github.com/py-econometrics/fastmatch@fadcd13c6468c99a0f01c5293115e03de561e764" pyarrow
```

fastmatch has no exact strata or calipers, so it runs in scenario G only.
It matches on the covariates whitened with MatchIt's covariance, with a
Euclidean metric. Its own `"mahalanobis"` metric (the faiss backend's
default) weights distances by the covariance instead of its inverse (see
`docs/implementations.md`).

**Paths.** `MS_BENCH_RSCRIPT`, `MS_BENCH_PYMATCHIT_PYTHON` and
`MS_BENCH_FASTMATCH_PYTHON` point at other installations.

**Stata.** kmatch, teffects and psmatch2 are not benchmarked. Their outputs
are references for the tests (`tests/reference/stata/`).
