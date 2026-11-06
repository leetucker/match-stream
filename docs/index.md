# match-stream

Matching on data larger than memory: nearest-neighbor matching, on a
propensity score or on the covariates themselves, with or without
replacement, calipers on the score and on covariates, exact strata, kernel,
ridge and coarsened exact matching, balance diagnostics,
and treatment effects with Abadie–Imbens standard errors. Results are tested
against R's MatchIt and Matching and Stata's teffects and kmatch.

```python
from match_stream import match_nn

m = match_nn("treat ~ age + tenure + log_wage0", "extract/*.parquet",
             exact=["state", "naics2"], replace=False,
             score_model="treat ~ age + tenure + log_wage0 | firm_id",
             caliper={"ps": 0.2, "tenure": 2}, id="person_id")
m.matches().sink_parquet("pairs.parquet")
```

## When to use this, and when not to

To choose a method, see [Choosing a matching method](choosing.md).

Use match-stream when the data are large (tens of millions of rows and up,
or more than fits in memory) and you need more than plain 1:1 matching on
a score:

- k nearest neighbors with calipers on the score and on covariates;
- greedy matching without replacement, in any of five orders;
- exact strata, with a score model that absorbs high-dimensional fixed
  effects;
- Mahalanobis, kernel, ridge or coarsened exact matching;
- matches only, with no outcome, written straight to Parquet;
- Abadie-Imbens or clustered influence-function standard errors.

Use something else when:

- **Plain 1:1 matching on a score, with replacement.** Polars'
  `join_asof` is faster, and match-stream's own benchmarks agree:

  ```python
  pairs = treated.sort("ps").join_asof(controls.sort("ps"), on="ps",
                                       by=["r", "i"], strategy="nearest",
                                       tolerance=0.01)
  ```

- **Optimal, full, genetic or cardinality matching.** These solve a
  global problem, which MatchIt (with optmatch) does and match-stream does
  not.
- **Below about a million rows.** MatchIt is fine, and its ecosystem
  (cobalt, marginaleffects) is larger.

Benchmarks against MatchIt, R Matching, pymatchit, fastmatch and Polars, from
10,000 to 100 million units, are in [Benchmarks](benchmarks/index.md).

## How it is fast

- **Windows, not pairs.** Within exact strata and sorted by the score,
  each unit's candidates are a window found by binary search.
- **Cheap skipping.** Matching without replacement skips taken controls
  with path-compressed pointers.
- **Window sums.** Kernel and ridge weights come from window sums, with no
  pairs formed.
- **Narrow columns.** Only tens of bytes a row are held. The score model is
  fitted out of core by hdfe-stream, and large data are matched a batch of
  strata at a time.

See [How it works](how_it_works.md).

## Install

```bash
pip install match-stream
```

(Not on PyPI yet; install from the repository with `pip install -e .`.)

```{toctree}
:hidden:
:caption: Guide

choosing
examples/index
how_it_works
resources
implementations
```

```{toctree}
:hidden:
:caption: Benchmarks

benchmarks/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
changelog
```
