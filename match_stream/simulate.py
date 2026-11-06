"""`simulate_matching()`: a data set with a known treatment effect, for
examples, tests and benchmarks.

The design of tests/reference/make_data.py (which made the R reference
fixtures) at any size: exact strata ``r`` x ``i``, three normal covariates,
a logistic treatment with stratum effects, the true propensity score ``ps``
and an outcome ``y`` whose treatment effect is `effect` for everyone.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def simulate_matching(n: int = 20_000, *, n_r: int = 5, n_i: int = 4, effect: float = 1.0,
                      seed: int = 20261101) -> pl.DataFrame:
    """Simulated units for matching.

    With the defaults it reproduces ``tests/data/ref_data.csv``, the data the
    test suite's reference values were computed on.

    Parameters
    ----------
    n : int
        Units.
    n_r, n_i : int
        Levels of the two stratum columns ``r`` and ``i`` (``n_r * n_i`` strata).
    effect : float
        The treatment effect on ``y``.
    seed : int
        Random seed.

    Returns
    -------
    polars.DataFrame
        Columns ``id`` (1..n), ``r``, ``i``, ``d`` (treatment, 0/1), ``x1``,
        ``x2``, ``x3``, ``ps`` (the true propensity score), ``y``.
    """
    rng = np.random.default_rng(seed)
    r = rng.integers(0, n_r, n)
    i = rng.integers(0, n_i, n)
    x = rng.normal(size=(n, 3))
    # the stratum effects scale with the default 5 x 4 grid's spacing
    cell = (r * 0.1 - 0.2) + (i * 0.15 - 0.2)
    lin = -1.2 + 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2] + cell
    ps = 1 / (1 + np.exp(-lin))
    d = (rng.random(n) < ps).astype(np.int64)
    y = 1 + effect * d + x[:, 0] + 0.5 * x[:, 1] + 0.3 * x[:, 2] + cell + rng.normal(size=n)
    return pl.DataFrame({"id": np.arange(1, n + 1), "r": r, "i": i, "d": d,
                         "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "ps": ps, "y": y})
