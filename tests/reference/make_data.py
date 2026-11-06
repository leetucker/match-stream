"""Write the reference data set, tests/data/ref_data.csv, that the R reference
outputs (reference_r.R) are computed on.

20,000 units, 20 exact strata (r in 0..4 x i in 0..3), three normal covariates,
a logistic treatment with stratum effects and a true effect of 1. ``ps`` is the
true propensity score; ``id`` runs 1..n, as R's row names do.

    python tests/reference/make_data.py
"""
from pathlib import Path

import numpy as np
import polars as pl

N, SEED = 20_000, 20261101
OUT = Path(__file__).resolve().parents[1] / "data" / "ref_data.csv"


def make(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    r = rng.integers(0, 5, n)
    i = rng.integers(0, 4, n)
    x = rng.normal(size=(n, 3))
    cell = (r * 0.1 - 0.2) + (i * 0.15 - 0.2)
    lin = -1.2 + 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2] + cell
    ps = 1 / (1 + np.exp(-lin))
    d = (rng.random(n) < ps).astype(np.int64)
    y = 1 + 1.0 * d + x[:, 0] + 0.5 * x[:, 1] + 0.3 * x[:, 2] + cell + rng.normal(size=n)
    return pl.DataFrame({"id": np.arange(1, n + 1), "r": r, "i": i, "d": d,
                         "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "ps": ps, "y": y})


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    make().write_csv(OUT, float_precision=17)
    print(f"wrote {OUT}")
