"""One match with fastmatch (github.com/py-econometrics/fastmatch), for
benchmarks/_harness.py, in its own environment (benchmarks/external/.fastmatch:
faiss and voyager are its dependencies, not the benchmarks'). Reads the
Parquet file into NumPy (timed as "read"), matches, and prints

    RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {"att": ...}}

    python fastmatch_fit.py '<spec as JSON>'

spec: path, scenario (G: 1:1 Mahalanobis on x1-x3 with replacement, no
strata; benchmarks/_scenarios.py), backend ("faiss", "scikit" or "voyager").

The covariates are whitened with MatchIt's Mahalanobis covariance (pooled
within the arms), inside the timed part, and fastmatch matches on them with
a Euclidean metric: that is the design MatchIt and match-stream make.
fastmatch's own "mahalanobis" metric (its faiss backend's default) weights
distances by the controls' covariance instead of its inverse
(docs/implementations.md), so it is not used. faiss computes in float32,
which can flip near-ties; voyager's HNSW index is approximate.

With `bias_adjustment` (effects_benchmark.py), fastmatch's bias-corrected estimator
with a linear outcome model on x1-x3, and its Otsu-Rai standard error.
"""

import json
import sys
import time


def hwm_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return 0.0


def whiten(X, w):
    """X times the inverse Cholesky factor of the pooled within-arm
    covariance (MatchIt's for distance="mahalanobis")."""
    import numpy as np
    dev = [X[w == a] - X[w == a].mean(axis=0) for a in (0, 1)]
    cov = (dev[0].T @ dev[0] + dev[1].T @ dev[1]) / (X.shape[0] - 2)
    L = np.linalg.cholesky(cov)
    return np.linalg.solve(L, X.T).T


def main(spec):
    import numpy as np
    import pyarrow.parquet as pq
    from fastmatch import Matching

    baseline = hwm_mb()
    t0 = time.perf_counter()
    t = pq.read_table(spec["path"], columns=["d", "y", "x1", "x2", "x3"])
    w = t.column("d").to_numpy().astype(np.int64)
    y = t.column("y").to_numpy()
    X = np.column_stack([t.column(c).to_numpy() for c in ("x1", "x2", "x3")])
    del t
    t_read = time.perf_counter() - t0

    assert spec["scenario"] == "G"
    backend = spec["backend"]
    kwargs = {"faiss": {"metric": "euclidean"}, "scikit": {"n_jobs": -1},
              "voyager": {}}[backend]
    bias = None
    if spec.get("bias_adjustment"):
        from sklearn.linear_model import LinearRegression
        bias = LinearRegression()
    t0 = time.perf_counter()
    Z = whiten(X, w)
    est, se = Matching(estimand="ATT", k=1, bias_corr_mod=bias, backend=backend).fit(
        y, w, Z, **kwargs)
    t_fit = time.perf_counter() - t0
    coefs = {"att": float(est)}
    if spec.get("bias_adjustment"):
        coefs["se"] = float(se)
    print("RESULT " + json.dumps({
        "t_read": t_read, "t_fit": t_fit, "peak_memory_mb": hwm_mb(),
        "baseline_memory_mb": baseline, "peak_disk_mb": 0.0, "matched": int(w.sum()),
        "converged": True, "coefs": coefs}))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
