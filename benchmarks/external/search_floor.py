"""The nearest-neighbor search alone, as a floor for the Mahalanobis designs
(B and G of benchmarks/_scenarios.py), in the environment of fastmatch
(benchmarks/external/.fastmatch: faiss and SciPy are there). Reads the Parquet
file (timed as "read"), whitens x1-x3 with MatchIt's Mahalanobis covariance
(pooled within the arms), and for every treated unit finds its single nearest
control, within its exact stratum in B and among all controls in G. The ATT is
the mean of the treated unit's outcome minus its match's. There is no
caliper, tie rule, weight or bookkeeping: whatever a matcher does on top of
the search costs more than this.

    python search_floor.py '<spec as JSON>'

spec: path, scenario ("B" or "G"), backend ("kdtree": scipy.spatial.cKDTree;
"faiss": IndexFlatL2, exact, float32), threads. Prints

    RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {"att": ...}}
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
    """X times the inverse Cholesky factor of the pooled within-arm covariance."""
    import numpy as np
    dev = [X[w == a] - X[w == a].mean(axis=0) for a in (0, 1)]
    cov = (dev[0].T @ dev[0] + dev[1].T @ dev[1]) / (X.shape[0] - 2)
    L = np.linalg.cholesky(cov)
    return np.linalg.solve(L, X.T).T


def nearest(backend, controls, treated, threads):
    """The index in `controls` of each row of `treated`'s nearest row."""
    import numpy as np
    if backend == "kdtree":
        from scipy.spatial import cKDTree
        return cKDTree(controls).query(treated, k=1, workers=threads)[1]
    import faiss
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatL2(controls.shape[1])
    index.add(np.ascontiguousarray(controls, dtype=np.float32))
    return index.search(np.ascontiguousarray(treated, dtype=np.float32), 1)[1][:, 0]


def main(spec):
    import numpy as np
    import pyarrow.parquet as pq

    assert spec["scenario"] in ("B", "G")
    stratified = spec["scenario"] == "B"
    threads = int(spec.get("threads") or 1)
    baseline = hwm_mb()
    t0 = time.perf_counter()
    cols = ["d", "y", "x1", "x2", "x3"] + (["r", "i"] if stratified else [])
    t = pq.read_table(spec["path"], columns=cols)
    w = t.column("d").to_numpy().astype(np.int64)
    y = t.column("y").to_numpy()
    X = np.column_stack([t.column(c).to_numpy() for c in ("x1", "x2", "x3")])
    key = None
    if stratified:
        r, i = t.column("r").to_numpy(), t.column("i").to_numpy()
        key = r.astype(np.int64) * (int(i.max()) + 1) + i
    del t
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    Z = whiten(X, w)
    diff = np.empty(int(w.sum()))
    if not stratified:
        tr, co = np.flatnonzero(w == 1), np.flatnonzero(w == 0)
        diff[:] = y[tr] - y[co[nearest(spec["backend"], Z[co], Z[tr], threads)]]
    else:
        order = np.argsort(key, kind="stable")
        k, w_o, y_o, Z_o = key[order], w[order], y[order], Z[order]
        cuts = np.flatnonzero(np.diff(k)) + 1
        pos = 0
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, k.size]):
            wt = w_o[lo:hi]
            tr, co = lo + np.flatnonzero(wt == 1), lo + np.flatnonzero(wt == 0)
            if tr.size == 0 or co.size == 0:
                continue
            m = co[nearest(spec["backend"], Z_o[co], Z_o[tr], threads)]
            diff[pos:pos + tr.size] = y_o[tr] - y_o[m]
            pos += tr.size
        diff = diff[:pos]
    att = float(diff.mean())
    t_fit = time.perf_counter() - t0
    print("RESULT " + json.dumps({
        "t_read": t_read, "t_fit": t_fit, "peak_memory_mb": hwm_mb(),
        "baseline_memory_mb": baseline, "peak_disk_mb": 0.0, "matched": int(diff.size),
        "converged": True, "coefs": {"att": att}}))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
