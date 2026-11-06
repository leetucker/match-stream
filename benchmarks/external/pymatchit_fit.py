"""One match with pymatchit (pymatchit-causal), for benchmarks/_harness.py,
in its own environment (benchmarks/external/.pymatchit). Reads the Parquet
file into pandas (timed as "read"), matches with the scenario's design as
MatchIt's (external/match.R), computes the ATT with the matching weights,
and prints

    RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {"att": ...}}

    python pymatchit_fit.py '<spec as JSON>'

spec: path, scenario (A, B, C, E, F or G; benchmarks/_scenarios.py), cal_sd,
cal_x.
"""

import json
import sys
import time
import warnings


def hwm_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return 0.0


COLUMNS = ["r", "i", "d", "score", "y", "x1", "x2", "x3"]


def main(spec):
    import numpy as np
    import pandas as pd
    from pymatchit import MatchIt

    warnings.simplefilter("ignore")
    baseline = hwm_mb()
    t0 = time.perf_counter()
    df = pd.read_parquet(spec["path"], columns=COLUMNS)
    df["d"] = df["d"].astype(int)
    t_read = time.perf_counter() - t0

    s, cal_sd, cal_x = spec["scenario"], spec["cal_sd"], spec["cal_x"]
    cal = {"distance": cal_sd, "x1": cal_x, "x2": cal_x}
    score = df["score"].to_numpy()
    designs = {
        "A": dict(method="nearest", distance=score, replace=True, caliper=cal_sd,
                  exact=["r", "i"]),
        "B": dict(method="nearest", distance="mahalanobis", replace=True, exact=["r", "i"]),
        # pymatchit's CEM takes no exact=: the strata enter as categories, as
        # in MatchIt's call (external/match.R)
        "C": dict(method="cem", cutpoints={"x1": 5, "x2": 5, "x3": 5}),
        "E": dict(method="nearest", distance=score, replace=True, ratio=3, caliper=cal,
                  exact=["r", "i"]),
        "F": dict(method="nearest", distance=score, replace=False, m_order="largest",
                  caliper=cal, exact=["r", "i"]),
        "G": dict(method="nearest", distance="mahalanobis", replace=True),
        "H": dict(method="nearest", distance="mahalanobis", replace=False,
                  caliper={"x1": cal_x, "x2": cal_x}, exact=["r", "i"]),
    }
    fml = "d ~ x1 + x2 + x3"
    if s == "C":
        df["r"] = df["r"].astype("category")
        df["i"] = df["i"].astype("category")
        fml += " + r + i"
    t0 = time.perf_counter()
    m = MatchIt(df, estimand="ATT", **designs[s])
    m.fit(fml)
    w = np.asarray(m.weights, dtype=float)
    t = df["d"].to_numpy() == 1
    y = df["y"].to_numpy()
    k = w > 0
    att = np.average(y[t & k], weights=w[t & k]) - np.average(y[~t & k], weights=w[~t & k])
    t_fit = time.perf_counter() - t0
    print("RESULT " + json.dumps({
        "t_read": t_read, "t_fit": t_fit, "peak_memory_mb": hwm_mb(),
        "baseline_memory_mb": baseline, "peak_disk_mb": 0.0, "matched": int((t & k).sum()),
        "converged": True, "coefs": {"att": float(att)}}))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
