"""The benchmarks' data and designs, the same for every library (see
external/match.R and external/pymatchit_fit.py for their spelling there).

The data (`data_path`) are simulated units in 500 exact strata (50 regions
by 10 industries), about 20% treated, with the true propensity score as a
column (``score``), three matching covariates, an outcome with a treatment
effect of 1, and ten float32 columns no design reads (a wide source: what
each library reads is its business). They are written once per size in
chunks of 5,000,000 rows, so 100M rows never sit in memory.

The scenarios:

    A  1:1 nearest neighbor on the score with replacement, exact on (r, i),
       a caliper of 0.2 SD of the score
    B  1:1 Mahalanobis on x1-x3 with replacement, exact on (r, i)
    C  coarsened exact matching: x1-x3 in 5 equal-width bins, exact on (r, i);
       Polars' group_by alone is the floor
    D  Epanechnikov kernel matching on the score within (r, i), kmatch's
       pair-matching bandwidth (match-stream only: no R or Python package
       has kernel matching)
    E  3:1 nearest neighbor with replacement, calipers on the score (0.2 SD)
       and on x1 and x2 (0.25, raw), exact on (r, i)
    F  1:1 greedy without replacement, largest score first (MatchIt's
       default), the calipers of E, exact on (r, i)
    G  1:1 Mahalanobis on x1-x3 with replacement, no exact strata: the design
       fastmatch supports (it has no strata or calipers). fastmatch is given
       the covariates whitened with MatchIt's covariance (pooled within the
       arms; computed in its timed part): its own "mahalanobis" metric weights
       distances by the covariance rather than its inverse (see
       docs/implementations.md), so its Euclidean metric on whitened data is
       the comparable design
    H  1:1 Mahalanobis on x1-x3 without replacement (greedy, in data order),
       exact on (r, i), calipers of 0.25 on x1 and x2 (raw): the package's
       main use, where the memory and the time as the sample grows matter

B and G also run the nearest-neighbor search alone (external/search_floor.py:
SciPy's cKDTree and an exact faiss index, one thread per core, on the whitened
covariates, no calipers, ties or weights), as the floor under any matcher.

Matching (R) has no G: its Mahalanobis uses the full-sample covariance, a
different design (it is in B, for the timing).

Every configuration reports the ATT with the matching weights (match-stream:
``effect("y", vcov="if")``; the others: the weighted difference in means, as
MatchIt's documentation computes it) and the matched treated units, so the
CSV's max_abs_diff checks that the libraries made the same design.

match-stream is timed after a warm-up match of a tiny sample in the same
process with the same options (numba compiles its kernels, or loads them from
its cache, on the first call in a process; that is not counted, as in the
sibling packages' benchmarks). R and pymatchit compile nothing.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

N_R, N_I = 50, 10           # exact strata: 50 regions x 10 industries = 500 cells
CAL_SD = 0.2                # caliper on the score, in SDs of the score
CAL_X = 0.25                # calipers on x1 and x2, raw units
CHUNK = 5_000_000           # rows simulated and written at a time
SEED = 20261103

SCENARIOS = ("A", "B", "C", "D", "E", "F", "G", "H")
CALIPERS = {"score": CAL_SD, "x1": CAL_X, "x2": CAL_X}

# match_stream's options per scenario (the formula is "d ~ x1 + x2 + x3")
MS_OPTIONS = {
    "A": dict(score="score", replace=True, caliper={"score": CAL_SD}),
    "B": dict(distance="mahalanobis", replace=True),
    "C": dict(fn="cem", cutpoints=5),
    "D": dict(fn="kernel", score="score", bandwidth="pair_distance"),
    "E": dict(score="score", replace=True, n_matches=3, caliper=CALIPERS),
    "F": dict(score="score", replace=False, match_order="largest", caliper=CALIPERS),
    "G": dict(distance="mahalanobis", replace=True),
    "H": dict(distance="mahalanobis", replace=False, caliper={"x1": CAL_X, "x2": CAL_X}),
}
# the exact strata of each scenario
EXACT = {s: ["r", "i"] for s in "ABCDEFH"} | {"G": None}
# the narrow columns a design reads (for the libraries that read them first)
COLUMNS = ["r", "i", "d", "score", "y", "x1", "x2", "x3"]


def simulate_chunk(n, start, seed):
    """`n` units with ids from `start` (a pyarrow Table)."""
    import pyarrow as pa
    rng = np.random.default_rng(seed)
    r = rng.integers(0, N_R, n).astype(np.int32)
    i = rng.integers(0, N_I, n).astype(np.int32)
    x = rng.normal(size=(n, 3))
    cell = (r * 0.02 - 0.5) + (i * 0.05 - 0.25)
    lin = -1.5 + 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * x[:, 2] + cell
    score = 1 / (1 + np.exp(-lin))
    d = rng.random(n) < score
    y = 1 + 1.0 * d + x[:, 0] + 0.5 * x[:, 1] + 0.3 * x[:, 2] + cell + rng.normal(size=n)
    cols = {"id": np.arange(start, start + n, dtype=np.int64), "r": r, "i": i, "d": d,
            "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "score": score, "y": y}
    cols.update({f"z{k}": rng.normal(size=n).astype(np.float32) for k in range(10)})
    return pa.table(cols)


def data_path(n, data_dir):
    """The Parquet file of `n` units, written on first use (row groups of
    500,000 rows; chunk c is seeded SEED + c, so a size's data never change)."""
    import pyarrow.parquet as pq
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"units_{n}.parquet"
    if path.exists():
        return path
    part = path.with_name(path.name + ".part")
    writer = None
    for c, start in enumerate(range(0, n, CHUNK)):
        table = simulate_chunk(min(CHUNK, n - start), start, SEED + c)
        if writer is None:
            writer = pq.ParquetWriter(part, table.schema)
        writer.write_table(table, row_group_size=500_000)
    writer.close()
    part.replace(path)
    return path


# --------------------------------------------------------------------------
# match-stream
# --------------------------------------------------------------------------

def _att(m):
    fit = m.effect("y", vcov="if")
    return float(fit.beta[0])


def _matched(m):
    c = m.counts()
    return int(c.get("matched_treated", 0))


def _matcher(options):
    """The match function an options dict calls (its "fn": nn, kernel or cem) and
    the options for it."""
    import match_stream as ms

    options = dict(options)
    kind = options.pop("fn", "nn")
    if kind == "cem":
        options.pop("memory", None)            # CEM streams a group-by: no memory path
    return getattr(ms, "match_" + kind), options


def warm_up(options, exact):
    """Match a tiny sample first, in the timed process, with the same options."""
    import polars as pl

    match, options = _matcher(options)
    small = pl.from_arrow(simulate_chunk(4_000, 0, 1))
    m = match("d ~ x1 + x2 + x3", small, exact=exact, verbose=False, **options)
    _att(m)


def run_match_stream(scenario, path, workdir, memory="auto", max_memory_gb=None):
    """One match and its ATT; the report run_child prints."""
    options = dict(MS_OPTIONS[scenario], memory=memory, max_memory_gb=max_memory_gb)
    warm_up({k: v for k, v in options.items() if k not in ("memory", "max_memory_gb")},
            EXACT[scenario])
    t0 = time.perf_counter()
    match, options = _matcher(options)
    m = match("d ~ x1 + x2 + x3", str(path), exact=EXACT[scenario], workdir=str(workdir),
              verbose=False, **options)
    t_match = time.perf_counter() - t0
    att = _att(m)
    t_fit = time.perf_counter() - t0
    return {"t_read": 0.0, "t_fit": t_fit, "t_match": t_match, "matched": _matched(m),
            "converged": True, "coefs": {"att": att}}


# --------------------------------------------------------------------------
# Polars join_asof (scenario A only: 1-NN with replacement within strata,
# a tolerance as the caliper; a primitive, not a matcher)
# --------------------------------------------------------------------------

def run_polars_asof(scenario, path):
    import polars as pl
    assert scenario == "A"
    t0 = time.perf_counter()
    df = pl.read_parquet(path, columns=["r", "i", "d", "score", "y"])
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    cal = CAL_SD * df["score"].std()
    t = df.filter(pl.col("d")).sort("score")
    c = df.filter(~pl.col("d")).select("r", "i", "score", pl.col("y").alias("y_c"),
                                       pl.col("score").alias("s_c")).sort("score")
    j = t.join_asof(c, on="score", by=["r", "i"], strategy="nearest", tolerance=cal)
    j = j.filter(pl.col("y_c").is_not_null())
    att = float((j["y"] - j["y_c"]).mean())
    t_fit = time.perf_counter() - t0
    return {"t_read": t_read, "t_fit": t_fit, "matched": j.height, "converged": True,
            "coefs": {"att": att}}


# --------------------------------------------------------------------------
# Polars group_by (scenario C: coarsened exact matching is a group-by; this is
# only the group-by: x1-x3 cut into equal-width bins over their ranges, cells
# of (r, i, bins) with both arms kept, the ATT as the treated mean minus the
# controls' mean weighted by their cell's treated count)
# --------------------------------------------------------------------------

def run_polars_cem(scenario, path):
    import polars as pl
    assert scenario == "C"
    t0 = time.perf_counter()
    df = pl.read_parquet(path, columns=["r", "i", "d", "y", "x1", "x2", "x3"])
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    covs = ["x1", "x2", "x3"]
    lo = df.select(pl.col(c).min() for c in covs).row(0)
    hi = df.select(pl.col(c).max() for c in covs).row(0)
    bins = [((pl.col(c) - a) / (b - a) * 5).floor().clip(0, 4).cast(pl.Int32).alias(f"b_{c}")
            for c, a, b in zip(covs, lo, hi)]
    cells = (df.lazy().with_columns(bins)
             .group_by("r", "i", "b_x1", "b_x2", "b_x3")
             .agg(pl.col("d").sum().alias("n_t"), (~pl.col("d")).sum().alias("n_c"),
                  pl.col("y").filter(pl.col("d")).sum().alias("y_t"),
                  pl.col("y").filter(~pl.col("d")).sum().alias("y_c"))
             .filter((pl.col("n_t") > 0) & (pl.col("n_c") > 0))
             .select(pl.col("n_t").sum().alias("n"), pl.col("y_t").sum().alias("y_t"),
                     (pl.col("n_t") * pl.col("y_c") / pl.col("n_c")).sum().alias("y_c"))
             .collect().row(0, named=True))
    att = (cells["y_t"] - cells["y_c"]) / cells["n"]
    t_fit = time.perf_counter() - t0
    return {"t_read": t_read, "t_fit": t_fit, "matched": int(cells["n"]), "converged": True,
            "coefs": {"att": float(att)}}


# --------------------------------------------------------------------------
# the configurations
# --------------------------------------------------------------------------

def configs(scenario):
    """{label: kind} for a scenario; a kind is "<scenario>:<library>"."""
    s = scenario
    out = {"match-stream": f"{s}:ms"}
    # CEM runs on cells in one streaming group-by, and G has no strata: no
    # by-stratum path
    if s not in ("C", "G"):
        out["match-stream (by_stratum)"] = f"{s}:ms_by_stratum"
    if s in ("A", "B", "C", "E", "F", "G", "H"):
        out["MatchIt (R)"] = f"{s}:matchit"
        out["pymatchit"] = f"{s}:pymatchit"
    if s in ("A", "B", "E", "F", "H"):
        out["Matching (R)"] = f"{s}:matching"
    if s == "A":
        out["Matching Matchby (R)"] = f"{s}:matchby"
        out["Polars join_asof"] = f"{s}:polars"
    if s == "C":
        out["Polars group_by"] = f"{s}:polars"
    if s in ("B", "G"):
        # the search alone, with no bookkeeping: what any matcher costs at least
        out["raw k-d tree (SciPy)"] = f"{s}:floor_kdtree"
        out["raw exact search (faiss)"] = f"{s}:floor_faiss"
    if s == "G":
        out["fastmatch (faiss)"] = f"{s}:fastmatch_faiss"
        out["fastmatch (scikit-learn)"] = f"{s}:fastmatch_scikit"
        out["fastmatch (voyager, approximate)"] = f"{s}:fastmatch_voyager"
    return out


def commands(path, external_command, external_missing):
    """The command for a configuration run outside Python's child: R or
    pymatchit (None: the benchmark's own child; a string: skipped, why)."""
    def command(label, kind):
        scenario, lib = kind.split(":", 1)
        spec = dict(path=path, scenario=scenario, cal_sd=CAL_SD, cal_x=CAL_X)
        if lib in ("matchit", "matching", "matchby"):
            return external_command("R", fit=lib, **spec) or external_missing("R")
        if lib == "pymatchit":
            return external_command("pymatchit", **spec) or external_missing("pymatchit")
        if lib.startswith("floor_"):
            return (external_command("floor", backend=lib.split("_", 1)[1], **spec)
                    or external_missing("floor"))
        if lib.startswith("fastmatch_"):
            return (external_command("fastmatch", backend=lib.split("_", 1)[1], **spec)
                    or external_missing("fastmatch"))
        return None
    return command


def runner(config, n, path, workdir):
    """run_child's runner for the Python configurations."""
    scenario, lib = config.split(":", 1)
    if lib == "ms":
        return run_match_stream(scenario, path, workdir)
    if lib == "ms_by_stratum":
        return run_match_stream(scenario, path, workdir, memory="by_stratum")
    if lib == "polars":
        if scenario == "C":
            return run_polars_cem(scenario, path)
        return run_polars_asof(scenario, path)
    raise ValueError(f"no Python runner for {config!r}")
