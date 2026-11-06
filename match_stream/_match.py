"""The drivers: sort, order, run the kernel, and return the pairs.

The focal group (treated for the ATT, controls for the ATC) is matched to the
other group within strata. Controls are sorted by (stratum, score); focal
units by (stratum, m.order). `match_frame` matches any frame of whole strata;
`run` feeds it either the whole projection (``in_memory``) or batches of whole
strata read back from disk with hdfe-stream's ``iter_groups``
(``by_stratum``). Strata are independent, so both paths give the same pairs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from . import _kdtree
from ._prep import ROW, SCORE, STRATUM, Narrow, order_key, with_strata
from ._spec import Spec
from .kernels import metric, nn

# rows per batch on the by_stratum path (a batch is whole strata, so it can be
# larger: one stratum is never split)
BATCH_ROWS = 2_000_000


def base_dir(workdir=None) -> Path:
    """Where run directories go: `workdir`, else $HDFE_STREAM_WORKDIR (shared
    with the sibling packages, so `cleanup()` finds them), else the system
    temporary directory."""
    return Path(workdir or os.environ.get("HDFE_STREAM_WORKDIR") or tempfile.gettempdir())


def calipers(spec: Spec, nar: Narrow):
    """The score caliper in score units, the caliper columns, and their widths
    (standard deviations over all kept units where the caliper is in SDs)."""
    cal = np.inf
    if spec.caliper is not None:
        cal = spec.caliper * nar.score_sd if spec.caliper_std else spec.caliper
    cols = list(spec.cov_calipers)
    widths = [spec.cov_calipers[c] * (nar.cov_sd[c] if spec.cov_calipers_std.get(c) else 1.0)
              for c in cols]
    return float(cal), cols, np.asarray(widths, np.float64)


def match_frame(spec: Spec, df: pl.DataFrame, cal: float, cols: list[str],
                calx: np.ndarray, zcols: list[str] | None = None,
                score_var: float = 0.0) -> pl.DataFrame:
    """Pairs for the units of `df`, a frame of whole strata: the focal unit's
    and its match's ROW, the match's rank 1..k, the score distance, and the
    ids (``focal_id``, ``match_id``) when ``id=`` was given."""
    df, n_strata = with_strata(spec, df)
    focal = df[spec.treatment].to_numpy()
    if spec.estimand == "atc":
        focal = ~focal
    g = df[STRATUM].to_numpy()
    s = df[SCORE].to_numpy().astype(np.float64)
    rows = df[ROW].to_numpy()
    X = (np.ascontiguousarray(df.select(cols).to_numpy().astype(np.float64)) if cols
         else np.zeros((df.height, 0)))

    c = np.flatnonzero(~focal)
    c = c[np.lexsort((s[c], g[c]))]
    starts = np.searchsorted(g[c], np.arange(n_strata + 1))
    t = np.flatnonzero(focal)
    t = t[np.lexsort((rows[t], order_key(spec, s[t], rows[t]), g[t]))]
    tstarts = np.searchsorted(g[t], np.arange(n_strata + 1))

    if spec.match_metric:
        Z = np.ascontiguousarray(df.select(zcols).to_numpy().astype(np.float64))
        if spec.mahalanobis_vars and np.isfinite(cal):
            # mahalanobis_vars: the score caliper is one more caliper column, checked
            # like a covariate caliper (|gap| <= width)
            X = np.ascontiguousarray(np.column_stack([X, s]))
            calx = np.append(calx, cal)
            cols = [*cols, SCORE]
        margs = (tstarts, np.ascontiguousarray(Z[t]), np.ascontiguousarray(X[t]), starts,
                 np.ascontiguousarray(Z[c]), np.ascontiguousarray(X[c]), calx, spec.n_matches)
        if spec.replace and not cols and 0 < Z.shape[1] <= _kdtree.KDTREE_MAX_DIM:
            # with replacement and no covariate calipers: the k-d tree, the
            # same matches as the brute-force kernel in O(N log N)
            J = _kdtree.knn_kdtree(tstarts, margs[1], starts, margs[4], spec.n_matches)
        elif spec.replace and 0 < Z.shape[1] <= _kdtree.KDTREE_MAX_DIM:
            # with replacement and covariate calipers: candidate lists, the
            # first k that pass (the brute force's matches)
            J = _kdtree.knn_with_calipers(*margs[:7], spec.n_matches)
        elif spec.replace:
            J = metric.knn_replace(*margs)
        elif 0 < Z.shape[1] <= _kdtree.KDTREE_MAX_DIM:
            # without replacement: k-d tree candidate lists, the brute force's
            # matches (and "closest", which only this path offers)
            J = _kdtree.match_without_replacement(*margs[:7], spec.n_matches, spec.match_in_rounds,
                                                  spec.match_order == "closest")
        elif spec.match_order == "closest":
            raise NotImplementedError(f"match_order='closest' on more than "
                                      f"{_kdtree.KDTREE_MAX_DIM} covariates")
        else:
            J = metric.greedy_no_replace(*margs, spec.match_in_rounds)
        ti, rank = np.nonzero(J >= 0)
        fi, cj = t[ti], c[J[ti, rank]]
        dist = np.sqrt(((Z[fi] - Z[cj]) ** 2).sum(axis=1))
        return _pairs(spec, df, rows, fi, cj, rank, dist)
    args = (tstarts, s[t], np.ascontiguousarray(X[t]), starts, s[c],
            np.ascontiguousarray(X[c]), cal, calx, spec.n_matches)
    if spec.keep_ties:
        # R Matching's tolerance is on squared distances standardized by the
        # score's variance over all units (its inverse-variance weights)
        off, pos = nn.knn_replace_ties(*args, spec.tie_tolerance * score_var)
        cnt = np.diff(off)
        ti = np.repeat(np.arange(t.shape[0]), cnt)
        rank = np.arange(pos.shape[0]) - np.repeat(off[:-1], cnt)
        fi, cj = t[ti], c[pos]
        return _pairs(spec, df, rows, fi, cj, rank, np.abs(s[fi] - s[cj]))
    if spec.replace:
        J = nn.knn_replace(*args)
    elif spec.match_order == "closest":
        J = nn.closest_no_replace(*args, spec.match_in_rounds)
    else:
        J = nn.greedy_no_replace(*args, spec.match_in_rounds)

    ti, rank = np.nonzero(J >= 0)
    fi, cj = t[ti], c[J[ti, rank]]
    return _pairs(spec, df, rows, fi, cj, rank, np.abs(s[fi] - s[cj]))


def _pairs(spec, df, rows, fi, cj, rank, dist) -> pl.DataFrame:
    out = {"focal": rows[fi], "match": rows[cj], "rank": (rank + 1).astype(np.int32),
           "distance": dist}
    if spec.id is not None:
        ids = df[spec.id]
        out["focal_id"] = ids.gather(fi)
        out["match_id"] = ids.gather(cj)
    return pl.DataFrame(out)


def choose_memory(spec: Spec, nar: Narrow, max_memory_gb=None, memory_reserve=None) -> str:
    """"auto": all strata at once if the projection fits the memory headroom
    (stream-limits' MemoryBudget), else a batch of strata at a time."""
    if spec.memory != "auto":
        return spec.memory
    if not spec.exact:
        return "in_memory"              # one stratum: nothing to batch
    from stream_limits import MemoryBudget

    budget = MemoryBudget(limit_gb=max_memory_gb, reserve=memory_reserve)
    return "in_memory" if budget.fits(nar.bytes_in_memory() / 1e9) else "by_stratum"


def run(spec: Spec, nar: Narrow, memory: str, workdir=None) -> pl.DataFrame:
    cal, cols, calx = calipers(spec, nar)
    if memory == "in_memory" or not spec.exact:
        return match_frame(spec, nar.lf.collect(engine="streaming"), cal, cols, calx,
                           nar.zcols, nar.score_sd**2)

    from hdfe_stream.core.store import PolarsEngine, iter_groups
    from hdfe_stream.core.workspace import Run

    run_dir = Run(base_dir(workdir), auto_cleanup=True)
    try:
        path = run_dir.path / "narrow.parquet"
        engine = PolarsEngine()
        engine.sink(nar.lf, str(path), "match-stream narrow projection")
        parts = [match_frame(spec, frame, cal, cols, calx, nar.zcols, nar.score_sd**2)
                 for frame in iter_groups(engine, path, spec.exact, BATCH_ROWS, run_dir.path,
                                          nar.n_strata)]
    finally:
        run_dir.cleanup()
    parts = [p for p in parts if p.height]
    if not parts:
        return match_frame(spec, nar.lf.head(0).collect(), cal, cols, calx, nar.zcols,
                           nar.score_sd**2)
    return pl.concat(parts)
