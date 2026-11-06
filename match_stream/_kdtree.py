"""Nearest neighbors on a multivariate distance with a k-d tree: the fast
path for matching with replacement and no covariate calipers.

The covariates arrive whitened, so every metric is Euclidean in z. One
scipy `cKDTree` holds the controls of every stratum, the stratum being an
extra coordinate: 0 between units of one stratum (so their distances are
exactly the covariate distances) and farther than any covariate distance
between strata (so a query never prefers another stratum's unit when its
own stratum has enough). A query of all focal units runs on every core
(``workers=_threads.WORKERS``), in O(N log N) where the brute-force kernel
(kernels/metric.py) is quadratic within each stratum.

The result is the brute-force kernel's, tie for tie: neighbors are ranked
by distance, and equal distances go to the control earlier in the
stratum's order. scipy does not promise an order for ties, so the
neighbors are re-sorted by (distance, position), and where a tie straddles
the k-th place the tied controls are found with a ball query and the
earliest taken. A stratum with fewer than k controls gives what it has.
k-d trees lose their edge in many dimensions; `_match.py` uses this path
up to KDTREE_MAX_DIM whitened covariates.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import _threads

KDTREE_MAX_DIM = 15


def knn_kdtree(tstarts, tz, starts, cz, kmax: int) -> np.ndarray:
    """The k nearest controls of each focal unit within its stratum: focal
    rows of `tz` in stratum blocks ``tstarts``, controls of `cz` in blocks
    ``starts``. Returns (n_focal, kmax) control positions, -1 where a
    stratum has fewer controls."""
    nt, nc = tz.shape[0], cz.shape[0]
    out = np.full((nt, kmax), -1, np.int64)
    if nt == 0 or nc == 0:
        return out
    G = starts.shape[0] - 1
    gt = np.repeat(np.arange(G), np.diff(tstarts)).astype(np.float64)
    gc = np.repeat(np.arange(G), np.diff(starts)).astype(np.float64)
    both = np.vstack([tz, cz]) if tz.size else cz
    span = float(np.sqrt(((both.max(0) - both.min(0)) ** 2).sum())) if both.size else 0.0
    sep = 2.0 * span + 1.0
    tree = cKDTree(np.column_stack([cz, gc * sep]))
    P = np.column_stack([tz, gt * sep])
    kk = min(kmax + 1, nc)               # one extra: is there a tie at the k-th place?
    d, idx = tree.query(P, k=kk, workers=_threads.WORKERS)
    d, idx = d.reshape(nt, kk), idx.reshape(nt, kk)
    d = np.where(gc[idx] == gt[:, None], d, np.inf)   # other strata: not candidates
    order = np.lexsort((idx, d), axis=-1)             # by distance, then position
    d = np.take_along_axis(d, order, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    k = min(kmax, kk)
    out[:, :k] = np.where(np.isfinite(d[:, :k]), idx[:, :k], -1)
    if kk > kmax:
        tie = np.isfinite(d[:, kmax - 1]) & (d[:, kmax - 1] == d[:, kmax])
        for i in np.flatnonzero(tie):
            # every control of the stratum at (or within) the k-th distance;
            # exact squared distances decide, then the position
            cand = np.asarray(tree.query_ball_point(P[i], r=d[i, kmax - 1] * (1 + 1e-9) + 1e-300))
            cand = cand[gc[cand] == gt[i]]
            dd = ((cz[cand] - tz[i]) ** 2).sum(1)
            pick = cand[np.lexsort((cand, dd))][:kmax]
            out[i, :pick.size] = pick
    return out


CANDIDATES = 16       # each focal unit's list; at least 4 k


def _tree(tstarts, tz, starts, cz):
    """The controls' tree (strata as an extra coordinate) and the focal
    units' query points, with each row's stratum."""
    G = starts.shape[0] - 1
    gt = np.repeat(np.arange(G), np.diff(tstarts)).astype(np.float64)
    gc = np.repeat(np.arange(G), np.diff(starts)).astype(np.float64)
    both = np.vstack([tz, cz])
    span = float(np.sqrt(((both.max(0) - both.min(0)) ** 2).sum())) if both.size else 0.0
    sep = 2.0 * span + 1.0
    return cKDTree(np.column_stack([cz, gc * sep])), np.column_stack([tz, gt * sep]), gt, gc


def _candidates(tstarts, tz, starts, cz, kmax):
    """Each focal unit's candidate list: its K nearest controls of its
    stratum (K = max(CANDIDATES, 4k)), re-sorted by the brute force's
    squared distance and position; the distances, each list's last one, and
    whether it holds the whole stratum."""
    from .kernels import metric_wor as mw

    nt, nc = tz.shape[0], cz.shape[0]
    tree, P, gt, gc = _tree(tstarts, tz, starts, cz)
    K = min(max(CANDIDATES, 4 * kmax), nc)
    _, idx = tree.query(P, k=K, workers=_threads.WORKERS)
    idx = idx.reshape(nt, K).astype(np.int64)
    idx[idx >= nc] = -1                                # fewer points than K
    ok = idx >= 0
    same = np.zeros_like(ok)
    same[ok] = gc[idx[ok]] == np.repeat(gt, K).reshape(nt, K)[ok]
    cand = np.where(same, idx, -1)
    n_ctrl = np.diff(starts)[gt.astype(np.int64)]
    complete = same.sum(1) >= n_ctrl
    d2 = mw.sort_candidates(np.ascontiguousarray(tz), np.ascontiguousarray(cz), cand)
    last = np.where(np.isfinite(d2), d2, -np.inf).max(axis=1)
    return cand, d2, last, complete


def knn_with_calipers(tstarts, tz, tx, starts, cz, cx, calx, kmax: int) -> np.ndarray:
    """With replacement and covariate calipers: the brute force's matches
    (kernels.metric.knn_replace) from candidate lists (kernels/metric_wor.py
    knn_candidates). Returns (n_focal, kmax) control positions."""
    from .kernels import metric_wor as mw

    nt, nc = tz.shape[0], cz.shape[0]
    if nt == 0 or nc == 0:
        return np.full((nt, kmax), -1, np.int64)
    cand, d2, last, complete = _candidates(tstarts, tz, starts, cz, kmax)
    return mw.knn_candidates(tstarts, np.ascontiguousarray(tz), np.ascontiguousarray(tx),
                             starts, np.ascontiguousarray(cz), np.ascontiguousarray(cx), calx,
                             int(kmax), cand, d2, last, complete)


def match_without_replacement(tstarts, tz, tx, starts, cz, cx, calx, kmax: int,
                              rounds: bool, closest: bool) -> np.ndarray:
    """Greedy matching without replacement on whitened covariates from k-d
    tree candidate lists (kernels/metric_wor.py): the brute force's matches
    (kernels.metric.greedy_no_replace) for the focal units' order, and
    pairs in ascending distance with `closest`. Returns (n_focal, kmax)
    control positions, -1 for none."""
    from .kernels import metric_wor as mw

    nt, nc = tz.shape[0], cz.shape[0]
    if nt == 0 or nc == 0:
        return np.full((nt, kmax), -1, np.int64)
    cand, d2, last, complete = _candidates(tstarts, tz, starts, cz, kmax)
    kernel = mw.closest_candidates if closest else mw.greedy_candidates
    return kernel(tstarts, np.ascontiguousarray(tz), np.ascontiguousarray(tx), starts,
                  np.ascontiguousarray(cz), np.ascontiguousarray(cx), calx, int(kmax),
                  bool(rounds), cand, d2, last, complete)


def same_arm_variances(z: np.ndarray, g: np.ndarray, t: np.ndarray, y: np.ndarray,
                       J: int) -> np.ndarray:
    """Each unit's conditional outcome variance for Abadie-Imbens standard
    errors on a multivariate distance (R Matching's Var.calc = J): the sample
    variance (n - 1 divisor) of its outcome and those of the other units of
    its arm and stratum within its J-th nearest distance on the whitened
    covariates `z`, ties at that distance included (fewer when the stratum
    has fewer). NaN for a unit alone in its arm and stratum. One k-d tree per
    arm, the stratum an extra coordinate (as `knn_kdtree`)."""
    n = t.size
    out = np.full(n, np.nan)
    span = float(np.sqrt(((z.max(0) - z.min(0)) ** 2).sum())) if z.size else 0.0
    sep = 2.0 * span + 1.0
    for arm in (True, False):
        idx = np.flatnonzero(t == arm)
        if idx.size == 0:
            continue
        P = np.column_stack([z[idx], g[idx].astype(np.float64) * sep])
        tree = cKDTree(P)
        kk = min(J + 2, idx.size)          # the unit itself, J others, one more for ties
        d, nb = tree.query(P, k=kk, workers=_threads.WORKERS)
        d, nb = d.reshape(idx.size, kk), nb.reshape(idx.size, kk)
        gi = g[idx]
        self_ = nb == np.arange(idx.size)[:, None]
        other = (gi[nb] == gi[:, None]) & ~self_
        d = np.where(other, d, np.inf)
        order = np.lexsort((nb, d), axis=-1)
        d = np.take_along_axis(d, order, axis=1)
        nb = np.take_along_axis(nb, order, axis=1)
        n_other = np.isfinite(d).sum(1)
        m = np.minimum(J, n_other)                       # neighbors before ties
        rows = np.arange(idx.size)
        dJ = np.where(m > 0, d[rows, np.maximum(m - 1, 0)], np.inf)
        # a tie at the J-th distance may reach beyond the query: ball query
        more = (m == J) & (kk - 1 > J) & (d[:, min(J, kk - 1)] == dJ)
        more |= (m == J) & (n_other >= kk - 1) & (kk - 1 <= J)
        yy = y[idx]
        for i in range(idx.size):
            if m[i] == 0:
                continue
            if more[i]:
                cand = np.asarray(tree.query_ball_point(P[i], r=dJ[i] * (1 + 1e-12) + 1e-300))
                cand = cand[(cand != i) & (gi[cand] == gi[i])]
                dd = np.sqrt(((z[idx[cand]] - z[idx[i]]) ** 2).sum(1))
                cand = cand[dd <= dJ[i] * (1 + 1e-12)]
            else:
                cand = nb[i, :m[i]]
            vals = np.concatenate([[yy[i]], yy[cand]])
            out[idx[i]] = vals.var(ddof=1)
    return out
