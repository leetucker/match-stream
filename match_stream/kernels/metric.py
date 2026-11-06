"""Nearest-neighbor matching on a multivariate distance, within exact strata.

The covariates arrive whitened (``z = W x``), so the Mahalanobis,
inverse-variance and Euclidean distances are all Euclidean distances between
rows of z. Within a stratum every focal unit is compared with every control
(blocked brute force, ``M_g x N_g x p`` operations): exact, simple, and fine
when strata are moderate, which exact matching makes them. A k-d tree is the
faster choice for large strata: ``_match.py`` uses scipy's (``_kdtree.py``,
with candidate lists for calipers and matching without replacement) up to
15 whitened covariates, and this kernel beyond, where trees lose their edge
(the candidate kernels also fall back on this kernel's logic, unit by unit).

Calipers on covariates are checked on the raw values (``tx``, ``cx``), with
MatchIt's rule: a control that fails one is skipped. Ties in distance go to
the control earlier in the stratum's order (data order).
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _dist2(tz, k, cz, j):
    d = 0.0
    for q in range(tz.shape[1]):
        e = tz[k, q] - cz[j, q]
        d += e * e
    return d


@njit(cache=True)
def _passes(tx, k, cx, j, calx):
    for q in range(calx.shape[0]):
        if abs(tx[k, q] - cx[j, q]) > calx[q]:
            return False
    return True


@njit(parallel=True, cache=True)
def knn_replace(tstarts, tz, tx, starts, cz, cx, calx, kmax):
    """k nearest controls, with replacement. Returns (n_focal, kmax) positions."""
    G = starts.shape[0] - 1
    out = np.full((tz.shape[0], kmax), -1, np.int64)
    for g in prange(G):
        lo, hi = starts[g], starts[g + 1]
        best = np.empty(kmax)
        for k in range(tstarts[g], tstarts[g + 1]):
            got = 0
            for j in range(lo, hi):
                if not _passes(tx, k, cx, j, calx):
                    continue
                d = _dist2(tz, k, cz, j)
                if got < kmax:
                    pos = got
                    got += 1
                elif d < best[kmax - 1]:
                    pos = kmax - 1
                else:
                    continue
                # insertion into the sorted top-k (strict <: earlier controls win ties)
                while pos > 0 and d < best[pos - 1]:
                    best[pos] = best[pos - 1]
                    out[k, pos] = out[k, pos - 1]
                    pos -= 1
                best[pos] = d
                out[k, pos] = j
    return out


@njit(parallel=True, cache=True)
def greedy_no_replace(tstarts, tz, tx, starts, cz, cx, calx, kmax, rounds=True):
    """Greedy without replacement, focal units in the given order, in MatchIt's
    rounds for a ratio k (see kernels.nn.greedy_no_replace); with `match_in_rounds`
    False each focal unit takes all its matches at its turn."""
    G = starts.shape[0] - 1
    out = np.full((tz.shape[0], kmax), -1, np.int64)
    for g in prange(G):
        lo, hi = starts[g], starts[g + 1]
        taken = np.zeros(hi - lo, np.bool_)
        n_outer = kmax if rounds else 1
        n_inner = 1 if rounds else kmax
        for rnd in range(n_outer):
            for k in range(tstarts[g], tstarts[g + 1]):
                if rnd > 0 and out[k, rnd - 1] < 0:
                    continue
                for q in range(n_inner):
                    bj, bd = -1, np.inf
                    for j in range(lo, hi):
                        if taken[j - lo] or not _passes(tx, k, cx, j, calx):
                            continue
                        d = _dist2(tz, k, cz, j)
                        if d < bd:
                            bj, bd = j, d
                    if bj < 0:
                        break
                    out[k, rnd if rounds else q] = bj
                    taken[bj - lo] = True
    return out
