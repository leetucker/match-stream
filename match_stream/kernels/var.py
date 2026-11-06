"""Conditional outcome variances for Abadie-Imbens standard errors.

For ``n_variance_matches = J > 0`` (R Matching's ``Var.calc``), a unit's conditional
variance is the sample variance (n - 1 denominator) of its own outcome and
those of its J nearest units in the same arm, ties at the J-th distance
included. Units arrive sorted by (stratum, score) within one arm, with
stratum g at ``starts[g]:starts[g+1]``; neighbors are sought within the
stratum by score gap (R Matching searches all same-arm units, and with exact
matching columns in X the other strata are far; they differ only when a
stratum has fewer than J + 1 units of the arm).
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def sigma2_score(starts, s, y, J):
    n = s.shape[0]
    out = np.full(n, np.nan)
    G = starts.shape[0] - 1
    for g in prange(G):
        lo, hi = starts[g], starts[g + 1]
        for p in range(lo, hi):
            left, right, got = p - 1, p + 1, 0
            sm, ss, cnt = y[p], y[p] * y[p], 1
            last = -1.0
            while True:
                dl = s[p] - s[left] if left >= lo else np.inf
                dr = s[right] - s[p] if right < hi else np.inf
                d = dl if dl < dr else dr
                if d == np.inf:
                    break
                if got >= J and d > last:
                    break                        # past the J-th distance and its ties
                if dl < dr:
                    j = left
                    left -= 1
                else:
                    j = right
                    right += 1
                sm += y[j]
                ss += y[j] * y[j]
                cnt += 1
                got += 1
                last = d
            if cnt > 1:
                mean = sm / cnt
                out[p] = (ss / cnt - mean * mean) * cnt / (cnt - 1)
    return out
