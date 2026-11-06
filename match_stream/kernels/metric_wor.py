"""Matching without replacement on a multivariate distance, from k-d tree
candidate lists (built in _kdtree.py), with the brute-force kernel's
results (kernels/metric.py) exactly.

Each focal unit arrives with its K nearest controls of its stratum from the
tree. They are re-sorted here by the squared distance computed as the
brute-force kernel computes it (`_dist2`), then by position, so that ties
and near-ties resolve the same way. Matching then walks each unit's list
for the first control still free that passes the covariate calipers: that
is the nearest such control as long as the list is known to hold every
control at least as near. When the list runs out, or the control found sits
at the list's last distance (a control just as near might lie outside it),
the unit falls back to scanning its stratum, exactly as the brute force
does. The fallback is rare unless controls are scarce relative to focal
units; it changes the time, never the result.

Orders: the focal units' order (data or random; MatchIt's rounds for a
ratio k, or one pass with `match_in_rounds` False), and "closest", pairs in
ascending distance (a heap of each unit's current best free control, as
kernels.nn.closest_no_replace on a score).
"""

import numpy as np
from numba import njit, prange

from .metric import _dist2, _passes
from .nn import _heappop, _heappush

_REL = 1e-12        # a candidate this close to the list's last distance is not trusted


@njit(parallel=True, cache=True)
def sort_candidates(tz, cz, cand):
    """Squared distances of each focal unit's candidates (as `_dist2`), the
    candidates sorted by (distance, position); -1 pads stay last."""
    nf, K = cand.shape
    d2 = np.full((nf, K), np.inf)
    for i in prange(nf):
        for a in range(K):
            j = cand[i, a]
            if j >= 0:
                d2[i, a] = _dist2(tz, i, cz, j)
        # insertion sort on (d2, position): K is small
        for a in range(1, K):
            dv, jv = d2[i, a], cand[i, a]
            b = a - 1
            # pads (-1) carry an infinite distance, so they sort last
            while b >= 0 and (d2[i, b] > dv or (d2[i, b] == dv and cand[i, b] > jv)):
                d2[i, b + 1] = d2[i, b]
                cand[i, b + 1] = cand[i, b]
                b -= 1
            d2[i, b + 1] = dv
            cand[i, b + 1] = jv
    return d2


@njit(cache=True)
def _scan(i, lo, hi, tz, tx, cz, cx, calx, taken):
    """The brute force for one unit: the nearest free control of the stratum
    passing the calipers (strict <: the earlier position wins ties)."""
    bj, bd = -1, np.inf
    for j in range(lo, hi):
        if taken[j] or not _passes(tx, i, cx, j, calx):
            continue
        d = _dist2(tz, i, cz, j)
        if d < bd:
            bj, bd = j, d
    return bj


@njit(cache=True)
def _next(i, lo, hi, ptr, cand, d2, last, complete, tz, tx, cz, cx, calx, taken):
    """Unit i's nearest free control passing the calipers, from its list
    (advancing its pointer past taken and failing controls), else the scan."""
    K = cand.shape[1]
    while ptr[i] < K:
        j = cand[i, ptr[i]]
        if j < 0:
            break
        if taken[j] or not _passes(tx, i, cx, j, calx):
            ptr[i] += 1
            continue
        if complete[i] or d2[i, ptr[i]] < last[i] * (1.0 - _REL):
            return j
        return _scan(i, lo, hi, tz, tx, cz, cx, calx, taken)
    if complete[i]:
        return -1
    return _scan(i, lo, hi, tz, tx, cz, cx, calx, taken)


@njit(parallel=True, cache=True)
def greedy_candidates(tstarts, tz, tx, starts, cz, cx, calx, kmax, rounds, cand, d2, last,
                      complete):
    """Greedy without replacement, focal units in the given order: MatchIt's
    rounds for a ratio k, or one pass (`match_in_rounds` False)."""
    G = starts.shape[0] - 1
    nf = tz.shape[0]
    out = np.full((nf, kmax), -1, np.int64)
    taken = np.zeros(cz.shape[0], np.bool_)       # strata are disjoint: no races
    ptr = np.zeros(nf, np.int64)
    for g in prange(G):
        lo, hi = starts[g], starts[g + 1]
        if hi == lo:
            continue
        n_outer = kmax if rounds else 1
        n_inner = 1 if rounds else kmax
        for rnd in range(n_outer):
            for i in range(tstarts[g], tstarts[g + 1]):
                if rnd > 0 and out[i, rnd - 1] < 0:
                    continue
                for q in range(n_inner):
                    j = _next(i, lo, hi, ptr, cand, d2, last, complete, tz, tx, cz, cx,
                              calx, taken)
                    if j < 0:
                        break
                    out[i, rnd if rounds else q] = j
                    taken[j] = True
    return out


@njit(parallel=True, cache=True)
def closest_candidates(tstarts, tz, tx, starts, cz, cx, calx, kmax, rounds, cand, d2, last,
                       complete):
    """Without replacement in ascending pair distance (m.order "closest"):
    repeatedly the closest (focal, free control) pair; ties go to the
    earlier focal unit, then the earlier control. MatchIt's rounds for a
    ratio k, or one pass (kmatch's wor) with `match_in_rounds` False."""
    G = starts.shape[0] - 1
    nf = tz.shape[0]
    out = np.full((nf, kmax), -1, np.int64)
    taken = np.zeros(cz.shape[0], np.bool_)
    ptr = np.zeros(nf, np.int64)
    got = np.zeros(nf, np.int64)
    for g in prange(G):
        lo, hi = starts[g], starts[g + 1]
        t0, t1 = tstarts[g], tstarts[g + 1]
        if hi == lo or t1 == t0:
            continue
        for rnd in range(kmax if rounds else 1):
            heap = [(0.0, 0, 0)]              # (distance, focal, control); typed by the seed
            heap.pop()
            for i in range(t0, t1):
                if rnd > 0 and out[i, rnd - 1] < 0:
                    continue
                j = _next(i, lo, hi, ptr, cand, d2, last, complete, tz, tx, cz, cx, calx,
                          taken)
                if j >= 0:
                    _heappush(heap, (_dist2(tz, i, cz, j), i, j))
            while len(heap) > 0:
                d, i, j = _heappop(heap)
                if taken[j]:
                    j2 = _next(i, lo, hi, ptr, cand, d2, last, complete, tz, tx, cz, cx,
                               calx, taken)
                    if j2 >= 0:
                        _heappush(heap, (_dist2(tz, i, cz, j2), i, j2))
                    continue
                out[i, rnd if rounds else got[i]] = j
                got[i] += 1
                taken[j] = True
                if not rounds and got[i] < kmax:
                    j2 = _next(i, lo, hi, ptr, cand, d2, last, complete, tz, tx, cz, cx,
                               calx, taken)
                    if j2 >= 0:
                        _heappush(heap, (_dist2(tz, i, cz, j2), i, j2))
    return out


@njit(cache=True)
def _scan_k(i, lo, hi, tz, tx, cz, cx, calx, kmax, row):
    """The brute force for one unit with replacement: its k nearest controls
    of the stratum passing the calipers, ranked by (distance, position), in
    `row` (as kernels.metric.knn_replace)."""
    best = np.full(kmax, np.inf)
    got = 0
    for j in range(lo, hi):
        if not _passes(tx, i, cx, j, calx):
            continue
        d = _dist2(tz, i, cz, j)
        if got < kmax:
            pos = got
            got += 1
        elif d < best[kmax - 1]:
            pos = kmax - 1
        else:
            continue
        while pos > 0 and d < best[pos - 1]:
            best[pos] = best[pos - 1]
            row[pos] = row[pos - 1]
            pos -= 1
        best[pos] = d
        row[pos] = j


@njit(parallel=True, cache=True)
def knn_candidates(tstarts, tz, tx, starts, cz, cx, calx, kmax, cand, d2, last, complete):
    """With replacement: each focal unit's k nearest controls passing the
    calipers, the first k of its sorted list that pass, as long as the k-th
    lies strictly inside the list's last distance (or the list holds the
    whole stratum); else the brute-force scan for that unit."""
    nf = tz.shape[0]
    G = starts.shape[0] - 1
    out = np.full((nf, kmax), -1, np.int64)
    K = cand.shape[1]
    for g in prange(G):
        lo, hi = starts[g], starts[g + 1]
        if hi == lo:
            continue
        for i in range(tstarts[g], tstarts[g + 1]):
            got = 0
            dk = np.inf
            for a in range(K):
                j = cand[i, a]
                if j < 0:
                    break
                if not _passes(tx, i, cx, j, calx):
                    continue
                out[i, got] = j
                got += 1
                dk = d2[i, a]
                if got == kmax:
                    break
            trusted = complete[i] or (got == kmax and dk < last[i] * (1.0 - _REL))
            if not trusted:
                for q in range(kmax):
                    out[i, q] = -1
                _scan_k(i, lo, hi, tz, tx, cz, cx, calx, kmax, out[i])
    return out
