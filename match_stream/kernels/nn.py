"""Nearest-neighbor matching on a scalar distance (the score), within exact strata.

Every kernel takes the controls sorted by (stratum, score), with stratum g's
controls at ``starts[g]:starts[g+1]``, and the treated units in processing
order with their own offsets ``tstarts``. A treated unit may match a control
when the score gap is at most `cal` and, for every caliper covariate q,
``|tx[k, q] - cx[j, q]| <= calx[q]``. Candidates are visited nearest first
(by score gap) and a control that fails a covariate caliper is skipped, not
fatal: MatchIt's rule (R's Matching instead drops the treated unit). Ties in
the score gap go to the control on the right, the larger score.

Outputs are positions into the sorted controls, -1 for no match. Strata are
independent, so the kernels run strata in parallel (``prange`` over strata).
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _lower_bound(cs, lo, hi, s):
    """First position in cs[lo:hi] with cs >= s (hi if none)."""
    a, b = lo, hi
    while a < b:
        mid = (a + b) >> 1
        if cs[mid] < s:
            a = mid + 1
        else:
            b = mid
    return a


@njit(cache=True)
def _passes(tx, k, cx, j, calx):
    for q in range(calx.shape[0]):
        if abs(tx[k, q] - cx[j, q]) > calx[q]:
            return False
    return True


@njit(cache=True)
def _query_strata(tstarts, n):
    """The stratum of each of the n queries, from their CSR offsets."""
    qg = np.empty(n, np.int64)
    for g in range(tstarts.shape[0] - 1):
        for k in range(tstarts[g], tstarts[g + 1]):
            qg[k] = g
    return qg


@njit(parallel=True, cache=True)
def knn_replace(tstarts, ts, tx, starts, cs, cx, cal, calx, kmax):
    """k nearest controls with replacement. Returns (n_treated, kmax) positions.

    Parallel over the treated units, not the strata: with few strata (or
    none) a loop over them would leave most threads idle."""
    n = ts.shape[0]
    qg = _query_strata(tstarts, n)
    out = np.full((n, kmax), -1, np.int64)
    for k in prange(n):
        g = qg[k]
        lo, hi = starts[g], starts[g + 1]
        s = ts[k]
        a = _lower_bound(cs, lo, hi, s)
        left, right, got = a - 1, a, 0
        while got < kmax:
            dl = s - cs[left] if left >= lo else np.inf
            dr = cs[right] - s if right < hi else np.inf
            if dr <= dl:
                j, dist = right, dr
                right += 1
            else:
                j, dist = left, dl
                left -= 1
            if dist == np.inf or not dist <= cal:    # inf: no controls left
                break
            if _passes(tx, k, cx, j, calx):
                out[k, got] = j
                got += 1
    return out


@njit(cache=True)
def _find(P, j):
    """Root of j with path compression: the nearest free slot in one direction."""
    root = j
    while P[root] != root:
        root = P[root]
    while P[j] != root:
        nxt = P[j]
        P[j] = root
        j = nxt
    return root


@njit(cache=True)
def _nearest_free(R, Lp, cs, lo, m, s, tx, k, cx, cal, calx):
    """The nearest free control (position within the stratum) passing the
    calipers, or -1. R[j]: next free at or right of j (m = none); Lp[j+1]:
    next free at or left of j, plus one (0 = none)."""
    a = _lower_bound(cs, lo, lo + m, s) - lo
    r = _find(R, a)
    l = _find(Lp, a) - 1
    while True:
        dl = s - cs[lo + l] if l >= 0 else np.inf
        dr = cs[lo + r] - s if r < m else np.inf
        if dr <= dl:
            j, dist = r, dr
        else:
            j, dist = l, dl
        # no free control left (both sides exhausted): without this check an
        # infinite caliper (none given) would accept the sentinel position m
        if dist == np.inf or not dist <= cal:
            return -1
        if _passes(tx, k, cx, lo + j, calx):
            return j
        if j == r:
            r = _find(R, r + 1)
        else:
            l = _find(Lp, l) - 1


@njit(parallel=True, cache=True)
def greedy_no_replace(tstarts, ts, tx, starts, cs, cx, cal, calx, kmax, rounds=True):
    """Greedy matching without replacement, treated in the given order.

    For a ratio k, MatchIt's rounds: in round r every treated unit still
    matching (it found a control in each earlier round) takes its nearest free
    control, in the same order. With `match_in_rounds` False, each treated unit takes
    its k nearest free controls at its turn. Taken controls are skipped with
    path-compressed next-free pointers, so a step costs about a binary search."""
    G = starts.shape[0] - 1
    out = np.full((ts.shape[0], kmax), -1, np.int64)
    for g in prange(G):
        lo, hi = starts[g], starts[g + 1]
        m = hi - lo
        if m == 0:
            continue
        R = np.arange(m + 1)
        Lp = np.arange(m + 1)
        n_outer = kmax if rounds else 1
        n_inner = 1 if rounds else kmax
        for rnd in range(n_outer):
            for k in range(tstarts[g], tstarts[g + 1]):
                if rnd > 0 and out[k, rnd - 1] < 0:
                    continue
                for q in range(n_inner):
                    j = _nearest_free(R, Lp, cs, lo, m, ts[k], tx, k, cx, cal, calx)
                    if j < 0:
                        break
                    out[k, rnd if rounds else q] = lo + j
                    R[j] = j + 1
                    Lp[j + 1] = j
    return out


@njit(parallel=True, cache=True)
def closest_no_replace(tstarts, ts, tx, starts, cs, cx, cal, calx, kmax, rounds=True):
    """Greedy matching without replacement in order of pair distance
    (MatchIt's m.order = "closest"; kmatch's wor): repeatedly the closest
    remaining (treated, free control) pair is matched. A heap holds each
    treated unit's current nearest free control; a popped pair whose control
    was taken meanwhile is refreshed and pushed back. With a ratio k, rounds
    as in `greedy_no_replace`, each round in order of distance; with
    `match_in_rounds` False one pass, a treated unit pushing its next nearest free
    control after each match until it has k (kmatch's wor, moremata's
    mm_greedy)."""
    G = starts.shape[0] - 1
    out = np.full((ts.shape[0], kmax), -1, np.int64)
    for g in prange(G):
        lo, hi = starts[g], starts[g + 1]
        m = hi - lo
        t0, t1 = tstarts[g], tstarts[g + 1]
        if m == 0 or t1 == t0:
            continue
        R = np.arange(m + 1)
        Lp = np.arange(m + 1)
        taken = np.zeros(m, np.bool_)
        got = np.zeros(t1 - t0, np.int64)
        for rnd in range(kmax if rounds else 1):
            heap = [(0.0, 0, 0)]              # (distance, treated, control); typed by the seed
            heap.pop()
            for k in range(t0, t1):
                if rnd > 0 and out[k, rnd - 1] < 0:
                    continue
                j = _nearest_free(R, Lp, cs, lo, m, ts[k], tx, k, cx, cal, calx)
                if j >= 0:
                    _heappush(heap, (abs(cs[lo + j] - ts[k]), k, j))
            while len(heap) > 0:
                d, k, j = _heappop(heap)
                if taken[j]:
                    j2 = _nearest_free(R, Lp, cs, lo, m, ts[k], tx, k, cx, cal, calx)
                    if j2 >= 0:
                        _heappush(heap, (abs(cs[lo + j2] - ts[k]), k, j2))
                    continue
                col = rnd if rounds else got[k - t0]
                out[k, col] = lo + j
                got[k - t0] += 1
                taken[j] = True
                R[j] = j + 1
                Lp[j + 1] = j
                if not rounds and got[k - t0] < kmax:
                    j2 = _nearest_free(R, Lp, cs, lo, m, ts[k], tx, k, cx, cal, calx)
                    if j2 >= 0:
                        _heappush(heap, (abs(cs[lo + j2] - ts[k]), k, j2))
    return out


# heapq on a list of (float, int, int) tuples: numba compiles these directly;
# ties in distance go to the lower treated position (processing order), then
# the lower control position.
@njit(cache=True)
def _heappush(heap, item):
    heap.append(item)
    pos = len(heap) - 1
    while pos > 0:
        parent = (pos - 1) >> 1
        if heap[pos] < heap[parent]:
            heap[pos], heap[parent] = heap[parent], heap[pos]
            pos = parent
        else:
            break


@njit(cache=True)
def _heappop(heap):
    last = heap.pop()
    if len(heap) == 0:
        return last
    top = heap[0]
    heap[0] = last
    pos, n = 0, len(heap)
    while True:
        c = 2 * pos + 1
        if c >= n:
            break
        if c + 1 < n and heap[c + 1] < heap[c]:
            c += 1
        if heap[c] < heap[pos]:
            heap[pos], heap[c] = heap[c], heap[pos]
            pos = c
        else:
            break
    return top


@njit(cache=True)
def _knn_one(k, lo, hi, s, cs, tx, cx, cal, calx, kmax, tie_eps2, buf):
    """The k nearest controls of focal unit k with replacement into `buf`,
    plus every further control whose squared distance is within `tie_eps2` of
    the k-th's (R Matching's ties: its `tolerance` is in squared
    standardized units; the caller converts). Returns how many."""
    a = _lower_bound(cs, lo, hi, s)
    left, right, got = a - 1, a, 0
    dk2 = np.inf
    while True:
        dl = s - cs[left] if left >= lo else np.inf
        dr = cs[right] - s if right < hi else np.inf
        if dr <= dl:
            j, dist = right, dr
        else:
            j, dist = left, dl
        if dist == np.inf or not dist <= cal:       # inf: no controls left
            break
        if got >= kmax and not dist * dist <= dk2 + tie_eps2:
            break
        if dr <= dl:
            right += 1
        else:
            left -= 1
        if _passes(tx, k, cx, j, calx):
            if got < buf.shape[0]:
                buf[got] = j
            got += 1
            if got == kmax:
                dk2 = dist * dist
    return got


@njit(parallel=True, cache=True)
def knn_replace_ties(tstarts, ts, tx, starts, cs, cx, cal, calx, kmax, tie_eps2):
    """`knn_replace` keeping ties at the k-th distance. Returns CSR arrays
    (offsets, positions): focal unit i's matches are
    positions[offsets[i]:offsets[i+1]]. Parallel over the focal units (see
    `knn_replace`)."""
    n = ts.shape[0]
    qg = _query_strata(tstarts, n)
    counts = np.zeros(n, np.int64)
    buf = np.empty(0, np.int64)             # the counting pass stores nothing
    for k in prange(n):
        g = qg[k]
        counts[k] = _knn_one(k, starts[g], starts[g + 1], ts[k], cs, tx, cx, cal, calx, kmax,
                             tie_eps2, buf)
    offsets = np.zeros(n + 1, np.int64)
    for i in range(n):
        offsets[i + 1] = offsets[i] + counts[i]
    pos = np.empty(offsets[n], np.int64)
    for k in prange(n):
        g = qg[k]
        _knn_one(k, starts[g], starts[g + 1], ts[k], cs, tx, cx, cal, calx, kmax, tie_eps2,
                 pos[offsets[k]:offsets[k + 1]])
    return offsets, pos
