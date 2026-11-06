"""Kernel and ridge matching on a scalar score, without pairs.

Every kernel offered is a polynomial in u = (s_j - s_i) / h on each side of
the focal unit (the triangle kernel differs left and right), so the sums
kernel and ridge matching need over a window |u| < 1 are combinations of
the window's moments sum_j v_j u_j^a, for a few values v and powers a. This
module computes those moments without forming pairs:
the points are sorted within strata, a window is two binary searches, and
its moments come from

- blocks of BLOCK consecutive points of one stratum whose moments about
  the block's own midpoint are precomputed; a block wholly inside the window is shifted to
  the query point with the binomial theorem (both offsets are within one
  bandwidth, so no digits are lost), and
- direct sums over the partial blocks at the window's ends.

Global running sums would be simpler, but differences of prefix sums lose
the digits of a narrow window in a long sorted array (the cancellation is
roughly N / (points in the window) times eps, times (spread / h)^2 for the
higher powers). The cost is O(BLOCK + window / BLOCK) per query instead of
O(log n), and the result is accurate to about 1e-13 relative whatever the
sample size.

Conventions (kmatch's): a point is in the window when |s_j - s_i| < h
strictly; the window is split into a left part (s_j < s_i) and a right part
(s_j >= s_i). Ridge matching follows Frolich (2004), as kmatch's
``Kmatch_Ridge``: for focal unit i with kernel weights K_j, W = sum K_j,
pbar = sum K_j s_j / W, d = s_i - pbar, S = sum K_j (s_j - pbar)^2,

    w_ij = K_j / W + K_j (s_j - pbar) d / (S + r h |d|),

falling back to the kernel weights when every point in the window has the
same score (S = 0, as kmatch's collinearity check).
"""

import numpy as np
from numba import njit, prange

BLOCK = 64


@njit(cache=True)
def _binom(A):
    C = np.zeros((A + 1, A + 1))
    for a in range(A + 1):
        C[a, 0] = 1.0
        for k in range(1, a + 1):
            C[a, k] = C[a - 1, k - 1] + (C[a - 1, k] if k <= a - 1 else 0.0)
    return C


@njit(cache=True)
def blocks(starts, n):
    """Blocks of at most BLOCK points that never cross a stratum boundary
    (so a stratum's sums are the same whatever else is in the frame, and
    the memory paths agree to the bit): each block's bounds, and each
    point's block."""
    nb = 0
    for g in range(starts.shape[0] - 1):
        nb += (starts[g + 1] - starts[g] + BLOCK - 1) // BLOCK
    lo = np.empty(nb, np.int64)
    hi = np.empty(nb, np.int64)
    of = np.empty(n, np.int64)
    b = 0
    for g in range(starts.shape[0] - 1):
        p = starts[g]
        while p < starts[g + 1]:
            e = min(p + BLOCK, starts[g + 1])
            lo[b], hi[b] = p, e
            of[p:e] = b
            b += 1
            p = e
    return lo, hi, of


@njit(parallel=True, cache=True)
def block_totals(s, V, h, A, blo, bhi):
    """Per block: its midpoint and sum_j V[j, c] ((s_j - mid) / h)^a."""
    m = V.shape[1]
    nb = blo.shape[0]
    T = np.zeros((nb, A + 1, m))
    mid = np.zeros(nb)
    for b in prange(nb):
        lo, hi = blo[b], bhi[b]
        o = 0.5 * (s[lo] + s[hi - 1])
        mid[b] = o
        for j in range(lo, hi):
            t = (s[j] - o) / h
            for c in range(m):
                p = V[j, c]
                for a in range(A + 1):
                    T[b, a, c] += p
                    p *= t
    return T, mid


@njit(cache=True)
def _range_moments(s, V, T, mid, BK, C, lo, hi, q, h, A, out):
    """Add sum_{j in [lo, hi)} V[j, c] ((s_j - q) / h)^a to out[a, c]."""
    m = V.shape[1]
    blo, bhi, bof = BK
    j = lo
    while j < hi:
        b = bof[j]
        bs = blo[b]
        be = bhi[b]
        if j == bs and be <= hi:
            delta = (mid[b] - q) / h
            pw = np.empty(A + 1)
            pw[0] = 1.0
            for a in range(1, A + 1):
                pw[a] = pw[a - 1] * delta
            for a in range(A + 1):
                for k in range(a + 1):
                    f = C[a, k] * pw[a - k]
                    for c in range(m):
                        out[a, c] += f * T[b, k, c]
            j = be
        else:
            e = min(be, hi)
            for jj in range(j, e):
                u = (s[jj] - q) / h
                for c in range(m):
                    p = V[jj, c]
                    for a in range(A + 1):
                        out[a, c] += p
                        p *= u
            j = e


@njit(cache=True)
def _first_ge(s, lo, hi, x):
    while lo < hi:
        mid = (lo + hi) >> 1
        if s[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _first_gt(s, lo, hi, x):
    while lo < hi:
        mid = (lo + hi) >> 1
        if s[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _window(s, V, T, mid, BK, C, lo, hi, q, h, A, ML, MR):
    """Left and right moments of the window |s_j - q| < h within [lo, hi);
    returns the window's index bounds."""
    ML[:, :] = 0.0
    MR[:, :] = 0.0
    L = _first_gt(s, lo, hi, q - h)
    Q = _first_ge(s, L, hi, q)
    R = _first_ge(s, Q, hi, q + h)
    _range_moments(s, V, T, mid, BK, C, L, Q, q, h, A, ML)
    _range_moments(s, V, T, mid, BK, C, Q, R, q, h, A, MR)
    return L, R


@njit(cache=True)
def _ksum(cL, cR, ML, MR, b, c):
    """sum_j K(u_j) u_j^b V[j, c] from the moments."""
    out = 0.0
    for a in range(cL.shape[0]):
        out += cL[a] * ML[a + b, c] + cR[a] * MR[a + b, c]
    return out


@njit(cache=True)
def _fit(K0, K1, K2, Y0, Y1, h, r):
    """From sum K, sum K u, sum K u^2, sum K y, sum K u y: the fitted value
    at the query and the weight coefficients (alpha, beta) such that
    w_j = K_j (alpha + beta * v_j), v_j = (s_query - s_j) / h. r < 0: kernel."""
    W = K0
    ubar = K1 / W
    if r < 0:
        return Y0 / W, 1.0 / W, 0.0
    Su = K2 - W * ubar * ubar
    if not Su > 1e-12 * K2:
        return Y0 / W, 1.0 / W, 0.0          # one score in the window: kernel weights
    d = -h * ubar                            # s_query - pbar
    e = d / (h * h * Su + r * h * abs(d))
    m = Y0 / W + e * h * (Y1 - ubar * Y0)
    return m, 1.0 / W + e * d, -h * e


@njit(parallel=True, cache=True)
def focal_pass(cs, cy, cstarts, fs, fg, h, cL, cR, r):
    """For each focal unit (score fs, stratum fg) against the sorted other
    arm (cs, outcome cy or zeros, stratum g at cstarts[g]:cstarts[g+1]):
    the number of units in its window, the fitted value of y, and the
    weight coefficients alpha, beta (NaN when the window is empty)."""
    A = cL.shape[0] + 1
    V = np.empty((cs.shape[0], 2))
    V[:, 0] = 1.0
    V[:, 1] = cy
    BK = blocks(cstarts, cs.shape[0])
    T, mid = block_totals(cs, V, h, A, BK[0], BK[1])
    C = _binom(A)
    nf = fs.shape[0]
    cnt = np.zeros(nf, np.int64)
    fit = np.full(nf, np.nan)
    al = np.full(nf, np.nan)
    be = np.full(nf, np.nan)
    for i in prange(nf):
        g = fg[i]
        ML = np.zeros((A + 1, 2))
        MR = np.zeros((A + 1, 2))
        L, R = _window(cs, V, T, mid, BK, C, cstarts[g], cstarts[g + 1], fs[i], h, A, ML,
                       MR)
        cnt[i] = R - L
        if R == L:
            continue
        fit[i], al[i], be[i] = _fit(_ksum(cL, cR, ML, MR, 0, 0), _ksum(cL, cR, ML, MR, 1, 0),
                                    _ksum(cL, cR, ML, MR, 2, 0), _ksum(cL, cR, ML, MR, 0, 1),
                                    _ksum(cL, cR, ML, MR, 1, 1), h, r)
    return cnt, fit, al, be


@njit(parallel=True, cache=True)
def loo_pass(cs, cy, cstarts, cg, h, cL, cR, r):
    """Leave-one-out fitted values within the sorted arm itself: for each
    unit, the kernel or ridge fit at its score from the other units of its
    stratum in the window, and their number (fit NaN when there are none)."""
    A = cL.shape[0] + 1
    n = cs.shape[0]
    V = np.empty((n, 2))
    V[:, 0] = 1.0
    V[:, 1] = cy
    BK = blocks(cstarts, n)
    T, mid = block_totals(cs, V, h, A, BK[0], BK[1])
    C = _binom(A)
    cnt = np.zeros(n, np.int64)
    fit = np.full(n, np.nan)
    k0 = cR[0]                                   # K(0): the unit's own kernel weight
    for j in prange(n):
        g = cg[j]
        ML = np.zeros((A + 1, 2))
        MR = np.zeros((A + 1, 2))
        L, R = _window(cs, V, T, mid, BK, C, cstarts[g], cstarts[g + 1], cs[j], h, A, ML,
                       MR)
        cnt[j] = R - L - 1
        if cnt[j] == 0:
            continue
        # the unit itself sits at u = 0 on the right: only the u^0 sums change
        K0 = _ksum(cL, cR, ML, MR, 0, 0) - k0
        Y0 = _ksum(cL, cR, ML, MR, 0, 1) - k0 * cy[j]
        if not K0 > 0:
            continue
        fit[j] = _fit(K0, _ksum(cL, cR, ML, MR, 1, 0), _ksum(cL, cR, ML, MR, 2, 0), Y0,
                      _ksum(cL, cR, ML, MR, 1, 1), h, r)[0]
    return cnt, fit


@njit(parallel=True, cache=True)
def control_pass(fs, fal, fbe, fstarts, cs, cg, h, cL, cR):
    """Each other-arm unit's matching weight, sum_i w_ij over the focal
    units whose windows contain it, and their number. The focal units
    (sorted, stratum g at fstarts[g]:fstarts[g+1]) carry alpha, beta from
    `focal_pass` (zero for unmatched ones, which then count for nothing).
    From the other unit's side v = (s_i - s_j) / h = -u: a focal unit to
    its left (v < 0) sees it at u > 0, the right half of the kernel."""
    dK = cL.shape[0] - 1
    A = dK + 1
    nf = fs.shape[0]
    V = np.zeros((nf, 3))
    for i in range(nf):
        if fal[i] == fal[i]:                  # matched (not NaN)
            V[i, 0] = fal[i]
            V[i, 1] = fbe[i]
            V[i, 2] = 1.0
    BK = blocks(fstarts, nf)
    T, mid = block_totals(fs, V, h, A, BK[0], BK[1])
    C = _binom(A)
    # K(-v) on each side, as a polynomial in v
    pL = np.empty(dK + 1)
    pR = np.empty(dK + 1)
    for a in range(dK + 1):
        sg = -1.0 if a % 2 else 1.0
        pL[a] = cR[a] * sg
        pR[a] = cL[a] * sg
    n = cs.shape[0]
    w = np.zeros(n)
    cnt = np.zeros(n, np.int64)
    for j in prange(n):
        g = cg[j]
        ML = np.zeros((A + 1, 3))
        MR = np.zeros((A + 1, 3))
        _window(fs, V, T, mid, BK, C, fstarts[g], fstarts[g + 1], cs[j], h, A, ML, MR)
        tot = 0.0
        for a in range(dK + 1):
            tot += pL[a] * (ML[a, 0] + ML[a + 1, 1]) + pR[a] * (MR[a, 0] + MR[a + 1, 1])
        w[j] = tot
        cnt[j] = int(round(ML[0, 2] + MR[0, 2]))
    return w, cnt


@njit(parallel=True, cache=True)
def nearest_gap(cs, cstarts, fs, fg):
    """Each focal unit's distance to the nearest other-arm unit of its
    stratum (inf when there is none): 1:1 matching with replacement."""
    nf = fs.shape[0]
    out = np.full(nf, np.inf)
    for i in prange(nf):
        lo, hi = cstarts[fg[i]], cstarts[fg[i] + 1]
        if hi == lo:
            continue
        p = _first_ge(cs, lo, hi, fs[i])
        best = np.inf
        if p < hi:
            best = cs[p] - fs[i]
        if p > lo:
            best = min(best, fs[i] - cs[p - 1])
        out[i] = best
    return out


@njit(cache=True)
def _kval(cL, cR, u):
    """K(u) from the polynomial coefficients (ascending powers), left of the
    focal unit (u < 0) and right."""
    c = cL if u < 0 else cR
    out, p = 0.0, 1.0
    for a in range(c.shape[0]):
        out += c[a] * p
        p *= u
    return out


@njit(parallel=True, cache=True)
def score_cv_pass(cs, cstarts, fs, fstarts, h, cL, cR, r):
    """kmatch's cross-validation without an outcome (Kmatch_i_PS_cvs), one
    bandwidth: for each focal unit, the weighted mean score of the other-arm
    units in its window (C1, NaN when the window is empty); for each
    other-arm unit, the sum over the focal units whose window holds it of
    the change its removal makes to their C1 (cc1), and the number of focal
    units it is the only unit in the window of (cc2: removing it unmatches
    them). The change is w (s_j - C1) / (1 - w), or, when some weight is
    within 1e-8 of 1, the mean recomputed without the unit, as kmatch.
    Strata run in parallel; one stratum's units are touched by its own
    focal units only. This forms the pairs within the bandwidth (the
    criterion is not a sum of window moments)."""
    G = cstarts.shape[0] - 1
    c1 = np.full(fs.shape[0], np.nan)
    cc1 = np.zeros(cs.shape[0])
    cc2 = np.zeros(cs.shape[0])
    for g in prange(G):
        lo, hi = cstarts[g], cstarts[g + 1]
        for i in range(fstarts[g], fstarts[g + 1]):
            q = fs[i]
            a = _first_gt(cs, lo, hi, q - h)
            b = _first_ge(cs, a, hi, q + h)
            n = b - a
            if n == 0:
                continue
            if n == 1:
                c1[i] = cs[a]
                cc2[a] += 1.0
                cc1[a] += cs[a]
                continue
            K = np.empty(n)
            W = 0.0
            P = 0.0
            for j in range(n):
                K[j] = _kval(cL, cR, (cs[a + j] - q) / h)
                W += K[j]
                P += K[j] * cs[a + j]
            P /= W
            S = 0.0
            for j in range(n):
                S += K[j] * (cs[a + j] - P) ** 2
            w = np.empty(n)
            same = True
            for j in range(1, n):
                if cs[a + j] != cs[a]:
                    same = False
                    break
            if r < 0 or same:
                for j in range(n):
                    w[j] = K[j] / W
            else:
                d = q - P
                den = S + r * h * abs(d)
                for j in range(n):
                    w[j] = K[j] / W + K[j] * (cs[a + j] - P) * (d / den)
            sw = 0.0
            sp = 0.0
            for j in range(n):
                sw += w[j]
                sp += w[j] * cs[a + j]
            m = sp / sw
            c1[i] = m
            slow = False
            for j in range(n):
                if abs(1.0 - w[j]) < 1e-8:
                    slow = True
                    break
            for j in range(n):
                if slow:
                    cc1[a + j] += m - (sp - w[j] * cs[a + j]) / (sw - w[j])
                else:
                    cc1[a + j] += w[j] * (cs[a + j] - m) / (1.0 - w[j])
    return c1, cc1, cc2
