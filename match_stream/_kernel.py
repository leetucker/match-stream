"""Kernel and ridge matching on the score: bandwidths and weights.

Each focal unit (treated for the ATT, controls for the ATC) is compared with
a kernel-weighted average of the other arm's units of its stratum within
one bandwidth h of its score; ridge matching replaces the kernel average by
Frolich's (2004) ridge-regularized local-linear fit. A focal unit with no
other-arm unit in its window is unmatched (kmatch's common support). The
result is a weight per matched unit; there is no pair table. The numerics
are in kernels/kernel.py (window moments, no pairs).

Bandwidths (MatchIt has no kernel matching; the conventions are kmatch's
unless noted):

- a number: h in score units (kmatch, psmatch2), not in standard
  deviations (unlike MatchIt's calipers);
- ``"pair_distance"`` (the default; kmatch's ``pm``): `PairDistance.factor` (1.5)
  times the `PairDistance.quantile` (0.90) quantile of the nonzero distances of 1:1
  nearest-neighbor matching with replacement, pooled over strata, for the
  matching direction at hand; Stata's default quantile definition
  (Hyndman-Fan type 2); plus kmatch's tiny increase that keeps h > 0;
- ``"cross_validation"`` (kmatch's ``cv``): leave-one-out cross-validation on an
  outcome (Frolich 2004, 2005): the h on a grid minimizing the mean squared error of predicting
  each other-arm unit's outcome from the others of its arm, with the same
  kernel or ridge fit. ``CrossValidation.weighted`` weights each unit by its
  matching weight at that h (Galdo, Smith and Black 2008, section 4.2, as
  kmatch reads it). Deviating from kmatch, which adds a penalty for units
  lost to common support, a penalty on h above the score's SD and a step
  search: the criterion averages over a fixed set of units (those with a
  neighbor at the smallest h on the grid, so every h is judged on the same
  units), no penalties, and the grid is explicit: `CrossValidation.grid`, or by default
  the pair-distance bandwidth times 1.5^k, k = -4..4, extended four steps at a time on
  a side whose edge holds the minimum (up to 1.5^+-20), then refined by four
  rounds of midpoints around the minimum (kmatch also refines); a warning
  if the minimum stays at an edge. Every evaluated bandwidth is in `.cv`.
- ``"cross_validation"`` without an outcome (kmatch's ``bwidth(cv)`` with no variable,
  read from its Mata code: Kmatch_cv, Kmatch_cvs, Kmatch_i_PS_cvs): the
  criterion is the balance of the score's mean. With M0 the focal units'
  mean score, C1_i a matched focal unit's kernel- (or ridge-) weighted mean
  score of its window, and M1_j the mean of the C1 with other-arm unit j
  left out (in kmatch's closed form, and a focal unit whose only neighbor
  is j dropped), the criterion is the mean over the other arm of
  (M0 - M1_j)^2. It needs each pair's weight, so the window's pairs are
  formed (kernels/kernel.py ``score_cv_pass``); the frames return sums
  (`frame_score_cv`), so memory stays bounded by stratum. kmatch applies
  no support-loss penalty here; ``penalty="support_loss"`` adds its limit
  factor and its search, the default neither.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ._prep import ROW, SCORE, STRATUM, with_strata
from ._spec import Spec
from .kernels import kernel as kk

# K(u) on [-1, 0) and [0, 1) as polynomial coefficients in u (kmatch's
# definitions, support +/- 1). Cosine, Parzen and the Gaussian are not
# polynomials on each side, so the window moments cannot give them.
KERNELS = {
    "epan": ([0.75, 0.0, -0.75], [0.75, 0.0, -0.75]),
    "rectangle": ([0.5], [0.5]),
    "triangle": ([1.0, 1.0], [1.0, -1.0]),
    "biweight": ([0.9375, 0.0, -1.875, 0.0, 0.9375],) * 2,
    "triweight": ([1.09375, 0.0, -3.28125, 0.0, 3.28125, 0.0, -1.09375],) * 2,
}
KERNEL_ALIASES = {"uniform": "rectangle", "epanechnikov": "epan"}
# Seifert and Gasser's (2000) ridge parameters, kmatch's defaults
RIDGE_DEFAULT = {"epan": 5 / 16, "rectangle": 1 / 4, "triangle": 3 / 8, "biweight": 21 / 64,
                 "triweight": 429 / 1280}
CV_STEPS = np.arange(-4, 5)
CV_STEP = 1.5
CV_EXTEND = 4          # grid steps added on a side whose edge holds the minimum
CV_LIMIT = 20          # at most pm x 1.5^+-20
CV_REFINE = 4          # rounds of midpoints around the grid's minimum


@dataclass
class Bandwidth:
    """The bandwidth used and how it was chosen. For "cross_validation": the grid and the
    criterion at each grid point (`cv`)."""

    h: float
    rule: str
    pair_distance: float | None = None
    cv: pl.DataFrame | None = None
    params: dict = field(default_factory=dict)


def kernel_poly(spec: Spec):
    cL, cR = KERNELS[spec.kernel]
    return np.asarray(cL, np.float64), np.asarray(cR, np.float64)


def ridge_param(spec: Spec) -> float:
    if spec.method != "ridge":
        return -1.0
    return RIDGE_DEFAULT[spec.kernel] if spec.ridge is None else float(spec.ridge)


@dataclass
class _Arms:
    """A frame of whole strata split into the focal and other arm, each
    sorted by (stratum, score), with stratum offsets."""

    f_idx: np.ndarray
    f_s: np.ndarray
    f_g: np.ndarray
    f_starts: np.ndarray
    o_idx: np.ndarray
    o_s: np.ndarray
    o_g: np.ndarray
    o_starts: np.ndarray


def _arms(spec: Spec, df: pl.DataFrame) -> tuple[pl.DataFrame, _Arms]:
    df, n_strata = with_strata(spec, df)
    focal = df[spec.treatment].to_numpy()
    if spec.estimand == "atc":
        focal = ~focal
    g = df[STRATUM].to_numpy()
    s = df[SCORE].to_numpy().astype(np.float64)
    out = []
    for sel in (focal, ~focal):
        idx = np.flatnonzero(sel)
        idx = idx[np.lexsort((s[idx], g[idx]))]
        out += [idx, np.ascontiguousarray(s[idx]), np.ascontiguousarray(g[idx]),
                np.searchsorted(g[idx], np.arange(n_strata + 1))]
    return df, _Arms(*out)


def frame_gaps(spec: Spec, df: pl.DataFrame) -> np.ndarray:
    """Distances of each focal unit to its nearest other-arm unit (finite only)."""
    _, a = _arms(spec, df)
    d = kk.nearest_gap(a.o_s, a.o_starts, a.f_s, a.f_g)
    return d[np.isfinite(d)]


def quantile_type2(x: np.ndarray, q: float) -> float:
    """Stata's default quantile (moremata's mm_quantile definition 2,
    Hyndman and Fan's type 2): the inverse ECDF, averaging at jumps."""
    x = np.sort(x)
    n = x.shape[0]
    pn = q * n
    j = int(np.floor(pn))
    if pn > j:
        return float(x[min(max(j, 0), n - 1)])          # x[ceil(pn)], 1-based
    return float(0.5 * (x[min(max(j - 1, 0), n - 1)] + x[min(j, n - 1)]))


def pm_bandwidth(gaps: np.ndarray, q: float, f: float) -> float:
    """kmatch's pair-matching bandwidth from the 1:1 distances."""
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        h = 0.0
    elif q >= 1:
        h = float(gaps.max()) * f
    else:
        h = quantile_type2(gaps, q) * f
    # kmatch: h + epsilon(h) + smallestdouble(), Stata's 2^-52 |h| and 2^-1022
    return h + h * 2.0**-52 + 2.0**-1022


def frame_weights(spec: Spec, df: pl.DataFrame, h: float) -> pl.DataFrame:
    """The matched units of a frame: ROW, treated, the raw weight (1 for a
    focal unit; for another unit, the sum of its weights over the focal
    units it serves, each focal unit's weights summing to 1), focal flag,
    the number of units in the window (focal) or of focal units served, and
    the id when ``id=`` was given."""
    df, a = _arms(spec, df)
    cL, cR = kernel_poly(spec)
    r = ridge_param(spec)
    cnt, _, al, be = kk.focal_pass(a.o_s, np.zeros_like(a.o_s), a.o_starts, a.f_s, a.f_g, h,
                                   cL, cR, r)
    w, ccnt = kk.control_pass(a.f_s, al, be, a.f_starts, a.o_s, a.o_g, h, cL, cR)
    fm, om = cnt > 0, ccnt > 0
    rows = df[ROW].to_numpy()
    ft = spec.estimand != "atc"
    parts = [
        pl.DataFrame({"row": rows[a.f_idx[fm]], "treated": np.full(fm.sum(), ft),
                      "weight": np.ones(fm.sum()), "focal": np.ones(fm.sum(), bool),
                      "n": cnt[fm]}),
        pl.DataFrame({"row": rows[a.o_idx[om]], "treated": np.full(om.sum(), not ft),
                      "weight": w[om], "focal": np.zeros(om.sum(), bool), "n": ccnt[om]}),
    ]
    if spec.id is not None:
        ids = df[spec.id]
        parts[0] = parts[0].with_columns(ids.gather(a.f_idx[fm]).alias(spec.id))
        parts[1] = parts[1].with_columns(ids.gather(a.o_idx[om]).alias(spec.id))
    return pl.concat(parts)


def frame_cv(spec: Spec, df: pl.DataFrame, grid: np.ndarray, y: str,
             weighted: bool, ev_h: float | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Per grid bandwidth, the sum of (weighted) squared leave-one-out errors
    over this frame's evaluation units and the sum of their weights; and the
    number of evaluation units (other-arm units with a neighbor of their
    arm within `ev_h`, default the smallest bandwidth of `grid`)."""
    df, a = _arms(spec, df)
    cL, cR = kernel_poly(spec)
    r = ridge_param(spec)
    oy = np.ascontiguousarray(df[y].to_numpy().astype(np.float64)[a.o_idx])
    h0 = float(np.min(grid)) if ev_h is None else float(ev_h)
    cnt0, _ = kk.loo_pass(a.o_s, oy, a.o_starts, a.o_g, h0, cL, cR, r)
    ev = cnt0 > 0
    sse, wsum = np.zeros(grid.shape[0]), np.zeros(grid.shape[0])
    for k, h in enumerate(grid):
        _, fit = kk.loo_pass(a.o_s, oy, a.o_starts, a.o_g, float(h), cL, cR, r)
        if weighted:
            fcnt, _, al, be = kk.focal_pass(a.o_s, oy, a.o_starts, a.f_s, a.f_g, float(h),
                                            cL, cR, r)
            om, _ = kk.control_pass(a.f_s, al, be, a.f_starts, a.o_s, a.o_g, float(h), cL, cR)
        else:
            om = np.ones_like(fit)
        e2 = (oy - fit) ** 2
        sel = ev & np.isfinite(e2)
        sse[k] = float(np.sum(om[sel] * e2[sel]))
        wsum[k] = float(np.sum(om[sel]))
    return sse, wsum, int(ev.sum())


def frame_cv_kmatch(spec: Spec, df: pl.DataFrame, h: float, y: str,
                    weighted: bool) -> np.ndarray:
    """kmatch's cross-validation sums for one bandwidth on this frame:
    (sum of weighted squared leave-one-out errors, sum of weights, other-arm
    units, those with another unit of their arm in the window, focal units,
    those with an other-arm unit in the window). Unweighted, every other-arm
    unit with a leave-one-out fit counts with weight 1; weighted, those used
    as matches at h, with their matching weight (kmatch's _Kmatch_cvo)."""
    df, a = _arms(spec, df)
    cL, cR = kernel_poly(spec)
    r = ridge_param(spec)
    oy = np.ascontiguousarray(df[y].to_numpy().astype(np.float64)[a.o_idx])
    cnt, fit = kk.loo_pass(a.o_s, oy, a.o_starts, a.o_g, float(h), cL, cR, r)
    e2 = (oy - fit) ** 2
    ev = (cnt > 0) & np.isfinite(e2)
    fcnt, _, al, be = kk.focal_pass(a.o_s, oy, a.o_starts, a.f_s, a.f_g, float(h), cL, cR, r)
    if weighted:
        mw, used = kk.control_pass(a.f_s, al, be, a.f_starts, a.o_s, a.o_g, float(h), cL, cR)
        sel = ev & (used > 0)
        num, den = float(np.sum(mw[sel] * e2[sel])), float(np.sum(mw[sel]))
    else:
        num, den = float(np.sum(e2[ev])), float(np.sum(ev))
    return np.array([num, den, a.o_s.shape[0], ev.sum(), a.f_s.shape[0], (fcnt > 0).sum()],
                    np.float64)


def frame_score_cv(spec: Spec, df: pl.DataFrame, h: float):
    """The sums of the score criterion (cross-validation without an
    outcome, kmatch's ``bwidth(cv)`` with no variable) for one bandwidth on
    this frame: the focal units' score total and count, the total and count
    of their matched means C1 (the weighted mean score of the other-arm
    units in the window), the other-arm units' count, and of their
    leave-one-out changes cc1 (kernels/kernel.py ``score_cv_pass``), for
    those that unmatch no focal unit, the count, sum and sum of squares;
    for the few that do (cc2 > 0), the pairs (cc1, cc2) themselves. Sums,
    so that frames of strata can be added."""
    df, a = _arms(spec, df)
    cL, cR = kernel_poly(spec)
    c1, cc1, cc2 = kk.score_cv_pass(a.o_s, a.o_starts, a.f_s, a.f_starts, float(h), cL, cR,
                                    ridge_param(spec))
    ok = np.isfinite(c1)
    z = cc2 == 0
    t = np.array([a.f_s.sum(), a.f_s.size, c1[ok].sum(), ok.sum(), a.o_s.size, z.sum(),
                  cc1[z].sum(), (cc1[z] ** 2).sum()], np.float64)
    return t, np.column_stack([cc1[~z], cc2[~z]])


def score_criterion(parts, h: float, sd: float = 0.0) -> float:
    """kmatch's criterion without an outcome (Kmatch_cvs) from the frames'
    `frame_score_cv` sums: with M0 the focal units' mean score and M1_j the
    mean of their matched means C1 with other-arm unit j left out,
    mean_j (M0 - M1_j)^2 over the other-arm units (those whose removal would
    leave no focal unit matched are left out, as Stata's mean() leaves out
    missing values). `sd` > 0: kmatch's limit factor max(1, (h / sd)^0.015).
    NaN when no focal unit is matched."""
    t = sum((p[0] for p in parts), np.zeros(8))
    extra = np.concatenate([p[1] for p in parts] or [np.zeros((0, 2))])
    f_sum, n_f, s1, n1, n_o, n0, sc, sc2 = t
    if not (n1 > 0 and n_f > 0):
        return np.nan
    m0 = f_sum / n_f
    A = m0 - s1 / n1
    # unit j with cc2 = 0: (A + cc1_j / n1)^2, summed from the moments
    total = n0 * A * A + 2 * A * sc / n1 + sc2 / (n1 * n1)
    count = n0
    if extra.shape[0]:
        den = n1 - extra[:, 1]
        ok = den > 0
        total += float(np.sum((m0 - (s1 - extra[ok, 0]) / den[ok]) ** 2))
        count += int(ok.sum())
    if not count > 0:
        return np.nan
    mise = total / count
    if sd > 0:
        mise *= max(1.0, (h / sd) ** 0.015)
    return float(mise)


def kmatch_criterion(t: np.ndarray, h: float, weighted: bool, sd: float) -> float:
    """kmatch's MISE from the pooled sums of `frame_cv_kmatch`: the mean
    error, times the support-loss penalty (all over evaluable other-arm
    units; weighted: all over supported focal units), times the limit
    max(1, (h / SD of the score)^0.015). NaN when nothing is evaluable."""
    num, den, n_o, n_o_ev, n_f, n_f_sup = t
    if not den > 0:
        return np.nan
    mise = num / den
    if weighted:
        mise *= n_f / n_f_sup if n_f_sup > 0 else np.nan
    else:
        mise *= n_o / n_o_ev
    if sd > 0:
        mise *= max(1.0, (h / sd) ** 0.015)
    return float(mise)


def kmatch_search(start: float, crit, n: int = 15, f: float = 1.5):
    """kmatch's bandwidth search (Kmatch_cv_search): from the pm bandwidth,
    steps up by `f` (down if the first step is worse) until the criterion
    rises, then halves the bracket around the best point, n evaluations in
    all. Returns the grid and criteria in evaluation order."""
    grid, mise = np.full(n, np.nan), np.full(n, np.nan)
    up, mn, l, ll = 1, 0, 0, 0
    tiny = np.finfo(float).tiny

    def lt(a, b):                        # Stata: missing is larger than any number
        return (not np.isnan(a)) and (np.isnan(b) or a < b)
    for i in range(n):
        if i == 0:
            g = start
        elif i == 1:
            g = grid[0] * f
        elif up == 1:
            if lt(mise[i - 1], mise[i - 2]):
                g = grid[i - 1] * f
            elif i == 2:
                ll, g, up = 0, grid[0] / f, 0
            else:
                up, mn, l, ll = None, i - 2, i - 1, i - 3
                g = (grid[mn] + grid[l]) / 2
        elif up == 0:
            if lt(mise[i - 1], mise[i - 2]):
                g = grid[i - 1] / f
            else:
                up = None
                ll = 0 if ll == 0 else i - 3
                mn, l = i - 2, i - 1
                g = (grid[mn] + grid[l]) / 2
        else:
            if lt(mise[i - 1], mise[mn]):
                ll, l, mn = l, mn, i - 1
            else:
                l, ll = ll, i - 1
            g = (grid[mn] + grid[l]) / 2
        grid[i] = max(g, tiny)
        mise[i] = crit(grid[i])
    return grid, mise


def kmatch_pick(grid: np.ndarray, mise: np.ndarray) -> float:
    """kmatch's choice: the smallest criterion (ties: the smaller bandwidth);
    if every criterion is missing, the largest bandwidth."""
    if np.all(np.isnan(mise)):
        return float(np.max(grid))
    order = np.lexsort((grid, np.where(np.isnan(mise), np.inf, mise)))
    return float(grid[order[0]])


def choose_bandwidth(spec: Spec, frames, score_sd: float = 0.0) -> Bandwidth:
    """The bandwidth for the match: `frames` is a callable that yields the
    frames of whole strata (called once per pass needed)."""
    bw = spec.bandwidth
    params = {"quantile": spec.bandwidth_quantile, "factor": spec.bandwidth_factor}
    if isinstance(bw, float):
        return Bandwidth(h=bw, rule="fixed")
    gaps = np.concatenate([frame_gaps(spec, f) for f in frames()] or [np.zeros(0)])
    pm = pm_bandwidth(gaps, spec.bandwidth_quantile, spec.bandwidth_factor)
    if not pm > 0:
        raise ValueError("no focal unit has another-arm unit at a positive distance in its "
                         "stratum, so the pair-distance bandwidth is 0; give bandwidth= a number")
    if bw == "pair_distance":
        return Bandwidth(h=pm, rule="pair_distance", pair_distance=pm, params=params)
    if spec.cv_outcome is None:
        return _score_bandwidth(spec, frames, pm, params, score_sd)
    if spec.cv_penalty == "support_loss":
        w = spec.cv_weighted

        def crit(h):
            t = sum((frame_cv_kmatch(spec, f, h, spec.cv_outcome, w) for f in frames()),
                    np.zeros(6))
            return kmatch_criterion(t, h, w, score_sd)
        if spec.cv_grid is not None:
            grid = np.asarray(spec.cv_grid, np.float64)
            mise = np.array([crit(h) for h in grid])
        else:
            grid, mise = kmatch_search(pm, crit)
        table = pl.DataFrame({"bandwidth": grid, "criterion": mise})
        params |= {"outcome": spec.cv_outcome, "weighted": w, "penalty": "support_loss"}
        return Bandwidth(h=kmatch_pick(grid, mise), rule="cross_validation",
                         pair_distance=pm, cv=table, params=params)

    def evaluate(grid, ev_h=None):
        sse, wsum, n_ev = np.zeros(grid.size), np.zeros(grid.size), 0
        for f in frames():
            a, b, n = frame_cv(spec, f, grid, spec.cv_outcome, spec.cv_weighted, ev_h)
            sse += a
            wsum += b
            n_ev += n
        with np.errstate(invalid="ignore", divide="ignore"):
            crit = np.where(wsum > 0, sse / wsum, np.nan)
        if not np.isfinite(crit).any():
            raise ValueError("cross-validation found no unit with a neighbor of its own arm "
                             "at the smallest bandwidth on the grid; widen CrossValidation(grid=)")
        return crit, wsum, n_ev

    if spec.cv_grid is not None:
        grid = np.unique(np.asarray(spec.cv_grid, np.float64))
        if not (grid.size and np.all(grid > 0)):
            raise ValueError("CrossValidation grid must hold positive bandwidths")
        crit, wsum, n_ev = evaluate(grid)
        k = int(np.nanargmin(crit))
        if k in (0, grid.size - 1):
            warnings.warn(f"the cross-validated bandwidth {grid[k]:.6g} is at the edge of the "
                          f"grid [{grid[0]:.6g}, {grid[-1]:.6g}]; extend "
                          "CrossValidation(grid=)", stacklevel=3)
    else:
        # the pm bandwidth times 1.5^k, k = -4..4, extended by CV_EXTEND steps on
        # a side while the minimum sits at that edge (kmatch's search also steps
        # by 1.5 until the criterion turns up). Every pass is recomputed on the
        # whole grid, so the units judged are those of its smallest bandwidth.
        lo, hi = int(CV_STEPS[0]), int(CV_STEPS[-1])
        while True:
            grid = pm * CV_STEP ** np.arange(lo, hi + 1, dtype=np.float64)
            crit, wsum, n_ev = evaluate(grid)
            k = int(np.nanargmin(crit))
            if k == grid.size - 1 and hi < CV_LIMIT:
                hi = min(hi + CV_EXTEND, CV_LIMIT)
            elif k == 0 and lo > -CV_LIMIT:
                lo = max(lo - CV_EXTEND, -CV_LIMIT)
            else:
                break
        if k in (0, grid.size - 1):
            warnings.warn(f"the cross-validated bandwidth {grid[k]:.6g} is at the edge of the "
                          f"search range (pair distance x 1.5^{CV_LIMIT}); give "
                          "CrossValidation(grid=)", stacklevel=3)
        else:
            # refine between the minimum's neighbors: each round evaluates the
            # midpoints on either side of the current best and keeps the best
            # of the bracket (as kmatch's search refines), on the same units
            ev_h = float(grid[0])
            pts = dict(zip(grid.tolist(), zip(crit.tolist(), wsum.tolist())))
            left, best, right = grid[k - 1], grid[k], grid[k + 1]
            for _ in range(CV_REFINE):
                new = np.array([(left + best) / 2, (best + right) / 2])
                c2, w2, _ = evaluate(new, ev_h)
                pts |= dict(zip(new.tolist(), zip(c2.tolist(), w2.tolist())))
                cand = sorted([left, new[0], best, new[1], right])
                vals = [pts[x][0] for x in cand]
                j = int(np.nanargmin(vals[1:4])) + 1
                left, best, right = cand[j - 1], cand[j], cand[j + 1]
            grid = np.array(sorted(pts))
            crit = np.array([pts[x][0] for x in grid])
            wsum = np.array([pts[x][1] for x in grid])
            k = int(np.searchsorted(grid, best))
    table = pl.DataFrame({"bandwidth": grid, "criterion": crit, "weight": wsum})
    params |= {"outcome": spec.cv_outcome, "weighted": spec.cv_weighted, "n_eval": n_ev}
    return Bandwidth(h=float(grid[k]), rule="cross_validation", pair_distance=pm, cv=table,
                     params=params)


def _score_bandwidth(spec: Spec, frames, pm: float, params: dict, score_sd: float) -> Bandwidth:
    """Cross-validation without an outcome: kmatch's balance criterion on the
    score (`score_criterion`). With penalty="support_loss", kmatch's limit
    factor and its 15-step search (or the grid); by default no limit
    factor, and the default search of the outcome criterion (the pm
    bandwidth times 1.5^k, extended while the minimum is at an edge, then
    refined). kmatch has no support-loss penalty here."""
    kmatch = spec.cv_penalty == "support_loss"

    def crit(h):
        return score_criterion([frame_score_cv(spec, f, h) for f in frames()], h,
                               score_sd if kmatch else 0.0)
    params |= {"outcome": None, "criterion": "score balance",
               "penalty": "support_loss" if kmatch else None}
    if spec.cv_grid is not None:
        grid = np.unique(np.asarray(spec.cv_grid, np.float64))
        if not (grid.size and np.all(grid > 0)):
            raise ValueError("CrossValidation grid must hold positive bandwidths")
        mise = np.array([crit(h) for h in grid])
    elif kmatch:
        grid, mise = kmatch_search(pm, crit)
    else:
        pts = {}

        def ev(hs):
            for h in hs:
                if h not in pts:
                    pts[h] = crit(h)
        lo, hi = int(CV_STEPS[0]), int(CV_STEPS[-1])
        while True:
            g = pm * CV_STEP ** np.arange(lo, hi + 1, dtype=np.float64)
            ev(g.tolist())
            vals = np.array([pts[h] for h in g])
            if np.all(np.isnan(vals)):
                raise ValueError("cross-validation found no matched focal unit at any "
                                 "bandwidth searched; give CrossValidation(grid=)")
            k = int(np.nanargmin(vals))
            if k == g.size - 1 and hi < CV_LIMIT:
                hi = min(hi + CV_EXTEND, CV_LIMIT)
            elif k == 0 and lo > -CV_LIMIT:
                lo = max(lo - CV_EXTEND, -CV_LIMIT)
            else:
                break
        if k in (0, g.size - 1):
            warnings.warn(f"the cross-validated bandwidth {g[k]:.6g} is at the edge of the "
                          f"search range (pair distance x 1.5^{CV_LIMIT}); give "
                          "CrossValidation(grid=)", stacklevel=4)
        else:
            left, best, right = g[k - 1], g[k], g[k + 1]
            for _ in range(CV_REFINE):
                new = [(left + best) / 2, (best + right) / 2]
                ev(new)
                cand = sorted([left, new[0], best, new[1], right])
                vals = [pts[x] for x in cand]
                j = int(np.nanargmin(vals[1:4])) + 1
                left, best, right = cand[j - 1], cand[j], cand[j + 1]
        grid = np.array(sorted(pts))
        mise = np.array([pts[h] for h in grid])
    table = pl.DataFrame({"bandwidth": grid, "criterion": mise})
    return Bandwidth(h=kmatch_pick(grid, mise), rule="cross_validation",
                     pair_distance=pm, cv=table, params=params)


def run_kernel(spec: Spec, nar, memory: str, workdir=None):
    """The bandwidth and the matched units' weights, on either memory path."""
    if memory == "in_memory" or not spec.exact:
        df = nar.lf.collect(engine="streaming")

        def frames():
            yield df
    else:
        from hdfe_stream.core.store import PolarsEngine, iter_groups
        from hdfe_stream.core.workspace import Run

        from ._match import BATCH_ROWS, base_dir

        run_dir = Run(base_dir(workdir), auto_cleanup=True)
        path = run_dir.path / "narrow.parquet"
        engine = PolarsEngine()
        engine.sink(nar.lf, str(path), "match-stream narrow projection")

        def frames():
            yield from iter_groups(engine, path, spec.exact, BATCH_ROWS, run_dir.path,
                                   nar.n_strata)
    try:
        bw = choose_bandwidth(spec, frames, nar.score_sd)
        parts = [frame_weights(spec, f, bw.h) for f in frames()]
    finally:
        if memory != "in_memory" and spec.exact:
            run_dir.cleanup()
    parts = [p for p in parts if p.height]
    units = (pl.concat(parts).sort("row") if parts else
             pl.DataFrame(schema={"row": pl.Int64, "treated": pl.Boolean, "weight": pl.Float64,
                                  "focal": pl.Boolean, "n": pl.Int64}))
    return bw, units
