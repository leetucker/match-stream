"""The Abadie-Imbens (2016) correction of the matching variance for an
estimated propensity score.

Matching on a score that was estimated in a first step (a logit or probit
of the treatment on z, coefficients gamma) changes the estimator's large-
sample variance. Abadie and Imbens (2016) give the adjustment; this module
computes it exactly as Stata's ``teffects psmatch`` does (its Methods and
formulas, [TE] teffects nnmatch; reproduced to 1e-14 in
tests/test_teffects.py):

    ATE:   V_adj = V - c' V_g c
    ATET:  V_adj = V - c_d' V_g c_d + (d delta / d gamma)' V_g (d delta / d gamma)

V is the Abadie-Imbens (2006) variance with heteroskedastic conditional
variances, V_g the score model's variance (the inverse of its observed
information), and, with f = dF/d(z'gamma), p = F(z'gamma):

    c    = 1/n  sum_i f_i [cov_i(z, y1) / p_i + cov_i(z, y0) / (1 - p_i)]
    c_d  = 1/n1 sum_i z_i f_i (ybar1_-i - ybar0_-i - delta)
         + 1/n1 sum_i f_i [cov_i(z, y1) + p_i / (1 - p_i) cov_i(z, y0)]
    d delta / d gamma = 1/n1 sum_i z_i f_i [(2 t_i - 1)(y_i - ybar_i^z) - delta]

- cov_i(z, y_a) is the sample covariance (divisor size - 1) over a set of
  units of arm a near unit i on the score: if i is in arm a, i itself and
  its h - 1 nearest units of a; otherwise its h nearest units of a (ties at
  the last distance kept).
- ybar_a_-i is the mean outcome of the h nearest units of arm a, i itself
  left out.
- ybar_i^z is the mean outcome of i's k nearest units of the other arm on
  the score model's covariates z (Euclidean distance; k the number of
  matches; a k-d tree, scipy's cKDTree, queried in parallel).

Every term is a sum X~' v over units, X~ the score model's design and v a
number per unit (a set's covariance cov(z, y) = sum_j z_j (y_j - ybar) /
(size - 1) adds its members' outcome deviations to their v), so each
quadratic form is v' X~ V_g X~' v: computed from the observed information
without absorbed fixed effects, and with them by a fixed-effect regression
(`BinaryFirstStage.quad`, out of core, weighted by the observed
information). With fixed effects the covariate match uses
the covariates only, within the exact strata.

Here h = n_variance_matches + 1, and the conditional variances inside V use n_variance_matches
other units: teffects psmatch's ``vce(robust, nn(h))`` counts the unit
itself in its h, where teffects nnmatch's ``nn(J)`` and R Matching's
``Var.calc`` (n_variance_matches here) do not. With exact strata every set is found
within the unit's stratum (teffects psmatch has no exact matching: an
extension, not checked against it). The ATC is the ATET of the flipped
treatment (score 1 - p), which teffects does not offer.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numba as nb
import numpy as np
import polars as pl
from hdfe_stream.core.utils import as_lazy
from scipy.spatial import cKDTree

from . import _threads
from ._core.mestimation import BinaryFirstStage
from ._prep import ROW, SCORE, STRATUM, with_strata
from .kernels import nn

_EMPTY = np.zeros(0)


def _near(qs, ps, s, g, n_strata, k):
    """For each query unit of `qs`, its k nearest units of `ps` of the same
    stratum on the score, ties at the k-th distance kept (the parallel numba
    kernel). `qs` and `ps` are unit indices sorted by (stratum, score), as
    `_sorted` gives them. Returns CSR (offsets, unit indices) in `qs`'s
    order: callers take the queries in that order rather than have the sets
    put back in theirs (a sort of every set member)."""
    starts = np.searchsorted(g[ps], np.arange(n_strata + 1))
    tstarts = np.searchsorted(g[qs], np.arange(n_strata + 1))
    off, pos = nn.knn_replace_ties(tstarts, np.ascontiguousarray(s[qs]),
                                   np.zeros((qs.size, 0)), starts,
                                   np.ascontiguousarray(s[ps]),
                                   np.zeros((ps.size, 0)), np.inf, _EMPTY, int(k), 0.0)
    return off, ps[pos]


def _sorted(order, mask):
    """The units with `mask`, in the order of `order` (all units sorted by
    stratum and score, stably): the same as sorting them alone, without the
    sort."""
    return order[mask[order]]


@nb.njit(parallel=True, cache=True)
def _set_means(off, mem, y, skip):
    """The mean of y over each set of the CSR (offsets, members): NaN for
    an empty set. With `skip` (one unit per set; empty for none), that
    unit is left out of its set: the leave-one-out mean. Each set's sum is
    taken in its members' order, as NumPy's bincount would."""
    n = off.size - 1
    out = np.empty(n)
    leave = skip.size > 0
    for i in nb.prange(n):
        tot, c = 0.0, 0
        for j in range(off[i], off[i + 1]):
            u = mem[j]
            if leave and u == skip[i]:
                continue
            tot += y[u]
            c += 1
        out[i] = tot / c if c > 0 else np.nan
    return out


@nb.njit(cache=True)
def _cov_terms(off, mem, y, kappa, acc):
    """acc_j += kappa_i (y_j - ybar_i) / (size_i - 1) for every member j of
    set i of more than one unit, in order (one thread: sets share members;
    the same additions, in the same order, as NumPy's bincount)."""
    for i in range(off.size - 1):
        a, b = off[i], off[i + 1]
        c = b - a
        if c <= 1:
            continue
        tot = 0.0
        for j in range(a, b):
            tot += y[mem[j]]
        ybar = tot / c
        scale = kappa[i] / (c - 1)
        for j in range(a, b):
            acc[mem[j]] += scale * (y[mem[j]] - ybar)


def _mean(off, mem, y, without=None):
    """Each set's mean of y; with `without`, a unit per set left out of it
    (`_set_means`)."""
    skip = np.zeros(0, np.int64) if without is None else np.ascontiguousarray(without, np.int64)
    return _set_means(off, np.ascontiguousarray(mem, np.int64), np.ascontiguousarray(y, np.float64),
                      skip)


def _add_cov(v, off, mem, y, kappa):
    """v_j += kappa_i (y_j - ybar_i) / (size_i - 1) for every member j of
    set i: then sum_i kappa_i cov_i(z, y) = sum_j z_j v_j. Sets of one
    unit have no covariance and add nothing."""
    acc = np.zeros(v.size)
    _cov_terms(off, np.ascontiguousarray(mem, np.int64), np.ascontiguousarray(y, np.float64),
               np.ascontiguousarray(kappa, np.float64), acc)
    v += acc


def _units(ms, df: pl.DataFrame, first: BinaryFirstStage):
    """For every unit used: treatment, stratum, score, eta, p, f, y and the
    score model's covariates z (the design without fixed effects)."""
    src = (as_lazy(ms._source).with_row_index(ROW).with_columns(pl.col(ROW).cast(pl.Int64)))
    keep = df.select(ROW).lazy()
    lf = src.join(keep, on=ROW, how="inner")
    if first.fe:
        lf = first.with_eta(lf)
    lf = lf.select(ROW, *([pl.col("__eta")] if first.fe else []),
                   *[x.cast(pl.Float64).alias(f"__zx{j}") for j, x in enumerate(first.x)])
    # the source's rows first, then the join: as one query (df's join to the
    # source's, sorted) Polars held 1.7 GB on 5 million units, against 0.15
    # and 0.4 GB in two
    z = lf.collect(engine="streaming")
    d = df.join(z, on=ROW, how="left").sort(ROW)
    del z
    d, n_strata = with_strata(ms.spec, d)
    Z = d.select([f"__zx{j}" for j in range(len(first.x))]).to_numpy()
    eta = d["__eta"].to_numpy() if first.fe else Z @ first.beta
    if first.link == "logit":
        from scipy.special import expit
        p = expit(eta)
        f = p * (1 - p)
    else:
        from scipy.special import ndtr
        p = ndtr(eta)
        f = np.exp(-0.5 * eta * eta) / np.sqrt(2 * np.pi)
    return dict(t=d[ms.spec.treatment].to_numpy().astype(bool), g=d[STRATUM].to_numpy(),
                s=d[SCORE].to_numpy().astype(np.float64), p=p, f=f, eta=eta,
                y=d["__y"].to_numpy(), Z=np.ascontiguousarray(Z), n_strata=n_strata,
                rows=d[ROW].to_numpy())


def _sets(u, h, kappa, loo, fallback=None):
    """Add to v the covariance terms (kappa[a]: per unit, the coefficient of
    cov_i(z, y_a)), each unit's own-arm set being itself and its h - 1
    nearest, its other-arm set the h nearest; with `loo`, also return the
    leave-one-out means of y over the h nearest of each arm.

    A unit alone in its arm within its stratum has an own-arm set of itself
    alone: no covariance, and no leave-one-out mean (NaN). `fallback`
    (effect's lone_unit_variance) fills it: "nearest", its own-arm sets from the
    nearest units of its arm in any stratum; "pooled", its whole arm."""
    t, g, s, y = u["t"], u["g"], u["s"], u["y"]
    n = t.size
    v = np.zeros(n)
    ybar = {True: np.zeros(n), False: np.zeros(n)}
    order = np.lexsort((s, g))              # once: every search's units are a subset
    for a in (True, False):
        arm = t == a
        pts = _sorted(order, arm)
        for own in (True, False):
            q = _sorted(order, arm if own else ~arm)
            if q.size == 0 or pts.size == 0:
                continue
            # own arm: the h nearest include the unit itself, at distance 0
            off, mem = _near(q, pts, s, g, u["n_strata"], h)
            _add_cov(v, off, mem, y, kappa[a][q])
            if loo:
                k = h + 1 if own else h            # own arm: h others besides i
                off2, mem2 = _near(q, pts, s, g, u["n_strata"], k)
                ybar[a][q] = _mean(off2, mem2, y, without=q if own else None)
            if own and fallback is not None and pts.size > 1:
                lone = q[np.bincount(g[pts], minlength=u["n_strata"])[g[q]] == 1]
                if lone.size:
                    _lone_sets(v, ybar[a], lone, pts, s, y, kappa[a], h, loo, fallback)
    return v, ybar


def _lone_sets(v, ybar, lone, pts, s, y, kappa, h, loo, fallback):
    """The own-arm sets of the units alone in their arm and stratum (their
    within-stratum sets, of themselves only, added nothing), from `pts`,
    their whole arm: the nearest on the score in any stratum, or all."""
    if fallback == "nearest":
        # one stratum: sorted by the score alone, ties in unit order (as
        # `_sorted`; the callers' arrays are in stratum order)
        zero = np.zeros(s.size, np.int64)
        lone, pts = np.sort(lone), np.sort(pts)
        lone = lone[np.argsort(s[lone], kind="stable")]
        pts = pts[np.argsort(s[pts], kind="stable")]
        off, mem = _near(lone, pts, s, zero, 1, h)
        _add_cov(v, off, mem, y, kappa[lone])
        if loo:
            off2, mem2 = _near(lone, pts, s, zero, 1, h + 1)
            ybar[lone] = _mean(off2, mem2, y, without=lone)
        return
    # pooled: every lone unit's set is the whole arm, one covariance for all
    n_a = pts.size
    yp = y[pts]
    v[pts] += kappa[lone].sum() * (yp - yp.mean()) / (n_a - 1)
    if loo:
        ybar[lone] = (yp.sum() - y[lone]) / (n_a - 1)


def _z_matches(u, k):
    """ybar_i^z: the mean outcome of each unit's k nearest other-arm units of
    its stratum on the covariates (Euclidean), by a k-d tree for each arm of
    each stratum, the strata's trees built and queried in parallel threads
    (scipy's cKDTree releases the GIL). A unit whose stratum has fewer than
    k other-arm units averages over those it has, and one with none gets 0;
    a unit with no other-arm unit in any stratum, NaN."""
    t, g, Z, y = u["t"], u["g"], u["Z"], u["y"]
    # constant columns (the intercept) do not move a Euclidean distance
    X = np.ascontiguousarray(Z[:, np.ptp(Z, axis=0) > 0]) if Z.size else Z
    out = np.full(t.size, np.nan)
    order = np.argsort(g, kind="stable")
    bounds = np.searchsorted(g[order], np.arange(u["n_strata"] + 1))
    workers = _threads.WORKERS if _threads.WORKERS > 0 else (os.cpu_count() or 1)
    n_pool = max(1, min(workers, u["n_strata"]))
    per_tree = max(1, workers // n_pool)        # few strata: each query in parallel

    def stratum(j, a):
        units = order[bounds[j]:bounds[j + 1]]
        q, o = units[t[units] == a], units[t[units] != a]
        if q.size == 0:
            return
        if o.size == 0:
            out[q] = 0.0
            return
        kk = min(int(k), o.size)
        _, idx = cKDTree(X[o]).query(X[q], k=kk, workers=per_tree)
        out[q] = y[o[idx.reshape(q.size, kk)]].mean(axis=1)

    for a in (True, False):
        if not (t == a).any() or not (t != a).any():
            continue
        if n_pool == 1:
            for j in range(u["n_strata"]):
                stratum(j, a)
        else:
            with ThreadPoolExecutor(n_pool) as pool:
                list(pool.map(stratum, range(u["n_strata"]), [a] * u["n_strata"]))
    return out


def _information(u, link) -> np.ndarray:
    """The score model's observed information over the units used (as
    _core.mestimation.BinaryFirstStage.information; Stata's logit and probit
    report its inverse as their variance). Without fixed effects."""
    Z, p, f, t, eta = u["Z"], u["p"], u["f"], u["t"].astype(np.float64), u["eta"]
    if link == "logit":
        a = f
    else:
        v = p * (1 - p)
        g = f / v
        gp = (-eta * f * v - f * f * (1 - 2 * p)) / (v * v)
        a = f * g - (t - p) * gp
    return (Z * a[:, None]).T @ Z


def _quads(ms, u, first, vs: dict, workdir, u_fit=None) -> dict:
    """v' X~ V_g X~' v for each named per-unit v (over the units of `u`, as
    the score model saw them: not the ATC's flipped treatment). V_g is the
    first stage's own variance, over `u_fit`, the units it was fitted on,
    when they are more than `u`'s (discard= without reestimate_score)."""
    u_fit = u if u_fit is None else u_fit
    if not first.fe:
        H = _information(u_fit, first.link)
        out = {}
        for k, v in vs.items():
            J = u["Z"].T @ v
            out[k] = float(J @ np.linalg.solve(H, J))
        return out
    src = (as_lazy(ms._source).with_row_index(ROW).with_columns(pl.col(ROW).cast(pl.Int64)))
    vf = pl.DataFrame({ROW: u["rows"], **{f"__v_{k}": v for k, v in vs.items()}})
    rows = pl.DataFrame({ROW: u_fit["rows"]}).lazy()
    lf = (src.join(rows, on=ROW, how="semi").join(vf.lazy(), on=ROW, how="left")
          .with_columns([pl.col(f"__v_{k}").fill_null(0.0) for k in vs]))
    return first.quad(lf, {k: pl.col(f"__v_{k}") for k in vs}, workdir=workdir)


def _restrict(u, keep):
    """The units of `u` with `keep` (every per-unit array subset)."""
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
            for k, v in u.items()}


def _full(v, keep):
    out = np.zeros(keep.size)
    out[keep] = v
    return out


def ps_adjustment(ms, df: pl.DataFrame, estimand: str, est: float, n_variance_matches: int,
                  workdir=None, matched_focal: dict | None = None,
                  fit_units: pl.LazyFrame | None = None, lone_unit_variance=None) -> float:
    """The change in the variance of the matching estimate `est` for the
    estimated score (to be added to the Abadie-Imbens variance computed with
    `n_variance_matches` other units). `df` holds every unit used (ROW, treatment,
    SCORE, strata, ``__y``).

    `matched_focal` maps a focal arm (True: treated, False: controls) to the
    ROWs of its matched units, when a caliper left some unmatched. The
    estimate then concerns the matched units, so the correction's sums,
    neighbor sets and counts run over the units the estimate uses (the ATT:
    every control and the matched treated; the ATE: the units matched in
    their own direction), while V_g stays the score model's own, over every
    unit it was fitted on (as trimming in _core.ipw). The caliper's
    dependence on the estimated score is not accounted for (the restricted
    sample is taken as given, as usual).

    `fit_units` (ROW, treatment, SCORE, strata) are the units the score
    model was fitted on when they are more than those matched (units
    discarded for common support without re-estimating): V_g runs over them,
    and the discarded take no part in the second stage, like units a caliper
    left out.

    `lone_unit_variance` (None, "nearest", "pooled") fills the own-arm sets of the
    units alone in their arm within their stratum (see `_sets`)."""
    first = BinaryFirstStage.from_fit(ms.score_fit, ms.spec.treatment)
    if first.link not in ("logit", "probit"):
        raise NotImplementedError(f"the score correction for link {first.link!r}")
    u_all = _units(ms, df, first)
    u_fit = None
    if fit_units is not None:
        u_fit = _units(ms, fit_units.with_columns(pl.lit(0.0).alias("__y"))
                       .collect(engine="streaming"), first)
    keep = np.ones(u_all["t"].size, bool)
    if matched_focal:
        for arm, rows in matched_focal.items():
            arm_mask = u_all["t"] == arm
            keep &= ~arm_mask | np.isin(u_all["rows"], rows)
    u = _restrict(u_all, keep)
    h = int(n_variance_matches) + 1
    p, f = u["p"], u["f"]
    if estimand == "ate":
        n = p.size
        v, _ = _sets(u, h, {True: f / (n * p), False: f / (n * (1 - p))}, loo=False,
                     fallback=lone_unit_variance)
        return -_quads(ms, u_all, first, {"c": _full(v, keep)}, workdir, u_fit)["c"]
    if estimand == "atc":
        # the ATET of the flipped treatment: controls focal, score 1 - p (the
        # same f for a symmetric link), the effect -est; the derivatives flip
        # sign with gamma, so the quadratic forms in the first stage's own
        # variance are unchanged (V_g from the unflipped units: a probit's
        # observed information depends on the treatment)
        u = dict(u, t=~u["t"], p=1 - p)
        p, est = u["p"], -est
    t, y = u["t"], u["y"]
    n1 = t.sum()
    v_cd, ybar = _sets(u, h, {True: f / n1, False: f * p / ((1 - p) * n1)}, loo=True,
                       fallback=lone_unit_variance)
    v_cd += f * (ybar[True] - ybar[False] - est) / n1
    yz = _z_matches(u, ms.spec.n_matches)
    v_dd = f * ((2 * t - 1) * (y - yz) - est) / n1
    # V_g from the units as the score model saw them: all of them, unflipped
    q = _quads(ms, u_all, first, {"cd": _full(v_cd, keep), "dd": _full(v_dd, keep)}, workdir,
               u_fit)
    return -q["cd"] + q["dd"]
