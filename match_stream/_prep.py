"""Pass 0: from the source to the narrow columns the match needs.

The score model (if any) is fitted out of core by hdfe-stream's
`feglm_stream`, and its prediction is added lazily, so the source is never
held in memory. The narrow projection holds the row position in the source,
the treatment, the score, the exact-matching columns, the caliper covariates
and the id: tens of bytes a row. Rows with a missing value in
any of these, or a null score (a fixed-effect level the score model dropped,
as a singleton or separated), are left out and counted.

One streaming query computes what must be global whichever memory path runs:
the counts, the score's standard deviation over all kept units (MatchIt's
caliper unit), the caliper covariates' standard deviations and the number of
strata. The projection is then either collected (``in_memory``) or sunk to
Parquet and read back a batch of whole strata at a time (``by_stratum``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from hdfe_stream.core.utils import as_lazy

from ._spec import DISCARD, Spec

ROW = "__ms_row"            # the row's position in the source (Int64)
SCORE = "__ms_score"
STRATUM = "__ms_stratum"


@dataclass
class Narrow:
    lf: pl.LazyFrame            # ROW, treatment (bool), SCORE, exact, caliper columns, id
    columns: list[str]
    n_source: int               # rows read
    n_kept: int                 # rows with every needed value
    score_sd: float             # SD of the score over all kept units
    cov_sd: dict[str, float] = field(default_factory=dict)
    n_strata: int = 1
    score_fit: object | None = None
    n_treated: int = 0
    zcols: list[str] = field(default_factory=list)   # whitened covariates (metric distances)
    # common support: every unit with its values (the discarded too, with the
    # score MatchIt reports for them), and the discarded by arm
    all_lf: pl.LazyFrame | None = None
    discarded_treated: int = 0
    discarded_control: int = 0
    first_stage_all: bool = False       # the score model was fitted on all_lf's units

    @property
    def n_dropped(self) -> int:
        return self.n_source - self.n_kept

    def bytes_in_memory(self) -> float:
        """Memory to match all strata at once: the narrow columns and the
        sort keys and kernel inputs (the formula `profile=True` uses, so the
        profile's path and memory="auto"'s choice agree)."""
        from .profile import _SORT_BYTES

        return self.n_kept * (8 * len(self.columns) + _SORT_BYTES)


# The score model's convergence tolerance (feglm_stream's iwls_tol, relative
# deviance change). hdfe-stream's default, 1e-8, leaves a probit's
# coefficients about 5e-7 from the maximum (logits converge faster); score
# differences of that size flip some nearest neighbors, so match-stream
# fits to 1e-10: one or two more iterations, and Stata's logit and probit
# reproduced to about 1e-9 (tests/test_teffects.py).
SCORE_TOL = 1e-10


def fit_score(spec: Spec, lf: pl.LazyFrame, workdir=None, verbose: bool = True):
    """The propensity-score model, fitted with feglm_stream (absorbed fixed
    effects after '|'), and the source with the score added lazily."""
    from hdfe_stream import feglm_stream

    fit = feglm_stream(spec.score_model, lf, family=spec.link, workdir=workdir, verbose=verbose,
                       iwls_tol=SCORE_TOL, iwls_maxiter=100)
    kind = "link" if spec.distance == "logit_ps" else "response"
    return fit, fit.predict(lf, type=kind, name=SCORE)


def narrow(spec: Spec, data, workdir=None, verbose: bool = True) -> Narrow:
    lf = as_lazy(data).with_row_index(ROW).with_columns(pl.col(ROW).cast(pl.Int64))
    schema = lf.collect_schema()
    missing = [c for c in spec.columns() if c not in schema]
    if missing:
        raise ValueError(f"columns not in the data: {missing}")

    fit, zcols = None, []
    if spec.fits_score:
        fit, lf = fit_score(spec, lf, workdir, verbose)
    if spec.mahalanobis_vars:
        # Mahalanobis on mahalanobis_vars within the score's caliper: the score as
        # usual, and the mahalanobis_vars whitened (recomputed over the units kept
        # when common support discards some)
        if not spec.fits_score:
            lf = lf.with_columns(pl.col(spec.distance).cast(pl.Float64).alias(SCORE))
        lf, zcols = with_whitened(spec, lf, spec.mahalanobis_vars)
    elif spec.fits_score:
        pass
    elif spec.is_metric:
        W = whitener(spec, lf)
        zcols = [f"__ms_z{q}" for q in range(W.shape[0])]
        lf = lf.with_columns([sum(float(W[a, b]) * pl.col(c).cast(pl.Float64)
                                  for b, c in enumerate(spec.covariates)).alias(zcols[a])
                              for a in range(W.shape[0])] + [pl.lit(0.0).alias(SCORE)])
    else:
        lf = lf.with_columns(pl.col(spec.distance).cast(pl.Float64).alias(SCORE))

    keep = list(dict.fromkeys([ROW, spec.treatment, SCORE, *spec.exact, *spec.cov_calipers,
                               *zcols, *([spec.id] if spec.id is not None else []),
                               *([spec.cv_outcome] if spec.cv_outcome else []),
                               *([spec.discard] if spec.discard not in DISCARD else []),
                               *spec.mahalanobis_vars]))
    proj = lf.select(keep)
    pschema = proj.collect_schema()
    floats = [c for c in keep if pschema[c].is_float()]
    ok = pl.all_horizontal([pl.col(c).is_not_null() for c in keep]
                           + [pl.col(c).is_not_nan() for c in floats])
    t = pl.col(spec.treatment)
    tnum = t.cast(pl.Int8) if pschema[spec.treatment] == pl.Boolean else t
    stats = [pl.len().alias("n_source"), ok.sum().alias("n_kept"),
             pl.col(SCORE).filter(ok).std().alias("sd"),
             tnum.filter(ok).min().alias("tmin"), tnum.filter(ok).max().alias("tmax"),
             tnum.filter(ok).cast(pl.Int64).sum().alias("n_treated")]
    stats += [pl.col(c).filter(ok).std().alias(f"sd:{c}") for c in spec.cov_calipers]
    if spec.exact:
        key = pl.struct(spec.exact) if len(spec.exact) > 1 else pl.col(spec.exact[0])
        stats.append(key.filter(ok).n_unique().alias("n_strata"))
    row = proj.select(stats).collect(engine="streaming").row(0, named=True)

    if row["n_kept"] and (row["tmin"] < 0 or row["tmax"] > 1):
        raise ValueError(f"the treatment {spec.treatment!r} must be 0/1 or boolean")
    out = proj.filter(ok).with_columns(t.cast(pl.Boolean))
    nar = Narrow(lf=out, columns=keep, n_source=int(row["n_source"]), n_kept=int(row["n_kept"]),
                 score_sd=float(row["sd"] or 0.0),
                 cov_sd={c: float(row[f"sd:{c}"] or 0.0) for c in spec.cov_calipers},
                 n_strata=int(row.get("n_strata") or 1), score_fit=fit,
                 n_treated=int(row["n_treated"] or 0), zcols=zcols, all_lf=out)
    if spec.discard != "none":
        _discard(spec, nar, data, keep, ok, workdir, verbose)
    return nar


def _discard(spec: Spec, nar: Narrow, data, keep, ok, workdir, verbose) -> None:
    """Common support, as MatchIt's ``discard`` (its discard(): the bounds
    are strict, a unit at a bound is kept) and ``reestimate_score``: `nar.lf`
    becomes the units kept, `nar.all_lf` every unit (the discarded with
    their first score, as MatchIt reports them), and the caliper scales are
    the standard deviations over the units kept (MatchIt's
    process.caliper)."""
    out = nar.all_lf
    t = pl.col(spec.treatment)
    if spec.discard in DISCARD:
        b = out.select(pl.col(SCORE).filter(t).min().alias("min1"),
                       pl.col(SCORE).filter(t).max().alias("max1"),
                       pl.col(SCORE).filter(~t).min().alias("min0"),
                       pl.col(SCORE).filter(~t).max().alias("max0")) \
            .collect(engine="streaming").row(0, named=True)
        s = pl.col(SCORE)
        if spec.discard == "both":
            lo, hi = max(b["min0"], b["min1"]), min(b["max0"], b["max1"])
            disc = (s < lo) | (s > hi)
        elif spec.discard == "control":
            disc = ~t & ((s < b["min1"]) | (s > b["max1"]))
        else:
            disc = t & ((s < b["min0"]) | (s > b["max0"]))
    else:
        disc = pl.col(spec.discard).cast(pl.Boolean)
    flagged = out.with_columns(disc.fill_null(False).alias("__ms_disc"))
    c = (flagged.select(pl.col("__ms_disc").filter(t).sum().alias("t"),
                        pl.col("__ms_disc").filter(~t).sum().alias("c"))
         .collect(engine="streaming").row(0, named=True))
    nar.discarded_treated, nar.discarded_control = int(c["t"]), int(c["c"])
    kept = flagged.filter(~pl.col("__ms_disc")).drop("__ms_disc")
    nar.first_stage_all = spec.fits_score
    if spec.reestimate_score and (c["t"] or c["c"]):
        # refit on the units kept (MatchIt: the distance options subset to
        # them) and match on the new score; the discarded keep their first
        src = as_lazy(data).with_row_index(ROW).with_columns(pl.col(ROW).cast(pl.Int64))
        src = src.join(kept.select(ROW), on=ROW, how="semi")
        fit, lf2 = fit_score(spec, src, workdir, verbose)
        kept = (lf2.select(keep).filter(ok).with_columns(t.cast(pl.Boolean)))
        nar.score_fit = fit
        nar.first_stage_all = False
        nar.all_lf = pl.concat([kept, flagged.filter(pl.col("__ms_disc")).drop("__ms_disc")],
                               how="vertical_relaxed").sort(ROW)
    if spec.mahalanobis_vars:
        # MatchIt's covariance for mahalanobis_vars is over the units not discarded
        kept, _ = with_whitened(spec, kept, spec.mahalanobis_vars)
    nar.lf = kept
    # calipers in standard deviations: over the units kept
    sd = (kept.select(pl.col(SCORE).std().alias("sd"),
                      *[pl.col(x).std().alias(f"sd:{x}") for x in spec.cov_calipers])
          .collect(engine="streaming").row(0, named=True))
    nar.score_sd = float(sd["sd"] or 0.0)
    nar.cov_sd = {x: float(sd[f"sd:{x}"] or 0.0) for x in spec.cov_calipers}


def with_whitened(spec: Spec, lf: pl.LazyFrame, cols: list[str]):
    """`lf` with the Mahalanobis-whitened `cols` as __ms_z0, __ms_z1, ...
    (MatchIt's pooled within-group covariance over `lf`'s units)."""
    W = whitener(spec, lf, cols, "mahalanobis")
    zcols = [f"__ms_z{q}" for q in range(W.shape[0])]
    return lf.with_columns([sum(float(W[a, b]) * pl.col(c).cast(pl.Float64)
                                for b, c in enumerate(cols)).alias(zcols[a])
                            for a in range(W.shape[0])]), zcols


def whitener(spec: Spec, lf: pl.LazyFrame, cols: list[str] | None = None,
             kind: str | None = None) -> np.ndarray:
    """W with ``z = W x``: Euclidean distance between z is the requested
    distance between x. Mahalanobis uses the pooled within-group covariance
    ((n1-1) S1 + (n0-1) S0) / (n - 2), MatchIt's convention (verified against
    MatchIt 4.8.2); "ivariance" the inverse of each covariate's variance over
    all units; "euclidean" the identity. One streaming pass of sums and
    cross-products per arm, over units with every covariate present."""
    xs = spec.covariates if cols is None else cols
    kind = spec.distance if kind is None else kind
    p = len(xs)
    if kind == "euclidean":
        return np.eye(p)
    t = pl.col(spec.treatment).cast(pl.Boolean)
    ok = pl.all_horizontal([pl.col(c).is_not_null() for c in [spec.treatment, *xs]])
    aggs = []
    for arm, cond in (("t", t), ("c", ~t)):
        sel = ok & cond
        aggs.append(sel.sum().alias(f"n{arm}"))
        aggs += [pl.col(a).cast(pl.Float64).filter(sel).sum().alias(f"s{arm}{i}")
                 for i, a in enumerate(xs)]
        aggs += [(pl.col(a).cast(pl.Float64) * pl.col(b)).filter(sel).sum().alias(f"q{arm}{i}_{j}")
                 for i, a in enumerate(xs) for j, b in enumerate(xs) if j >= i]
    r = lf.select(aggs).collect(engine="streaming").row(0, named=True)
    cov = {}
    for arm in ("t", "c"):
        n = r[f"n{arm}"]
        m = np.array([r[f"s{arm}{i}"] for i in range(p)])
        Q = np.empty((p, p))
        for i in range(p):
            for j in range(i, p):
                Q[i, j] = Q[j, i] = r[f"q{arm}{i}_{j}"]
        cov[arm] = (n, Q - np.outer(m, m) / max(n, 1))       # (n, (n-1) S)
    (nt, At), (nc, Ac) = cov["t"], cov["c"]
    n = nt + nc
    if kind == "ivariance":
        # each covariate's variance over all units: within-arm sums of squares
        # plus the between-arm part
        mt = np.array([r[f"st{i}"] for i in range(p)]) / max(nt, 1)
        mc = np.array([r[f"sc{i}"] for i in range(p)]) / max(nc, 1)
        var = (np.diag(At) + np.diag(Ac) + nt * nc / n * (mt - mc) ** 2) / (n - 1)
        if np.any(var <= 0):
            raise ValueError("a matching covariate has zero variance")
        return np.diag(1 / np.sqrt(var))
    S = (At + Ac) / (n - 2)
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        raise ValueError("the matching covariates' covariance is singular (a covariate is "
                         "constant or a linear combination of others)") from None
    return np.linalg.inv(L)


def with_strata(spec: Spec, df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """`df` with dense stratum codes 0..G-1 (codes are local to `df`: strata
    are independent, so a frame of whole strata can be matched on its own)."""
    if not spec.exact:
        return df.with_columns(pl.lit(0, pl.Int64).alias(STRATUM)), 1
    key = pl.struct(spec.exact) if len(spec.exact) > 1 else pl.col(spec.exact[0])
    df = df.with_columns((key.rank("dense").cast(pl.Int64) - 1).alias(STRATUM))
    return df, (int(df[STRATUM].max()) + 1 if df.height else 0)


def _mix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64's finalizer: a well-mixed 64-bit hash of each value."""
    z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def order_key(spec: Spec, score: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The processing order of the focal units within a stratum (MatchIt's
    m.order). Ties keep data order (the sort is stable on the row position).
    "random" hashes the row with the seed, so the order is the same whichever
    memory path runs and however the strata are batched."""
    if spec.match_order == "largest":
        return -score
    if spec.match_order == "smallest":
        return score
    if spec.match_order == "random":
        seed = np.uint64((spec.seed or 0) & 0xFFFFFFFFFFFFFFFF)
        with np.errstate(over="ignore"):
            return _mix64(rows.astype(np.uint64) ^ _mix64(np.array([seed]))[0]).astype(np.float64)
    return rows.astype(np.float64)      # "data"; "closest" orders pairs inside its kernel
