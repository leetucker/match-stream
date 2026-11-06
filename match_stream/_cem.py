"""Coarsened exact matching (Iacus, King and Porro), as MatchIt's method="cem".

CEM is a group-by: each matching covariate is cut into bins, a cell is a
combination of bins and exact-matching values, and a cell is kept when it has
both treated and control units. Two streaming passes over the source (the
covariates' ranges, then the cells' counts) and a third that writes the
matched units' weights; nothing the size of the source is held.

Cutpoints follow MatchIt for a number of bins: equal-width bins over the
covariate's range among the units used (R's ``cut(x, breaks = k)``; a value on
a boundary goes to the upper bin here and to the lower one in R, which matters
only for values exactly on a boundary, of measure zero for continuous data).
A list gives the cut points themselves (bins are left-closed).

Weights (ATT): a treated unit in a matched cell weighs 1; a control weighs
(treated in its cell / controls in its cell) x (matched controls / matched
treated), MatchIt's weights. The ATC swaps the roles.
"""

from __future__ import annotations

import polars as pl
from hdfe_stream.core.utils import as_lazy

from ._spec import Spec

ROW = "__ms_row"
CELL = "__ms_cell"


def _bins(x: str, cut, lo: float, hi: float) -> pl.Expr:
    if isinstance(cut, int):
        if cut < 1:
            raise ValueError(f"cutpoints for {x!r} must be a positive number of bins")
        span = hi - lo
        if span <= 0:
            return pl.lit(0, pl.Int32)
        return ((pl.col(x) - lo) / span * cut).floor().clip(0, cut - 1).cast(pl.Int32)
    edges = sorted(float(e) for e in cut)
    return pl.col(x).cut(edges, left_closed=True).to_physical().cast(pl.Int32)


def cem(spec: Spec, data, cutpoints) -> tuple[pl.DataFrame, dict, pl.LazyFrame]:
    """The matched units' weights (row, id, treated, weight, subclass = cell)
    and the counts."""
    covs = spec.covariates
    if not covs and not spec.exact:
        raise ValueError("CEM needs matching covariates in the formula or exact=")
    if cutpoints is None:
        cutpoints = {}
    elif isinstance(cutpoints, int):
        cutpoints = {c: cutpoints for c in covs}
    unknown = set(cutpoints) - set(covs)
    if unknown:
        raise ValueError("cutpoints names columns that are not matching covariates: "
                         f"{sorted(unknown)}")
    missing_cut = [c for c in covs if c not in cutpoints]
    if missing_cut:
        # MatchIt's default is Sturges' rule through R's hist(); not reproduced
        # exactly, so ask for the number of bins instead of guessing
        raise ValueError(f"give cutpoints for every covariate (a number of bins or the cut "
                         f"points): missing {missing_cut}")

    lf = as_lazy(data).with_row_index(ROW).with_columns(pl.col(ROW).cast(pl.Int64))
    keep = list(dict.fromkeys([ROW, spec.treatment, *spec.exact, *covs,
                               *([spec.id] if spec.id is not None else [])]))
    schema = lf.collect_schema()
    missing = [c for c in keep if c not in schema]
    if missing:
        raise ValueError(f"columns not in the data: {missing}")
    floats = [c for c in keep if schema[c].is_float()]
    ok = pl.all_horizontal([pl.col(c).is_not_null() for c in keep]
                           + [pl.col(c).is_not_nan() for c in floats])
    proj = lf.select(keep)
    n_source = proj.select(pl.len()).collect(engine="streaming").item()
    proj = proj.filter(ok).with_columns(pl.col(spec.treatment).cast(pl.Boolean))
    rng = proj.select([pl.col(c).min().alias(f"lo:{c}") for c in covs]
                      + [pl.col(c).max().alias(f"hi:{c}") for c in covs]
                      + [pl.len().alias("n")]).collect(engine="streaming").row(0, named=True)

    bins = [_bins(c, cutpoints[c], rng[f"lo:{c}"], rng[f"hi:{c}"]).alias(f"__b:{c}") for c in covs]
    keys = [*spec.exact, *[f"__b:{c}" for c in covs]]
    coarse = proj.with_columns(bins)
    t = pl.col(spec.treatment)
    cells = (coarse.group_by(keys)
             .agg(t.sum().cast(pl.Int64).alias("n_t"), (~t).sum().cast(pl.Int64).alias("n_c"))
             .filter((pl.col("n_t") > 0) & (pl.col("n_c") > 0))
             .collect(engine="streaming")
             .sort(keys).with_row_index(CELL, offset=1)
             .with_columns(pl.col(CELL).cast(pl.Int64)))
    mt, mc = int(cells["n_t"].sum()), int(cells["n_c"].sum())
    focal_t = spec.estimand != "atc"
    focal_n, other_n = ("n_t", "n_c") if focal_t else ("n_c", "n_t")
    m_focal, m_other = (mt, mc) if focal_t else (mc, mt)
    is_focal = t if focal_t else ~t
    w = (pl.when(is_focal).then(1.0)
         .otherwise(pl.col(focal_n) / pl.col(other_n) * (m_other / max(m_focal, 1))))
    out = (coarse.join(cells.lazy(), on=keys, how="inner")
           .select(pl.col(ROW).alias("row"), *([spec.id] if spec.id is not None else []),
                   t.alias("treated"), w.alias("weight"), pl.col(CELL).alias("subclass"))
           .collect(engine="streaming").sort("row"))
    n_t = int(proj.select(t.cast(pl.Int64).sum()).collect(engine="streaming").item())
    counts = {"treated": n_t, "control": int(rng["n"]) - n_t, "matched_treated": mt,
              "matched_control": mc, "unmatched_treated": n_t - mt,
              "unmatched_control": int(rng["n"]) - n_t - mc, "cells": cells.height,
              "dropped_missing": n_source - int(rng["n"])}
    return out, counts, proj
