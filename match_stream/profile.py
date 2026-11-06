"""``profile=True`` in `match_nn`, `match_kernel` and `match_cem`: the memory and disk a
match would need, without matching.

The result is hdfe-stream's `Profile` (``hdfe_stream.core.profile``): a list
of components, each an array or file with an expected size and an upper
bound, added up by phase, printed, and turned into a resource request
(``p.recommend().pbs`` is the PBS ``select=`` line). This module supplies
match-stream's model of what a match holds, from three streaming passes
over the columns matching reads (two when the score is fitted):

1. the rows kept (every needed value present), the score's range and
   standard deviation when it is a column (`hdfe_stream.core.profile.scan_source`
   and one aggregation);
2. per exact stratum: units by arm, which give the largest stratum (the
   memory bound of ``memory="by_stratum"``) and the brute-force pair count;
3. with a score column and a caliper on it (or a fixed kernel bandwidth),
   a 1,000-bin histogram of the score per stratum and arm, which counts the
   candidate pairs within the caliper (an expected figure and an upper
   bound): the work of the search, reported instead of a time estimate.

A fitted score adds the score model's own profile (`feglm_stream(...,
profile=True)`) as the phase "score"; its pairs cannot be counted before it
is fitted. The rows a score model drops (a fixed-effect singleton, say) are
not known before fitting either: the figures count every row with its
values, an upper bound.

What is modeled, per phase "match": the narrow columns (8 bytes a value,
the row position, treatment, score, strata, caliper covariates, whitened
covariates, id), the sort keys and kernel inputs, the pair table (or the
matched units' weights), the k-d tree and its candidate lists for a
multivariate distance, and for ``by_stratum`` the largest batch of strata
in memory and the narrow copy (and its partitioned copy) on disk. Not
modeled: `effect()` and `balance()`, which stream the source again and
hold the units used and their outcome (16 bytes a unit) on top.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from hdfe_stream.core.profile import Component, Profile, baseline_gb, scan_source
from hdfe_stream.core.utils import as_lazy

from ._spec import DISCARD, Spec

_BINS = 1000
_SORT_BYTES = 40        # per unit: stratum code, order key, two permutations, a copy of the score


def narrow_width(spec: Spec, p_z: int) -> int:
    """Bytes a unit of the narrow projection takes (8 per value)."""
    n_cols = 3 + len(spec.exact) + len(spec.cov_calipers) + p_z
    n_cols += (spec.id is not None) + (spec.cv_outcome is not None)
    n_cols += spec.discard not in DISCARD
    return 8 * n_cols


def _needed(spec: Spec) -> list[str]:
    """The source columns the match reads (not the score model's)."""
    if spec.method == "cem":            # cells of the covariates and strata
        return list(dict.fromkeys([spec.treatment, *spec.exact, *spec.covariates,
                                   *([spec.id] if spec.id is not None else [])]))
    cols = [c for c in spec.columns() if c != spec.distance or not spec.fits_score]
    if spec.is_metric:
        cols += [c for c in spec.covariates if c not in cols]
    return list(dict.fromkeys(cols))


def _z_dims(spec: Spec) -> int:
    if spec.mahalanobis_vars:
        return len(spec.mahalanobis_vars)
    return len(spec.covariates) if spec.is_metric else 0


def profile_match(spec: Spec, data, *, cutpoints=None, workdir=None, max_memory_gb=None,
                  memory_reserve=None, n_threads=None, verbose=False) -> Profile:
    """The `Profile` of a match (see the module docstring)."""
    from stream_limits import MemoryBudget, detect_cpus

    from ._match import BATCH_ROWS

    lf = as_lazy(data)
    cols = _needed(spec)
    missing = [c for c in cols if c not in lf.collect_schema()]
    if missing:
        raise ValueError(f"columns not in the data: {missing}")
    score_col = None if spec.fits_score or spec.match_metric and not spec.mahalanobis_vars \
        else (spec.distance if spec.method != "cem" else None)
    scan = scan_source(lf, present=[pl.col(c) for c in cols],
                       minmax=[score_col] if score_col else [])
    n = scan.n_rows
    t = pl.col(spec.treatment).cast(pl.Boolean)
    ok = pl.all_horizontal([pl.col(c).is_not_null() for c in cols])
    kept = lf.filter(ok)

    # pass 2: units by arm per stratum
    if spec.exact:
        tal = (kept.group_by(spec.exact)
               .agg(t.sum().cast(pl.Int64).alias("n_t"), pl.len().cast(pl.Int64).alias("n"))
               .collect(engine="streaming"))
    else:
        tal = kept.select(t.sum().cast(pl.Int64).alias("n_t"),
                          pl.len().cast(pl.Int64).alias("n")).collect(engine="streaming")
    n_t_s = tal["n_t"].to_numpy()
    n_s = tal["n"].to_numpy()
    n_c_s = n_s - n_t_s
    focal_t = spec.estimand != "atc"
    n_f_s = n_t_s if focal_t else n_c_s
    n_o_s = n_c_s if focal_t else n_t_s
    n_f, n_o = int(n_f_s.sum()), int(n_o_s.sum())
    largest = int(n_s.max()) if n_s.size else 0
    notes = {"rows": n, "strata": int(n_s.size), "treated": int(n_t_s.sum()),
             "controls": int(n_c_s.sum()), "largest_stratum_rows": largest,
             "focal_units": n_f, "brute_force_pairs": int((n_f_s * n_o_s).sum())}
    warnings = []

    # pass 3: candidate pairs within the caliper (score column only)
    width = None
    if score_col and spec.method == "nn" and spec.caliper is not None:
        sd = float(kept.select(pl.col(score_col).std()).collect(engine="streaming").item() or 0)
        width = spec.caliper * sd if spec.caliper_std else spec.caliper
    elif score_col and spec.is_kernel and isinstance(spec.bandwidth, float):
        width = spec.bandwidth
    if width is not None and n:
        lo, hi = scan.minmax[score_col]
        lo, hi = float(lo), float(hi)
        bw = (hi - lo) / _BINS if hi > lo else 1.0
        b = ((pl.col(score_col) - lo) / bw).floor().clip(0, _BINS - 1).cast(pl.Int32)
        h = (kept.group_by([*spec.exact, b.alias("__b")])
             .agg(t.sum().cast(pl.Int64).alias("t"), pl.len().cast(pl.Int64).alias("n"))
             .collect(engine="streaming"))
        key = (pl.struct(spec.exact).rank("dense") if spec.exact else pl.lit(1)) \
            .cast(pl.Int64) - 1
        h = h.with_columns(key.alias("__g"))
        G = int(h["__g"].max()) + 1
        M = np.zeros((2, G, _BINS))
        gi, bi = h["__g"].to_numpy(), h["__b"].to_numpy()
        np.add.at(M[0], (gi, bi), h["t"].to_numpy())
        np.add.at(M[1], (gi, bi), h["n"].to_numpy() - h["t"].to_numpy())
        F, O = (M[0], M[1]) if focal_t else (M[1], M[0])
        r = width / bw
        cs = np.concatenate([np.zeros((G, 1)), np.cumsum(O, axis=1)], axis=1)
        idx = np.arange(_BINS)

        def window(radius):
            a = np.clip(idx - radius, 0, _BINS)
            z = np.clip(idx + radius + 1, 0, _BINS)
            return cs[:, z] - cs[:, a]
        up = (F * window(int(np.ceil(r)))).sum()
        # expected: a fraction of the edge bins, the width over the bin
        frac = r - np.floor(r)
        inner = window(int(np.floor(r)))
        edge = window(int(np.floor(r)) + 1) - inner
        exp_ = (F * (inner + frac * edge)).sum()
        notes["candidate_pairs"] = {"expected": float(exp_), "upper": float(up),
                                    "within": float(width)}

    # the match phase's components
    threads = int(n_threads) if n_threads else detect_cpus()
    pz = _z_dims(spec)
    wbytes = narrow_width(spec, pz)
    comps = []
    budget = MemoryBudget(limit_gb=max_memory_gb, reserve=memory_reserve)
    path = spec.memory
    in_mem = n * (wbytes + _SORT_BYTES)
    if path == "auto":
        path = "in_memory" if not spec.exact or budget.fits(in_mem / 1e9) else "by_stratum"
    notes["memory_path"] = path
    rows_mem = n if path == "in_memory" or spec.method == "cem" else \
        min(n, max(largest, BATCH_ROWS) + largest)
    # the narrow frame and the numpy copies the kernels take
    comps.append(Component("narrow columns", 2 * rows_mem * wbytes, 2.5 * rows_mem * wbytes,
                           phases=("match",),
                           formula=f"{rows_mem:,} rows x {wbytes} B, frame and arrays"))
    comps.append(Component("sort keys and kernel inputs", rows_mem * _SORT_BYTES,
                           rows_mem * _SORT_BYTES * 1.6, phases=("match",)))
    if path == "by_stratum" and spec.method != "cem":
        comps.append(Component("narrow copy on disk", n * wbytes * 0.6, n * wbytes * 2.0,
                               where="disk", phases=("match",),
                               formula="the Parquet copy and its partitions by stratum"))
        if largest * wbytes > budget.headroom_gb() * 1e9:
            warnings.append(f"the largest stratum ({largest:,} rows) alone needs more memory "
                            "than is free: use finer exact strata")
    idb = 16 if spec.id is not None else 0
    if spec.method == "nn":
        per = 40 + idb
        k = spec.n_matches
        exp_pairs = n_f * k
        up_pairs = exp_pairs * (2 if spec.keep_ties else 1)
        # built, then joined to the subclasses (a copy)
        comps.append(Component("pair table", 2 * exp_pairs * per, 2 * up_pairs * per,
                               phases=("match",),
                               formula=f"{n_f:,} focal units x {k} x {per} B, twice"))
        if spec.match_metric and pz:
            if pz <= 15:
                tree = n_o * (pz + 1) * 8 * 2
                kk = max(16, 4 * k)
                # the query's distances and indices, the sorted lists and
                # their recomputed distances
                cand = n_f * kk * 32
                comps.append(Component("k-d tree", tree, phases=("match",),
                                       formula=f"{n_o:,} points x {pz + 1} dims"))
                plain = spec.replace and not spec.cov_calipers and not spec.mahalanobis_vars
                comps.append(Component("candidate lists", n_f * (k + 1) * 32 if plain else cand,
                                       n_f * (k + 1) * 40 if plain else cand * 1.25,
                                       phases=("match",)))
            else:
                warnings.append(f"{pz} covariates: brute-force search, "
                                f"{notes['brute_force_pairs']:,} distance evaluations")
    elif spec.is_kernel:
        comps.append(Component("kernel arrays and weights", n * 12 * 8, n * 16 * 8,
                               phases=("match",)))
    else:   # CEM
        comps.append(Component("cells and weights", n * (40 + idb), phases=("match",)))

    prof = Profile(model=f"match-stream: {spec.method}"
                   + (f", {spec.n_matches}:1 {'with' if spec.replace else 'without'} replacement"
                      if spec.method == "nn" else "") + f", {path}",
                   components=comps, scan=scan, threads=threads, baseline_gb=baseline_gb(),
                   memory_limit_gb=budget.limit_gb, memory_reserve=budget.reserve,
                   warnings=warnings, notes=notes)
    if spec.fits_score:
        from hdfe_stream import feglm_stream

        sp = feglm_stream(spec.score_model, data, family=spec.link, workdir=workdir, profile=True,
                          verbose=verbose)
        prof.components = [Component(f"score model: {c.name}", c.expected, c.upper, c.where,
                                     ("score",), c.formula) for c in sp.components] + comps
        prof.notes["score_model"] = sp.model
        prof.warnings = list(sp.warnings) + prof.warnings
    return prof
