"""`match_nn()`, `match_kernel()` and `match_cem()`: the entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import polars as pl

from . import _threads
from ._cem import cem
from ._match import choose_memory, run
from ._prep import narrow
from ._spec import METRICS, SCORE_DISTANCES, parse_spec
from .bandwidth import CrossValidation, PairDistance
from .matches import CEMSet, KernelSet, MatchSet

if TYPE_CHECKING:
    from hdfe_stream.core.profile import Profile

_FOCAL = {"treated": "att", "control": "atc"}

# every argument of `_execute`, with the value a method that does not offer it uses
_DEFAULTS = dict(
    method="nn", exact=None, distance="ps", score_model=None, link="logit", n_matches=1,
    replace=False, match_order=None, seed=None, caliper=None, estimand="att", id=None,
    memory="auto", keep_ties=False, tie_tolerance=0.0, min_matches=1, match_in_rounds=True,
    discard="none", reestimate_score=False, mahalanobis_vars=None, kernel="epan", ridge=None,
    bandwidth=None, cutpoints=None, workdir=None, max_memory_gb=None, memory_reserve=None,
    n_threads=None, profile=False, verbose=True)


def _estimand(focal: str) -> str:
    if focal not in _FOCAL:
        raise ValueError(f"focal must be 'treated' or 'control', not {focal!r}")
    return _FOCAL[focal]


def _distance(distance: str, score: str | None, score_model: str | None, kinds) -> str:
    """The spec's distance: `distance` itself, or the name of the supplied `score`
    column."""
    if score is None:
        if distance not in kinds:
            extra = "" if distance in METRICS or distance in SCORE_DISTANCES else (
                "; to match on a score column already in the data give score=")
            raise ValueError(f"distance must be one of {tuple(kinds)}, not {distance!r}{extra}")
        return distance
    if score_model is not None:
        raise ValueError("score= is a score column already in the data; score_model= fits one")
    if distance != "ps":
        raise ValueError("score= is the score itself: leave distance= at its default "
                         "(it says how match-stream fits a score)")
    return str(score)


def match_nn(fml: str, data, *, exact: Sequence[str] | str | None = None,
             distance: str = "ps", score: str | None = None, score_model: str | None = None,
             score_link: str = "logit", n_matches: int = 1, replace: bool = False,
             match_order: str | None = None, seed: int | None = None,
             caliper: float | Mapping[str, float | tuple[float, str]] | None = None,
             focal: str = "treated", id: str | None = None, keep_ties: bool = False,
             tie_tolerance: float = 0.0, min_matches: int = 1, match_in_rounds: bool = True,
             discard: str = "none", reestimate_score: bool = False,
             mahalanobis_vars: Sequence[str] | str | None = None,
             memory: str = "auto", workdir=None, max_memory_gb: float | None = None,
             memory_reserve: float | None = None, n_threads: int | None = None,
             profile: bool = False, verbose: bool = True) -> MatchSet | Profile:
    """Nearest-neighbor matching, out of core: each focal unit takes its nearest
    other-arm units, on a propensity score or a multivariate distance.

    Matching only: no outcome is needed, and the result holds the pairs and
    weights. Where conventions differ between matching packages, the default is
    R's MatchIt's; "Conventions and other implementations" in the documentation
    lists the differences.

    Parameters
    ----------
    fml : str
        ``"treat ~ x1 + x2"``: the treatment (0/1 or boolean) and the matching
        covariates (column names). The covariates are the default score model
        and the balance table's rows. Exact strata do not go here: ``|`` means
        absorbed fixed effects in this package family, so it is allowed only
        in `score_model`.
    data : str, Path, polars.LazyFrame or polars.DataFrame
        A Parquet path or glob (never fully loaded), or a frame.
    exact : column name or list of names, optional
        Exact-matching strata: units match only within cells of these columns.
        ``"a^b"`` is the same as ``["a", "b"]``.
    distance : {"ps", "logit_ps", "mahalanobis", "ivariance", "euclidean"}
        What to match on. ``"ps"``: the propensity score match-stream fits
        (`score_model`, `score_link`); ``"logit_ps"``: its linear predictor;
        ``"mahalanobis"``, ``"ivariance"`` (each covariate scaled by its
        variance) or ``"euclidean"``: the covariates themselves. To match on a
        score column already in the data, give `score`.
    score : str, optional
        A score column already in the data to match on, instead of fitting one.
    score_model : str, optional
        The score model, a formula for hdfe-stream's ``feglm_stream``, which
        absorbs fixed effects after ``|``: ``"treat ~ x1 + x2 | firm"``.
        Default: a logit of the treatment on the matching covariates.
    score_link : {"logit", "probit"}
        The score model's link.
    n_matches : int, default 1
        Matches per focal unit.
    replace : bool, default False
        Match with replacement. Without replacement, matching is greedy in
        the order `match_order`.
    match_order : {"largest", "smallest", "closest", "random", "data"}, optional
        The order focal units are matched in: by descending or ascending score,
        by ascending pair distance, at random (`seed`), or in data order.
        Default: "largest" ("smallest" when the controls are focal) for a
        score, "data" for a multivariate distance (where largest/smallest do
        not apply).
    seed : int, optional
        Seed for ``match_order="random"``.
    caliper : float or dict, optional
        A number is a caliper on the score, in standard deviations of the
        score over all units. A dict names calipers: ``{"ps": 0.2, "age": 3}``;
        covariate calipers are in raw units. To change a caliper's units give
        ``(width, "sd")`` or ``(width, "raw")``: ``{"ps": 0.2, "income":
        (0.5, "sd")}``. A pair is allowed when every gap is at most its
        caliper. A control that fails a covariate caliper is skipped and the
        next nearest is tried; the focal unit is not dropped.
    focal : {"treated", "control"}
        The units that choose: treated units are matched to controls (the ATT's
        match), or controls to treated units (the ATC's).
    id : str, optional
        A unit id column, carried into the match table and the weights.
    keep_ties : bool, default False
        With replacement, keep every control at exactly the k-th nearest
        distance (otherwise exactly k are kept, the ones with the larger
        score first). Each focal unit's matches then share its weight.
    tie_tolerance : float, default 0
        With ties: a control is tied when its squared distance, divided by
        the score's variance, is within this of the k-th's. The default, 0,
        keeps exact ties only.
    min_matches : int, default 1
        With ``n_matches > 1``: drop a focal unit that ends with fewer than
        this many matches (a caliper or exact strata can leave it short). The
        default keeps every focal unit with at least one match;
        ``min_matches=n_matches`` keeps only fully matched units. Matches tied
        at the k-th distance (`keep_ties`) count. Applied after matching:
        without replacement, the controls a dropped unit took are not offered
        to other units (the greedy pass is not rerun).
    match_in_rounds : bool, default True
        Without replacement and ``n_matches > 1``: match in rounds (every
        focal unit takes its 1st match, then those matched take their 2nd,
        ...; each round in `match_order`). False: one pass, each focal unit
        taking its k matches at its turn; with ``match_order="closest"`` that
        is all pairs in ascending distance, a unit taking controls until it
        has k. Identical for ``n_matches=1``.
    discard : {"none", "both", "control", "treated"} or a column name
        Common support: before matching, drop the units whose score lies
        outside the other arm's range ("treated": treated units outside the
        controls' range; "control": the reverse; "both": either, outside the
        overlap of the two ranges; a unit at a bound is kept). A column name
        drops the units where it is true. Discarded units are not matched, and
        are counted apart (`counts`, the balance table's "Discarded" row);
        calipers in standard deviations use the units kept.
    reestimate_score : bool, default False
        With `discard`, refit the score model on the units kept and match on
        the new score.
    mahalanobis_vars : list of column names, optional
        Match on the Mahalanobis distance between these covariates, within
        the caliper on the score: the score still orders the matching
        (`match_order`), carries the caliper and decides `discard`. The
        covariance is pooled within the treatment groups, over the units kept.
    memory : {"auto", "in_memory", "by_stratum"}
        How the narrow columns (tens of bytes a unit) are held while matching:
        all strata at once (fastest), or a batch of whole strata at a time,
        read back from a Parquet copy on disk (memory bounded by the largest
        stratum and the batch). Both give the same matches. "auto" takes
        "in_memory" when it fits the memory headroom (stream-limits).
    workdir : path, optional
        Scratch directory for the score model's files and the by-stratum copy.
        Default: $HDFE_STREAM_WORKDIR, else the system temporary directory.
    max_memory_gb, memory_reserve : float, optional
        The memory limit and the share kept in reserve that ``memory="auto"``
        plans with (stream-limits' ``MemoryBudget``; default: the limits it
        detects, including a PBS job's).
    n_threads : int, optional
        Threads for matching and for `effect()`: numba's, BLAS's and the k-d
        trees' (stream-limits' ``thread_scope``). Default: as they are (every
        core). Polars' threads are fixed when polars is imported: set
        ``STREAM_LIMITS_MAX_THREADS`` in the job script.
    profile : bool, default False
        Return a `Profile` (hdfe-stream's) of the memory and disk the match
        would need, from a scan of the data, without matching (see
        `match_stream.profile`).
    verbose : bool, default True
        Print the score model's progress (hdfe-stream's ``feglm_stream``).

    Returns
    -------
    MatchSet, or a Profile with ``profile=True``
    """
    return _execute(
        fml, data, method="nn", exact=exact,
        distance=_distance(distance, score, score_model, (*SCORE_DISTANCES, *METRICS)),
        score_model=score_model, link=score_link, n_matches=n_matches, replace=replace,
        match_order=match_order, seed=seed, caliper=caliper, estimand=_estimand(focal), id=id,
        keep_ties=keep_ties, tie_tolerance=tie_tolerance, min_matches=min_matches,
        match_in_rounds=match_in_rounds, discard=discard, reestimate_score=reestimate_score,
        mahalanobis_vars=mahalanobis_vars, memory=memory, workdir=workdir,
        max_memory_gb=max_memory_gb, memory_reserve=memory_reserve, n_threads=n_threads,
        profile=profile, verbose=verbose)


def match_kernel(fml: str, data, *, exact: Sequence[str] | str | None = None,
                 distance: str = "ps", score: str | None = None, score_model: str | None = None,
                 score_link: str = "logit", kernel: str = "epan",
                 ridge: bool | float | None = None,
                 bandwidth: float | str | PairDistance | CrossValidation = "pair_distance",
                 focal: str = "treated", id: str | None = None, memory: str = "auto",
                 workdir=None, max_memory_gb: float | None = None,
                 memory_reserve: float | None = None, n_threads: int | None = None,
                 profile: bool = False, verbose: bool = True) -> KernelSet | Profile:
    """Kernel or ridge matching on the score, out of core: each focal unit is
    compared with a weighted average of the other arm's units of its stratum
    within one bandwidth of its score.

    Matching only: no outcome is needed, and the result holds the matched units
    and their weights; there is no match table. Parameters not described here
    are as in `match_nn`.

    Parameters
    ----------
    fml, data, exact, distance, score, score_model, score_link, focal, id, memory, \
workdir, max_memory_gb, memory_reserve, n_threads, profile, verbose
        As in `match_nn`. `distance` is ``"ps"`` or ``"logit_ps"`` here (kernel
        matching is on a score), or give `score`.
    kernel : {"epan", "rectangle", "triangle", "biweight", "triweight"}
        The kernel, with support +/- 1 ("uniform" is "rectangle").
    ridge : None, True or float
        None: kernel matching. True: ridge matching, Frolich's regularized
        local-linear weights, with the kernel's ridge parameter (Seifert and
        Gasser's: 5/16 for "epan"). A number: ridge matching with that
        parameter; 0 is local-linear matching.
    bandwidth : float, "pair_distance", "cross_validation", PairDistance or CrossValidation
        The kernel's half-width. A number is in score units. ``"pair_distance"``
        (the default): 1.5 times the 90th percentile of the nonzero 1:1
        nearest-neighbor distances; ``PairDistance(factor=, quantile=)``
        changes those two numbers. ``"cross_validation"``: the bandwidth that
        predicts best a unit's outcome from the others (or, without an
        outcome, that moves the matched mean of the score least);
        ``CrossValidation(outcome=, grid=, weighted=, penalty=)`` sets the
        outcome, the bandwidths tried and the rest. The criterion without an
        outcome forms the pairs within each bandwidth tried, so it costs more
        than the others on large strata. A bandwidth is chosen for the focal
        arm at hand.

    Returns
    -------
    KernelSet, or a Profile with ``profile=True``
    """
    if ridge is None or ridge is False:
        method, ridge_value = "kernel", None
    elif ridge is True:
        method, ridge_value = "ridge", None
    else:
        method, ridge_value = "ridge", ridge
    return _execute(
        fml, data, method=method, exact=exact,
        distance=_distance(distance, score, score_model, SCORE_DISTANCES),
        score_model=score_model, link=score_link, estimand=_estimand(focal), id=id, kernel=kernel,
        ridge=ridge_value, bandwidth=bandwidth, memory=memory, workdir=workdir,
        max_memory_gb=max_memory_gb, memory_reserve=memory_reserve, n_threads=n_threads,
        profile=profile, verbose=verbose)


def match_cem(fml: str, data, *, cutpoints: int | Mapping[str, int | Sequence[float]],
              exact: Sequence[str] | str | None = None, focal: str = "treated",
              id: str | None = None, workdir=None, max_memory_gb: float | None = None,
              memory_reserve: float | None = None, n_threads: int | None = None,
              profile: bool = False, verbose: bool = True) -> CEMSet | Profile:
    """Coarsened exact matching, out of core: units match exactly on binned
    covariates (and the `exact` columns); cells with both arms are kept, and
    weighted so the matched controls sum to the matched treated units.

    Parameters
    ----------
    fml, data, exact, focal, id, workdir, max_memory_gb, memory_reserve, n_threads, \
profile, verbose
        As in `match_nn`.
    cutpoints : int or dict
        The bins of each matching covariate: a number of equal-width bins over
        its range, or a list of cut points; an int applies to every
        covariate. Required.

    Returns
    -------
    CEMSet, or a Profile with ``profile=True``
    """
    return _execute(
        fml, data, method="cem", exact=exact, estimand=_estimand(focal), id=id,
        cutpoints=cutpoints, workdir=workdir, max_memory_gb=max_memory_gb,
        memory_reserve=memory_reserve, n_threads=n_threads, profile=profile, verbose=verbose)


def _execute(fml, data, **given):
    """Parse the request, then match (or profile) it. Every argument not given
    takes its `_DEFAULTS` value."""
    unknown = set(given) - set(_DEFAULTS)
    if unknown:
        raise TypeError(f"unexpected arguments: {sorted(unknown)}")
    a = {**_DEFAULTS, **given}
    spec = parse_spec(
        fml, exact=a["exact"], method=a["method"], distance=a["distance"],
        score_model=a["score_model"], link=a["link"], n_matches=a["n_matches"],
        replace=a["replace"], match_order=a["match_order"], seed=a["seed"], caliper=a["caliper"],
        estimand=a["estimand"], id=a["id"], memory=a["memory"], keep_ties=a["keep_ties"],
        tie_tolerance=a["tie_tolerance"], min_matches=a["min_matches"],
        match_in_rounds=a["match_in_rounds"], discard=a["discard"],
        reestimate_score=a["reestimate_score"], mahalanobis_vars=a["mahalanobis_vars"],
        kernel=a["kernel"], ridge=a["ridge"], bandwidth=a["bandwidth"])
    if a["profile"]:
        from .profile import profile_match

        return profile_match(spec, data, cutpoints=a["cutpoints"], workdir=a["workdir"],
                             max_memory_gb=a["max_memory_gb"], memory_reserve=a["memory_reserve"],
                             n_threads=a["n_threads"], verbose=a["verbose"])
    with _threads.scope(a["n_threads"]):
        out = _match(spec, fml, data, a)
    out._n_threads = a["n_threads"]
    return out


def _match(spec, fml, data, a):
    """The match itself (inside the thread scope)."""
    if spec.method == "cem":
        units, counts, used = cem(spec, data, a["cutpoints"])
        return CEMSet(spec, units, counts, a["cutpoints"], used, source=data)
    workdir = a["workdir"]
    # what `effect()` needs to match again with the other focal group
    call = (fml, data, {k: v for k, v in a.items() if k not in ("estimand", "profile")})
    nar = narrow(spec, data, workdir=workdir, verbose=a["verbose"])
    path = choose_memory(spec, nar, a["max_memory_gb"], a["memory_reserve"])
    if spec.is_kernel:
        from ._kernel import run_kernel

        bw, units = run_kernel(spec, nar, path, workdir=workdir)
        ks = KernelSet(spec, nar, units, bw, path, source=data)
        ks._call = call
        return ks
    pairs = run(spec, nar, path, workdir=workdir)
    if spec.min_matches > 1:
        pairs = pairs.filter(pl.len().over("focal") >= spec.min_matches)
    ms = MatchSet(spec, nar, pairs, path, source=data)
    ms._workdir = workdir                  # scratch for effect()'s regressions
    ms._call = call
    return ms
