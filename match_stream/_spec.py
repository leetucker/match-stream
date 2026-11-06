"""The parsed and checked request: what to match, on what, with which rules.

`parse_spec` turns the arguments of `match_nn`, `match_kernel` and `match_cem`
into a `Spec` and raises a
ValueError naming the argument for anything it cannot honor, before any data
is read. Conventions follow R's MatchIt: calipers on the
distance are in standard deviations of the distance over all units, calipers
on covariates are raw, and the default match order is "largest" for a
propensity-score distance (``"smallest"`` for the ATC) and "data" otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

METHODS = ("nn", "cem", "kernel", "ridge")
SCORE_DISTANCES = ("ps", "logit_ps")
METRICS = ("mahalanobis", "ivariance", "euclidean")    # multivariate distances on the covariates
ORDERS = ("largest", "smallest", "closest", "random", "data")
ESTIMANDS = ("att", "atc")                      # the focal group (the ATE: effect())
MEMORY = ("auto", "in_memory", "by_stratum")
DISCARD = ("none", "both", "control", "treated")
_SCORE_KEYS = ("ps", "distance", "logit_ps")    # caliper keys meaning "the distance"


@dataclass
class Spec:
    treatment: str
    covariates: list[str]
    exact: list[str]
    method: str
    distance: str                       # "ps", "logit_ps" or the name of a score column
    score_model: str | None                  # the score model's formula (feglm_stream), if fitted
    link: str
    n_matches: int
    replace: bool
    match_order: str
    seed: int | None
    caliper: float | None               # on the distance, in its units after std (None = none)
    caliper_std: bool                   # caliper on the distance in SDs of the distance
    cov_calipers: dict[str, float] = field(default_factory=dict)   # column -> width
    cov_calipers_std: dict[str, bool] = field(default_factory=dict)
    estimand: str = "att"
    id: str | None = None
    memory: str = "auto"
    keep_ties: bool = False
    tie_tolerance: float = 0.0       # R Matching's distance.tolerance (squared, standardized)
    min_matches: int = 1             # drop focal units with fewer matches (kmatch: k)
    match_in_rounds: bool = True              # without replacement, k > 1: MatchIt's rounds
    discard: str = "none"            # MatchIt's: none/both/control/treated, or a column
    reestimate_score: bool = False   # refit the score on the units kept
    # Mahalanobis within the score caliper
    mahalanobis_vars: list[str] = field(default_factory=list)
    # kernel and ridge matching (_kernel.py)
    kernel: str = "epan"
    ridge: float | None = None          # None: the kernel's default (method="ridge")
    bandwidth: float | str = "pair_distance"   # a number (score units), or the rule's name
    bandwidth_quantile: float = 0.9
    bandwidth_factor: float = 1.5
    cv_outcome: str | None = None
    cv_grid: list[float] | None = None
    cv_weighted: bool = False
    cv_penalty: str | None = None       # None, or "support_loss"

    @property
    def is_kernel(self) -> bool:
        return self.method in ("kernel", "ridge")

    @property
    def is_metric(self) -> bool:
        return self.distance in METRICS

    @property
    def match_metric(self) -> bool:
        """Matching on a multivariate distance: a metric `distance`, or
        `mahalanobis_vars` (Mahalanobis on them, the score for order and caliper)."""
        return self.is_metric or bool(self.mahalanobis_vars)

    @property
    def fits_score(self) -> bool:
        if self.method == "cem" or self.is_metric:
            return False
        return self.distance in SCORE_DISTANCES

    def columns(self) -> list[str]:
        """The source columns the match itself reads (not the score model's)."""
        cols = [self.treatment, *self.exact, *self.cov_calipers]
        if self.is_metric:
            cols += self.covariates
        elif not self.fits_score:
            cols.append(self.distance)
        if self.id is not None:
            cols.append(self.id)
        if self.cv_outcome is not None:
            cols.append(self.cv_outcome)
        if self.discard not in DISCARD:
            cols.append(self.discard)
        cols += self.mahalanobis_vars
        return list(dict.fromkeys(cols))


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def parse_formula(fml: str) -> tuple[str, list[str]]:
    """``"treat ~ x1 + x2"`` -> ("treat", ["x1", "x2"]). Exact strata do not go
    in the formula: in this package family ``|`` always means absorbed fixed
    effects."""
    if not isinstance(fml, str) or "~" not in fml:
        raise ValueError(f"the formula must look like 'treat ~ x1 + x2', not {fml!r}")
    if "|" in fml:
        raise ValueError(
            "'|' is not allowed in the matching formula: give exact-matching strata "
            "with exact=[...]. ('|' means absorbed fixed effects in the score model, "
            "score_model='treat ~ x | fe', as in feglm_stream.)")
    lhs, rhs = (s.strip() for s in fml.split("~", 1))
    if not _NAME.match(lhs):
        raise ValueError(f"the treatment must be a column name, not {lhs!r}")
    terms = [t.strip() for t in rhs.split("+") if t.strip() and t.strip() != "1"]
    for t in terms:
        if not _NAME.match(t):
            raise ValueError(
                f"matching covariates must be column names for now, not {t!r}; "
                "create the column first (a transformed or categorical score-model "
                "term can go in score_model=)")
    return lhs, terms


def _exact_list(exact) -> list[str]:
    if exact is None:
        return []
    if isinstance(exact, str):
        exact = [exact]
    out = []
    for e in exact:
        for part in str(e).split("^"):          # "a^b" is the cells of a by b, as fixest
            part = part.strip()
            if not _NAME.match(part):
                raise ValueError(f"exact= takes column names (or 'a^b'), not {e!r}")
            out.append(part)
    return list(dict.fromkeys(out))


def _caliper_value(key, value) -> tuple[float, str | None]:
    """A caliper's width and unit (None: the default for its key) from a number
    or a ``(width, "sd" | "raw")`` pair."""
    unit = None
    if isinstance(value, (tuple, list)):
        if len(value) != 2 or value[1] not in ("sd", "raw"):
            raise ValueError(f"caliper {key!r}: give a width, or (width, 'sd') or "
                             f"(width, 'raw'), not {value!r}")
        value, unit = value
    width = float(value)
    if not width >= 0:
        raise ValueError(f"caliper widths must be non-negative, not {key}={width}")
    return width, unit


def parse_spec(fml, *, exact, method, distance, score_model, link, n_matches, replace, match_order,
               seed, caliper, estimand, id, memory, keep_ties=False,
               tie_tolerance=0.0, min_matches=1, match_in_rounds=True, discard="none",
               reestimate_score=False, mahalanobis_vars=None, kernel="epan", ridge=None,
               bandwidth=None) -> Spec:
    treatment, covariates = parse_formula(fml)
    m_order_given = match_order
    method = str(method).lower()
    if method not in METHODS:
        raise NotImplementedError(
            f"method={method!r} is not available; available: {METHODS}")
    estimand = str(estimand).lower()
    if estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS} for matching, not {estimand!r}")
    if link not in ("logit", "probit"):
        raise ValueError(f"score_link must be 'logit' or 'probit', not {link!r}")
    if not isinstance(n_matches, (int,)) or isinstance(n_matches, bool) or n_matches < 1:
        raise ValueError(f"n_matches must be a positive integer, not {n_matches!r}")
    if memory not in MEMORY:
        raise ValueError(f"memory must be one of {MEMORY}, not {memory!r}")
    if distance == "robust_mahalanobis":
        raise NotImplementedError("distance='robust_mahalanobis' (the rank-based variant) "
                                  "is not available")
    if distance in METRICS and not covariates:
        raise ValueError(f"distance={distance!r} needs matching covariates in the formula")
    if score_model is not None and distance not in SCORE_DISTANCES:
        raise ValueError("score_model= fits a score; use it with distance='ps' or 'logit_ps'")
    if method == "cem":
        distance, score_model = "none", None             # CEM matches on cells, not a distance
    if distance in SCORE_DISTANCES and score_model is None:
        # MatchIt's default: a logit of the treatment on the matching covariates
        if not covariates:
            raise ValueError("distance='ps' needs covariates in the formula, or score_model=")
        score_model = f"{treatment} ~ {' + '.join(covariates)}"

    if match_order is None:
        # MatchIt: "largest" for a score, estimated or supplied (a score column
        # counts as supplied), "smallest" for the ATC; "data" for other distances
        match_order = ("data" if distance in METRICS
                   else "smallest" if estimand == "atc" else "largest")
    match_order = str(match_order).lower()
    if match_order not in ORDERS:
        raise ValueError(f"match_order must be one of {ORDERS}, not {match_order!r}")
    if distance in METRICS and match_order in ("largest", "smallest") and not replace:
        raise ValueError(
            f"match_order={match_order!r} with distance={distance!r}: "
            "largest/smallest need a score (use 'data', 'random' or 'closest')")

    # calipers: a number is a caliper on the distance; a dict names columns. A
    # value is a width in the default units (the distance in standard deviations,
    # a covariate raw) or (width, "sd" | "raw")
    dist_cal, dist_std, cov_cal, cov_std = None, True, {}, {}
    if caliper is not None:
        items = caliper.items() if isinstance(caliper, dict) else [("ps", caliper)]
        for key, value in items:
            width, unit = _caliper_value(key, value)
            if key in _SCORE_KEYS or key == distance:
                if distance in METRICS:
                    raise ValueError(f"a caliper on the distance needs a score; with "
                                     f"distance={distance!r} give calipers on covariates")
                dist_cal = width
                dist_std = True if unit is None else unit == "sd"
            else:
                if not _NAME.match(str(key)):
                    raise ValueError(f"caliper keys must be column names, not {key!r}")
                cov_cal[str(key)] = width
                cov_std[str(key)] = False if unit is None else unit == "sd"

    spec = Spec(treatment=treatment, covariates=covariates, exact=_exact_list(exact),
                method=method, distance=distance, score_model=score_model, link=link,
                n_matches=int(n_matches),
                replace=bool(replace), match_order=match_order, seed=seed, caliper=dist_cal,
                caliper_std=dist_std, cov_calipers=cov_cal, cov_calipers_std=cov_std,
                estimand=estimand, id=id, memory=memory)
    if keep_ties and not (replace and method == "nn" and distance not in METRICS):
        raise NotImplementedError("keep_ties=True is available for nearest-neighbor matching on a "
                                  "score with replacement")
    spec.keep_ties = bool(keep_ties)
    if not tie_tolerance >= 0:
        raise ValueError("tie_tolerance must be non-negative")
    spec.tie_tolerance = float(tie_tolerance)
    if (not isinstance(min_matches, int) or isinstance(min_matches, bool)
            or not 1 <= min_matches <= spec.n_matches):
        raise ValueError(f"min_matches must be an integer from 1 to n_matches={spec.n_matches}, "
                         f"not {min_matches!r}")
    if min_matches > 1 and method != "nn":
        raise ValueError("min_matches is for nearest-neighbor matching")
    spec.min_matches = min_matches
    if mahalanobis_vars is not None:
        if isinstance(mahalanobis_vars, str):
            mv = [t.strip() for t in mahalanobis_vars.replace("~", " ").split("+") if t.strip()]
        else:
            mv = [str(c) for c in mahalanobis_vars]
        if not mv or not all(_NAME.match(c) for c in mv):
            raise ValueError("mahalanobis_vars must be column names (a list, or 'x1 + x2')")
        if method != "nn" or distance in METRICS:
            raise ValueError("mahalanobis_vars= is for nearest-neighbor matching with a score "
                             "distance (the score orders the matching and carries the caliper)")
        if keep_ties:
            raise NotImplementedError("keep_ties=True is for matching on a score, "
                                      "not with mahalanobis_vars")
        spec.mahalanobis_vars = list(dict.fromkeys(mv))
    discard = "none" if discard is None else str(discard)
    if discard != "none":
        if method != "nn":
            raise ValueError("discard= is for nearest-neighbor matching (kernel matching has "
                             "its own support rule; CEM keeps only cells with both arms)")
        if discard in DISCARD and distance in METRICS:
            raise ValueError("discard='both'/'control'/'treated' compares propensity scores: "
                             "give a score distance, or discard= a column of units to drop")
    if reestimate_score and not (spec.fits_score and discard in DISCARD[1:]):
        raise ValueError("reestimate_score=True refits a score match-stream fits (distance='ps' or "
                         "'logit_ps') after discard='both', 'control' or 'treated'")
    spec.discard, spec.reestimate_score = discard, bool(reestimate_score)
    if not match_in_rounds and (replace or method != "nn"):
        raise ValueError("match_in_rounds=False is for nearest-neighbor matching "
                         "without replacement")
    spec.match_in_rounds = bool(match_in_rounds)
    if spec.is_kernel:
        _kernel_options(spec, kernel=kernel, ridge=ridge, bandwidth=bandwidth,
                        n_matches=n_matches, match_order=m_order_given,
                        replace=replace, caliper=caliper)
    clash = {treatment} & set(spec.exact)
    if clash:
        raise ValueError(f"the treatment cannot be an exact-matching column: {sorted(clash)}")
    return spec


def _kernel_options(spec: Spec, *, kernel, ridge, bandwidth, n_matches, match_order, replace,
                    caliper) -> None:
    """Checks and fills the kernel/ridge options of `spec` (see _kernel.py)."""
    from ._kernel import KERNEL_ALIASES, KERNELS
    from .bandwidth import CrossValidation, PairDistance

    if spec.is_metric:
        raise NotImplementedError("kernel and ridge matching are on a score; "
                                  "a multivariate distance is not available")
    if n_matches != 1 or match_order is not None or spec.keep_ties:
        raise ValueError("n_matches=, match_order= and keep_ties= are for nearest-neighbor "
                         "matching; kernel matching uses every unit within the bandwidth")
    if caliper is not None:
        raise ValueError("kernel matching has no caliper: the bandwidth bounds the window")
    del replace     # kernel matching reuses units by construction
    name = KERNEL_ALIASES.get(str(kernel).lower(), str(kernel).lower())
    if name not in KERNELS:
        extra = (" (cosine, Parzen and the Gaussian are not polynomials on each side of 0, "
                 "which the window sums need)" if name in ("cosine", "parzen", "gaussian")
                 else "")
        raise ValueError(f"kernel must be one of {sorted(KERNELS)}, not {kernel!r}{extra}")
    spec.kernel = name
    if ridge is not None:
        if spec.method != "ridge":
            raise ValueError("a ridge parameter needs ridge matching")
        if not float(ridge) >= 0:
            raise ValueError("ridge must be non-negative (0: local-linear matching)")
        spec.ridge = float(ridge)
    if bandwidth is None:
        bandwidth = "pair_distance"
    if isinstance(bandwidth, str):
        rule = bandwidth.lower()
        if rule == "pair_distance":
            bandwidth = PairDistance()
        elif rule == "cross_validation":
            bandwidth = CrossValidation()
        else:
            raise ValueError("bandwidth must be a number, 'pair_distance', 'cross_validation', "
                             f"PairDistance(...) or CrossValidation(...), not {bandwidth!r}")
    if isinstance(bandwidth, PairDistance):
        spec.bandwidth = "pair_distance"
        spec.bandwidth_quantile = float(bandwidth.quantile)
        spec.bandwidth_factor = float(bandwidth.factor)
    elif isinstance(bandwidth, CrossValidation):
        spec.bandwidth = "cross_validation"
        spec.cv_outcome = None if bandwidth.outcome is None else str(bandwidth.outcome)
        spec.cv_grid = None if bandwidth.grid is None else list(bandwidth.grid)
        spec.cv_weighted = bool(bandwidth.weighted)
        spec.cv_penalty = bandwidth.penalty
    else:
        try:
            width = float(bandwidth)
        except (TypeError, ValueError):
            raise ValueError("bandwidth must be a number, 'pair_distance', 'cross_validation', "
                             f"PairDistance(...) or CrossValidation(...), not {bandwidth!r}"
                             ) from None
        if not width > 0:
            raise ValueError("bandwidth must be positive")
        spec.bandwidth = width
