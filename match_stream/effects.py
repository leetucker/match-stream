"""Treatment effects from matches, with Abadie-Imbens standard errors.

The estimator is the matching estimator of Abadie and Imbens (2006): each
focal unit's effect is its outcome minus the mean outcome of its matches
(for the ATC, the matches' mean minus its own), and the ATT (ATC) is their
mean over the matched focal units. The ATE matches both ways, every unit
being focal once.

The variance follows R Matching's ``Match()`` (the reference; verified to
1e-8 in tests/test_effects.py), Abadie and Imbens (2006, section 4):

    V = [ sum_{used units i} sigma_i^2 (K_i^2 - KK_i)
          + sum_{focal i} (tau_i - tau)^2 ] / N_f^2         (ATT, ATC)
    V = [ sum_i sigma_i^2 (K_i^2 + 2 K_i - KK_i)
          + sum_i (tau_i - tau)^2 ] / N^2                    (ATE)

with K_i = sum over the focal units j that use i of 1/M_j (M_j: j's number
of matches), KK_i = sum of 1/M_j^2, and N_f the number of matched focal
units. sigma_i^2 is homoskedastic for ``n_variance_matches=0``: half the weighted mean,
over matched pairs (j, m) with weight 1/M_j, of (y_j - y_m - tau)^2 (signs
flipped for control focal units); for ``n_variance_matches=J > 0`` it is each
unit's own, from its J nearest same-arm units (kernels/var.py).

Bias adjustment (Abadie and Imbens 2011; ``bias_adjustment=``) follows R
Matching's ``BiasAdjust`` (verified to 1e-10 / 1e-8 in tests/test_effects.py).
For each arm that supplies matches, a least-squares regression of the outcome
on (1, Z) is fitted over the units of that arm used as matches, weighted by
K_i, pooled over strata. Each match's outcome is then adjusted to the focal
unit's covariates,

    y_m  ->  y_m + (Z_focal - Z_m)' beta_(m's arm),

and the estimate, the homoskedastic sigma^2 and the dispersion term above
all use the adjusted outcomes; sigma_i^2 for ``n_variance_matches > 0`` uses the raw
ones (as Matching). For the ATT only the control regression is needed, for
the ATC only the treated one, for the ATE both.

Abadie-Imbens standard errors assume matching with replacement, as in Stata's
teffects. The other choice, ``vcov="if"`` or ``vcov={"CRV1": "a+b"}``, works
for every design (NN with or without replacement, CEM; kernel and ridge
matching when they land): the estimate is the difference of the weighted
means of the matched units' outcomes (the same number as above for NN), and
its variance is the influence function's with the matching weights held
fixed (``_core/influence.py``), clustered on the unit (``"if"``), the
subclass, source columns, or several of these. NN rows are one per unit and
subclass, as MatchIt's ``get_matches()``: a control used by three treated
units appears three times. The ATE matched both ways is kmatch's
p ATT + (1 - p) ATC, its variance that of the combination with the treated
share p estimated (`_mean_difference`). For one direction, the standard error
equals that of the weighted
regression ``y ~ treated`` on those rows with HC1 / CRV1 standard errors
(fixest's and ``sandwich::vcovCL``'s conventions; checked against
hdfe-stream). This is MatchIt's recommended analysis (``lm`` with the
matching weights, ``vcov = ~subclass`` without replacement, ``~subclass +
id`` with replacement) and kmatch's analytic standard errors (which use
G/(G-1) without the (N-1)/(N-2) regression factor). Holding the weights
fixed ignores that the matches were estimated: valid for pair-clustered NN
without replacement (Abadie and Spiess 2022), an approximation otherwise.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl
from hdfe_stream.core.results_base import RegressionResult
from hdfe_stream.core.utils import as_lazy

from ._core.influence import (
    WeightedMeanDifference,
    cluster_dims,
    group_mean_phis,
    if_vcov,
    joint_if_vcov,
    weighted_mean_difference,
)
from ._prep import ROW, SCORE, STRATUM, with_strata
from .kernels.var import sigma2_score


@dataclass
class MatchResult(RegressionResult):
    """A treatment effect estimated from matches, on hdfe-stream's
    `RegressionResult`: its reporting (`summary`, `tidy`, `confint`,
    `wald_test`, `summary_dict`, `to_pyfixest`) and ``hdfe_stream.etable``
    work as for the sibling packages' results.

    `beta` holds one estimate, named ``"<ATT|ATC|ATE>:<outcome>"`` (in the
    pyfixest view, and so in `etable`, just the estimand: the outcome heads
    the column); `vcov` its variance, of the kind `vcov` names. Inference is
    normal (`tidy`'s "t value" column, hdfe-stream's name for it, is a z
    statistic).

    `n_obs` is what the estimate uses: for Abadie-Imbens SEs the units (the
    focal units and their matches); for influence-function SEs the rows of
    the estimation table (MatchIt's ``get_matches()``: with replacement a
    control used by several focal units has a row in each subclass; for CEM
    and kernel matching the units with a nonzero weight). `n_clusters`
    follows hdfe-stream: the clustering (``"subclass"``, ``"subclass+id"``)
    to the number of clusters of each dimension, ``{"unit": (n,)}`` by unit,
    empty for Abadie-Imbens.

    Attributes
    ----------
    estimand : str
        "att", "atc" or "ate".
    n_variance_matches : int
        Same-arm matches for the conditional variances (Abadie-Imbens).
    n_focal : int
        Focal units matched (both arms for the ATE).
    bias_adjustment : list of str or None
        The bias-adjustment covariates.
    bias_coefs : dict
        The bias-adjustment slopes, by the arm they were fitted in.
    vcov_type : str
        "ai", "if" or "CRV1:<dims>" (hdfe-stream's attribute).
    score_correction : bool
        Whether the Abadie-Imbens (2016) correction for the estimated score
        was applied.
    n_lone : int
        Units alone in their arm within their exact stratum, whose
        conditional variance (n_variance_matches >= 1) has no same-arm neighbor.
    estimands : list of str or None
        With several estimands (or potential outcomes), the estimands, in
        the order of the coefficients; `estimand` then joins them.
    n_focal_by : dict
        With several estimands, the focal units of each ("nate": the units
        compared).
    labels : list of str or None
        The coefficients' names in the pyfixest view (`etable`), when they
        are not just "<outcome> (<estimand>)".
    lone_unit_variance : str or None
        How their variances were filled ("nearest" or "pooled"); None if
        there were none, or none was asked for (the SE is then NaN when the
        variance needs them).
    """

    estimand: str = "att"
    n_variance_matches: int = 0
    n_focal: int = 0
    bias_adjustment: list[str] | None = None
    bias_coefs: dict = field(default_factory=dict)
    n_clusters: dict = field(default_factory=dict)
    score_correction: bool = False
    n_lone: int = 0
    lone_unit_variance: str | None = None
    outcomes: list | None = None
    estimands: list | None = None
    n_focal_by: dict = field(default_factory=dict)
    labels: list | None = None
    _normal_inference = True

    def _summary_lines(self) -> list[str]:
        if self.n_focal_by:
            focal = ", ".join(f"{e.upper()} {n:,}" for e, n in self.n_focal_by.items())
            lines = [f"Estimands: {', '.join(e.upper() for e in self.estimands)}   "
                     f"Matched focal units: {focal}"]
        else:
            lines = [f"Estimand: {self.estimand.upper()}   Matched focal units: {self.n_focal:,}"]
        if self.vcov_type == "ai":
            how = ("homoskedastic" if self.n_variance_matches == 0
                   else f"heteroskedastic, {self.n_variance_matches} same-arm matches")
            lines.append(f"Standard errors: Abadie-Imbens ({how})"
                         + (", corrected for the estimated score (Abadie-Imbens 2016)"
                            if self.score_correction else ""))
        else:
            by = ", ".join(f"{d} ({g:,})" for key, gs in self.n_clusters.items()
                           for d, g in zip(key.split("+"), gs))
            lines.append(f"Standard errors: influence function, weights fixed, clustered by {by}")
        if self.bias_adjustment:
            lines.append("Bias adjustment (Abadie-Imbens 2011) on "
                         + ", ".join(self.bias_adjustment))
        if self.n_lone and self.lone_unit_variance:
            how = ("their nearest same-arm units in any stratum"
                   if self.lone_unit_variance == "nearest" else "their arm's mean variance")
            lines.append(f"{self.n_lone:,} units alone in their arm within their stratum: "
                         f"variance from {how} (lone_unit_variance={self.lone_unit_variance!r})")
        return lines

    def _pyfixest_attrs(self) -> dict:
        # pyfixest reads "ATT:y" as an interaction ("ATT x y"); the outcome
        # is the table's column, so the row is the estimand alone; with
        # several outcomes, one row per outcome: "y1 (ATT)"
        est = self.estimand.upper()
        names = [f"{o} ({est})" for o in self.outcomes] if self.outcomes else [est]
        if self.labels:
            names = list(self.labels)
        return {**super()._pyfixest_attrs(), "_coefnames": names}

    def _summary_fields(self) -> dict:
        return {"estimand": self.estimand, "n_focal": self.n_focal,
                "n_variance_matches": self.n_variance_matches, "n_obs": self.n_obs,
                "bias_adjustment": self.bias_adjustment, "vcov_type": self.vcov_type,
                "n_clusters": self.n_clusters, "score_correction": self.score_correction,
                "n_lone": self.n_lone, "lone_unit_variance": self.lone_unit_variance,
                "estimands": self.estimands}


def _source_rows(ms, name: str) -> pl.LazyFrame:
    """The source with its row position as column `name`: the source itself,
    or, inside a several-outcome `effect`, the columns it read once
    (``ms._y_cache``)."""
    cache = getattr(ms, "_y_cache", None)
    if cache is not None:
        return cache.lazy().rename({ROW: name}) if name != ROW else cache.lazy()
    return as_lazy(ms._source).with_row_index(name).with_columns(pl.col(name).cast(pl.Int64))


def _outcomes(ms, outcome: str, z: list[str] = ()) -> pl.DataFrame:
    """ROW, treatment, score, stratum, the outcome (``__y``) and the
    bias-adjustment covariates (``__z0``, ``__z1``, ...) of every unit used."""
    src = (_source_rows(ms, ROW)
           .select(ROW, pl.col(outcome).cast(pl.Float64).alias("__y"),
                   *(pl.col(c).cast(pl.Float64).alias(f"__z{j}") for j, c in enumerate(z))))
    df = (ms._narrow_lf.join(src, on=ROW, how="left")
          .collect(engine="streaming"))
    for name, col in [(outcome, "__y"), *((c, f"__z{j}") for j, c in enumerate(z))]:
        if df[col].null_count():
            what = "outcome" if col == "__y" else "bias-adjustment covariate"
            raise ValueError(f"the {what} {name!r} is missing for {df[col].null_count():,} "
                             "units used in matching; drop them before matching")
    return df


def _sigma2(ms, df: pl.DataFrame, n_variance_matches: int,
            across_strata: bool = False) -> np.ndarray:
    """Each unit's conditional variance from its n_variance_matches nearest same-arm
    units of its stratum (NaN for a unit alone in its arm there), or, with
    `across_strata`, of any stratum (lone_unit_variance="nearest")."""
    if ms.spec.match_metric:
        # on the whitened covariates (for mahalanobis_vars: the mahalanobis_vars), within strata
        from ._kdtree import same_arm_variances

        df, _ = with_strata(ms.spec, df)
        g = df[STRATUM].to_numpy()
        zc = sorted((c for c in df.columns if c.startswith("__ms_z")), key=lambda c: int(c[6:]))
        return same_arm_variances(df.select(zc).to_numpy().astype(np.float64),
                                  np.zeros_like(g) if across_strata else g,
                                  df[ms.spec.treatment].to_numpy(),
                                  df["__y"].to_numpy(), int(n_variance_matches))
    df, n_strata = with_strata(ms.spec, df)
    out = np.full(df.height, np.nan)
    t = df[ms.spec.treatment].to_numpy()
    g = df[STRATUM].to_numpy()
    if across_strata:
        g, n_strata = np.zeros_like(g), 1
    s = df[SCORE].to_numpy().astype(np.float64)
    y = df["__y"].to_numpy()
    for arm in (True, False):
        idx = np.flatnonzero(t == arm)
        idx = idx[np.lexsort((s[idx], g[idx]))]
        starts = np.searchsorted(g[idx], np.arange(n_strata + 1))
        out[idx] = sigma2_score(starts, s[idx], y[idx], int(n_variance_matches))
    return out


VAR_FALLBACKS = (None, "nearest", "pooled")


def _fill_lone(ms, df: pl.DataFrame, s2: np.ndarray, n_variance_matches: int,
               fallback: str) -> np.ndarray:
    """The conditional variances of the units alone in their arm within
    their exact stratum (NaN in `s2`): "nearest", from their n_variance_matches
    nearest same-arm units in any stratum, on the matching distance;
    "pooled", the mean of their arm's other estimated variances."""
    lone = np.isnan(s2)
    out = s2.copy()
    if fallback == "nearest":
        out[lone] = _sigma2(ms, df, n_variance_matches, across_strata=True)[lone]
        return out
    t = df[ms.spec.treatment].to_numpy().astype(bool)
    for arm in (True, False):
        here = t == arm
        if (lone & here).any():
            out[lone & here] = np.mean(s2[here & ~lone]) if (here & ~lone).any() else np.nan
    return out


def _weighted(pairs: pl.DataFrame) -> pl.DataFrame:
    """focal, match and w: a pair's weight is column "w" when given (R
    Matching's), else 1 / (its focal unit's number of matches)."""
    p = pairs.select("focal", "match", *(["w"] if "w" in pairs.columns else []))
    if "w" not in p.columns:
        p = p.with_columns((1.0 / pl.len().over("focal")).alias("w"))
    return p


def _unit_values(values, name: str) -> pl.DataFrame:
    """A value per unit as a frame, columns ``unit`` and `name`: from a frame
    of (unit, value) columns, or a dict {unit: value} (small inputs, as the
    tests'). A frame, not a dict: a Python dict of millions of units costs
    some 200 bytes each."""
    if isinstance(values, pl.DataFrame):
        unit, v = values.columns
        return values.select(pl.col(unit).cast(pl.Int64).alias("unit"),
                             pl.col(v).cast(pl.Float64).alias(name))
    return pl.DataFrame({"unit": list(values), name: list(values.values())},
                        schema={"unit": pl.Int64, name: pl.Float64})


def _lookup(frame: pl.DataFrame, key: str, values: pl.DataFrame, name: str) -> pl.DataFrame:
    """`frame` with column `name`: `values`' value of each row's unit `key`,
    in `frame`'s row order; every unit must have one."""
    out = frame.join(values.rename({"unit": key, values.columns[1]: name}), on=key,
                     how="left", maintain_order="left")
    if out[name].null_count():
        raise KeyError(f"no value for {out[name].null_count():,} units in column {key!r}")
    return out


def bias_coefs(sets, y, z: pl.DataFrame) -> list[np.ndarray]:
    """The bias-adjustment slopes for each direction in `sets` (see
    `ai_estimate`): a regression of y on (1, Z) over the units used as
    matches, weighted by K_i (the sum of the weights of their pairs), as R
    Matching's BiasAdjust. `z` holds ``unit`` and the Z columns. Returns the
    slopes (the intercept cancels from the adjustment)."""
    out = []
    zc = [c for c in z.columns if c != "unit"]
    yv = _unit_values(y, "__v")
    for pairs, _ in sets:
        k = (_weighted(pairs).group_by("match").agg(pl.col("w").sum().alias("K"))
             .join(z, left_on="match", right_on="unit", how="left").sort("match"))
        if k.height <= len(zc):
            raise ValueError(f"bias adjustment on {len(zc)} covariates needs more than "
                             f"{len(zc)} matched units in each arm used as matches; "
                             f"there are {k.height}")
        sw = np.sqrt(k["K"].to_numpy())
        X = np.column_stack([np.ones(k.height), k.select(zc).to_numpy()]) * sw[:, None]
        yy = _lookup(k.select("match"), "match", yv, "y")["y"].to_numpy() * sw
        # least squares (the minimum-norm solution if Z is collinear among the
        # matches; Matching then adds a tiny ridge or uses a g-inverse)
        coef, _, rank, _ = np.linalg.lstsq(X, yy, rcond=None)
        if rank < X.shape[1]:
            warnings.warn("the bias-adjustment covariates are collinear among the matched "
                          "units; using the minimum-norm least-squares solution", stacklevel=3)
        out.append(coef[1:])
    return out


def _adjustment(p: pl.DataFrame, z: pl.DataFrame, beta: np.ndarray) -> pl.DataFrame:
    """`p` with a column "adj": (Z_focal - Z_match)' beta for each pair."""
    zc = [c for c in z.columns if c != "unit"]
    zf = z.rename({c: f"{c}_f" for c in zc})
    zm = z.rename({c: f"{c}_m" for c in zc})
    p = (p.with_row_index("__i").join(zf, left_on="focal", right_on="unit", how="left")
         .join(zm, left_on="match", right_on="unit", how="left").sort("__i"))
    adj = pl.sum_horizontal([(pl.col(f"{c}_f") - pl.col(f"{c}_m")) * float(b)
                             for c, b in zip(zc, beta)])
    return p.with_columns(adj.alias("adj")).drop("__i", *[f"{c}_f" for c in zc],
                                                      *[f"{c}_m" for c in zc])


def _pieces(pairs: pl.DataFrame, y, sign: float, z=None, beta=None):
    """Per focal unit: tau_i; per used unit: K_i and KK_i; per pair: its
    weight and signed outcome difference. With `z` and `beta`, match
    outcomes are bias-adjusted to the focal unit's covariates. `y` is a
    frame of (unit, outcome) or a dict (`_unit_values`)."""
    yv = _unit_values(y, "__v")
    p = _lookup(_lookup(_weighted(pairs), "focal", yv, "yf"), "match", yv, "ym")
    if z is not None:
        p = _adjustment(p, z, beta).with_columns(pl.col("ym") + pl.col("adj"))
    tau = p.group_by("focal").agg(
        (sign * (pl.col("yf").first() - (pl.col("w") * pl.col("ym")).sum() / pl.col("w").sum()))
        .alias("tau"))
    k = p.group_by("match").agg(pl.col("w").sum().alias("K"), (pl.col("w") ** 2).sum().alias("KK"))
    rows = p.select(pl.col("w"), (sign * (pl.col("yf") - pl.col("ym"))).alias("d"))
    return tau, k, rows


def ai_estimate(sets, y, estimand: str, sigma2=None, n_unmatched: int = 0,
                z: pl.DataFrame | None = None, betas=None) -> tuple[float, float, int]:
    """The matching estimate, its Abadie-Imbens variance and the number of
    focal units, from `sets`: (pairs, focal_is_treated) for each direction
    matched (one for the ATT or ATC, two for the ATE). `y` gives each unit's
    outcome and `sigma2` its conditional variance (n_variance_matches > 0;
    None is homoskedastic), each a frame of (unit, value) columns or a dict.

    `z` (columns ``unit`` and the covariates) turns on the bias adjustment;
    `betas`, one slope vector per direction, defaults to `bias_coefs`.

    `n_unmatched` reproduces R Matching's treatment of focal units a caliper
    left unmatched: they count in the variance's N with an effect of 0 (each
    adds tau^2 to the dispersion term), although the estimate leaves them
    out. match-stream's default (0) leaves them out of both; the switch exists
    to test the formula against Matching (tests/test_effects.py)."""
    if z is not None and betas is None:
        betas = bias_coefs(sets, y, z)
    taus, ks, rows = [], [], []
    for j, (pairs, focal_treated) in enumerate(sets):
        tau, k, r = _pieces(pairs, y, 1.0 if focal_treated else -1.0,
                            z, None if z is None else betas[j])
        taus.append(tau)
        ks.append(k)
        rows.append(r)
    tau = pl.concat(taus)
    rows = pl.concat(rows)
    k = pl.concat(ks).group_by("match").agg(pl.col("K").sum(), pl.col("KK").sum())
    n_f = tau.height
    tau_v = tau["tau"].to_numpy()
    est = float(tau_v.mean())
    if sigma2 is None:
        # homoskedastic: from each matched pair's own difference (not the
        # focal unit's mean over its matches), weighted 1/M, as R Matching
        w, d = rows["w"].to_numpy(), rows["d"].to_numpy()
        sig = np.full(k.height, 0.5 * float(np.sum(w * (d - est) ** 2) / np.sum(w)))
    else:
        sig = _lookup(k.select("match"), "match", _unit_values(sigma2, "__v"), "s")["s"]
        sig = sig.to_numpy()
    K, KK = k["K"].to_numpy(), k["KK"].to_numpy()
    extra = 2 * K if estimand == "ate" else 0.0
    n = n_f + n_unmatched
    var = (float(np.sum(sig * (K * K + extra - KK))) + float(np.sum((tau_v - est) ** 2))
           + n_unmatched * est**2) / n**2
    return est, var, n_f


def _design(spec) -> str:
    """How the units were matched, for the result's title."""
    if spec.method == "cem":
        return "coarsened exact matching"
    if spec.is_kernel:
        return f"{spec.method} matching on {spec.distance}"
    return f"nearest-neighbor matching on {spec.distance}"


def _parse_vcov(vcov, spec) -> tuple[str, list[str] | None]:
    """("ai", None), ("if", None) (by unit) or ("CRV1", [dims]). The default:
    AI for NN with replacement; clustered on the subclass (pair membership)
    for NN without replacement (Abadie and Spiess 2022, MatchIt's advice);
    by unit for CEM (MatchIt's advice for subclass designs)."""
    if vcov is None:
        if spec.method == "cem" or spec.is_kernel:
            return "if", None
        return ("ai", None) if spec.replace else ("CRV1", ["subclass"])
    if isinstance(vcov, dict):
        if len(vcov) != 1 or next(iter(vcov)) != "CRV1":
            raise ValueError(f'vcov as a dict must be {{"CRV1": "a+b"}}, not {vcov!r}')
        return "CRV1", cluster_dims(vcov["CRV1"])
    if isinstance(vcov, str):
        v = vcov.strip()
        if v.lower() == "ai":
            return "ai", None
        if v.lower() in ("if", "hetero", "hc1"):
            return "if", None
        if v.upper().startswith("CRV1:"):
            return "CRV1", cluster_dims(v.split(":", 1)[1])
    raise ValueError(f"vcov must be 'ai', 'if', or {{'CRV1': 'a+b'}}, not {vcov!r}")


def _bias_cols(bias_adjustment, spec) -> list[str]:
    if bias_adjustment is True:
        if not spec.covariates:
            raise ValueError("bias_adjustment=True adjusts on the matching covariates, and the "
                             "formula has none; pass bias_adjustment=[column names]")
        return list(spec.covariates)
    if bias_adjustment is False or bias_adjustment is None:
        return []
    zcols = [bias_adjustment] if isinstance(bias_adjustment, str) else list(bias_adjustment)
    if not zcols or not all(isinstance(c, str) for c in zcols):
        raise ValueError("bias_adjustment must be True, False, or a list of column names")
    return zcols


def _check_estimand(estimand, spec) -> str:
    estimand = (estimand or spec.estimand).lower()
    if estimand not in ("att", "atc", "ate"):
        raise ValueError(f"estimand must be 'att', 'atc' or 'ate', not {estimand!r}")
    return estimand


def _estimand_list(estimand, spec) -> list[str] | None:
    """The estimands of a joint request (`_joint`): a list of "att", "atc",
    "ate" and "nate" (kmatch's naive effect: the unmatched, unweighted
    difference of means), or the string "nate"; None for one matched
    estimand, which `_effect` handles."""
    if isinstance(estimand, str) and estimand.lower() == "nate":
        return ["nate"]
    if not isinstance(estimand, (list, tuple)):
        return None
    names = [str(e).lower() for e in estimand]
    if not names or len(set(names)) != len(names):
        raise ValueError("estimand: give one estimand, or a list of distinct ones")
    for e in names:
        if e not in ("att", "atc", "ate", "nate"):
            raise ValueError(f"estimand must be 'att', 'atc', 'ate' or 'nate', not {e!r}")
    return names


def _sets(ms, estimand):
    """The matches behind `estimand`: the match itself, or one or two
    re-matches (the ATE matches both ways)."""
    spec = ms.spec
    if estimand == spec.estimand:
        return [ms]
    if estimand == "ate":
        return [ms, ms._rematch("atc" if spec.estimand == "att" else "att")]
    return [ms._rematch(estimand)]


def _ps_correction_plan(ms, sets, estimand, n_variance_matches, zcols, request) -> bool:
    """Whether to apply the Abadie-Imbens (2016) correction for the estimated
    score. `request` None: when it applies (a score match-stream fitted, no
    absorbed fixed effects, n_variance_matches >= 1, no bias adjustment, every focal
    unit matched; absorbed fixed effects in the score are supported), with a
    warning when a fitted score goes uncorrected; True:
    required (an error otherwise); False: never."""
    import warnings

    if request is False:
        return False
    spec = ms.spec
    reasons = []
    if not spec.fits_score or ms.score_fit is None:
        if request:
            raise ValueError("score_correction is for a score match-stream fits (distance='ps' "
                             "or 'logit_ps'); a score column is taken as known")
        return False
    if spec.mahalanobis_vars:
        if request:
            raise ValueError("score_correction (Abadie-Imbens 2016) is for matching on the score; "
                             "with mahalanobis_vars the matching is on the covariates")
        return False
    if n_variance_matches < 1:
        reasons.append("it needs n_variance_matches >= 1 (the correction goes with the "
                       "heteroskedastic variance)")
    if zcols:
        reasons.append("bias adjustment is used (the correction does not cover it)")
    if not reasons:
        return True
    msg = "the standard error is not corrected for the estimated score: " + "; ".join(reasons)
    if request:
        raise ValueError(msg)
    warnings.warn(msg, stacklevel=4)
    return False


def effect(ms, outcome, *, estimand: str | list[str] | None = None,
           n_variance_matches: int = 0, bias_adjustment: bool | str | list[str] = False,
           vcov=None, score_correction: bool | None = None,
           lone_unit_variance: str | None = None,
           potential_outcomes: bool = False) -> MatchResult:
    """The effect (see `MatchSet.effect`), inside the match's thread limit.
    `outcome`: a column, or a list of columns (`_several`); `estimand`: one
    of "att", "atc", "ate", or a list of those and "nate" (`_joint`)."""
    from . import _threads

    kw = dict(estimand=estimand, n_variance_matches=n_variance_matches,
              bias_adjustment=bias_adjustment, vcov=vcov, score_correction=score_correction,
              lone_unit_variance=lone_unit_variance)
    names = None
    if isinstance(outcome, (list, tuple)):
        names = [str(o) for o in outcome]
        if not names or len(set(names)) != len(names):
            raise ValueError("outcome: give one column, or a list of distinct columns")
    estimands = _estimand_list(estimand, ms.spec)
    with _threads.scope(getattr(ms, "_n_threads", None)):
        if estimands is not None or potential_outcomes:
            return _joint(ms, names or [str(outcome)],
                          estimands or [_check_estimand(None, ms.spec)], kw, potential_outcomes)
        if names is not None:
            if len(names) > 1:
                return _several(ms, names, kw)
            outcome = names[0]
        return _effect(ms, outcome, **kw)


def _several(ms, outcomes: list[str], kw: dict) -> MatchResult:
    """Several outcomes in one read of the source, with their joint vcov.

    Every variance here is a quadratic form in the outcome vector y: the
    influence functions are linear in y; the Abadie-Imbens variance's
    conditional variances (n_variance_matches = 0 or J, with lone_unit_variance), its
    dispersion term, the bias adjustment (linear in y) and the AI (2016)
    correction are quadratic. A quadratic form V(y) has exactly one
    symmetric bilinear form, so the covariance of the effects on y1 and y2
    is [V(y1 + y2) - V(y1) - V(y2)] / 2: the single-outcome computation run
    on y1, y2 and y1 + y2. That is the clustered (influence-function)
    covariance kmatch reports, and for Abadie-Imbens the covariance that
    keeps the variance of the effect on any combination a y1 + b y2 equal to
    the Abadie-Imbens variance computed on that combination. No package
    reports the Abadie-Imbens cross-outcome covariance; it is checked
    through this identity against R Matching's SEs on y1, y2 and y1 + y2
    (tests/test_several_outcomes.py).

    The outcome columns (with the bias-adjustment covariates and the
    clustering columns) are read once, for the units the match kept, into
    ``ms._y_cache`` with the pairwise sums; each single-outcome computation
    reads them from there."""
    spec = ms.spec
    kind, dims = _parse_vcov(kw["vcov"], spec)
    cols = list(outcomes) + [c for c in _bias_cols(kw["bias_adjustment"], spec)
                             if c not in outcomes]
    if kind == "CRV1":
        cols += [c for c in dims if c != "subclass" and c not in cols]
    if hasattr(ms, "_narrow_lf"):
        keep = ms._narrow_lf.select(ROW)
    else:                                     # CEM: the units with a weight
        keep = ms.weights().select(pl.col("row").cast(pl.Int64).alias(ROW))
    k = len(outcomes)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    sums = {f"__ms_sum{i}_{j}": (outcomes[i], outcomes[j]) for i, j in pairs}
    src = as_lazy(ms._source).with_row_index(ROW).with_columns(pl.col(ROW).cast(pl.Int64))
    cache = (src.join(keep, on=ROW, how="semi")
             .select(ROW, *cols, *((pl.col(a).cast(pl.Float64) + pl.col(b).cast(pl.Float64))
                                   .alias(name) for name, (a, b) in sums.items()))
             .collect(engine="streaming"))
    ms._y_cache = cache
    try:
        fits = [_effect(ms, o, **kw) for o in outcomes]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # each outcome warned already
            joint = {name: _effect(ms, name, **kw).vcov[0, 0] for name in sums}
    finally:
        ms._y_cache = None
    v = np.diag([f.vcov[0, 0] for f in fits])
    for (i, j), name in zip(pairs, sums):
        v[i, j] = v[j, i] = (joint[name] - v[i, i] - v[j, j]) / 2
    if np.all(np.isfinite(v)):
        ev = np.linalg.eigvalsh(v)
        if ev.min() < -1e-10 * max(ev.max(), 0.0):
            warnings.warn("the joint vcov of the outcomes is not positive semidefinite "
                          f"(smallest eigenvalue {ev.min():.3g}); this can happen with the "
                          "Abadie-Imbens (2016) correction, which subtracts a term", stacklevel=3)
    f0 = fits[0]
    est = f0.estimand.upper()
    return replace(f0, fml=f"{', '.join(outcomes)}: " + _design(spec),
                   depvar=", ".join(outcomes), coefnames=[f"{est}:{o}" for o in outcomes],
                   beta=np.array([f.beta[0] for f in fits]), vcov=v,
                   all_vcovs={f0.vcov_type: (v, f0.df_t)},
                   bias_coefs={o: f.bias_coefs for o, f in zip(outcomes, fits)},
                   outcomes=list(outcomes))


def _joint(ms, outcomes: list[str], estimands: list[str], kw: dict,
           potential_outcomes: bool) -> MatchResult:
    """Several estimands (and outcomes, and potential-outcome means) in one
    call, with their joint influence-function vcov (kmatch's
    ``kmatch ... (y1 y2), ate att atc nate po``).

    Every coefficient is a weighted group mean or the difference of two,
    with the weights held fixed, so each has a per-row influence
    contribution (`_core.influence.group_mean_phis`). The estimands' tables
    share units (a control can match for the ATT and be a focal unit of the
    ATC), and clusters are the unit or source columns, so summing the
    contributions of all tables by cluster gives every covariance
    (`joint_if_vcov`). Each coefficient's variance is exactly that of its own
    single fit. The potential-outcome means, ``<estimand>:<y>(1)`` and
    ``(0)``, are the weighted means of the treated and the control units of
    the estimand's table (for "nate", the raw means), with the effect's
    small-sample factor; the effect is their difference.

    Coefficients are ordered by outcome, then estimand in the order given,
    each followed by its two means if asked."""
    spec = ms.spec
    kind, dims = _parse_vcov(kw["vcov"], spec)
    if kind == "ai":
        raise ValueError("several estimands, 'nate' and potential_outcomes use influence-"
                         "function standard errors: pass vcov='if' or vcov={'CRV1': ...} "
                         "(Abadie-Imbens standard errors are for one matched estimand at a "
                         "time; no package reports their covariance across estimands)")
    if kw["n_variance_matches"]:
        raise ValueError("n_variance_matches is for Abadie-Imbens standard errors (vcov='ai')")
    if _bias_cols(kw["bias_adjustment"], spec):
        raise NotImplementedError("bias_adjustment comes with Abadie-Imbens standard errors "
                                  "only (vcov='ai')")
    if kw["score_correction"]:
        raise ValueError("score_correction is for Abadie-Imbens standard errors (vcov='ai')")
    if kw["lone_unit_variance"] is not None:
        raise ValueError("lone_unit_variance is for Abadie-Imbens standard errors (vcov='ai')")
    if kind == "CRV1" and "subclass" in dims and (len(estimands) > 1 or "nate" in estimands):
        raise ValueError("a subclass belongs to one estimand's match: with several estimands "
                         "or 'nate', cluster on source columns, or by unit (vcov='if')")

    tables = {e: _if_rows(ms, e, kind, dims) for e in estimands}
    for e, (rows, _, _) in tables.items():
        if rows.height == 0:
            raise ValueError(f"no matched units for the {e.upper()}")
    cl = list(dims) if kind == "CRV1" else [UNIT]
    src_cols = [c for c in cl if c not in ("subclass", UNIT)]
    units = pl.concat([t[0].select(UNIT) for t in tables.values()]).unique()
    src = (_source_rows(ms, UNIT)
           .join(units.lazy(), on=UNIT, how="semi")
           .select(UNIT, *(pl.col(o).cast(pl.Float64) for o in outcomes), *src_cols)
           .collect(engine="streaming"))
    for o in outcomes:
        if src[o].null_count():
            raise ValueError(f"the outcome {o!r} is missing for {src[o].null_count():,} "
                             "units in the estimates; drop them before matching")

    names, labels, beta, factors, parts = [], [], [], [], []
    n_focal_by = {}
    j = 0
    for o in outcomes:
        for e in estimands:
            rows, n_focal, neg = tables[e]
            n_focal_by[e] = n_focal
            ren = {SUB: "subclass"} if "subclass" in cl and SUB in rows.columns else {}
            tab = rows.join(src.select(UNIT, o, *src_cols), on=UNIT, how="left").rename(ren)
            r, phi = _mean_difference(tab, o, cl, neg)
            ec = e.upper()
            coefs = [(f"{ec}:{o}", f"{o} ({ec})", r.estimate, phi["phi1"] - phi["phi0"])]
            if potential_outcomes:
                coefs += [(f"{ec}:{o}(1)", f"{o}(1) ({ec})", r.mean1, phi["phi1"]),
                          (f"{ec}:{o}(0)", f"{o}(0) ({ec})", r.mean0, phi["phi0"])]
            cols = {}
            for name, label, est, col in coefs:
                names.append(name)
                labels.append(label)
                beta.append(est)
                factors.append(r.variance.adj)
                tab = tab.with_columns(col.alias(f"__c{j}"))
                cols[j] = f"__c{j}"
                j += 1
            parts.append((tab, cols))
    v = joint_if_vcov(parts, j, cl, np.array(factors))
    label = "if" if kind == "if" else "CRV1:" + "+".join(dims)
    n_obs = units.height
    return MatchResult(
        fml=f"{', '.join(outcomes)}: " + _design(spec), depvar=", ".join(outcomes),
        coefnames=names, beta=np.array(beta), vcov=v.vcov, vcov_type=label, df_t=10**9,
        n_obs=n_obs, all_vcovs={label: (v.vcov, 10**9)}, estimand=", ".join(estimands),
        n_focal=next(iter(n_focal_by.values())),
        n_clusters=_cluster_counts(kind, dims, v, units),
        outcomes=list(outcomes) if len(outcomes) > 1 else None,
        estimands=list(estimands), n_focal_by=dict(n_focal_by), labels=labels)


def _effect(ms, outcome: str, *, estimand: str | None = None, n_variance_matches: int = 0,
            bias_adjustment: bool | str | list[str] = False, vcov=None,
            score_correction: bool | None = None,
            lone_unit_variance: str | None = None) -> MatchResult:
    spec = ms.spec
    if n_variance_matches < 0 or int(n_variance_matches) != n_variance_matches:
        raise ValueError("n_variance_matches must be a non-negative integer")
    if lone_unit_variance not in VAR_FALLBACKS:
        raise ValueError("lone_unit_variance must be None, 'nearest' or 'pooled', "
                         f"not {lone_unit_variance!r}")
    estimand = _check_estimand(estimand, spec)
    kind, dims = _parse_vcov(vcov, spec)
    if lone_unit_variance is not None and (n_variance_matches < 1 or kind != "ai"):
        raise ValueError("lone_unit_variance is for Abadie-Imbens standard errors with "
                         "n_variance_matches >= 1 (vcov='ai'): it fills the conditional "
                         "variances of units alone in their arm within their stratum")
    zcols = _bias_cols(bias_adjustment, spec)
    if kind != "ai":
        if n_variance_matches:
            raise ValueError("n_variance_matches is for Abadie-Imbens standard errors (vcov='ai')")
        if zcols:
            # the bias-adjusted estimator is not a weighted mean difference:
            # its weights-fixed influence function would leave out the
            # regression; it has Abadie-Imbens standard errors only
            raise NotImplementedError("bias_adjustment comes with Abadie-Imbens standard errors "
                                      "only (vcov='ai')")
        if score_correction:
            raise ValueError("score_correction is for Abadie-Imbens standard errors (vcov='ai')")
        return _if_effect(ms, outcome, estimand, kind, dims)
    if spec.method != "nn" or not spec.replace:
        raise ValueError("Abadie-Imbens standard errors need nearest-neighbor matching with "
                         "replacement (replace=True); use vcov='if' "
                         "or vcov={'CRV1': 'subclass'}")

    df = _outcomes(ms, outcome, zcols)
    y = df.select(ROW, "__y")
    sets = _sets(ms, estimand)

    sigma2 = None
    n_lone = 0
    lone_rows = np.array([], dtype=np.int64)
    if n_variance_matches > 0:
        s2 = _sigma2(ms, df, int(n_variance_matches))
        lone = np.isnan(s2)
        n_lone = int(lone.sum())
        lone_rows = df[ROW].to_numpy()[lone]
        if n_lone and lone_unit_variance is not None:
            s2 = _fill_lone(ms, df, s2, int(n_variance_matches), lone_unit_variance)
        sigma2 = pl.DataFrame({ROW: df[ROW], "__s2": s2})
    pair_sets = [(m._pairs, m.focal_is_treated) for m in sets]
    z = betas = None
    coefs = {}
    if zcols:
        z = df.select(pl.col(ROW).alias("unit"), *(f"__z{j}" for j in range(len(zcols))))
        betas = bias_coefs(pair_sets, y, z)
        # keyed by the arm the regression is fitted in (the arm of the matches)
        coefs = {("control" if ft else "treated"): dict(zip(zcols, b.tolist()))
                 for (_, ft), b in zip(pair_sets, betas)}
    est, var, n_f = ai_estimate(pair_sets, y, estimand, sigma2, z=z, betas=betas)
    corrected = _ps_correction_plan(ms, sets, estimand, int(n_variance_matches), zcols,
                                    score_correction)
    if corrected:
        from .inference import ps_adjustment

        # a caliper that left focal units unmatched restricts the sample
        matched = {}
        for m in sets:
            arm = "treated" if m.focal_is_treated else "control"
            if m.counts()[f"unmatched_{arm}"]:
                matched[m.focal_is_treated] = m._pairs["focal"].unique().to_numpy()
        fit_units = None
        if getattr(ms, "_first_stage_all", False) and spec.discard != "none":
            # the score was fitted on every unit, the discarded too
            fit_units = ms._all_lf.select(ROW, spec.treatment, SCORE, *spec.exact)
        var = var + ps_adjustment(ms, df, estimand, est, int(n_variance_matches),
                                  workdir=getattr(ms, "_workdir", None),
                                  matched_focal=matched or None, fit_units=fit_units,
                                  lone_unit_variance=lone_unit_variance)

    if sigma2 is not None and np.isnan(var):
        # a unit alone in its arm within its exact stratum has no same-arm
        # neighbor, so its conditional variance is undefined
        import warnings

        carried = sum(int(p["match"].is_in(lone_rows).sum()) for p, _ in pair_sets)
        units = "1 unit is" if n_lone == 1 else f"{n_lone:,} units are"
        warnings.warn(f"the standard error is NaN: {units} alone in their arm "
                      f"within their exact stratum, so their conditional variance "
                      f"(n_variance_matches={int(n_variance_matches)}) is undefined, "
                      f"and the variance needs some "
                      f"of them (they are the matches in {carried:,} pairs). Use coarser "
                      "strata, n_variance_matches=0, vcov='if', or "
                      "lone_unit_variance='nearest' (their nearest "
                      "same-arm units in other strata) or 'pooled' (their arm's mean "
                      "variance)", stacklevel=3)
    name = f"{estimand.upper()}:{outcome}"
    vcov = np.array([[var]])
    # the units the estimate uses: the focal units and their matches
    n_used = pl.concat([pl.concat([p["focal"], p["match"]]) for p, _ in pair_sets]).n_unique()
    return MatchResult(fml=f"{outcome}: " + _design(spec), depvar=outcome,
                       coefnames=[name], beta=np.array([est]), vcov=vcov, vcov_type="ai",
                       df_t=10**9, n_obs=n_used, all_vcovs={"ai": (vcov, 10**9)},
                       estimand=estimand, n_variance_matches=int(n_variance_matches), n_focal=n_f,
                       bias_adjustment=zcols or None, bias_coefs=coefs, score_correction=corrected,
                       lone_unit_variance=lone_unit_variance if n_lone else None, n_lone=n_lone)


# ------------------------------------------------------- influence functions
UNIT, SUB, W, TR = "__unit", "__subclass", "__w", "__treated"
DIR = "__focal_treated"      # the ATE's rows: from the match of the treated (True) or controls


def match_rows(sets) -> pl.DataFrame:
    """The estimation table of NN matches, one row per unit and subclass
    (MatchIt's ``get_matches()``): each focal unit with weight 1 and each of
    its matches with weight 1 / (its number of matches). A unit used by
    several focal units (with replacement) has a row in each of their
    subclasses. The ATE stacks both directions, with distinct subclasses.
    Columns: UNIT (row in the source), SUB, TR, W."""
    out, off = [], 0
    for pairs, focal_treated in sets:
        p = pairs.select("focal", "match", "subclass").with_columns(
            (1.0 / pl.len().over("focal")).alias(W), pl.col("subclass") + off)
        foc = (p.group_by("focal").agg(pl.col("subclass").first())
               .select(pl.col("focal").alias(UNIT), pl.col("subclass").alias(SUB),
                       pl.lit(focal_treated).alias(TR), pl.lit(1.0).alias(W),
                       pl.lit(focal_treated).alias(DIR)))
        mat = p.select(pl.col("match").alias(UNIT), pl.col("subclass").alias(SUB),
                       pl.lit(not focal_treated).alias(TR), pl.col(W),
                       pl.lit(focal_treated).alias(DIR))
        out += [foc, mat]
        off = int(p["subclass"].max()) if p.height else off
    return pl.concat(out).sort(SUB, UNIT)


def _if_rows(ms, estimand: str, kind: str, dims):
    """The estimation table behind an influence-function effect: rows with
    UNIT (position in the source), TR, W and, for NN and CEM, SUB; the
    number of focal units; whether weights may be negative (kernel). The
    estimand "nate" is every unit used in matching, weight 1."""
    spec = ms.spec
    if estimand == "nate":
        lf = ms._narrow_lf if hasattr(ms, "_narrow_lf") else ms._used
        rows = (lf.select(pl.col(ROW).alias(UNIT), pl.col(spec.treatment).cast(pl.Boolean)
                          .alias(TR), pl.lit(1.0).alias(W)).collect(engine="streaming"))
        return rows, rows.height, False
    if spec.method == "cem":
        if estimand != spec.estimand:
            raise ValueError("CEM's effect is for the estimand it matched for "
                             f"({spec.estimand.upper()}); match again with focal=")
        rows = ms.weights().collect().select(
            pl.col("row").alias(UNIT), pl.col("subclass").alias(SUB),
            pl.col("treated").alias(TR), pl.col("weight").alias(W))
        n_focal = rows.filter(pl.col(TR) == (spec.estimand != "atc")).height
    elif spec.is_kernel:
        if kind == "CRV1" and "subclass" in dims:
            raise ValueError("kernel matching has no subclasses; cluster on source columns")
        # raw weights: each focal unit's weights sum to 1, so the directions
        # of the ATE stack on the same scale (as match_rows for NN)
        sets = _sets(ms, estimand)
        rows = pl.concat([m._units.select(pl.col("row").alias(UNIT), pl.col("treated").alias(TR),
                                          pl.col("weight").alias(W),
                                          pl.lit(m.focal_is_treated).alias(DIR))
                          for m in sets])
        n_focal = sum(m._units.filter(pl.col("focal")).height for m in sets)
    else:
        sets = [(m._pairs, m.focal_is_treated) for m in _sets(ms, estimand)]
        rows = match_rows(sets)
        n_focal = sum(p["focal"].n_unique() for p, _ in sets)
    return rows, n_focal, spec.is_kernel


def _mean_difference(tab: pl.DataFrame, y: str, cl, allow_negative: bool):
    """The effect of an estimation table and its per-row influence
    contributions: (`WeightedMeanDifference`, a frame with ``phi1`` and
    ``phi0``, those of the two potential-outcome means; the effect's is
    ``phi1 - phi0``).

    One direction (the ATT, the ATC, CEM's weights, "nate"): the weighted
    difference in means. The ATE matched both ways (DIR tells the directions
    apart) is kmatch's: ATE = p ATT + (1 - p) ATC, p = N1 / (N1 + N0) the
    share of treated among the focal units (the same number as the
    difference in means of the stacked rows), and its influence function
    p phi_ATT + (1 - p) phi_ATC + (ATT - ATC) phi_p, with phi_p the
    estimated share's, (1 - p) / N on a focal treated unit and -p / N on a
    focal control; likewise for each potential-outcome mean. The small-
    sample factor is that of the stacked rows, as for the other estimands."""
    if DIR not in tab.columns or tab[DIR].n_unique() < 2:
        r = weighted_mean_difference(tab, y, W, TR, cl, allow_negative=allow_negative)
        return r, group_mean_phis(tab, y, W, TR, r)
    d, g, w = pl.col(DIR), pl.col(TR), pl.col(W)
    Y = pl.col(y).cast(pl.Float64)
    a = weighted_mean_difference(tab.filter(d), y, W, TR, cl, allow_negative=allow_negative)
    c = weighted_mean_difference(tab.filter(~d), y, W, TR, cl, allow_negative=allow_negative)
    n = a.weight1 + c.weight0                   # the focal units' (weight 1 each)
    p = a.weight1 / n
    phi_p = pl.when(d & g).then(w * (1 - p)).when(~d & ~g).then(-w * p).otherwise(0.0) / n
    phi1 = (pl.when(g & d).then(p * w * (Y - a.mean1) / a.weight1)
            .when(g).then((1 - p) * w * (Y - c.mean1) / c.weight1).otherwise(0.0)
            + (a.mean1 - c.mean1) * phi_p)
    phi0 = (pl.when(~g & d).then(p * w * (Y - a.mean0) / a.weight0)
            .when(~g).then((1 - p) * w * (Y - c.mean0) / c.weight0).otherwise(0.0)
            + (a.mean0 - c.mean0) * phi_p)
    phis = tab.select(phi1.alias("phi1"), phi0.alias("phi0"))
    m1, m0 = p * a.mean1 + (1 - p) * c.mean1, p * a.mean0 + (1 - p) * c.mean0
    var = if_vcov(tab.with_columns((phis["phi1"] - phis["phi0"]).alias("__phi")), ["__phi"],
                  cl, df_k=2)
    r = WeightedMeanDifference(estimate=m1 - m0, mean1=m1, mean0=m0,
                               weight1=a.weight1 + c.weight1, weight0=a.weight0 + c.weight0,
                               variance=var)
    return r, phis


def _if_effect(ms, outcome: str, estimand: str, kind: str, dims) -> MatchResult:
    """The weighted difference in means of the matched units and its
    influence-function variance with the weights held fixed (kmatch's
    analytic standard errors; MatchIt's weighted regression with sandwich
    standard errors)."""
    spec = ms.spec
    rows, n_focal, allow_negative = _if_rows(ms, estimand, kind, dims)
    if rows.height == 0:
        raise ValueError("no matched units")

    cl = list(dims) if kind == "CRV1" else [UNIT]
    src_cols = [c for c in cl if c not in ("subclass", UNIT)]
    src = _source_rows(ms, UNIT).select(UNIT, pl.col(outcome).alias("__y"), *src_cols)
    tab = (rows.lazy().join(src, on=UNIT, how="left")
           .rename({SUB: "subclass"} if "subclass" in cl and SUB in rows.columns else {})
           .collect(engine="streaming"))
    if tab["__y"].null_count():
        raise ValueError(f"the outcome {outcome!r} is missing for {tab['__y'].null_count():,} "
                         "matched units; drop them before matching")
    r, _ = _mean_difference(tab, "__y", cl, allow_negative)
    v = r.variance
    if kind == "if":
        label = "if"
    else:
        label = "CRV1:" + "+".join(dims)
    vcov = v.vcov
    return MatchResult(fml=f"{outcome}: " + _design(spec), depvar=outcome,
                       coefnames=[f"{estimand.upper()}:{outcome}"], beta=np.array([r.estimate]),
                       vcov=vcov, vcov_type=label, df_t=10**9, n_obs=tab.height,
                       all_vcovs={label: (vcov, 10**9)}, estimand=estimand, n_focal=n_focal,
                       n_clusters=_cluster_counts(kind, dims, v, tab))


def _cluster_counts(kind, dims, v, tab) -> dict:
    """hdfe-stream's convention for `n_clusters` (what its reporting and
    pyfixest view read): the clustering's key, ``"a+b"``, to the number of
    clusters of each dimension; ``{"unit": (n,)}`` for the influence function
    by unit."""
    if kind == "CRV1":
        return {"+".join(dims): tuple(int(v.n_clusters[d]) for d in dims)}
    return {"unit": (int(v.n_clusters.get(UNIT, tab[UNIT].n_unique())),)}
