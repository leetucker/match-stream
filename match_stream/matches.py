"""`MatchSet`: the result of a match -- the pairs, the weights, the counts.

A match needs no outcome: this object is the whole result when matches are
all that is wanted. Outcome analysis is `effect()`. Everything the user takes
away comes as a Polars LazyFrame, so it can be sunk to Parquet or joined back
to a source larger than memory. Only the pairs are held (a few tens of bytes
per pair), never the source.
"""

from __future__ import annotations

import polars as pl

from ._prep import Narrow
from ._spec import Spec


class MatchSet:
    """Matched pairs and weights from `match_nn`.

    Attributes
    ----------
    spec : the request as parsed (formula, strata, calipers, order, ...)
    memory : the path that ran, "in_memory" or "by_stratum"
    n_source : rows read from the source
    n_dropped : rows left out for a missing value or a null score
    n_strata : exact-matching strata (1 without exact=)
    score_fit : the score model (an hdfe-stream result) when one was fitted
    score_sd : the standard deviation of the score over all kept units, the
        unit of a caliper on the score
    """

    def __init__(self, spec: Spec, nar: Narrow, pairs: pl.DataFrame, memory: str, source=None):
        self.spec = spec
        self._narrow_lf = nar.lf            # lazy: re-read for balance, never held
        self._source = source
        self.memory = memory
        self.n_source = nar.n_source
        self.n_dropped = nar.n_dropped
        self.n_strata = nar.n_strata
        self.score_fit = nar.score_fit
        self.score_sd = nar.score_sd
        self._n_treated = nar.n_treated
        self._n_control = nar.n_kept - nar.n_treated
        self._all_lf = nar.all_lf if nar.all_lf is not None else nar.lf
        self._discarded = (nar.discarded_treated, nar.discarded_control)
        self._first_stage_all = nar.first_stage_all
        # a subclass per focal unit, numbered in data order
        sub = (pairs.select("focal").unique().sort("focal")
               .with_row_index("subclass", offset=1)
               .with_columns(pl.col("subclass").cast(pl.Int64)))
        self._pairs = pairs.join(sub, on="focal", how="left").sort(["focal", "rank"])

    @property
    def focal_is_treated(self) -> bool:
        return self.spec.estimand != "atc"

    # ------------------------------------------------------------------ pairs
    def matches(self) -> pl.LazyFrame:
        """The match table, one row per pair: ``treated_row`` and
        ``control_row`` (positions in the source, from 0), the ids when
        ``id=`` was given (``treated_<id>``, ``control_<id>``), ``rank`` (1 for
        the focal unit's nearest match), ``distance`` (the score gap) and
        ``subclass`` (one per focal unit, given with and without replacement)."""
        t, c = ("focal", "match") if self.focal_is_treated else ("match", "focal")
        ren = {t: "treated_row", c: "control_row"}
        cols = ["treated_row", "control_row"]
        if self.spec.id is not None:
            ren |= {f"{t}_id": f"treated_{self.spec.id}", f"{c}_id": f"control_{self.spec.id}"}
            cols += [f"treated_{self.spec.id}", f"control_{self.spec.id}"]
        return self._pairs.rename(ren).select(*cols, "rank", "distance", "subclass").lazy()

    # ---------------------------------------------------------------- weights
    def weights(self) -> pl.LazyFrame:
        """Matching weights of the matched units:
        a matched focal unit weighs 1; a unit matched to focal units weighs the
        sum, over those units, of 1 / (their number of matches), rescaled so
        that these weights sum to the number of such units. Unmatched units
        (weight 0) are left out. Columns: ``row``, the id (when ``id=`` was
        given), ``treated``, ``weight`` and ``subclass`` (the focal unit's
        group; for a unit used by several focal units, with replacement, the
        lowest)."""
        p = self._pairs
        has_id = self.spec.id is not None
        idf = [pl.col("focal_id").first().alias("id")] if has_id else []
        idm = [pl.col("match_id").first().alias("id")] if has_id else []
        focal = (p.group_by("focal")
                 .agg(pl.col("subclass").first(), *idf)
                 .rename({"focal": "row"})
                 .with_columns(pl.lit(1.0).alias("weight"),
                               pl.lit(self.focal_is_treated).alias("treated")))
        other = (p.with_columns((1.0 / pl.len().over("focal")).alias("__w"))
                 .group_by("match")
                 .agg(pl.col("__w").sum().alias("weight"), pl.col("subclass").min(), *idm)
                 .rename({"match": "row"})
                 .with_columns(pl.lit(not self.focal_is_treated).alias("treated")))
        if other.height:
            other = other.with_columns(pl.col("weight") * (other.height / other["weight"].sum()))
        cols = ["row", *(["id"] if has_id else []), "treated", "weight", "subclass"]
        w = pl.concat([focal.select(cols), other.select(cols)]).sort("row")
        if has_id:
            w = w.rename({"id": self.spec.id})
        return w.lazy()

    # ---------------------------------------------------------------- effects
    def effect(self, outcome: str | list[str], *, estimand: str | list[str] | None = None,
               n_variance_matches: int = 0, bias_adjustment: bool | str | list[str] = False,
               vcov=None, score_correction: bool | None = None,
               lone_unit_variance: str | None = None, potential_outcomes: bool = False):
        """The treatment effect on `outcome`, with Abadie-Imbens or
        influence-function standard errors.

        Parameters
        ----------
        outcome : str or list of str
            The outcome column in the source, or several: read in one pass, one estimate each, with
            their joint vcov. The covariances are those the variances imply
            (each variance is a quadratic form in the outcome): for
            influence-function SEs the stacked covariance; for
            Abadie-Imbens SEs the covariance that makes the variance of the
            effect on any combination a y1 + b y2 its Abadie-Imbens variance.
        estimand : {"att", "atc", "ate", "nate"} or list of them, optional
            Default: the match's own. Another one matches again (the ATE
            matches both ways). A list (``["ate", "att", "atc", "nate"]``)
            estimates them in one call, with their joint vcov: the
            coefficients are ordered by outcome, then estimand. ``"nate"`` is
            the naive effect: the plain difference of means of the
            units used in matching, no weights. Several estimands, "nate"
            and `potential_outcomes` need influence-function standard errors
            (``vcov="if"`` or ``{"CRV1": ...}``, on source columns, not the
            subclass, when there are several tables); the estimands
            covary through the units they share (docs/implementations.md).
        n_variance_matches : int, default 0
            0: homoskedastic conditional variance; J > 0: each unit's own,
            from its J nearest units in the same arm.
        bias_adjustment : bool, str or list of str, default False
            Abadie and Imbens's (2011) bias adjustment: each match's outcome
            is moved to its focal unit's covariates with a regression of the
            outcome on these columns, fitted (weighted by the times a unit is
            used) over the matched units of each arm.
            True adjusts on the matching covariates of the formula; a list
            names the columns (numeric, no missing values among the units
            used). The regression pools the strata; to let it differ by
            stratum, add stratum indicator columns. Needs ``vcov="ai"``.
        vcov : str or dict, optional
            ``"ai"``: Abadie and Imbens (2006), for matching with replacement
            (the default then). ``"if"``: the influence function of the
            weighted mean difference with the weights held fixed, by unit
            (HC1 on the rows of `effect`'s table). ``{"CRV1": "subclass"}``
            (the default without replacement), ``{"CRV1": "subclass+id"}``
            (advised with replacement), or any source columns:
            clustered, multi-way with ``+``. ``"subclass"`` means the
            match's subclass, never a source column of that name.
        score_correction : bool, optional
            Correct the Abadie-Imbens variance for a propensity score that
            match-stream estimated (Abadie and Imbens 2016). Default: whenever it applies (a
            fitted score, with or without absorbed fixed effects,
            ``n_variance_matches >= 1``, no bias adjustment), with a warning when a
            fitted score goes uncorrected. When a caliper leaves focal units
            unmatched, the correction concerns the matched units (the score
            model's variance stays its own). A score column passed as
            `distance` is taken as known.
        lone_unit_variance : {None, "nearest", "pooled"}, optional
            With ``n_variance_matches >= 1``, each unit's conditional variance comes
            from its nearest same-arm units within its exact stratum, so a
            unit alone in its arm there has none. None (the default): the
            standard error is NaN, with a warning, when the variance needs
            such a unit (the ATT needs the matched controls'; the ATC, the
            ATE and the Abadie-Imbens 2016 correction the treated units'
            too). ``"nearest"``: its n_variance_matches nearest same-arm units in any
            stratum, on the matching distance (the score, or the whitened
            covariates); this assumes the variance changes smoothly across
            strata. ``"pooled"``: the mean of its arm's other conditional
            variances. The 2016 correction's own-arm sets for such a unit
            are, likewise, its nearest units of its arm in any stratum, or
            its whole arm. Such a unit is often the match of many focal
            units, so the choice can move the standard error a lot.

        potential_outcomes : bool, default False
            Also report each estimand's potential-outcome means: the weighted
            mean outcome of its treated units, ``"ATT:y(1)"``, and of its
            control units, ``"ATT:y(0)"``, whose difference is the effect.
            Influence-function standard errors only.

        Returns
        -------
        MatchResult
            hdfe-stream's result API: ``summary()``, ``tidy()``,
            ``confint()``, ``etable``.

        The Abadie-Imbens variance needs matching with replacement (as in the
        literature it follows). The influence-function variance treats the matches as
        given (see ``match_stream.effects``)."""
        from .effects import effect

        return effect(self, outcome, estimand=estimand, n_variance_matches=n_variance_matches,
                      bias_adjustment=bias_adjustment, vcov=vcov,
                      score_correction=score_correction, lone_unit_variance=lone_unit_variance,
                      potential_outcomes=potential_outcomes)

    def _rematch(self, estimand: str) -> MatchSet:
        """The same match with another focal group."""
        from .api import _execute

        fml, data, kw = self._call
        return _execute(fml, data, estimand=estimand, **kw)

    # ---------------------------------------------------------------- balance
    def balance(self, covariates: list[str] | None = None):
        """Balance before and after matching: a `BalanceTable`. Rows: the score
        (``distance``, for score distances) and `covariates` (default: the formula's)."""
        from hdfe_stream.core.utils import as_lazy

        from ._prep import ROW, SCORE
        from .balance import T, balance_table

        covs = list(self.spec.covariates if covariates is None else covariates)
        src = (as_lazy(self._source).with_row_index(ROW)
               .with_columns(pl.col(ROW).cast(pl.Int64)).select(ROW, *covs))
        cols = [pl.col(ROW), pl.col(self.spec.treatment).alias(T)]
        terms = list(covs)
        if self.spec.fits_score or not self.spec.is_metric:
            cols.append(pl.col(SCORE).alias("distance"))
            terms = ["distance", *covs]
        # every unit, the discarded too (MatchIt's "All")
        units = self._all_lf.select(cols).join(src, on=ROW, how="left")
        w = self.weights().collect().rename({"row": ROW})
        pairs = self.matches().select(pl.col("treated_row").alias("t_row"),
                                      pl.col("control_row").alias("c_row")).collect()
        return balance_table(units, w, terms, self.focal_is_treated, pairs,
                             discarded=self._discarded)

    # ---------------------------------------------------------------- summary
    def counts(self) -> dict:
        """Units by arm: available (after missing values were dropped),
        matched, unmatched, discarded (common support; the
        unmatched exclude the discarded); the number of pairs."""
        p = self._pairs
        foc, oth = p["focal"].n_unique(), p["match"].n_unique()
        m_t, m_c = (foc, oth) if self.focal_is_treated else (oth, foc)
        n_t, n_c = self._n_treated, self._n_control
        d_t, d_c = self._discarded
        return {"treated": n_t, "control": n_c, "matched_treated": m_t, "matched_control": m_c,
                "unmatched_treated": n_t - m_t - d_t, "unmatched_control": n_c - m_c - d_c,
                "discarded_treated": d_t, "discarded_control": d_c,
                "pairs": p.height, "dropped_missing": self.n_dropped}

    def summary(self) -> None:
        """Print what was matched and how."""
        s, c = self.spec, self.counts()
        how = "with" if s.replace else "without"
        print(f"match-stream: {s.method}, {s.n_matches}:1 {how} replacement, "
              f"estimand {s.estimand.upper()}, order {s.match_order}")
        print(f"  distance: {s.distance}" + (f" (score model: {s.score_model}, {s.link})"
                                             if s.fits_score else ""))
        if s.exact:
            print(f"  exact: {', '.join(s.exact)} ({self.n_strata:,} strata)")
        if s.caliper is not None or s.cov_calipers:
            cal = []
            if s.caliper is not None:
                cal.append(f"{s.distance} {s.caliper:g}" + (" SD" if s.caliper_std else ""))
            cal += [f"{k} {v:g}" + (" SD" if s.cov_calipers_std.get(k) else "")
                    for k, v in s.cov_calipers.items()]
            print(f"  calipers: {', '.join(cal)}")
        if s.min_matches > 1:
            print(f"  focal units with fewer than {s.min_matches} matches dropped")
        if s.discard != "none":
            print(f"  common support: discard {s.discard}"
                  + (" (score re-estimated)" if s.reestimate_score else ""))
        print(f"  {'':10s}{'available':>12s}{'matched':>12s}{'unmatched':>12s}"
              f"{'discarded':>12s}")
        for arm in ("treated", "control"):
            print(f"  {arm:10s}{c[arm]:12,d}{c['matched_' + arm]:12,d}"
                  f"{c['unmatched_' + arm]:12,d}{c['discarded_' + arm]:12,d}")
        if self.n_dropped:
            print(f"  {self.n_dropped:,d} rows left out for missing values")

    def __repr__(self):
        c = self.counts()
        return (f"<MatchSet {c['matched_treated']:,} treated and {c['matched_control']:,} "
                f"controls matched, {c['pairs']:,} pairs>")


class CEMSet:
    """Coarsened exact matching from `match_cem`:
    the matched units and their weights. CEM matches cells, not pairs, so
    there is no match table; ``subclass`` is the cell.

    Attributes
    ----------
    spec : the request as parsed
    cutpoints : the bins asked for, by covariate
    """

    def __init__(self, spec: Spec, units: pl.DataFrame, counts: dict, cutpoints,
                 used: pl.LazyFrame | None = None, source=None):
        self.spec = spec
        self._source = source
        self._used = used                   # lazy: every unit used, for balance
        self.cutpoints = cutpoints
        self._units = units
        self._counts = counts
        self.n_dropped = counts["dropped_missing"]

    def weights(self) -> pl.LazyFrame:
        """The matched units: ``row`` (position in the source), the id,
        ``treated``, ``weight`` (matched treated units 1; controls scaled to the
        same total) and ``subclass`` (the cell, numbered 1..). Units in cells without both
        arms weigh 0 and are left out."""
        out = self._units
        return out.lazy()

    def balance(self, covariates: list[str] | None = None):
        """Balance before and after matching; the pair distance is over all
        treated-control pairs within each cell. Rows: `covariates` (default: the formula's)."""
        from .balance import T, balance_table

        covs = list(self.spec.covariates if covariates is None else covariates)
        units = self._used.select("__ms_row", pl.col(self.spec.treatment).alias(T), *covs)
        w = self._units.rename({"row": "__ms_row"})
        return balance_table(units, w, covs, self.spec.estimand != "atc", None)

    def effect(self, outcome: str | list[str], *, estimand: str | list[str] | None = None,
               vcov=None, potential_outcomes: bool = False):
        """The treatment effect on `outcome` (a column, or a list: several
        outcomes in one pass with their joint vcov, see `MatchSet.effect`)
        for the matched estimand: the
        difference of the CEM-weighted means, with the influence-function
        variance (weights fixed). `vcov`: ``"if"`` (by unit, the default:
        clustering on the cell is not advised), or ``{"CRV1":
        "a+b"}`` with ``"subclass"`` (the cell) or source columns. `estimand`
        may add ``"nate"`` (the plain difference of means of the units used)
        to the matched estimand, and `potential_outcomes` the treated and
        control means (see `MatchSet.effect`). Returns a `MatchResult`."""
        from .effects import effect

        return effect(self, outcome, estimand=estimand, vcov="if" if vcov is None else vcov,
                      potential_outcomes=potential_outcomes)

    def matches(self):
        raise TypeError("CEM matches cells, not pairs: use weights(), whose subclass is the cell")

    def counts(self) -> dict:
        """Units by arm (available, matched, unmatched) and matched cells."""
        return dict(self._counts)

    def summary(self) -> None:
        """Print what was matched."""
        c = self._counts
        print(f"match-stream: CEM, estimand {self.spec.estimand.upper()}, {c['cells']:,} "
              f"matched cells")
        if self.spec.exact:
            print(f"  exact: {', '.join(self.spec.exact)}")
        print(f"  {'':10s}{'available':>12s}{'matched':>12s}{'unmatched':>12s}")
        for arm in ("treated", "control"):
            print(f"  {arm:10s}{c[arm]:12,d}{c['matched_' + arm]:12,d}"
                  f"{c['unmatched_' + arm]:12,d}")

    def __repr__(self):
        c = self._counts
        return (f"<CEMSet {c['matched_treated']:,} treated and {c['matched_control']:,} "
                f"controls in {c['cells']:,} cells>")


class KernelSet:
    """Kernel or ridge matching from `match_kernel`: the matched units and their
    weights. Each focal unit is
    compared with a weighted average of the other arm's units of its
    stratum within the bandwidth, so there is no match table and no
    subclass.

    Attributes
    ----------
    spec : the request as parsed (kernel, ridge parameter, bandwidth rule)
    bandwidth : the bandwidth used, in score units
    bandwidth_info : how it was chosen (`_kernel.Bandwidth`: the rule, the
        pair-distance bandwidth, and for "cross_validation" the grid with its criterion, ``.cv``)
    memory, n_source, n_dropped, n_strata, score_fit, score_sd : as `MatchSet`
    """

    def __init__(self, spec: Spec, nar: Narrow, units: pl.DataFrame, bw, memory: str,
                 source=None):
        self.spec = spec
        self._narrow_lf = nar.lf
        self._source = source
        self._units = units                 # row, treated, weight (raw), focal, n, [id]
        self.bandwidth = bw.h
        self.bandwidth_info = bw
        self.memory = memory
        self.n_source = nar.n_source
        self.n_dropped = nar.n_dropped
        self.n_strata = nar.n_strata
        self.score_fit = nar.score_fit
        self.score_sd = nar.score_sd
        self._n_treated = nar.n_treated
        self._n_control = nar.n_kept - nar.n_treated

    @property
    def focal_is_treated(self) -> bool:
        return self.spec.estimand != "atc"

    def weights(self) -> pl.LazyFrame:
        """The matched units: ``row``, the id (when ``id=`` was given),
        ``treated``, ``weight``. A matched focal unit weighs 1; another unit
        weighs the sum of its kernel (or ridge) weights over the focal units
        it serves, rescaled so these weights sum to the number of such units
        (an estimate uses only their ratios). Ridge
        weights can be negative. Unmatched units are left out."""
        u = self._units
        oth = ~pl.col("focal")
        tot = u.filter(oth)["weight"].sum()
        n_oth = u.filter(oth).height
        scale = n_oth / tot if tot else 1.0
        cols = ["row", *([self.spec.id] if self.spec.id is not None else []), "treated",
                "weight"]
        return (u.with_columns(pl.when(oth).then(pl.col("weight") * scale)
                               .otherwise(pl.col("weight")).alias("weight"))
                .select(cols).sort("row").lazy())

    def matches(self):
        raise TypeError("kernel matching weights every unit within the bandwidth, so there are "
                        "no pairs: use weights()")

    def effect(self, outcome: str | list[str], *, estimand: str | list[str] | None = None,
               vcov=None, potential_outcomes: bool = False):
        """The treatment effect on `outcome` (a column, or a list: several
        outcomes in one pass with their joint vcov, see `MatchSet.effect`):
        the difference of the weighted means of the matched units, with the
        influence-function variance, weights held fixed (see
        ``match_stream.effects``). `vcov`: ``"if"`` (by unit, the default) or
        ``{"CRV1": "a+b"}`` on source columns. `estimand`
        "ate" matches the other way too (each direction with its own
        bandwidth rule); a list of estimands, with "nate", and
        `potential_outcomes` as in `MatchSet.effect`. Returns a
        `MatchResult`."""
        from .effects import effect

        return effect(self, outcome, estimand=estimand, vcov="if" if vcov is None else vcov,
                      potential_outcomes=potential_outcomes)

    def _rematch(self, estimand: str) -> KernelSet:
        from .api import _execute

        fml, data, kw = self._call
        return _execute(fml, data, estimand=estimand, **kw)

    def balance(self, covariates: list[str] | None = None):
        """Balance before and after matching, weighting by `weights`; no pair distance. Rows: the
        score (``distance``) and `covariates` (default: the formula's)."""
        from hdfe_stream.core.utils import as_lazy

        from ._prep import ROW, SCORE
        from .balance import T, balance_table

        covs = list(self.spec.covariates if covariates is None else covariates)
        src = (as_lazy(self._source).with_row_index(ROW)
               .with_columns(pl.col(ROW).cast(pl.Int64)).select(ROW, *covs))
        units = (self._narrow_lf.select(pl.col(ROW), pl.col(self.spec.treatment).alias(T),
                                        pl.col(SCORE).alias("distance"))
                 .join(src, on=ROW, how="left"))
        w = (self.weights().collect().rename({"row": ROW})
             .with_columns(pl.lit(None, pl.Int64).alias("subclass")))
        return balance_table(units, w, ["distance", *covs], self.focal_is_treated, False)

    def counts(self) -> dict:
        """Units by arm: available, matched, unmatched."""
        u = self._units
        m_t = u.filter(pl.col("treated")).height
        m_c = u.height - m_t
        n_t, n_c = self._n_treated, self._n_control
        return {"treated": n_t, "control": n_c, "matched_treated": m_t, "matched_control": m_c,
                "unmatched_treated": n_t - m_t, "unmatched_control": n_c - m_c,
                "dropped_missing": self.n_dropped}

    def summary(self) -> None:
        """Print what was matched and how."""
        s, c, b = self.spec, self.counts(), self.bandwidth_info
        how = (f"ridge (r = {s.ridge if s.ridge is not None else 'default'})"
               if s.method == "ridge" else "kernel")
        print(f"match-stream: {how} matching, {s.kernel} kernel, estimand {s.estimand.upper()}")
        print(f"  distance: {s.distance}   bandwidth: {b.h:.6g} ({b.rule})")
        if s.exact:
            print(f"  exact: {', '.join(s.exact)} ({self.n_strata:,} strata)")
        print(f"  {'':10s}{'available':>12s}{'matched':>12s}{'unmatched':>12s}")
        for arm in ("treated", "control"):
            print(f"  {arm:10s}{c[arm]:12,d}{c['matched_' + arm]:12,d}"
                  f"{c['unmatched_' + arm]:12,d}")

    def __repr__(self):
        c = self.counts()
        return (f"<KernelSet {self.spec.method}, bandwidth {self.bandwidth:.4g}: "
                f"{c['matched_treated']:,} treated and {c['matched_control']:,} controls matched>")
