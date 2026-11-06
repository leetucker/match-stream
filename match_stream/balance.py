"""Balance diagnostics: MatchIt's ``summary(m, un = TRUE)``, out of core.

For each covariate (and the score, as ``distance``), before matching (every
unit used, weight 1) and after (the matched units, with the matching
weights): the weighted means by arm, the standardized mean difference, the
variance ratio, the mean and maximum distance between the arms' weighted
empirical CDFs, and after matching the standardized mean distance between
matched units. The conventions are MatchIt's (verified against MatchIt 4.8.2
in tests/test_balance.py):

- weighted variance: weights normalized to sum to 1, sum w (x - m)^2 / (1 -
  sum w^2); for a 0/1 covariate, m (1 - m);
- the standardized mean difference divides by the standard deviation of the
  focal arm (treated for the ATT, controls for the ATC) over all units used,
  before matching, and so does the standardized pair distance;
- the eCDF differences are taken at each distinct value among all units used
  (after matching, unmatched units weigh 0 but still contribute their values),
  with each arm's weights summing to 1; for a 0/1 covariate both are
  |mean difference|;
- the pair distance is the mean |x_t - x_c| over matched pairs, or over all
  treated-control pairs within a subclass (CEM cells).

The statistics come from the generic weighted balance table in
``_core/balance.py`` (bound for hdfe_stream.core); this module chooses the
units, the weights and the scale as MatchIt does, and adds the pair
distance and the sample sizes. Everything is sums and sorts over the units
used, run by Polars' streaming engine, so the source is never held in
memory.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ._core.balance import EPS_DIFF, effective_sizes, weighted_balance

ROW = "__ms_row"
T = "__ms_t"
W = "__ms_w"
COLUMNS = ["Means Treated", "Means Control", "Std. Mean Diff.", "Var. Ratio",
           "eCDF Mean", "eCDF Max", "Std. Pair Dist."]


def _cross_abs_sum(xt: np.ndarray, xc: np.ndarray) -> float:
    """sum over (i, j) of |xt_i - xc_j|, by sorting: O((n+m) log(n+m))."""
    xc = np.sort(xc)
    pre = np.concatenate([[0.0], np.cumsum(xc)])
    k = np.searchsorted(xc, xt, side="right")       # controls <= xt_i
    total = pre[-1]
    return float(np.sum(xt * k - pre[k] + (total - pre[k]) - xt * (len(xc) - k)))


def pair_distance(values: pl.DataFrame, x: str, pairs: pl.DataFrame | None) -> float:
    """Mean |x_t - x_c| over matched pairs (`pairs`: treated ROW, control ROW),
    or, with pairs None, over all treated-control pairs within each subclass
    of `values` (ROW, T, subclass, x), pooled over subclasses."""
    if pairs is not None:
        v = values.select(ROW, x)
        d = (pairs.join(v.rename({ROW: "t_row", x: "xt"}), on="t_row")
             .join(v.rename({ROW: "c_row", x: "xc"}), on="c_row")
             .select((pl.col("xt") - pl.col("xc")).abs().mean()))
        return float(d.item())
    num = den = 0.0
    for (_,), g in values.group_by(["subclass"]):
        t = g[T].to_numpy()
        xs = g[x].cast(pl.Float64).to_numpy()
        if t.any() and (~t).any():
            num += _cross_abs_sum(xs[t], xs[~t])
            den += t.sum() * (~t).sum()
    return num / den if den else np.nan


class BalanceTable:
    """Balance before and after matching.

    Attributes
    ----------
    all : polars.DataFrame
        Every unit used, before matching: one row per term, with the columns
        (``Means Treated``, ``Means Control``, ``Std. Mean Diff.``,
        ``Var. Ratio``, ``eCDF Mean``, ``eCDF Max``; ``Std. Pair Dist.`` is
        null).
    matched : polars.DataFrame
        The matched units, weighted by the matching weights.
    sizes : polars.DataFrame
        Sample sizes by arm: all, matched (and its effective size,
        (sum w)^2 / sum w^2), unmatched.
    """

    def __init__(self, all_: pl.DataFrame, matched: pl.DataFrame, sizes: pl.DataFrame):
        self.all = all_
        self.matched = matched
        self.sizes = sizes

    def summary(self) -> None:
        """Print both tables and the sample sizes."""
        with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_hide_dataframe_shape=True,
                       tbl_hide_column_data_types=True):
            print("Balance before matching")
            print(self.all.drop("Std. Pair Dist."))
            print("\nBalance after matching")
            print(self.matched)
            print("\nSample sizes")
            print(self.sizes)

    def __repr__(self):
        return f"<BalanceTable: {self.matched.height} terms>"


def balance_table(units: pl.LazyFrame, weights: pl.DataFrame, terms: list[str],
                  focal_is_treated: bool, pairs: pl.DataFrame | None,
                  discarded: tuple[int, int] = (0, 0)) -> BalanceTable:
    """`units`: every unit used (ROW, T, the terms); `weights`: the matched
    units (ROW, weight, subclass); `pairs`: (t_row, c_row) for matched pairs,
    None to use subclasses (CEM), or False for no pair distance (kernel).

    The statistics are the core's weighted balance table; what is MatchIt's
    here is which units and which standard deviation: the denominator is the
    focal arm's SD over all units used, before matching (the pooled SD when
    that is 0), and the matched eCDF gaps are taken at the values of every
    unit used, the unmatched ones weighing 0."""
    matched = units.join(weights.lazy().select(ROW, pl.col("weight").alias(W), "subclass"),
                         on=ROW, how="inner")
    zero_filled = units.join(weights.lazy().select(ROW, pl.col("weight").alias(W)),
                             on=ROW, how="left").with_columns(pl.col(W).fill_null(0.0))
    before = weighted_balance(units, T, terms, None, scale="pooled")
    sds = {}
    for row in before.iter_rows(named=True):
        sd = np.sqrt(row["var1"] if focal_is_treated else row["var0"])
        if not np.isfinite(sd) or sd < EPS_DIFF:
            sd = np.sqrt((row["var1"] + row["var0"]) / 2)   # MatchIt's fallback: pooled
        sds[row["term"]] = sd
    after = weighted_balance(zero_filled, T, terms, W, scale=sds)

    def smd(tab):
        d = tab["mean1"] - tab["mean0"]
        sd = pl.Series([sds[t] for t in tab["term"]])
        # a term constant in both arms has sd 0: +-inf where the means
        # differ (the core's convention), d where they do not
        with np.errstate(divide="ignore", invalid="ignore"):
            return pl.Series(np.where(np.abs(d) > EPS_DIFF, d / sd, d))

    matched_df = None
    pdist = []
    for x in terms:
        pd_ = np.nan
        if pairs is not False:
            if matched_df is None:
                matched_df = matched.collect(engine="streaming")
            pd_ = pair_distance(matched_df, x, pairs)
            if np.isfinite(pd_) and abs(pd_) > 1e-8:
                # a term constant in both arms (sd 0): infinitely far apart
                pd_ = pd_ / sds[x] if sds[x] != 0 else np.inf
        pdist.append(pd_)

    def matchit(tab, pd_):
        return pl.DataFrame({"term": tab["term"], "Means Treated": tab["mean1"],
                             "Means Control": tab["mean0"], "Std. Mean Diff.": smd(tab),
                             "Var. Ratio": tab["var_ratio"].fill_null(np.nan),
                             "eCDF Mean": tab["ecdf_mean"], "eCDF Max": tab["ecdf_max"],
                             "Std. Pair Dist.": pl.Series(pd_, dtype=pl.Float64)})

    m = effective_sizes(matched, T, W)
    n = units.select(pl.col(T).sum().alias("n1"), (~pl.col(T)).sum().alias("n0")) \
        .collect(engine="streaming").row(0, named=True)
    n1, n0 = int(n["n1"]), int(n["n0"])
    d1, d0 = discarded
    # MatchIt's nn table; "All (ESS)" is "All" without sampling weights, and
    # the unmatched exclude the discarded
    sizes = pl.DataFrame({
        "sample": ["All (ESS)", "All", "Matched (ESS)", "Matched", "Unmatched", "Discarded"],
        "Control": [float(n0), float(n0), m.ess0, float(m.n0), float(n0 - m.n0 - d0), float(d0)],
        "Treated": [float(n1), float(n1), m.ess1, float(m.n1), float(n1 - m.n1 - d1), float(d1)],
    })
    return BalanceTable(matchit(before, [np.nan] * len(terms)), matchit(after, pdist), sizes)
