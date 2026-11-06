"""effect(lone_unit_variance=): the conditional variances (n_variance_matches >= 1) of units
alone in their arm within their exact stratum. The default leaves them
undefined (a NaN standard error, with a warning); "nearest" takes their
nearest same-arm units in any stratum, "pooled" their arm's mean variance.
Checked against the variance formula with the lone unit's variance filled by
hand; R Matching sets it to 0 (docs/implementations.md)."""

import warnings

import numpy as np
import polars as pl
import pytest

from match_stream import effects as E
from match_stream import match_nn


@pytest.fixture(scope="module")
def lone_data(ref_data):
    """The reference data with one treated unit left alone in stratum r == 0
    (its other treated units dropped)."""
    d = ref_data.with_row_index("__k")
    first = d.filter((pl.col("r") == 0) & (pl.col("d") == 1))["__k"][0]
    return d.filter((pl.col("r") != 0) | (pl.col("d") == 0) | (pl.col("__k") == first)).drop("__k")


def _atc(data, **kw):
    return match_nn("d ~ x1", data, exact=["r"], score="ps_true", replace=True,
                    focal="control", verbose=False, **kw)


def _se_with(m, fill):
    """The ATC's AI SE (n_variance_matches = 1) with the lone unit's variance `fill(...)`."""
    df = E._outcomes(m, "y", [])
    s2 = E._sigma2(m, df, 1)
    lone = np.flatnonzero(np.isnan(s2))
    assert lone.size == 1
    s2[lone[0]] = fill(df, s2, lone[0])
    rows = df["__ms_row"].to_list()
    sets = [(x._pairs, x.focal_is_treated) for x in E._sets(m, "atc")]
    _, var, _ = E.ai_estimate(sets, dict(zip(rows, df["__y"].to_list())), "atc",
                              dict(zip(rows, s2.tolist())))
    return np.sqrt(var)


def test_default_is_nan_with_a_warning(lone_data):
    m = _atc(lone_data)
    with pytest.warns(UserWarning,
                      match=r"1 unit is alone.*matches in [\d,]+ pairs.*lone_unit_variance"):
        fit = m.effect("y", n_variance_matches=1)
    assert np.isnan(fit.se[0]) and fit.n_lone == 1 and fit.lone_unit_variance is None


def test_nearest(lone_data):
    m = _atc(lone_data)

    def nearest(df, s2, i):
        # the unit's nearest treated unit on the score, in any stratum
        t = df["d"].to_numpy().astype(bool)
        s, y = df["__ms_score"].to_numpy(), df["__y"].to_numpy()
        same = np.flatnonzero(t & (np.arange(t.size) != i))
        j = same[np.argmin(np.abs(s[same] - s[i]))]
        return np.var([y[i], y[j]], ddof=1)
    fit = m.effect("y", n_variance_matches=1, lone_unit_variance="nearest")
    np.testing.assert_allclose(fit.se[0], _se_with(m, nearest), rtol=1e-12)
    assert fit.n_lone == 1 and fit.lone_unit_variance == "nearest"
    assert "lone_unit_variance='nearest'" in "\n".join(fit._summary_lines())


def test_pooled(lone_data):
    m = _atc(lone_data)

    def pooled(df, s2, i):
        t = df["d"].to_numpy().astype(bool)
        return np.nanmean(s2[t])
    fit = m.effect("y", n_variance_matches=1, lone_unit_variance="pooled")
    np.testing.assert_allclose(fit.se[0], _se_with(m, pooled), rtol=1e-12)


def test_matching_sets_it_to_zero(lone_data):
    # R Matching 4.10-15 on these data (Match with exact = c(FALSE, TRUE) on
    # (ps, r), ATC, Var.calc = 1, ties = TRUE): SE 0.031494192979; its
    # estimate differs from ours in the 6th digit (R Matching's tie on the
    # reference data, docs/implementations.md "Ties"), hence the tolerance
    se0 = _se_with(_atc(lone_data), lambda df, s2, i: 0.0)
    np.testing.assert_allclose(se0, 0.031494192979, rtol=1e-6)


def test_no_lone_units_no_change(ref_data):
    m = _atc(ref_data)
    base = m.effect("y", n_variance_matches=2)
    assert base.n_lone == 0
    for fb in ("nearest", "pooled"):
        fit = m.effect("y", n_variance_matches=2, lone_unit_variance=fb)
        np.testing.assert_allclose(fit.se[0], base.se[0], rtol=1e-13)   # summation order
        assert fit.lone_unit_variance is None


def test_metric_distance(lone_data):
    m = match_nn("d ~ x1 + x2", lone_data, exact=["r"], distance="mahalanobis",
                 replace=True, focal="control", verbose=False)
    for fb in ("nearest", "pooled"):
        assert np.isfinite(m.effect("y", n_variance_matches=1, lone_unit_variance=fb).se[0])


def test_ps_correction(lone_data, ref_data):
    # the 2016 correction's own-arm sets need the lone treated unit even for
    # the ATT; with a fallback they come from other strata, or the whole arm
    m = match_nn("d ~ x1 + x2", lone_data, exact=["r"], replace=True, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(m.effect("y", n_variance_matches=1).se[0])
    for fb in ("nearest", "pooled"):
        fit = m.effect("y", n_variance_matches=1, lone_unit_variance=fb)
        assert fit.score_correction and np.isfinite(fit.se[0])
    # no lone units: the correction is unchanged
    full = match_nn("d ~ x1 + x2", ref_data, exact=["r"], replace=True, verbose=False)
    base = full.effect("y", n_variance_matches=1).se[0]
    nearest = full.effect("y", n_variance_matches=1, lone_unit_variance="nearest")
    np.testing.assert_allclose(nearest.se[0], base, rtol=1e-13)


@pytest.mark.parametrize("kw", [dict(n_variance_matches=0), dict(n_variance_matches=1, vcov="if"),
                                dict(n_variance_matches=1, var_fallback_="other")])
def test_errors(lone_data, kw):
    m = _atc(lone_data)
    kw = dict(kw)
    fb = kw.pop("var_fallback_", "nearest")
    with pytest.raises(ValueError, match="lone_unit_variance"):
        m.effect("y", lone_unit_variance=fb, **kw)
