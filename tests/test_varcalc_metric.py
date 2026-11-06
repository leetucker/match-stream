"""Abadie-Imbens standard errors with conditional variances (n_variance_matches > 0)
when matching on covariates: against R Matching 4.10-15
(tests/reference/reference_r_varcalc.R; its default metric is
distance="ivariance"), and the neighbor sets against a direct computation
with heavy ties."""

import json

import numpy as np
import polars as pl
import pytest
from conftest import DATA

from match_stream import match_nn
from match_stream._kdtree import same_arm_variances

REF = json.loads((DATA / "ref_r_varcalc.json").read_text())
CASES = {   # name: (estimand, k, n_variance_matches)
    "att_m1_v0": ("att", 1, 0), "att_m1_v2": ("att", 1, 2), "att_m2_v4": ("att", 2, 4),
    "atc_m2_v3": ("atc", 2, 3), "ate_m1_v2": ("ate", 1, 2),
}


@pytest.mark.parametrize("name", list(CASES))
def test_r_matching(name):
    estimand, k, vc = CASES[name]
    df = pl.read_csv(DATA / "ref_data.csv")
    m = match_nn("d ~ x1 + x2 + x3", df, distance="ivariance", exact=["r", "i"],
                 replace=True, n_matches=k, focal="control" if estimand == "atc" else "treated")
    fit = m.effect("y", estimand=estimand, n_variance_matches=vc)
    np.testing.assert_allclose(fit.beta[0], REF[name]["est"], rtol=1e-10)
    np.testing.assert_allclose(np.sqrt(fit.vcov[0, 0]), REF[name]["se"], rtol=1e-10)


def _direct(z, g, t, y, J):
    out = np.full(t.size, np.nan)
    for i in range(t.size):
        o = np.flatnonzero((t == t[i]) & (g == g[i]))
        o = o[o != i]
        if o.size == 0:
            continue
        d = np.sqrt(((z[o] - z[i]) ** 2).sum(1))
        dJ = np.sort(d)[min(J, o.size) - 1]
        out[i] = np.concatenate([[y[i]], y[o[d <= dJ]]]).var(ddof=1)
    return out


@pytest.mark.parametrize("J", [1, 2, 4])
@pytest.mark.parametrize("discrete", [False, True])
def test_neighbor_sets_with_ties(J, discrete):
    rng = np.random.default_rng(J + 10 * discrete)
    n = 600
    z = rng.standard_normal((n, 2))
    if discrete:
        z = np.round(z)                                 # many exact ties
    g = rng.integers(0, 12, n)
    g[:3] = 99                                          # a stratum of three
    t = rng.random(n) < 0.4
    y = rng.standard_normal(n)
    np.testing.assert_allclose(same_arm_variances(z, g, t, y, J), _direct(z, g, t, y, J),
                               rtol=1e-12, equal_nan=True)


def test_mahvars_and_mahalanobis():
    df = pl.read_csv(DATA / "ref_data.csv")
    for kw in (dict(distance="mahalanobis"), dict(distance="ps", mahalanobis_vars=["x1", "x2"])):
        m = match_nn("d ~ x1 + x2 + x3", df, exact=["r", "i"], replace=True,
                     verbose=False, **kw)
        a, b = m.effect("y"), m.effect("y", n_variance_matches=2)
        assert np.isfinite(b.vcov[0, 0]) and b.vcov[0, 0] != a.vcov[0, 0]


def test_lone_unit_in_stratum_warns(ref_data):
    # one treated unit alone in its stratum: with n_variance_matches >= 1 its conditional
    # variance is undefined; the ATC variance needs it, so the SE is NaN, loudly
    import polars as pl

    from match_stream import match_nn
    keep = ref_data.with_row_index("__k").filter(
        (pl.col("r") != 0) | (pl.col("i") != 0) | ~pl.col("d").cast(pl.Boolean)
        | (pl.col("__k") == ref_data.with_row_index("__k").filter(
            (pl.col("r") == 0) & (pl.col("i") == 0) & pl.col("d").cast(pl.Boolean))["__k"][0]))
    m = match_nn("d ~ x1", keep.drop("__k"), exact=["r", "i"], score="ps_true",
                 replace=True, focal="control", verbose=False)
    with pytest.warns(UserWarning, match="alone in their arm"):
        fit = m.effect("y", vcov="ai", n_variance_matches=1)
    assert np.isnan(fit.se[0])
