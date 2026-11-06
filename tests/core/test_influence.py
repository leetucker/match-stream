"""match_stream._core.influence (bound for hdfe_stream.core.influence):
the weights-fixed weighted mean difference and its clustered variance,
against hdfe-stream's weighted least squares (itself checked against
pyfixest/fixest) and a direct numpy computation."""

import numpy as np
import polars as pl
import pytest
from hdfe_stream import feols_stream

from match_stream._core.influence import cluster_dims, if_vcov, weighted_mean_difference


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(7)
    n = 3000
    a = rng.integers(0, 40, n)
    b = rng.integers(0, 25, n)
    d = rng.random(n) < 0.3
    y = 1 + 0.5 * d + 0.3 * rng.standard_normal(40)[a] + 0.2 * rng.standard_normal(25)[b] \
        + rng.standard_normal(n)
    w = rng.exponential(1.0, n)
    return pl.DataFrame({"y": y, "d": d.astype(np.int64), "w": w, "a": a, "b": b})


@pytest.mark.parametrize("vcov,cluster", [("hetero", None), ({"CRV1": "a"}, "a"),
                                          ({"CRV1": "a+b"}, "a+b")])
def test_equals_weighted_least_squares(df, tmp_path, vcov, cluster):
    fit = feols_stream("y ~ d", df, workdir=tmp_path, weights="w", vcov=vcov)
    r = weighted_mean_difference(df, "y", "w", "d", cluster)
    i = fit.coefnames.index("d")
    np.testing.assert_allclose(r.estimate, fit.beta[i], rtol=1e-10)
    np.testing.assert_allclose(r.variance.vcov[0, 0], fit.vcov[i, i], rtol=1e-8)


def test_matches_a_direct_computation(df):
    y, d, w, a = (df[c].to_numpy() for c in ("y", "d", "w", "a"))
    d = d.astype(bool)
    m1, m0 = np.average(y[d], weights=w[d]), np.average(y[~d], weights=w[~d])
    phi = np.where(d, w * (y - m1) / w[d].sum(), -w * (y - m0) / w[~d].sum())
    s = np.bincount(a, weights=phi)
    G = len(s)
    r = weighted_mean_difference(df, "y", "w", "d", "a", df_k=None)
    np.testing.assert_allclose(r.estimate, m1 - m0, rtol=1e-12)
    np.testing.assert_allclose(r.variance.vcov[0, 0], G / (G - 1) * np.sum(s**2), rtol=1e-10)
    assert r.variance.n_clusters == {"a": G} and r.variance.df == G - 1


def test_two_way_is_inclusion_exclusion(df):
    d = df.with_columns(pl.col("y").alias("p"))
    va = if_vcov(d, ["p"], "a", df_k=None).vcov / (40 / 39)
    vb = if_vcov(d, ["p"], "b", df_k=None).vcov / (25 / 24)
    ab = d.with_columns((pl.col("a") * 100 + pl.col("b")).alias("ab"))
    vab = if_vcov(ab, ["p"], "ab", df_k=None)
    vab = vab.vcov / vab.adj
    two = if_vcov(d, ["p"], ["a", "b"], df_k=None)
    np.testing.assert_allclose(two.vcov, (25 / 24) * (va + vb - vab), rtol=1e-10)  # G_min = 25


def test_several_columns_and_no_clustering(df):
    d = df.with_columns((pl.col("y") * 2).alias("q"))
    v = if_vcov(d, ["y", "q"])
    Y = df["y"].to_numpy()
    n = len(Y)
    want = n / (n - 1) * np.array([[1, 2], [2, 4]]) * np.sum(Y**2)
    np.testing.assert_allclose(v.vcov, want, rtol=1e-10)


def test_checks(df):
    assert cluster_dims(" a + b ") == ["a", "b"] and cluster_dims(None) == []
    with pytest.raises(ValueError, match="repeated"):
        cluster_dims("a+a")
    with pytest.raises(ValueError, match="missing values"):
        if_vcov(df.with_columns(pl.lit(None).alias("c")), ["y"], "c")
    neg = df.with_columns(pl.when(pl.col("a") == 0).then(-0.1).otherwise(pl.col("w")).alias("w"))
    with pytest.raises(ValueError, match="negative weight"):
        weighted_mean_difference(neg, "y", "w", "d")
    r = weighted_mean_difference(neg, "y", "w", "d", allow_negative=True)   # ridge weights
    yv, dv, wv = (neg[c].to_numpy() for c in ("y", "d", "w"))
    dv = dv.astype(bool)
    want = (wv[dv] * yv[dv]).sum() / wv[dv].sum() - (wv[~dv] * yv[~dv]).sum() / wv[~dv].sum()
    np.testing.assert_allclose(r.estimate, want, rtol=1e-12)
    with pytest.raises(ValueError, match="positive total weight"):
        weighted_mean_difference(df.with_columns(pl.lit(1).alias("d")), "y", "w", "d")
