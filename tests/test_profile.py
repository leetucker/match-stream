"""profile=True (match_stream/profile.py) and n_threads= (match_stream/_threads.py)."""

import numpy as np
import polars as pl
import pytest
from hdfe_stream.core.profile import Profile

import match_stream._threads as th
from match_stream import match_cem, match_kernel, match_nn, simulate_matching


@pytest.fixture(scope="module")
def df():
    return simulate_matching(60_000, seed=4).rename({"ps": "s"})


def test_profile_is_a_profile_with_tallies(df):
    p = match_nn("d ~ x1 + x2 + x3", df, exact=["r", "i"], score="s", caliper=0.2,
                 profile=True)
    assert isinstance(p, Profile)
    g = df.group_by(["r", "i"]).agg(pl.len().alias("n"), pl.col("d").sum().alias("t"))
    n = p.notes
    assert n["rows"] == df.height and n["strata"] == g.height
    assert n["treated"] == df["d"].sum() and n["largest_stratum_rows"] == g["n"].max()
    assert n["brute_force_pairs"] == int((g["t"] * (g["n"] - g["t"])).sum())
    assert p.recommend().pbs.startswith("select=1:ncpus=")
    assert "pair table" in str(p)


def test_candidate_pairs_are_close_and_bounded(df):
    p = match_nn("d ~ x1", df, exact=["r", "i"], score="s", caliper=0.2, profile=True)
    cp = p.notes["candidate_pairs"]
    w = cp["within"]
    actual = 0
    for _, g in df.group_by(["r", "i"]):
        t = np.sort(g.filter(pl.col("d") == 1)["s"].to_numpy())
        c = np.sort(g.filter(pl.col("d") == 0)["s"].to_numpy())
        actual += int((np.searchsorted(c, t + w, side="right")
                       - np.searchsorted(c, t - w, side="left")).sum())
    assert cp["upper"] >= actual
    assert abs(cp["expected"] / actual - 1) < 0.05


@pytest.mark.parametrize("limit", [100.0, 0.001])
def test_profile_path_is_the_path_taken(df, limit):
    kw = dict(exact=["r", "i"], score="s", caliper=0.2, max_memory_gb=limit)
    p = match_nn("d ~ x1", df, profile=True, **kw)
    m = match_nn("d ~ x1", df, **kw)
    assert p.notes["memory_path"] == m.memory


def test_fitted_score_kernel_and_cem(df):
    p = match_nn("d ~ x1 + x2 + x3", df, exact=["r"], replace=True, profile=True,
                 verbose=False)
    assert "score" in p.phases() and "match" in p.phases()
    assert "candidate_pairs" not in p.notes            # the score is not known yet
    k = match_kernel("d ~ x1", df, score="s", bandwidth=0.01, profile=True)
    assert "candidate_pairs" in k.notes and k.memory_gb[0] > 0
    c = match_cem("d ~ x1 + x2", df, cutpoints=5, profile=True)
    assert c.memory_gb[0] > 0
    md = match_nn("d ~ x1 + x2 + x3", df, distance="mahalanobis", profile=True)
    assert any(x.name == "k-d tree" for x in md.components)


def test_n_threads(df):
    kw = dict(exact=["r"], distance="mahalanobis", n_matches=2)
    a = match_nn("d ~ x1 + x2", df, **kw).matches().collect()
    b = match_nn("d ~ x1 + x2", df, n_threads=1, **kw)
    assert th.WORKERS == -1                              # restored after the match
    assert a.equals(b.matches().collect())
    r = match_nn("d ~ x1 + x2", df, score="s", replace=True, n_threads=1)
    assert np.isfinite(r.effect("y", n_variance_matches=2).vcov[0, 0])
