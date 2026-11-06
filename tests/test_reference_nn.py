"""Nearest-neighbor matching against MatchIt 4.8.2 (tests/data/ref_matchit_*):
the same pairs, treated unit by treated unit, and the same weights."""

import numpy as np
import polars as pl
import pytest
from conftest import DATA, matchit_pairs, matchit_weights

from match_stream import match_cem, match_kernel, match_nn, simulate_matching

CAL3 = {"ps_true": 0.2, "x1": 0.25, "x2": 0.25}
SPECS = {
    "nn_ps_rep": dict(replace=True, caliper=0.2),
    "nn_ps_rep_k3_cal": dict(replace=True, n_matches=3, caliper=CAL3),
    "nn_ps_wor_largest": dict(match_order="largest", caliper=CAL3),
    "nn_ps_wor_smallest": dict(match_order="smallest", caliper=CAL3),
    "nn_ps_wor_closest": dict(match_order="closest", caliper=CAL3),
    "nn_ps_wor_data": dict(match_order="data", caliper=CAL3),
    "nn_ps_wor_k2": dict(n_matches=2, match_order="largest", caliper=0.2),
}


def ours(m) -> dict[int, set[int]]:
    t = m.matches().group_by("treated_id").agg(pl.col("control_id")).collect()
    return {a: set(b) for a, b in t.iter_rows()}


@pytest.fixture(scope="module")
def fits(ref_data):
    return {name: match_nn("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"],
                           score="ps_true", id="id", **kw)
            for name, kw in SPECS.items()}


@pytest.mark.parametrize("spec", list(SPECS))
def test_pairs_match_matchit(fits, spec):
    ref = {t: c for t, c in matchit_pairs(spec).items() if c}
    got = ours(fits[spec])
    assert got.keys() == ref.keys()
    assert all(got[t] == ref[t] for t in ref)


@pytest.mark.parametrize("spec", list(SPECS))
def test_weights_match_matchit(fits, spec):
    ref = matchit_weights(spec).select("id", pl.col("weight").alias("ref"))
    got = fits[spec].weights().collect().select("id", "weight")
    both = ref.join(got, on="id", how="full", coalesce=True)
    assert both["weight"].null_count() == 0 and both["ref"].null_count() == 0
    np.testing.assert_allclose(both["weight"].to_numpy(), both["ref"].to_numpy(), rtol=1e-10)


MAHA_WOR = DATA / "ref_matchit_nn_maha_wor_cal.csv.gz"


@pytest.mark.skipif(not MAHA_WOR.exists(), reason="run tests/reference/reference_r.R")
def test_mahalanobis_without_replacement_with_covariate_calipers(ref_data):
    # the benchmarks' scenario H
    m = match_nn("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], distance="mahalanobis",
                 replace=False, caliper={"x1": 0.25, "x2": 0.25}, id="id")
    ref = {t: c for t, c in matchit_pairs("nn_maha_wor_cal").items() if c}
    got = ours(m)
    assert got.keys() == ref.keys()
    assert all(got[t] == ref[t] for t in ref)


def test_subclass_partition_without_replacement(fits):
    ref = matchit_weights("nn_ps_wor_largest")
    got = fits["nn_ps_wor_largest"].weights().collect()

    def groups(df):
        return {frozenset(g) for g in df.group_by("subclass").agg("id")["id"].to_list()}

    assert groups(got) == groups(ref)


def test_fitted_score_with_absorbed_fe(ref_data):
    # MatchIt fitted glm(d ~ x1 + x2 + x3 + factor(r) + factor(i)); feglm_stream
    # absorbs r and i, the same likelihood, so the scores agree to ~1e-9
    m = match_nn("d ~ x1 + x2 + x3", ref_data, score_model="d ~ x1 + x2 + x3 | r + i",
                 replace=True, caliper=0.2, id="id")
    ref = {t: c for t, c in matchit_pairs("nn_glm_rep").items() if c}
    got = ours(m)
    assert got.keys() == ref.keys()
    assert sum(got[t] == ref[t] for t in ref) == len(ref)


def test_simulate_reproduces_reference_data(ref_data):
    sim = simulate_matching().rename({"ps": "ps_true"})
    for c in ref_data.columns:
        # the CSV keeps 17 decimal places, so tiny values lose relative precision
        np.testing.assert_allclose(sim[c].to_numpy(), ref_data[c].to_numpy(),
                                   rtol=1e-15, atol=1e-16)


@pytest.mark.parametrize("spec", ["nn_ps_rep_k3_cal", "nn_ps_wor_largest", "nn_ps_wor_closest"])
def test_by_stratum_path_gives_the_same_pairs(ref_data, tmp_path, spec):
    kw = dict(exact=["r", "i"], score="ps_true", id="id", **SPECS[spec])
    a = match_nn("d ~ x1", ref_data, memory="in_memory", **kw)
    b = match_nn("d ~ x1", ref_data, memory="by_stratum", workdir=tmp_path, **kw)
    assert b.memory == "by_stratum"
    sort = ["treated_row", "rank"]
    assert a.matches().collect().sort(sort).equals(b.matches().collect().sort(sort))
    wa, wb = a.weights().collect(), b.weights().collect()
    assert wa.drop("weight").equals(wb.drop("weight"))
    # the same weights; sums over the batches may differ in the last bit
    np.testing.assert_allclose(wa["weight"].to_numpy(), wb["weight"].to_numpy(), rtol=1e-12)


def test_random_order_is_reproducible_and_path_independent(ref_data, tmp_path):
    kw = dict(exact=["r", "i"], score="ps_true", match_order="random", seed=7, caliper=0.2)
    a = match_nn("d ~ x1", ref_data, **kw).matches().collect()
    b = match_nn("d ~ x1", ref_data, memory="by_stratum", workdir=tmp_path, **kw)
    c = match_nn("d ~ x1", ref_data, **{**kw, "seed": 8}).matches().collect()
    assert a.sort("treated_row").equals(b.matches().collect().sort("treated_row"))
    assert not a.sort("treated_row").equals(c.sort("treated_row"))


def test_cem_weights_match_matchit(ref_data):
    m = match_cem("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], cutpoints=5,
                  id="id")
    ref = matchit_weights("cem5").select("id", pl.col("weight").alias("ref"),
                                         pl.col("subclass").alias("ref_sub"))
    got = m.weights().collect()
    both = ref.join(got, on="id", how="full", coalesce=True)
    assert both["weight"].null_count() == 0 and both["ref"].null_count() == 0
    np.testing.assert_allclose(both["weight"].to_numpy(), both["ref"].to_numpy(), rtol=1e-10)
    # the same cells, whatever their numbering
    pairs = both.select("subclass", "ref_sub").unique()
    assert pairs["subclass"].n_unique() == pairs["ref_sub"].n_unique() == pairs.height


def test_mahalanobis_matches_matchit(ref_data):
    # MatchIt's covariance is the pooled within-group one
    m = match_nn("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], distance="mahalanobis",
                 replace=True, id="id")
    ref = {t: c for t, c in matchit_pairs("nn_maha_rep").items() if c}
    got = ours(m)
    assert got.keys() == ref.keys()
    assert all(got[t] == ref[t] for t in ref)


@pytest.mark.parametrize("distance", ["ivariance", "euclidean"])
def test_other_metrics_equal_brute_force(ref_data, distance):
    from scipy.spatial.distance import cdist
    sub = ref_data.filter(pl.col("r") == 0, pl.col("i") == 0)
    m = match_nn("d ~ x1 + x2 + x3", sub, distance=distance, replace=True)
    X = sub.select("x1", "x2", "x3").to_numpy()
    if distance == "ivariance":
        X = X / X.std(axis=0, ddof=1)
    d = sub["d"].to_numpy() == 1
    t, c = np.flatnonzero(d), np.flatnonzero(~d)
    best = c[cdist(X[t], X[c]).argmin(axis=1)]
    got = dict(m.matches().select("treated_row", "control_row").collect().iter_rows())
    assert got == dict(zip(t.tolist(), best.tolist()))


def test_mahalanobis_without_replacement_by_stratum_same(ref_data, tmp_path):
    kw = dict(exact=["r", "i"], distance="mahalanobis", caliper={"x1": 0.3})
    a = match_nn("d ~ x1 + x2 + x3", ref_data, memory="in_memory", **kw)
    b = match_nn("d ~ x1 + x2 + x3", ref_data, memory="by_stratum", workdir=tmp_path, **kw)
    pa = a.matches().collect().sort("treated_row")
    assert pa.equals(b.matches().collect().sort("treated_row"))
    assert pa["control_row"].n_unique() == pa.height
    x1 = ref_data["x1"].to_numpy()
    assert np.all(np.abs(x1[pa["treated_row"]] - x1[pa["control_row"]]) <= 0.3)


@pytest.mark.parametrize("spec", ["nn_ps_rep_k3_cal", "nn_ps_wor_k2"])
def test_min_matches_drops_short_focal_units(ref_data, fits, spec):
    # MatchIt keeps a treated unit with fewer than k matches (the default);
    # min_matches=k drops it (kmatch without keepall); the others keep their
    # matches (without replacement too: the filter runs after matching)
    kw = dict(SPECS[spec])
    k = kw["n_matches"]
    m = match_nn("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], score="ps_true",
                 id="id", min_matches=k, **kw)
    ref = {t: c for t, c in matchit_pairs(spec).items() if c}
    short = {t for t, c in ref.items() if len(c) < k}
    assert short                                   # the case exercises the option
    assert ours(m) == {t: c for t, c in ref.items() if t not in short}
    assert ours(fits[spec]) == ref                 # default unchanged
    c = m.counts()
    assert c["unmatched_treated"] == fits[spec].counts()["unmatched_treated"] + len(short)


def test_min_matches_checks(ref_data):
    with pytest.raises(ValueError, match="min_matches"):
        match_nn("d ~ x1", ref_data, score="ps_true", n_matches=2, min_matches=3)
    with pytest.raises(TypeError, match="min_matches"):
        match_kernel("d ~ x1", ref_data, score="ps_true", min_matches=2)
