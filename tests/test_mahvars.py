"""Mahalanobis matching within a score caliper (mahalanobis_vars=) against MatchIt
4.8.2 (tests/reference/reference_r_mahvars.R -> ref_r_mahvars.json and
ref_matchit_mahvars_*): the matches, the weights, the balance tables and the
sample sizes."""

import json

import numpy as np
import polars as pl
import pytest
from conftest import DATA
from test_balance import _compare

import match_stream._kdtree as kd
from match_stream import match_nn

REF = json.loads((DATA / "ref_r_mahvars.json").read_text())
MV = ["x1", "x2", "x3"]
SPECS = {
    "wor_cal": dict(score="ps_true", mahalanobis_vars=MV, caliper=0.2, exact=["r", "i"]),
    "rep_k2": dict(score="ps_true", mahalanobis_vars=MV, caliper=0.1, replace=True, n_matches=2,
                   exact=["r", "i"]),
    "glm_discard": dict(distance="ps", mahalanobis_vars=["x1", "x2"], caliper=0.1, discard="both"),
    "closest": dict(score="ps_true", mahalanobis_vars=MV, caliper=0.25, match_order="closest",
                    exact=["r", "i"]),
    "atc_rep": dict(score="ps_true", mahalanobis_vars=MV, caliper=0.2, replace=True,
                    focal="control"),
}


def _fit(ref_data, name):
    return match_nn("d ~ x1 + x2 + x3", ref_data, id="id", verbose=False, **SPECS[name])


@pytest.fixture(scope="module")
def fits(ref_data):
    return {name: _fit(ref_data, name) for name in SPECS}


def _ours(m):
    t = m.matches().collect()
    foc, oth = ("treated_id", "control_id") if m.focal_is_treated else \
        ("control_id", "treated_id")
    return {a: set(b) for a, b in t.group_by(foc).agg(pl.col(oth)).iter_rows()}


@pytest.mark.parametrize("name", list(SPECS))
def test_matches(fits, name):
    ref = pl.read_csv(DATA / f"ref_matchit_mahvars_{name}.csv.gz", null_values="NA")
    ref = {r[0]: {x for x in r[1:] if x is not None} for r in ref.iter_rows()}
    assert _ours(fits[name]) == {f: c for f, c in ref.items() if c}


@pytest.mark.parametrize("name", list(SPECS))
def test_weights_balance_and_sizes(fits, name):
    m = fits[name]
    ref = pl.read_csv(DATA / f"ref_matchit_mahvars_{name}_weights.csv.gz")
    got = m.weights().collect().select("id", "weight")
    both = ref.join(got, on="id", how="full", coalesce=True)
    np.testing.assert_allclose(both["weight_right"].to_numpy(), both["weight"].to_numpy(),
                               rtol=1e-10)
    b = m.balance()
    _compare(b.all, REF[name]["balance_all"])
    _compare(b.matched, REF[name]["balance_matched"])
    control, treated = REF[name]["nn"]
    np.testing.assert_allclose(b.sizes["Control"].to_numpy(), control, rtol=1e-10)
    np.testing.assert_allclose(b.sizes["Treated"].to_numpy(), treated, rtol=1e-10)


@pytest.mark.parametrize("name", ["wor_cal", "rep_k2"])
def test_brute_force_gives_the_same(ref_data, fits, monkeypatch, name):
    monkeypatch.setattr(kd, "KDTREE_MAX_DIM", 0)
    assert _ours(_fit(ref_data, name)) == _ours(fits[name])


def test_checks(ref_data):
    with pytest.raises(ValueError, match="score distance"):
        match_nn("d ~ x1", ref_data, distance="mahalanobis", mahalanobis_vars=MV)
    with pytest.raises(ValueError, match="column names"):
        match_nn("d ~ x1", ref_data, score="ps_true", mahalanobis_vars=["log(x1)"])
    m = match_nn("d ~ x1 + x2 + x3", ref_data, distance="ps", mahalanobis_vars="x1 + x2",
                 replace=True, verbose=False)
    assert m.spec.mahalanobis_vars == ["x1", "x2"]
    assert np.isfinite(m.effect("y", n_variance_matches=1).vcov[0, 0])     # on the mahalanobis_vars
    with pytest.raises(ValueError, match="matching on the score"):
        m.effect("y", score_correction=True)
    assert not m.effect("y").score_correction
