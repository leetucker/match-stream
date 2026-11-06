"""effect([y1, y2, ...]): several outcomes in one read of the source, with
their joint vcov.

The joint vcov rests on every variance being a quadratic form in the
outcome, so that Cov(y1, y2) = [V(y1 + y2) - V(y1) - V(y2)] / 2 (effects.
_several). Checked here: the diagonal against the single-outcome fits; the
quadratic form itself, on a combination it was not built from
(V(2 y1 - 3 y2) = c'Vc), for every kind of standard error; and the
Abadie-Imbens covariance against R Matching's SEs on y1, y2 and y1 + y2
(tests/reference/reference_r_several.R; no package reports the
cross-outcome covariance itself)."""

import json
import warnings

import numpy as np
import polars as pl
import pytest
from conftest import DATA

from match_stream import match_cem, match_kernel, match_nn

REF = DATA / "ref_r_several.json"


@pytest.fixture(scope="module")
def data(ref_data):
    return ref_data.with_columns(
        (0.5 * pl.col("y") + pl.col("x3")).alias("y2"),
        (pl.col("x1") ** 2).alias("y3"),
        (2 * pl.col("y") - 3 * (0.5 * pl.col("y") + pl.col("x3"))).alias("comb"))


@pytest.fixture(scope="module")
def nn(data):
    return match_nn("d ~ x1 + x2", data, exact=["r"], score="ps_true", replace=True,
                    id="id", verbose=False)


KINDS = {
    "ai": dict(),
    "ai_vc2": dict(n_variance_matches=2),
    "ai_bias": dict(n_variance_matches=1, bias_adjustment=True),
    "ai_ate": dict(estimand="ate", n_variance_matches=1),
    "ai_pooled": dict(n_variance_matches=1, lone_unit_variance="pooled"),
    "if": dict(vcov="if"),
    "crv1_twoway": dict(vcov={"CRV1": "subclass+id"}),
}


@pytest.mark.parametrize("kind", list(KINDS))
def test_diagonal_and_quadratic_form(nn, kind):
    kw = KINDS[kind]
    fit = nn.effect(["y", "y2", "y3"], **kw)
    assert fit.coefnames[1].endswith(":y2") and fit.outcomes == ["y", "y2", "y3"]
    for i, o in enumerate(fit.outcomes):
        one = nn.effect(o, **kw)
        assert fit.beta[i] == pytest.approx(one.beta[0], rel=1e-14)
        assert fit.vcov[i, i] == pytest.approx(one.vcov[0, 0], rel=1e-12)
    # a combination the joint vcov was not built from
    c = np.array([2.0, -3.0])
    comb = nn.effect("comb", **kw)
    two = nn.effect(["y", "y2"], **kw)
    assert comb.beta[0] == pytest.approx(c @ two.beta, rel=1e-10)
    assert comb.vcov[0, 0] == pytest.approx(c @ two.vcov @ c, rel=1e-9)
    assert np.allclose(two.vcov, two.vcov.T)


def test_kernel_cem_and_fitted_score(data):
    for m in (match_kernel("d ~ x1", data, score="ps_true", verbose=False),
              match_cem("d ~ x1 + x2", data, cutpoints=5, verbose=False),
              match_nn("d ~ x1 + x2", data, exact=["r"], replace=True, verbose=False)):
        kw = dict(n_variance_matches=1) if m.spec.method == "nn" else {}
        fit = m.effect(["y", "y2"], **kw)
        c = np.array([2.0, -3.0])
        assert m.effect("comb", **kw).vcov[0, 0] == pytest.approx(c @ fit.vcov @ c, rel=1e-9)
        if m.spec.method == "nn":
            assert fit.score_correction


@pytest.mark.skipif(not REF.exists(), reason="run tests/reference/reference_r_several.R")
@pytest.mark.parametrize("case,estimand,n_variance_matches,bias", [
    ("att_vc0", "att", 0, False), ("att_vc2", "att", 2, False),
    ("ate_vc2", "ate", 2, False), ("att_vc2_bias", "att", 2, True)])
def test_ai_covariance_against_r_matching(data, case, estimand, n_variance_matches, bias):
    ref = json.loads(REF.read_text())[case]
    m = match_nn("d ~ x1 + x2 + x3", data, exact=["r", "i"], score="ps_true",
                 replace=True, n_matches=4, keep_ties=True, verbose=False)
    fit = m.effect(["y", "y2"], estimand=estimand, n_variance_matches=n_variance_matches,
                   bias_adjustment=bias)
    v1, v2, vs = (ref[y]["se"] ** 2 for y in ("y", "y2", "ysum"))
    np.testing.assert_allclose(fit.beta, [ref["y"]["est"], ref["y2"]["est"]], rtol=1e-10)
    np.testing.assert_allclose(np.diag(fit.vcov), [v1, v2], rtol=1e-8)
    np.testing.assert_allclose(fit.vcov[0, 1], (vs - v1 - v2) / 2, rtol=1e-7)


def test_one_read_and_errors(nn, monkeypatch):
    import match_stream.effects as E
    calls = []
    orig = E._source_rows

    def spy(ms, name):
        calls.append(getattr(ms, "_y_cache", None) is not None)
        return orig(ms, name)
    monkeypatch.setattr(E, "_source_rows", spy)
    nn.effect(["y", "y2", "y3"], n_variance_matches=1)
    assert calls and all(calls)               # every outcome read from the cache
    assert nn._y_cache is None
    with pytest.raises(ValueError, match="distinct"):
        nn.effect(["y", "y"])
    assert nn.effect(["y"]).coefnames == ["ATT:y"]


def test_reporting(nn):
    from hdfe_stream import etable
    fit = nn.effect(["y", "y2"], n_variance_matches=1)
    view = fit.to_pyfixest()
    assert view._coefnames == ["y (ATT)", "y2 (ATT)"]
    tab = etable([fit, nn.effect("y")], type="df")
    rows = [r[1] for r in tab.index if r[0] == "coef"]
    assert rows[:2] == ["y (ATT)", "y2 (ATT)"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit.summary()


KM = DATA / "ref_kmatch_several.json"
KM_CASES = {   # name: (match_stream options, estimand)
    "nn_att": (dict(replace=True, keep_ties=True), "att"),
    "kernel_att": (dict(fn=match_kernel, kernel="epan", bandwidth=0.02), "att"),
    "ridge_atc": (dict(fn=match_kernel, ridge=True, kernel="epan", bandwidth=0.02), "atc"),
}


@pytest.mark.skipif(not KM.exists(), reason="run tests/reference/stata/kmatch_several.do")
@pytest.mark.parametrize("name", list(KM_CASES))
def test_influence_function_vcov_against_kmatch(data, name):
    # kmatch's e(V) for (y y2) in one call: its joint influence-function vcov.
    # Its scale: HC0 over the whole sample times n / (n - 1); match-stream's
    # (one factor for the whole matrix): G / (G - 1) (N - 1) / (N - 2)
    ref = json.loads(KM.read_text())[name]
    if ref["rc"]:
        pytest.fail(f"kmatch stopped with return code {ref['rc']}")
    kw, estimand = KM_CASES[name]
    kw = dict(kw)
    match = kw.pop("fn", match_nn)
    m = match("d ~ x1 + x2 + x3", data, exact=["r", "i"], score="ps_true",
              focal="control" if estimand == "atc" else "treated", verbose=False, **kw)
    fit = m.effect(["y", "y2"], vcov="if")
    (G,), N, n = fit.n_clusters["unit"], fit.n_obs, data.height
    v = fit.vcov / (G / (G - 1) * (N - 1) / (N - 2)) * n / (n - 1)
    # first run (kmatch 1.1.6, Stata 18): estimates and the whole matrix to 3e-15
    assert ref["names"].split()[0].startswith("y:") and ref["names"].split()[1].startswith("y2:")
    np.testing.assert_allclose(fit.beta, ref["b"][:2], rtol=1e-12)
    np.testing.assert_allclose(v, np.array(ref["V"])[:2, :2], rtol=1e-12)
