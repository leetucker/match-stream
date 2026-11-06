"""effect(estimand=[...], potential_outcomes=True): several estimands (ATT, ATC,
ATE, kmatch's naive NATE) and potential-outcome means in one call, with their
joint influence-function vcov (effects._joint).

Checked: each coefficient and its variance against the single-estimand fit
(the diagonal); the covariances against a unit-by-unit numpy computation of
the stacked influence functions; the identity effect = y(1) - y(0) on the
joint vcov; NATE against the unweighted regression of y on d (hdfe-stream);
errors for Abadie-Imbens SEs and subclass clusters; NN with and without
replacement, kernel and CEM. Against kmatch's e(V): tests/reference/stata/
kmatch_estimands.do (skipped until the user runs it)."""

import contextlib
import io
import json

import numpy as np
import polars as pl
import pytest
from conftest import DATA
from hdfe_stream import feols_stream

from match_stream import match_cem, match_kernel, match_nn


@pytest.fixture(scope="module")
def data(ref_data):
    return ref_data.with_columns((0.5 * pl.col("y") + pl.col("x3")).alias("y2"))


@pytest.fixture(scope="module")
def nn(data):
    return match_nn("d ~ x1 + x2", data, exact=["r"], score="ps_true", replace=True,
                    id="id", verbose=False)


ALL = ["att", "atc", "ate", "nate"]
VCES = {"if": "if", "crv_r": {"CRV1": "r"}, "crv_two": {"CRV1": "r+i"}}


@pytest.mark.parametrize("vcov", list(VCES))
def test_diagonal_is_the_single_fit(nn, vcov):
    v = VCES[vcov]
    fit = nn.effect(["y", "y2"], estimand=ALL, vcov=v, potential_outcomes=True)
    assert len(fit.beta) == 2 * 4 * 3
    assert np.allclose(fit.vcov, fit.vcov.T)
    k = 0
    for y in ("y", "y2"):
        for e in ALL:
            one = nn.effect(y, estimand=e, vcov=v)
            assert fit.coefnames[k] == f"{e.upper()}:{y}"
            assert fit.beta[k] == pytest.approx(one.beta[0], rel=1e-12)
            assert fit.vcov[k, k] == pytest.approx(one.vcov[0, 0], rel=1e-12)
            k += 3                                 # the means follow each effect


def _unit_phis(nn, est, y):
    """Per-unit influence contributions of the weights-fixed difference
    (rows of the estimation table summed by unit), brute force."""
    from match_stream.effects import _if_rows
    rows, _, _ = _if_rows(nn, est, "if", None)
    tab = rows.join(nn._source.with_row_index("__unit").select("__unit", y)
                    .with_columns(pl.col("__unit").cast(pl.Int64)), on="__unit")
    u, t, w, yy = (tab[c].to_numpy() for c in ("__unit", "__treated", "__w", y))
    t = t.astype(bool)
    m1 = (w * yy)[t].sum() / w[t].sum()
    m0 = (w * yy)[~t].sum() / w[~t].sum()
    phi = np.where(t, w * (yy - m1) / w[t].sum(), -w * (yy - m0) / w[~t].sum())
    return {int(k): float(v) for k, v in
            pl.DataFrame({"u": u, "p": phi}).group_by("u").agg(pl.col("p").sum()).iter_rows()}


def test_covariance_is_the_stacked_influence_function(nn):
    fit = nn.effect("y", estimand=["att", "atc", "nate"], vcov="if")
    ph = [_unit_phis(nn, e, "y") for e in ("att", "atc", "nate")]
    units = sorted(set().union(*ph))
    P = np.array([[p.get(u, 0.0) for u in units] for p in ph])
    raw = P @ P.T
    f = np.sqrt(np.diag(fit.vcov) / np.diag(raw))
    assert np.allclose(fit.vcov, raw * np.outer(f, f), rtol=1e-9)
    # the estimands do covary (they share units)
    assert abs(fit.vcov[0, 1]) > 0 and abs(fit.vcov[0, 2]) > 0


def test_effect_is_the_difference_of_the_potential_outcomes(nn):
    fit = nn.effect("y", estimand=["att", "ate"], vcov="if", potential_outcomes=True)
    assert fit.coefnames == ["ATT:y", "ATT:y(1)", "ATT:y(0)", "ATE:y", "ATE:y(1)", "ATE:y(0)"]
    for k in (0, 3):
        assert fit.beta[k] == pytest.approx(fit.beta[k + 1] - fit.beta[k + 2], rel=1e-12)
        c = np.zeros(6)
        c[k], c[k + 1], c[k + 2] = 0, 1, -1
        assert c @ fit.vcov @ c == pytest.approx(fit.vcov[k, k], rel=1e-10)
        # and the effect row covaries with the means as the difference does
        assert fit.vcov[k, k + 1] == pytest.approx(fit.vcov[k + 1, k + 1] - fit.vcov[k + 1, k + 2],
                                                   rel=1e-9)


def test_nate_is_the_unweighted_regression(nn, data, tmp_path):
    fit = nn.effect("y", estimand="nate", vcov="if")
    sub = data.select("d", "y", "r", "i")
    ols = feols_stream("y ~ d", sub, workdir=tmp_path, vcov="hetero")
    i = ols.coefnames.index("d")
    assert fit.coefnames == ["NATE:y"]
    assert fit.beta[0] == pytest.approx(ols.beta[i], rel=1e-10)
    assert fit.vcov[0, 0] == pytest.approx(ols.vcov[i, i], rel=1e-8)


def test_errors(nn, data):
    with pytest.raises(ValueError, match="influence-function"):
        nn.effect("y", estimand=["att", "atc"])                      # default vcov is ai
    with pytest.raises(ValueError, match="influence-function"):
        nn.effect("y", estimand="att", vcov="ai", potential_outcomes=True)
    with pytest.raises(ValueError, match="source columns"):
        nn.effect("y", estimand=["att", "atc"], vcov={"CRV1": "subclass"})
    with pytest.raises(ValueError, match="distinct"):
        nn.effect("y", estimand=["att", "att"], vcov="if")
    with pytest.raises(ValueError, match="'nate'"):
        nn.effect("y", estimand=["att", "foo"], vcov="if")
    with pytest.raises(ValueError, match="n_variance_matches"):
        nn.effect("y", estimand=["att", "atc"], vcov="if", n_variance_matches=1)


def test_one_estimand_with_means_and_a_subclass_cluster(data):
    wor = match_nn("d ~ x1 + x2", data, exact=["r"], score="ps_true", id="id",
                   verbose=False)
    fit = wor.effect("y", potential_outcomes=True)               # default: CRV1 subclass
    one = wor.effect("y")
    assert fit.beta[0] == pytest.approx(one.beta[0], rel=1e-12)
    assert fit.vcov[0, 0] == pytest.approx(one.vcov[0, 0], rel=1e-12)
    assert fit.n_clusters == one.n_clusters


def test_kernel_and_cem(data):
    ker = match_kernel("d ~ x1", data, score="ps_true", verbose=False)
    fit = ker.effect(["y", "y2"], estimand=["att", "ate", "nate"], potential_outcomes=True)
    assert len(fit.beta) == 18 and np.all(np.isfinite(fit.vcov))
    one = ker.effect("y2", estimand="ate")
    k = fit.coefnames.index("ATE:y2")
    assert fit.vcov[k, k] == pytest.approx(one.vcov[0, 0], rel=1e-10)
    cem = match_cem("d ~ x1 + x2", data, cutpoints=5, verbose=False)
    fit = cem.effect("y", estimand=["att", "nate"], potential_outcomes=True)
    assert fit.coefnames == ["ATT:y", "ATT:y(1)", "ATT:y(0)", "NATE:y", "NATE:y(1)", "NATE:y(0)"]
    assert fit.vcov[0, 0] == pytest.approx(cem.effect("y").vcov[0, 0], rel=1e-12)
    other = "atc" if cem.spec.estimand == "att" else "att"
    with pytest.raises(ValueError, match="CEM's effect"):
        cem.effect("y", estimand=[other, "nate"], vcov="if")


def data_rows(ms) -> int:
    return ms._narrow_lf.select(pl.len()).collect().item()


def test_reporting(nn):
    fit = nn.effect(["y", "y2"], estimand=["att", "atc", "nate"], vcov={"CRV1": "r+i"},
                    potential_outcomes=True)
    s = fit.tidy()
    assert s.shape[0] == len(fit.beta)
    pf = fit.to_pyfixest()
    assert "y(1) (ATT)" in list(pf._coefnames)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fit.summary()
    assert "Estimands: ATT, ATC, NATE" in buf.getvalue()
    assert fit.n_focal_by["att"] == nn.effect("y", estimand="att").n_focal
    assert fit.n_focal_by["nate"] == data_rows(nn)


REF = DATA / "ref_kmatch_estimands.json"
KM_CASES = {   # name: match_stream options
    "nn_all": dict(replace=True, keep_ties=True),
    "kernel_all": dict(fn=match_kernel, kernel="epan", bandwidth=0.02),
    "ridge_all": dict(fn=match_kernel, ridge=True, kernel="epan", bandwidth=0.02),
}


@pytest.mark.skipif(not REF.exists(), reason="run tests/reference/stata/kmatch_estimands.do")
@pytest.mark.parametrize("name", list(KM_CASES))
def test_against_kmatch(data, name):
    """kmatch's e(b) and e(V) for (y y2) with `ate att atc nate po` in one
    call: the effects and the potential-outcome means (kmatch's y:ATT,
    y:Y1(ATT), y:Y0(ATT), ...). kmatch's ATE is p ATT + (1 - p) ATC with the
    share p estimated, which match-stream follows. Its scale is HC0 over the
    whole sample times n / (n - 1) for the whole matrix; match-stream's, for
    each coefficient, its own fit's G / (G - 1) (N - 1) / (N - 2) (a mean
    takes its effect's). Undone below, entry by entry, to compare."""
    ref = json.loads(REF.read_text())[name]
    if ref["rc"]:
        pytest.fail(f"kmatch stopped with return code {ref['rc']}")
    kw = dict(KM_CASES[name])
    match = kw.pop("fn", match_nn)
    n = data.height
    m = match("d ~ x1 + x2 + x3", data, exact=["r", "i"], score="ps_true", verbose=False, **kw)
    scale = []
    for y in ("y", "y2"):
        for e in ALL:
            one = m.effect(y, estimand=e, vcov="if")
            (G,), N = one.n_clusters["unit"], one.n_obs
            scale += [G / (G - 1) * (N - 1) / (N - 2)] * 3
    names = ref["names"].split()
    want = [f"{y}:{k}" for y in ("y", "y2") for e in ALL
            for k in (e.upper(), f"Y1({e.upper()})", f"Y0({e.upper()})")]
    missing = [w for w in want if w not in names]
    assert not missing, f"kmatch names are {names}; not found: {missing}"
    idx = [names.index(w) for w in want]
    fit = m.effect(["y", "y2"], estimand=ALL, vcov="if", potential_outcomes=True)
    np.testing.assert_allclose(fit.beta, np.array(ref["b"])[idx], rtol=1e-9)
    f = np.sqrt(np.array(scale))
    v = fit.vcov / np.outer(f, f) * n / (n - 1)
    np.testing.assert_allclose(v, np.array(ref["V"])[np.ix_(idx, idx)], rtol=1e-9, atol=1e-15)
