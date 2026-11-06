"""effect() with influence-function standard errors (weights held fixed),
for NN with and without replacement and CEM.

The reference is the weighted regression y ~ treated on MatchIt's
estimation table (one row per unit and subclass, as get_matches()), with
HC1 / CRV1 standard errors, fitted here by hdfe-stream's feols_stream (itself
checked against fixest/pyfixest). The table is rebuilt from the public
matches() / weights(), independently of effects.match_rows. R references
(MatchIt + lm + sandwich::vcovCL) are scripted in
tests/reference/reference_r_if.R for the user to run; test_matchit_reference
uses them when present."""

import json

import numpy as np
import polars as pl
import pytest
from conftest import DATA
from hdfe_stream import feols_stream

from match_stream import match_cem, match_nn
from match_stream.effects import ai_estimate


def _long_from_matches(m, df):
    """MatchIt's get_matches(): treated and control rows of every subclass,
    controls weighted 1 / (the subclass's number of controls)."""
    mt = m.matches().collect()
    foc, oth = ("treated_row", "control_row") if m.focal_is_treated else \
        ("control_row", "treated_row")
    p = mt.with_columns((1.0 / pl.len().over("subclass")).alias("w"))
    rows = pl.concat([
        p.group_by("subclass").agg(pl.col(foc).first().alias("row"))
         .with_columns(pl.lit(1.0).alias("w")),
        p.select("subclass", pl.col(oth).alias("row"), "w")])
    return rows.join(df.with_row_index("row").with_columns(pl.col("row").cast(pl.Int64)),
                     on="row", how="left")


@pytest.fixture(scope="module")
def m_wor(ref_data):
    return match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", id="id")


@pytest.fixture(scope="module")
def m_rep(ref_data):
    return match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true",
                    replace=True, n_matches=3, caliper=0.2, id="id")


def test_without_replacement_default_is_pair_clustered(ref_data, m_wor, tmp_path):
    fit = m_wor.effect("y")
    assert fit.vcov_type == "CRV1:subclass"
    w = m_wor.weights().collect().join(
        ref_data.with_row_index("row").with_columns(pl.col("row").cast(pl.Int64)),
        on="row", how="left")
    # MatchIt's recipe: lm(y ~ d, weights) with vcovCL(cluster = ~subclass)
    ref = feols_stream("y ~ d", w, workdir=tmp_path, weights="weight",
                       vcov={"CRV1": "subclass"})
    i = ref.coefnames.index("d")
    np.testing.assert_allclose(fit.beta[0], ref.beta[i], rtol=1e-10)
    np.testing.assert_allclose(fit.vcov[0, 0], ref.vcov[i, i], rtol=1e-8)
    assert fit.n_clusters == {"subclass": (m_wor.counts()["matched_treated"],)}
    assert "clustered by subclass" in "\n".join(fit._summary_lines())


@pytest.mark.parametrize("vcov,cluster", [("if", "id"), ({"CRV1": "subclass"}, "subclass"),
                                         ({"CRV1": "subclass+id"}, "subclass+id"),
                                         ("CRV1:subclass+r", "subclass+r")])
def test_with_replacement_equals_weighted_regression(ref_data, m_rep, tmp_path, vcov, cluster):
    fit = m_rep.effect("y", vcov=vcov)
    long = _long_from_matches(m_rep, ref_data)
    ref = feols_stream("y ~ d", long, workdir=tmp_path, weights="w",
                       vcov={"CRV1": cluster})
    i = ref.coefnames.index("d")
    np.testing.assert_allclose(fit.beta[0], ref.beta[i], rtol=1e-10)
    np.testing.assert_allclose(fit.vcov[0, 0], ref.vcov[i, i], rtol=1e-8)
    assert fit.n_obs == long.height


@pytest.mark.parametrize("estimand", ["att", "atc", "ate"])
def test_estimate_equals_the_matching_estimate(ref_data, estimand):
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", replace=True,
                 n_matches=2, keep_ties=True, focal="control" if estimand == "atc" else "treated")
    a = m.effect("y", estimand=estimand)                 # AI
    b = m.effect("y", estimand=estimand, vcov="if")
    np.testing.assert_allclose(b.beta[0], a.beta[0], rtol=1e-10)
    assert b.n_focal == a.n_focal
    if estimand == "ate":                                 # every unit focal once
        assert b.n_focal == ref_data.height


def test_cem(ref_data, tmp_path):
    m = match_cem("d ~ x1 + x2", ref_data, cutpoints=5, exact=["r"])
    fit = m.effect("y")
    assert fit.vcov_type == "if"
    w = m.weights().collect().join(
        ref_data.with_row_index("row").with_columns(pl.col("row").cast(pl.Int64)),
        on="row", how="left")
    for vcov, ref_vcov in [(None, "hetero"), ({"CRV1": "subclass"}, {"CRV1": "subclass"})]:
        fit = m.effect("y", vcov=vcov)
        ref = feols_stream("y ~ d", w, workdir=tmp_path, weights="weight", vcov=ref_vcov)
        i = ref.coefnames.index("d")
        np.testing.assert_allclose(fit.beta[0], ref.beta[i], rtol=1e-10)
        np.testing.assert_allclose(fit.vcov[0, 0], ref.vcov[i, i], rtol=1e-8)


def test_checks(ref_data, m_rep, m_wor):
    with pytest.raises(ValueError, match="n_variance_matches"):
        m_rep.effect("y", vcov="if", n_variance_matches=2)
    with pytest.raises(NotImplementedError, match="bias_adjustment"):
        m_wor.effect("y", bias_adjustment=["x1"])
    with pytest.raises(ValueError, match="vcov must be"):
        m_rep.effect("y", vcov="robust")
    with pytest.raises(ValueError, match="missing values"):
        df = ref_data.with_columns(pl.when(pl.col("id") < 5000).then(None)
                                   .otherwise(pl.col("r")).alias("r2"))
        match_nn("d ~ x1", df, exact=["r", "i"], score="ps_true").effect(
            "y", vcov={"CRV1": "r2"})
    m = match_cem("d ~ x1", ref_data, cutpoints=5)
    with pytest.raises(ValueError, match="estimand"):
        from match_stream.effects import effect
        effect(m, "y", estimand="atc")


def test_without_replacement_rows_are_units(m_wor):
    # without replacement each unit has one row: the table is the weights()
    from match_stream.effects import match_rows
    rows = match_rows([(m_wor._pairs, True)])
    assert rows["__unit"].n_unique() == rows.height == m_wor.weights().collect().height


REF_IF = DATA / "ref_r_if.json"


CAL3 = {"ps_true": 0.2, "x1": 0.25, "x2": 0.25}
REF_SPECS = {   # as tests/test_reference_nn.py, the MatchIt specs of reference_r.R
    "nn_ps_wor_largest": dict(score="ps_true", match_order="largest", caliper=CAL3),
    "nn_ps_rep": dict(score="ps_true", replace=True, caliper=0.2),
    "nn_ps_rep_2way": dict(score="ps_true", replace=True, caliper=0.2),
    "cem5": dict(cutpoints=5),
}


@pytest.mark.skipif(not REF_IF.exists(), reason="run tests/reference/reference_r_if.R")
@pytest.mark.parametrize("name", list(REF_SPECS))
def test_matchit_reference(ref_data, name):
    refs = json.loads(REF_IF.read_text())
    if name not in refs:
        pytest.skip(f"{name} not in ref_r_if.json (fixest not installed when it was made)")
    ref = refs[name]
    kw = REF_SPECS[name]
    match = match_cem if "cutpoints" in kw else match_nn
    m = match("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], id="id", **kw)
    fit = m.effect("y", vcov=ref["vce"])
    np.testing.assert_allclose(fit.beta[0], ref["est"], rtol=1e-10)
    np.testing.assert_allclose(np.sqrt(fit.vcov[0, 0]), ref["se"], rtol=1e-8)


def test_ai_estimate_unchanged(ref_data, m_rep):
    # the AI path is untouched by vcov: same numbers as before for the default
    y = dict(enumerate(ref_data["y"].to_list()))
    est, var, _ = ai_estimate([(m_rep._pairs, True)], y, "att")
    fit = m_rep.effect("y")
    np.testing.assert_allclose([fit.beta[0], fit.vcov[0, 0]], [est, var], rtol=1e-12)
