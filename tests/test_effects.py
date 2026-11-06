"""Matching estimates and Abadie-Imbens standard errors against R Matching
4.10-15 (tests/data/ref_r.json["matching"])."""

import numpy as np
import polars as pl
import pytest
from conftest import DATA

from match_stream import match_nn
from match_stream.effects import ai_estimate

CASES = {
    # name: (estimand, k, caliper, n_variance_matches)
    "ai_att_m4_vc4": ("att", 4, None, 4),
    "ai_ate_m2_vc0": ("ate", 2, None, 0),
    "ai_atc_m2_vc2": ("atc", 2, None, 2),
}


@pytest.mark.parametrize("name", list(CASES))
def test_estimate_and_se_match_r_matching(ref_data, ref_r, name):
    estimand, k, caliper, n_variance_matches = CASES[name]
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", replace=True,
                 n_matches=k, caliper=caliper, keep_ties=True,       # Matching keeps ties
                 focal="control" if estimand == "atc" else "treated")
    fit = m.effect("y", estimand=estimand, n_variance_matches=n_variance_matches)
    ref = ref_r["matching"][name]
    np.testing.assert_allclose(fit.beta[0], ref["est"], rtol=1e-10)
    np.testing.assert_allclose(np.sqrt(fit.vcov[0, 0]), ref["se"], rtol=1e-8)


def test_variance_formula_on_r_matchings_own_pairs(ref_data, ref_r):
    # On these data R Matching counts treated unit 18735 as tied between two
    # controls whose distances differ by a factor of two, which we do not
    # reproduce; everything else about the M = 1 cases matches.
    # Here the variance formula runs on R's exact pairs and weights.
    r = pl.read_csv(DATA / "ref_matching_ai_att_m1_vc0_pairs.csv.gz", infer_schema_length=None)
    pairs = r.select((pl.col("treated") - 1).alias("focal"), (pl.col("control") - 1).alias("match"),
                     pl.col("weight").alias("w"))
    y = dict(enumerate(ref_data["y"].to_list()))
    # R Matching counts the 40 treated units the caliper dropped in N, with
    # an effect of 0 (match-stream leaves them out)
    n_dropped = ref_r["matching"]["ai_att_m1_vc0"]["n_dropped"]
    est, var, _ = ai_estimate([(pairs, True)], y, "att", n_unmatched=n_dropped)
    ref = ref_r["matching"]["ai_att_m1_vc0"]
    np.testing.assert_allclose(est, ref["est"], rtol=1e-10)
    np.testing.assert_allclose(np.sqrt(var), ref["se"], rtol=1e-8)


def test_m1_differs_from_r_only_by_its_tie(ref_data, ref_r):
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", replace=True,
                 caliper=0.2, keep_ties=True, id="id")
    fit = m.effect("y")
    y = dict(zip(ref_data["id"].to_list(), ref_data["y"].to_list()))
    # R averages controls 10885 and 12604 for unit 18735; we take the nearest, 10885
    shift = (y[10885] - (y[10885] + y[12604]) / 2) / m.counts()["matched_treated"]
    np.testing.assert_allclose(fit.beta[0] + shift, ref_r["matching"]["ai_att_m1_vc0"]["est"],
                               rtol=1e-10)


def test_result_reports(ref_data):
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", replace=True)
    fit = m.effect("y")
    assert fit.coefnames == ["ATT:y"] and fit.tidy().shape[0] == 1


def test_ai_without_replacement_is_refused(ref_data):
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true")
    with pytest.raises(ValueError, match="replace=True"):
        m.effect("y", vcov="ai")


# ------------------------------------------------------------ bias adjustment
Z = ["x1", "x2", "x3"]


def _z(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_row_index("unit").select(pl.col("unit").cast(pl.Int64), *Z)


def test_bias_adjusted_estimate_and_se_match_r_matching(ref_data, ref_r):
    # ai_att_biasadj is a 1-NN case, so R Matching again treats unit 18735 as
    # tied between controls 10885 and 12604. Its pairs are ours
    # plus that tie; on them the bias adjustment reproduces Matching exactly.
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", replace=True,
                 keep_ties=True, id="id")
    pairs = m._pairs.select("focal", "match")
    assert pairs.filter(pl.col("focal") == 18734)["match"].to_list() == [10884]   # rows = id - 1
    pairs = pl.concat([pairs, pl.DataFrame({"focal": [18734], "match": [12603]},
                                           schema=pairs.schema)])
    y = dict(enumerate(ref_data["y"].to_list()))
    est, var, _ = ai_estimate([(pairs, True)], y, "att", z=_z(ref_data))
    ref = ref_r["matching"]["ai_att_biasadj"]
    np.testing.assert_allclose(est, ref["est"], rtol=1e-10)
    np.testing.assert_allclose(np.sqrt(var), ref["se"], rtol=1e-8)


def test_effect_bias_adj_runs_the_same_formula(ref_data):
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", replace=True,
                 n_matches=2, keep_ties=True)
    fit = m.effect("y", bias_adjustment=Z)
    y = dict(enumerate(ref_data["y"].to_list()))
    est, var, n_f = ai_estimate([(m._pairs, True)], y, "att", z=_z(ref_data))
    np.testing.assert_allclose(fit.beta[0], est, rtol=1e-12)
    np.testing.assert_allclose(fit.vcov[0, 0], var, rtol=1e-12)
    assert fit.bias_adjustment == Z and set(fit.bias_coefs) == {"control"}
    assert "Bias adjustment" in "\n".join(fit._summary_lines())
    # bias_adjustment=True uses the formula's matching covariates
    m3 = match_nn("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], score="ps_true",
                  replace=True, n_matches=2, keep_ties=True)
    np.testing.assert_allclose(m3.effect("y", bias_adjustment=True).beta[0], fit.beta[0],
                               rtol=1e-12)


@pytest.mark.parametrize("estimand", ["att", "atc", "ate"])
def test_bias_adjustment_is_exact_when_the_outcome_is_linear(ref_data, estimand):
    # y = 1 + 2 x1 - x2 + 0.5 x3 + tau d with no noise: the regression in each
    # arm is exact, so the adjusted matches reproduce every counterfactual and
    # the estimate is tau however poor the matches are (here: on x1 only).
    tau = 0.7
    df = ref_data.with_columns(
        (1 + 2 * pl.col("x1") - pl.col("x2") + 0.5 * pl.col("x3") + tau * pl.col("d"))
        .alias("ylin"))
    m = match_nn("d ~ x1", df, score="x1", replace=True,
                 focal="control" if estimand == "atc" else "treated")
    raw = m.effect("ylin", estimand=estimand)
    adj = m.effect("ylin", estimand=estimand, bias_adjustment=Z)
    assert abs(raw.beta[0] - tau) > 1e-3
    np.testing.assert_allclose(adj.beta[0], tau, rtol=1e-10)
    want = {"att": {"control"}, "atc": {"treated"}, "ate": {"control", "treated"}}[estimand]
    assert set(adj.bias_coefs) == want
    for b in adj.bias_coefs.values():
        np.testing.assert_allclose([b["x1"], b["x2"], b["x3"]], [2, -1, 0.5], rtol=1e-10)


def test_bias_adj_argument_checks(ref_data):
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", replace=True)
    with pytest.raises(ValueError, match="list of column names"):
        m.effect("y", bias_adjustment=[1, 2])
    df = ref_data.with_columns(pl.when(pl.col("id") == 1).then(None).otherwise(pl.col("x2"))
                               .alias("x2"))
    m = match_nn("d ~ x1", df, exact=["r", "i"], score="ps_true", replace=True)
    with pytest.raises(ValueError, match="bias-adjustment covariate 'x2' is missing"):
        m.effect("y", bias_adjustment=Z)
