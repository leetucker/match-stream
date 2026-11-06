"""effect() against Stata's teffects nnmatch / psmatch
(tests/reference/stata/teffects_match.do; skipped until its outputs are in
tests/data/). AI standard errors are checked against teffects as well
as R Matching.

Conventions confirmed on the first run (read before the scripts were fixed
to import in double precision): vce(iid) is AI's homoskedastic variance,
vce(robust, nn(J)) is n_variance_matches=J, biasadj() is R Matching's BiasAdjust,
and teffects' 1-NN matches are match-stream's (it has no R Matching tie).
All agreed to about 2e-9, the single-precision import's limit."""

import json

import numpy as np
import polars as pl
import pytest
from conftest import DATA

from match_stream import match_nn

REF = DATA / "ref_teffects_match.json"
pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="run tests/reference/stata/teffects_match.do")


@pytest.fixture(scope="module")
def te():
    try:
        return json.loads(REF.read_text())
    except json.JSONDecodeError:
        # the first run's file: unfinished, and numbers without leading zeros
        pytest.skip("ref_teffects_match.json is not valid JSON (an unfinished run of an "
                    "older script): rerun tests/reference/stata/teffects_match.do")


def te_pairs(name) -> dict[int, set[int]]:
    t = pl.read_csv(DATA / f"ref_teffects_{name}_matches.csv.gz")
    mcols = [c for c in t.columns if c not in ("id", "d")]
    return {r[0]: {x for x in r[2:] if x is not None}
            for r in t.select("id", "d", *mcols).iter_rows()}


def our_pairs(m) -> dict[int, set[int]]:
    t = m.matches().collect()
    foc, oth = ("treated_id", "control_id") if m.focal_is_treated else \
        ("control_id", "treated_id")
    g = t.group_by(foc).agg(pl.col(oth))
    return {a: set(b) for a, b in g.iter_rows()}


# case: (estimand, k, n_variance_matches); all on the true score within strata r, i
NNM = {
    "nnm_atet_ps_m1_iid": ("att", 1, 0),
    "nnm_atet_ps_m4_vc4": ("att", 4, 4),
    "nnm_ate_ps_m2_iid": ("ate", 2, 0),
}


@pytest.mark.parametrize("name", list(NNM))
def test_nnmatch_on_a_known_score(ref_data, te, name):
    estimand, k, n_variance_matches = NNM[name]
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", replace=True,
                 n_matches=k, keep_ties=True, id="id")
    if estimand != "ate":
        treated = set(ref_data.filter(pl.col("d") == 1)["id"].to_list())
        ref = {f: c for f, c in te_pairs(name).items() if f in treated}
        assert our_pairs(m) == ref
    fit = m.effect("y", estimand=estimand, n_variance_matches=n_variance_matches)
    np.testing.assert_allclose(fit.beta[0], te[name]["est"], rtol=1e-8)
    # vce(iid) is taken to be AI's homoskedastic variance (R Matching's
    # Var.calc = 0) and vce(robust, nn(J)) n_variance_matches = J: unconfirmed
    np.testing.assert_allclose(np.sqrt(fit.vcov[0, 0]), te[name]["se"], rtol=1e-6)


def test_nnmatch_bias_adjustment(ref_data, te):
    name = "nnm_atet_ps_biasadj_iid"
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", replace=True,
                 keep_ties=True, id="id")
    fit = m.effect("y", bias_adjustment=["x1", "x2", "x3"])
    # teffects' bias-adjustment regression may differ from R Matching's
    # (weights K_i, matched units only): unconfirmed
    np.testing.assert_allclose(fit.beta[0], te[name]["est"], rtol=1e-8)
    np.testing.assert_allclose(np.sqrt(fit.vcov[0, 0]), te[name]["se"], rtol=1e-6)


def test_logit_score_is_teffects(ref_data, te, tmp_path):
    from hdfe_stream import feglm_stream

    for link in ("logit", "probit"):
        fit = feglm_stream("d ~ x1 + x2 + x3", ref_data, family=link, workdir=tmp_path)
        coef = te[f"{link}_coef"]
        np.testing.assert_allclose([fit.beta[fit.coefnames.index(c)] for c in coef],
                                   list(coef.values()), rtol=1e-6)


# psmatch cases re-run as nnmatch on the fitted score: no 2016 correction, so
# the existing AI variance applies (estimand, k, n_variance_matches, link)
KNOWN = {
    "psm_atet_logit_m1_known": ("att", 1, 2, "logit"),
    "psm_ate_logit_m1_known": ("ate", 1, 2, "logit"),
    "psm_atet_logit_m4_known": ("att", 4, 2, "logit"),
    "psm_ate_probit_m1_known": ("ate", 1, 2, "probit"),
    "psm_atet_logit_m1_iid_known": ("att", 1, 0, "logit"),
}


@pytest.mark.parametrize("name", list(KNOWN))
def test_matching_on_teffects_fitted_score(ref_data, te, name):
    estimand, k, n_variance_matches, link = KNOWN[name]
    ps = pl.read_csv(DATA / "ref_teffects_pscore.csv.gz")
    df = ref_data.join(ps, on="id", how="left")
    m = match_nn("d ~ x1", df, score=f"ps_{link}", replace=True, n_matches=k, keep_ties=True,
                 id="id")
    fit = m.effect("y", estimand=estimand, n_variance_matches=n_variance_matches)
    np.testing.assert_allclose(fit.beta[0], te[name]["est"], rtol=1e-8)
    np.testing.assert_allclose(np.sqrt(fit.vcov[0, 0]), te[name]["se"], rtol=1e-6)
    # psmatch's SE (with the 2016 correction) on the same matches
    psm = te[name.removesuffix("_known")]
    np.testing.assert_allclose(psm["est"], te[name]["est"], rtol=1e-12)


def test_caliper_units_are_sd_of_the_score(ref_data, te):
    # teffects stops (rc 459) when a unit has no match inside the caliper and
    # flags them with osample(): 149 here, exactly the treated units whose
    # nearest control in their stratum is more than 0.05 SD of the score away,
    # i.e. MatchIt's caliper units, match-stream's default
    assert te["nnm_atet_ps_cal_iid_rc"] == 459
    m = match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", replace=True,
                 caliper=0.05)
    assert m.counts()["unmatched_treated"] == te["nnm_atet_ps_cal_iid_n_osample"] == 149


def test_mahalanobis_recorded(te):
    # teffects' Mahalanobis uses the full-sample covariance; match-stream
    # MatchIt's pooled within-group one, so the matches differ
    print(te["nnm_atet_maha_m1_iid"])


# ------------------------------------------------ AI (2016): teffects psmatch
# psmatch's vce(robust, nn(2)) is n_variance_matches=1 here (psmatch counts the unit
# itself among its h); with vce(iid) psmatch does not correct
PSM = {   # name: (estimand, k, link, n_variance_matches, corrected)
    "psm_atet_logit_m1": ("att", 1, "logit", 1, True),
    "psm_ate_logit_m1": ("ate", 1, "logit", 1, True),
    "psm_atet_logit_m4": ("att", 4, "logit", 1, True),
    "psm_ate_probit_m1": ("ate", 1, "probit", 1, True),
    "psm_atet_logit_m1_iid": ("att", 1, "logit", 0, False),
}


@pytest.mark.parametrize("name", list(PSM))
def test_psmatch_ai2016(te, name):
    estimand, k, link, n_variance_matches, corrected = PSM[name]
    raw = pl.read_csv(DATA / "ref_data.csv")              # distance="ps" fits the score
    m = match_nn("d ~ x1 + x2 + x3", raw, distance="ps", score_link=link, replace=True, n_matches=k,
                 keep_ties=True, id="id", verbose=False)
    if corrected:
        fit = m.effect("y", estimand=estimand, n_variance_matches=n_variance_matches)
    else:
        with pytest.warns(UserWarning, match="not corrected"):
            fit = m.effect("y", estimand=estimand, n_variance_matches=n_variance_matches)
    assert fit.score_correction is corrected
    np.testing.assert_allclose(fit.beta[0], te[name]["est"], rtol=1e-10)
    np.testing.assert_allclose(np.sqrt(fit.vcov[0, 0]), te[name]["se"], rtol=1e-8)
