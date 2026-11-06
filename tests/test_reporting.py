"""MatchResult on hdfe-stream's RegressionResult: its reporting (summary,
tidy, confint, summary_dict, the pyfixest view and etable) for every kind of
estimate, and the conventions it shares with the sibling packages' results
(`n_clusters`, `n_obs`, normal inference)."""

import numpy as np
import polars as pl
import pytest
from hdfe_stream import etable
from scipy import special

from match_stream import match_cem, match_kernel, match_nn


@pytest.fixture(scope="module")
def fits(ref_data):
    rep = match_nn("d ~ x1 + x2", ref_data, exact=["r"], score="ps_true", replace=True,
                   verbose=False)
    wor = match_nn("d ~ x1 + x2", ref_data, exact=["r"], score="ps_true", id="id",
                   verbose=False)
    return {
        "ai": rep.effect("y"),
        "ai_bias": rep.effect("y", n_variance_matches=1, bias_adjustment=True),
        "if": rep.effect("y", vcov="if"),
        "crv1": wor.effect("y"),
        "twoway": wor.effect("y", vcov={"CRV1": "subclass+id"}),
        "cem": match_cem("d ~ x1 + x2", ref_data, cutpoints=5,
                         verbose=False).effect("y"),
        "kernel": match_kernel("d ~ x1", ref_data, score="ps_true",
                               verbose=False).effect("y"),
    }


@pytest.mark.parametrize("kind", ["ai", "ai_bias", "if", "crv1", "twoway", "cem", "kernel"])
def test_reporting_methods(fits, kind, capsys):
    fit = fits[kind]
    fit.summary()
    assert "Estimand: ATT" in capsys.readouterr().out
    t = fit.tidy()
    assert t["Coefficient"].to_list() == ["ATT:y"]
    # normal inference: the p-value and the interval use the normal
    z = fit.beta[0] / fit.se[0]
    assert t["Pr(>|t|)"][0] == pytest.approx(2 * special.ndtr(-abs(z)), rel=1e-12)
    ci = fit.confint()
    half = special.ndtri(0.975) * fit.se[0]
    assert ci[0, 1] == pytest.approx(fit.beta[0] - half, rel=1e-12)
    d = fit.summary_dict()
    assert d["estimand"] == "att" and d["n_obs"] == fit.n_obs
    view = fit.to_pyfixest()
    assert view._coefnames == ["ATT"]
    np.testing.assert_allclose(view._se, fit.se)


def test_cluster_counts_follow_hdfe_stream(fits):
    assert fits["ai"].n_clusters == {}
    (n,) = fits["if"].n_clusters["unit"]
    assert n > 0
    g = fits["twoway"].n_clusters["subclass+id"]
    assert len(g) == 2 and g[0] == fits["crv1"].n_clusters["subclass"][0]
    assert fits["twoway"].to_pyfixest()._G == [min(g), min(g)]
    assert "clustered by subclass" in "\n".join(fits["twoway"]._summary_lines())


def test_n_obs_is_what_the_estimate_uses(ref_data, fits):
    m = match_nn("d ~ x1 + x2", ref_data, exact=["r"], score="ps_true", replace=True,
                 id="id", verbose=False)
    p = m.matches().collect()
    assert fits["ai"].n_obs == pl.concat([p["treated_id"], p["control_id"]]).n_unique()
    assert fits["ai"].n_obs < ref_data.height


def test_etable(fits):
    tab = etable([fits["ai"], fits["if"], fits["twoway"], fits["kernel"]], type="df",
                 model_stats=["N", "se_type"])
    assert tab.shape[1] == 4
    coef_rows = [r for r in tab.index if r[0] == "coef"]
    assert [r[1] for r in coef_rows] == ["ATT"]
