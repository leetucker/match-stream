"""The Abadie-Imbens (2016) correction (match_stream/inference.py) beyond
what teffects psmatch offers (it is checked against psmatch in
tests/test_teffects.py): the ATC, exact strata, and when it applies."""

import warnings

import numpy as np
import polars as pl
import pytest

from match_stream import match_nn, simulate_matching


@pytest.fixture(scope="module")
def df():
    return simulate_matching(6000, seed=3)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_atc_is_the_flipped_att(df, link):
    # the ATC with controls focal equals minus the ATT of the flipped
    # treatment, and so does its corrected variance (the score refits as
    # gamma -> -gamma for a symmetric link)
    kw = dict(distance="ps", replace=True, n_matches=2, keep_ties=True, id="id", verbose=False,
              score_link=link)
    atc = match_nn("d ~ x1 + x2 + x3", df, focal="control", **kw).effect("y",
                                                                            n_variance_matches=1)
    flipped = df.with_columns((1 - pl.col("d")).alias("d"))
    att = match_nn("d ~ x1 + x2 + x3", flipped, **kw).effect("y", n_variance_matches=1)
    assert atc.score_correction and att.score_correction
    np.testing.assert_allclose(atc.beta[0], -att.beta[0], rtol=1e-10)
    np.testing.assert_allclose(atc.vcov[0, 0], att.vcov[0, 0], rtol=1e-6)


def test_exact_strata(df):
    # sets are found within strata; the correction is finite and lowers the
    # variance here
    kw = dict(exact=["r"], distance="ps", replace=True, keep_ties=True, id="id", verbose=False)
    m = match_nn("d ~ x1 + x2 + x3", df, **kw)
    a = m.effect("y", estimand="ate", n_variance_matches=1)
    b = m.effect("y", estimand="ate", n_variance_matches=1, score_correction=False)
    assert a.score_correction and not b.score_correction
    assert np.isfinite(a.vcov[0, 0]) and 0 < a.vcov[0, 0] < b.vcov[0, 0]
    assert "corrected for the estimated score" in "\n".join(a._summary_lines())


def test_when_it_applies(df):
    m = match_nn("d ~ x1 + x2 + x3", df, distance="ps", replace=True, verbose=False)
    with pytest.warns(UserWarning, match="n_variance_matches >= 1"):
        assert not m.effect("y").score_correction
    with pytest.raises(ValueError, match="n_variance_matches >= 1"):
        m.effect("y", score_correction=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not m.effect("y", score_correction=False).score_correction
    col = match_nn("d ~ x1", df, distance="ps", replace=True, verbose=False)
    known = match_nn("d ~ x1", df.rename({"ps": "p_true"}), score="p_true", replace=True)
    assert not known.effect("y", n_variance_matches=1).score_correction          # a column: known
    with pytest.raises(ValueError, match="score match-stream fits"):
        known.effect("y", n_variance_matches=1, score_correction=True)
    del col
    cal = match_nn("d ~ x1 + x2 + x3", df, distance="ps", replace=True, caliper=0.001,
                   verbose=False)
    assert cal.effect("y", n_variance_matches=1).score_correction      # calipers are supported
    with pytest.raises(ValueError, match="vcov='ai'"):
        m.effect("y", vcov="if", score_correction=True)


@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("estimand", ["ate", "att", "atc"])
def test_absorbed_fixed_effects_equal_dummies(df, estimand, link):
    # the score with r absorbed (the correction by a fixed-effect regression)
    # and with r as dummies (the dense observed information) are the same
    # model; within exact strata r the matches and neighbor sets are the
    # same too, so the corrected variances agree
    kw = dict(exact=["r"], distance="ps", replace=True, n_matches=2, keep_ties=True, id="id",
              score_link=link,
              verbose=False, focal="control" if estimand == "atc" else "treated")
    fe = match_nn("d ~ x1 + x2 + x3", df, score_model="d ~ x1 + x2 + x3 | r", **kw)
    du = match_nn("d ~ x1 + x2 + x3", df, score_model="d ~ x1 + x2 + x3 + C(r)", **kw)
    a = fe.effect("y", estimand=estimand, n_variance_matches=1)
    b = du.effect("y", estimand=estimand, n_variance_matches=1)
    assert a.score_correction and b.score_correction
    np.testing.assert_allclose(a.beta[0], b.beta[0], rtol=1e-10)
    np.testing.assert_allclose(a.vcov[0, 0], b.vcov[0, 0], rtol=1e-7)
    c = fe.effect("y", estimand=estimand, n_variance_matches=1, score_correction=False)
    assert a.vcov[0, 0] != pytest.approx(c.vcov[0, 0], rel=1e-4)    # it does something


# --------------------------------------------- calipers that drop focal units
def _brute_adjustment(df, m, estimand, h, keep, fit_keep=None):
    """The AI (2016) adjustment written out unit by unit: neighbor sets by
    sorting distances among the kept units, V_g over the units the score
    model was fitted on (`fit_keep`, default every unit). Logit, no strata."""
    from scipy.special import expit

    first_beta = np.asarray(m.score_fit.beta)
    X = df.select("x1", "x2", "x3").to_numpy()
    Z = np.column_stack([np.ones(df.height), X])
    p_all = expit(Z @ first_beta)
    fk = np.ones(df.height, bool) if fit_keep is None else fit_keep
    H = (Z[fk] * (p_all[fk] * (1 - p_all[fk]))[:, None]).T @ Z[fk]
    Vg = np.linalg.inv(H)
    t, y = df["d"].to_numpy().astype(bool)[keep], df["y"].to_numpy()[keep]
    p, Zk, Xk = p_all[keep], Z[keep], X[keep]
    f = p * (1 - p)
    n = t.size

    def near(i, arm, size, exclude_self):
        idx = np.flatnonzero(t == arm)
        if exclude_self:
            idx = idx[idx != i]
        return idx[np.argsort(np.abs(p[idx] - p[i]), kind="stable")[:size]]

    def cov(i, arm):
        mem = near(i, arm, h, False)       # own arm: includes i (distance 0)
        return ((Zk[mem] - Zk[mem].mean(0)) * (y[mem] - y[mem].mean())[:, None]).sum(0) \
            / (mem.size - 1)

    if estimand == "ate":
        c = sum(f[i] * (cov(i, True) / p[i] + cov(i, False) / (1 - p[i]))
                for i in range(n)) / n
        return -c @ Vg @ c
    n1 = t.sum()
    cd = np.zeros(Z.shape[1])
    dd = np.zeros(Z.shape[1])
    est = m.effect("y", estimand="att", n_variance_matches=h - 1, score_correction=False).beta[0]
    for i in range(n):
        yb = {a: y[near(i, a, h, t[i] == a)].mean() for a in (True, False)}
        cd += Zk[i] * f[i] * (yb[True] - yb[False] - est)
        cd += f[i] * (cov(i, True) + p[i] / (1 - p[i]) * cov(i, False))
        o = np.flatnonzero(t != t[i])
        nz = o[np.argsort(((Xk[o] - Xk[i]) ** 2).sum(1), kind="stable")[:m.spec.n_matches]]
        dd += Zk[i] * f[i] * ((2 * t[i] - 1) * (y[i] - y[nz].mean()) - est)
    cd /= n1
    dd /= n1
    return -cd @ Vg @ cd + dd @ Vg @ dd


@pytest.mark.parametrize("estimand", ["att", "ate"])
def test_caliper_correction_matches_a_direct_computation(estimand):
    df = simulate_matching(2500, seed=11)
    m = match_nn("d ~ x1 + x2 + x3", df, distance="ps", replace=True, n_matches=2,
                 keep_ties=True,
                 caliper=0.01, id="id", verbose=False)
    sets = [m] if estimand == "att" else [m, m._rematch("atc")]
    keep = np.ones(df.height, bool)
    for s in sets:
        arm = "treated" if s.focal_is_treated else "control"
        assert s.counts()[f"unmatched_{arm}"] > 0          # the caliper drops some
        focal = np.zeros(df.height, bool)
        focal[s._pairs["focal"].to_numpy()] = True
        keep &= ~(df["d"].to_numpy().astype(bool) == s.focal_is_treated) | focal
    a = m.effect("y", estimand=estimand, n_variance_matches=1)
    b = m.effect("y", estimand=estimand, n_variance_matches=1, score_correction=False)
    assert a.score_correction
    np.testing.assert_allclose(a.vcov[0, 0] - b.vcov[0, 0],
                               _brute_adjustment(df, m, estimand, 2, keep), rtol=1e-8)


def test_caliper_that_drops_nobody_changes_nothing(df):
    kw = dict(distance="ps", replace=True, n_matches=2, keep_ties=True, id="id", verbose=False)
    a = match_nn("d ~ x1 + x2 + x3", df, **kw).effect("y", n_variance_matches=1)
    b = match_nn("d ~ x1 + x2 + x3", df, caliper=10.0, **kw).effect("y", n_variance_matches=1)
    np.testing.assert_allclose(a.vcov, b.vcov, rtol=1e-12)


@pytest.mark.parametrize("estimand", ["att", "ate"])
def test_caliper_with_absorbed_fixed_effects(df, estimand):
    kw = dict(exact=["r"], distance="ps", replace=True, n_matches=2, keep_ties=True, id="id",
              caliper=0.02, verbose=False)
    fe = match_nn("d ~ x1 + x2 + x3", df, score_model="d ~ x1 + x2 + x3 | r", **kw)
    du = match_nn("d ~ x1 + x2 + x3", df, score_model="d ~ x1 + x2 + x3 + C(r)", **kw)
    assert fe.counts()["unmatched_treated"] > 0
    a = fe.effect("y", estimand=estimand, n_variance_matches=1)
    b = du.effect("y", estimand=estimand, n_variance_matches=1)
    np.testing.assert_allclose(a.vcov[0, 0], b.vcov[0, 0], rtol=1e-7)



@pytest.mark.parametrize("estimand", ["att", "ate"])
@pytest.mark.parametrize("reestimate_score", [False, True])
def test_discard_correction_matches_a_direct_computation(estimand, reestimate_score):
    # discard: the discarded take no part in the second stage; V_g runs over
    # the units the score was fitted on (all of them, or the kept with
    # reestimate_score)
    df = simulate_matching(2500, seed=13)
    m = match_nn("d ~ x1 + x2 + x3", df, distance="ps", replace=True, n_matches=2,
                 keep_ties=True,
                 discard="both", reestimate_score=reestimate_score, id="id", verbose=False)
    c = m.counts()
    assert c["discarded_treated"] + c["discarded_control"] > 0
    kept = np.ones(df.height, bool)
    disc_rows = set(range(df.height)) - set(m._narrow_lf.collect()["__ms_row"].to_list())
    kept[list(disc_rows)] = False
    a = m.effect("y", estimand=estimand, n_variance_matches=1)
    b = m.effect("y", estimand=estimand, n_variance_matches=1, score_correction=False)
    assert a.score_correction
    want = _brute_adjustment(df, m, estimand, 2, kept, fit_keep=kept if reestimate_score else None)
    np.testing.assert_allclose(a.vcov[0, 0] - b.vcov[0, 0], want, rtol=1e-8)
