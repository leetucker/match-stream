"""Kernel and ridge matching (match_stream/_kernel.py, kernels/kernel.py)
against a brute-force pairwise implementation written from kmatch 1.1.6's
Mata code (Kmatch_i_PS, Kmatch_Ridge, Kmatch_bw, the cv criterion), and the
effect's standard error against hdfe-stream's weighted regression.

Where there is no Stata reference these tests pin the formulas instead."""

import json

import numpy as np
import polars as pl
import pytest
from conftest import DATA
from hdfe_stream import feols_stream

from match_stream import CrossValidation, PairDistance, match_kernel
from match_stream._kernel import KERNELS, RIDGE_DEFAULT, pm_bandwidth, quantile_type2

KFUN = {
    "epan": lambda u: 0.75 * (1 - u**2),
    "rectangle": lambda u: 0.5 + 0 * u,
    "triangle": lambda u: 1 - np.abs(u),
    "biweight": lambda u: 0.9375 * (1 - u**2) ** 2,
    "triweight": lambda u: 1.09375 * (1 - u**2) ** 3,
}


def brute(df, focal_treated, h, kernel, r=None, strata=("g",)):
    """kmatch's weights pair by pair: {focal row: (window rows, weights)}."""
    out = {}
    key = list(strata)
    if "row" not in df.columns:
        df = df.with_row_index("row")
    for _, g in df.group_by(key):
        t = g.filter(pl.col("d") == focal_treated)
        c = g.filter(pl.col("d") != focal_treated)
        cs, crow = c["s"].to_numpy(), c["row"].to_numpy()
        for i, si in zip(t["row"].to_list(), t["s"].to_list()):
            sel = np.abs(cs - si) < h
            if not sel.any():
                continue
            # kmatch's formula, in scores centered at the focal unit (the same
            # weights; raw scores far from 0 lose digits in pbar and D)
            pj = cs[sel] - si
            w = KFUN[kernel](pj / h)
            if r is not None and len(pj) > 1 and np.var(pj) > 0:
                W = w.sum()
                P = (pj * w).sum() / W
                D = pj - P
                S = (D**2 * w).sum()
                dd = -P
                w = w / W + w * D * (dd / (S + r * h * abs(dd)))
            else:
                w = w / w.sum()
            out[i] = (crow[sel], w)
    return out


def brute_units(df, bw, focal_treated):
    """Raw weights by row: focal 1, others summed."""
    acc = {}
    for i, (rows, w) in bw.items():
        acc[i] = 1.0
        for j, wj in zip(rows, w):
            acc[j] = acc.get(j, 0.0) + wj
    return acc


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(3)
    n = 4000
    g = rng.integers(0, 5, n)
    s = np.round(rng.beta(2, 5, n), 3)            # ties in the score
    d = rng.random(n) < 0.2 + 0.5 * s
    y = 1 + d + 2 * s + rng.standard_normal(n)
    cl = rng.integers(0, 60, n)
    return pl.DataFrame({"d": d.astype(np.int64), "s": s, "g": g, "y": y, "cl": cl,
                         "x": rng.standard_normal(n)})


@pytest.mark.parametrize("kernel", list(KERNELS))
@pytest.mark.parametrize("method", ["kernel", "ridge"])
@pytest.mark.parametrize("estimand", ["att", "atc"])
def test_weights_match_brute_force(sim, kernel, method, estimand):
    h = 0.0371
    m = match_kernel("d ~ x", sim, exact="g", score="s", ridge=method == "ridge" or None,
                     kernel=kernel, bandwidth=h,
                     focal="control" if estimand == "atc" else "treated")
    ft = estimand == "att"
    ref = brute_units(sim, brute(sim, int(ft), h, kernel,
                                 RIDGE_DEFAULT[kernel] if method == "ridge" else None), ft)
    got = dict(zip(m._units["row"].to_list(), m._units["weight"].to_list()))
    assert got.keys() == ref.keys()
    np.testing.assert_allclose([got[k] for k in ref], list(ref.values()), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method,r", [("kernel", None), ("ridge", 5 / 16), ("ridge", 0.0)])
def test_estimate_is_kmatchs(sim, method, r):
    # h off the 0.001 grid of the scores: at |s_j - s_i| = h exactly, rounding
    # decides membership, and local-linear weights (r = 0) of a window with a
    # near-zero-weight point are ill-conditioned (true of kmatch too)
    h = 0.0503
    ridge = None if r is None else True if r == 5 / 16 else r
    m = match_kernel("d ~ x", sim, exact="g", score="s", ridge=ridge, bandwidth=h)
    bw = brute(sim, 1, h, "epan", r)
    y = sim["y"].to_numpy()
    att = np.mean([y[i] - np.dot(w, y[rows]) for i, (rows, w) in bw.items()])
    fit = m.effect("y")
    np.testing.assert_allclose(fit.beta[0], att, rtol=1e-10)
    assert fit.n_focal == len(bw)


def test_standard_error_equals_weighted_regression(sim, tmp_path):
    m = match_kernel("d ~ x", sim, exact="g", score="s", ridge=True, bandwidth=0.05)
    w = m.weights().collect().join(sim.with_row_index("row").with_columns(
        pl.col("row").cast(pl.Int64)), on="row", how="left")
    for vcov, ref_vcov in [(None, "hetero"), ({"CRV1": "cl"}, {"CRV1": "cl"})]:
        fit = m.effect("y", vcov=vcov)
        ref = feols_stream("y ~ d", w, workdir=tmp_path, weights="weight", vcov=ref_vcov)
        i = ref.coefnames.index("d")
        np.testing.assert_allclose(fit.beta[0], ref.beta[i], rtol=1e-10)
        np.testing.assert_allclose(fit.vcov[0, 0], ref.vcov[i, i], rtol=1e-8)


def test_ate_combines_both_directions(sim):
    kw = dict(exact="g", score="s", bandwidth=0.04)
    att = match_kernel("d ~ x", sim, **kw).effect("y")
    atc = match_kernel("d ~ x", sim, focal="control", **kw).effect("y")
    ate = match_kernel("d ~ x", sim, **kw).effect("y", estimand="ate")
    want = (att.n_focal * att.beta[0] + atc.n_focal * atc.beta[0]) / (att.n_focal + atc.n_focal)
    np.testing.assert_allclose(ate.beta[0], want, rtol=1e-10)


def test_pair_distance_bandwidth_is_kmatchs_pm(sim):
    m = match_kernel("d ~ x", sim, exact="g", score="s")
    gaps = []
    for _, g in sim.group_by("g"):
        t = g.filter(pl.col("d") == 1)["s"].to_numpy()
        c = g.filter(pl.col("d") == 0)["s"].to_numpy()
        gaps += [np.min(np.abs(c - x)) for x in t]
    gaps = np.array(gaps)
    q = np.quantile(gaps[gaps > 0], 0.9, method="averaged_inverted_cdf") * 1.5
    np.testing.assert_allclose(m.bandwidth, q, rtol=1e-14)
    assert m.bandwidth_info.rule == "pair_distance"
    m2 = match_kernel("d ~ x", sim, exact="g", score="s",
                      bandwidth=PairDistance(quantile=1.0, factor=2.0))
    np.testing.assert_allclose(m2.bandwidth, 2 * gaps.max(), rtol=1e-14)


def test_quantile_type2():
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 10, 37):
        x = rng.standard_normal(n)
        for q in (0.1, 0.5, 0.9, 0.95, 1.0):
            assert quantile_type2(x, q) == pytest.approx(
                np.quantile(x, q, method="averaged_inverted_cdf"), rel=1e-15)
    assert pm_bandwidth(np.array([0.0, 0.0]), 0.9, 1.5) > 0


def _loo_criterion(df, h, kernel, r, weighted, ev_h):
    """Brute force: mean (weighted) squared leave-one-out error over the
    controls with another control within ev_h."""
    num = den = 0.0
    bw = brute(df, 1, h, kernel, r) if weighted else None
    wc = brute_units(df, bw, True) if weighted else None
    for _, g in df.with_row_index("row").group_by("g"):
        c = g.filter(pl.col("d") == 0)
        cs, cy, crow = c["s"].to_numpy(), c["y"].to_numpy(), c["row"].to_numpy()
        for j in range(len(cs)):
            others = np.arange(len(cs)) != j
            if not (others & (np.abs(cs - cs[j]) < ev_h)).any():
                continue
            sel = others & (np.abs(cs - cs[j]) < h)
            pj, yj = cs[sel], cy[sel]
            K = KFUN[kernel]((pj - cs[j]) / h)
            if r is not None and len(pj) > 1 and np.var(pj) > 0:
                W = K.sum()
                P = (pj * K).sum() / W
                D = pj - P
                S = (D**2 * K).sum()
                dd = cs[j] - P
                w = K / W + K * D * (dd / (S + r * h * abs(dd)))
            else:
                w = K / K.sum()
            om = wc.get(crow[j], 0.0) if weighted else 1.0
            num += om * (cy[j] - np.dot(w, yj)) ** 2
            den += om
    return num / den


@pytest.mark.parametrize("method,weighted", [("kernel", False), ("ridge", False),
                                             ("kernel", True)])
def test_cross_validation_criterion(sim, method, weighted):
    grid = [0.02, 0.05, 0.12]
    m = match_kernel("d ~ x", sim, exact="g", score="s", ridge=method == "ridge" or None,
                     bandwidth=CrossValidation(outcome="y", grid=grid, weighted=weighted))
    tab = m.bandwidth_info.cv
    r = RIDGE_DEFAULT["epan"] if method == "ridge" else None
    for h, crit in zip(tab["bandwidth"].to_list(), tab["criterion"].to_list()):
        np.testing.assert_allclose(crit, _loo_criterion(sim, h, "epan", r, weighted, 0.02),
                                   rtol=1e-9)
    assert m.bandwidth == grid[int(np.argmin(tab["criterion"].to_numpy()))]


def test_memory_paths_agree(sim):
    kw = dict(exact="g", score="s", ridge=True,
              bandwidth=CrossValidation(outcome="y", grid=[0.03, 0.06]))
    a = match_kernel("d ~ x", sim, memory="in_memory", **kw)
    b = match_kernel("d ~ x", sim, memory="by_stratum", **kw)
    assert a.bandwidth == b.bandwidth and b.memory == "by_stratum"
    assert a.weights().collect().equals(b.weights().collect())


def test_precision_with_a_narrow_window_in_a_long_array():
    # 400k points far from 0, a bandwidth of a few points: global running sums
    # would lose the window here; the block moments must not
    rng = np.random.default_rng(1)
    n = 400_000
    s = 1000.0 + np.sort(rng.random(n))
    d = rng.random(n) < 0.1
    df = pl.DataFrame({"d": d.astype(np.int64), "s": s, "x": np.zeros(n), "g": np.zeros(n)})
    h = 2.5e-5
    m = match_kernel("d ~ x", df, score="s", ridge=True, kernel="biweight", bandwidth=h)
    sub = df.with_row_index("row").filter(pl.col("row") < 4000)   # brute force on a slice
    ref = brute_units(sub, brute(sub, 1, h, "biweight", RIDGE_DEFAULT["biweight"]), True)
    got = dict(zip(m._units["row"].to_list(), m._units["weight"].to_list()))
    inner = [k for k in ref if 200 < k < 3800]     # away from the slice's edges
    np.testing.assert_allclose([got[k] for k in inner], [ref[k] for k in inner], rtol=1e-9,
                               atol=1e-14)


def test_balance_and_reports(sim):
    m = match_kernel("d ~ x", sim, exact="g", score="s", bandwidth=0.05, id="g")
    b = m.balance()
    assert b.matched["Std. Pair Dist."].is_nan().all()
    c = m.counts()
    assert c["matched_treated"] == m._units.filter(pl.col("focal")).height
    w = m.weights().collect()
    ctrl = w.filter(~pl.col("treated"))
    np.testing.assert_allclose(ctrl["weight"].sum(), ctrl.height)
    with pytest.raises(TypeError, match="no pairs"):
        m.matches()


def test_checks(sim):
    kw = dict(exact="g", score="s")
    for name in ("caliper", "n_matches", "match_order", "keep_ties", "replace"):
        with pytest.raises(TypeError, match=name):        # nearest-neighbor options
            match_kernel("d ~ x", sim, **{name: 1}, **kw)
    with pytest.raises(ValueError, match="not polynomials"):
        match_kernel("d ~ x", sim, kernel="cosine", **kw)
    with pytest.raises(ValueError, match="non-negative"):
        match_kernel("d ~ x", sim, ridge=-0.1, **kw)
    with pytest.raises(ValueError, match="outcome"):
        CrossValidation(weighted=True)
    with pytest.raises(TypeError, match="cutpoints"):
        match_kernel("d ~ x", sim, cutpoints=3, **kw)
    m = match_kernel("d ~ x", sim, bandwidth=0.05, **kw)
    with pytest.raises(ValueError, match="no subclasses"):
        m.effect("y", vcov={"CRV1": "subclass"})
    with pytest.raises(ValueError, match="replace=True"):
        m.effect("y", vcov="ai")


# ------------------------------------------------------- Stata kmatch references
KM = DATA / "ref_kmatch_kernel.json"
KM_CASES = {   # name: (match_stream options, estimand)
    "kernel_att_pm": (dict(kernel="epan"), "att"),
    "ridge_att_pm": (dict(ridge=True, kernel="epan"), "att"),
    "kernel_atc_fixed": (dict(kernel="biweight", bandwidth=0.01), "atc"),
    "ll_att_fixed": (dict(ridge=0.0, kernel="epan", bandwidth=0.02), "att"),
}


@pytest.mark.skipif(not KM.exists(), reason="run tests/reference/stata/kmatch_kernel.do")
@pytest.mark.parametrize("name", list(KM_CASES))
def test_kmatch_reference(ref_data, name):
    ref = json.loads(KM.read_text())[name]
    kw, estimand = KM_CASES[name]
    m = match_kernel("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], score="ps_true",
                     focal="control" if estimand == "atc" else "treated", id="id", **kw)
    np.testing.assert_allclose(m.bandwidth, ref["bwidth"][0], rtol=1e-10)
    fit = m.effect("y")
    np.testing.assert_allclose(fit.beta[0], ref["est"], rtol=1e-8)
    np.testing.assert_allclose(kmatch_se(fit, ref_data.height), ref["se"], rtol=1e-8)
    # weights: the other arm's, each normalized to sum 1 (the scales differ)
    km = pl.read_csv(DATA / f"ref_kmatch_{name}_weights.csv.gz")
    other = 0 if estimand == "att" else 1
    km = km.filter(pl.col("d") == other).select("id", (pl.col("_W") / pl.col("_W").sum())
                                                .alias("ref"))
    got = m.weights().collect().filter(pl.col("treated") == bool(other))
    got = got.select("id", (pl.col("weight") / pl.col("weight").sum()).alias("w"))
    both = km.join(got, on="id", how="full", coalesce=True)
    assert both["w"].null_count() == 0 and both["ref"].null_count() == 0
    np.testing.assert_allclose(both["w"].to_numpy(), both["ref"].to_numpy(), rtol=1e-8,
                               atol=1e-12)


def kmatch_se(fit, n_sample):
    """match-stream's influence-function SE on kmatch's scale: kmatch sums
    the influence functions over the whole estimation sample (unmatched
    units contribute 0) with Stata's n/(n-1), where match-stream uses the
    weighted regression's G/(G-1)(N-1)/(N-2) over the matched rows."""
    (G,), N = fit.n_clusters["unit"], fit.n_obs
    hc0 = fit.vcov[0, 0] / (G / (G - 1) * (N - 1) / (N - 2))
    return np.sqrt(hc0 * n_sample / (n_sample - 1))


@pytest.mark.skipif(not KM.exists(), reason="run tests/reference/stata/kmatch_kernel.do")
@pytest.mark.parametrize("weighted,key", [(False, "kernel_att_cv"), (True, "kernel_att_cvw")])
def test_kmatch_cv_bandwidth_is_close(ref_data, weighted, key):
    # the same criterion; kmatch adds a support-loss penalty and its own
    # search path, match-stream judges every bandwidth on a fixed set of
    # units. On these data: 0.0569 vs 0.0552 and
    # 0.0724 vs 0.0758
    ref = json.loads(KM.read_text())
    m = match_kernel("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], score="ps_true",
                     bandwidth=CrossValidation(outcome="y", weighted=weighted))
    np.testing.assert_allclose(m.bandwidth, ref[key]["bwidth"][0], rtol=0.1)


@pytest.mark.skipif(not KM.exists(), reason="run tests/reference/stata/kmatch_kernel.do")
@pytest.mark.parametrize("weighted,key", [(False, "kernel_att_cv"), (True, "kernel_att_cvw")])
def test_support_loss_penalty_reproduces_kmatch(ref_data, weighted, key):
    # kmatch's criterion (support-loss penalty, limit factor) and its search
    ref = json.loads(KM.read_text())
    m = match_kernel("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], score="ps_true",
                     bandwidth=CrossValidation(outcome="y", weighted=weighted,
                                               penalty="support_loss"))
    np.testing.assert_allclose(m.bandwidth, ref[key]["bwidth"][0], rtol=1e-12)
    assert m.bandwidth_info.cv.height == 15 and m.bandwidth_info.params["penalty"] == "support_loss"


def test_support_loss_penalty_on_a_grid(sim):
    # with a grid, every point is evaluated and the smallest criterion wins
    m = match_kernel("d ~ x", sim, exact="g", score="s", ridge=True,
                     bandwidth=CrossValidation(outcome="y", grid=[0.02, 0.05, 0.12],
                                               penalty="support_loss"))
    t = m.bandwidth_info.cv
    assert t.height == 3
    assert m.bandwidth == t.sort("criterion")["bandwidth"][0]


def test_cv_penalty_checks():
    with pytest.raises(ValueError, match="penalty must be"):
        CrossValidation(outcome="y", penalty="stata")


def _score_cv_inputs(m):
    from match_stream._prep import SCORE, STRATUM, with_strata
    nar = m._narrow_lf.collect()
    df, _ = with_strata(m.spec, nar)
    focal = df[m.spec.treatment].to_numpy().astype(bool)
    if m.spec.estimand == "atc":
        focal = ~focal
    return nar, df[SCORE].to_numpy(), df[STRATUM].to_numpy(), focal


@pytest.mark.parametrize("method,kernel,estimand", [("kernel", "epan", "att"),
                                                    ("ridge", "epan", "att"),
                                                    ("kernel", "triangle", "atc"),
                                                    ("ridge", "biweight", "atc")])
def test_score_cv_is_kmatchs_criterion(ref_data, method, kernel, estimand):
    # the window kernel and the summed criterion against a loop-by-loop port
    # of kmatch's Kmatch_i_PS_cvs / Kmatch_cvs (tests/kmatch_cvs_port.py)
    from kmatch_cvs_port import kmatch_cvs

    from match_stream import _kernel as K
    d = ref_data.head(4000)
    m = match_kernel("d ~ x1", d, exact=["r", "i"], score="ps_true",
                     ridge=method == "ridge" or None, kernel=kernel,
                     focal="control" if estimand == "atc" else "treated",
                     bandwidth="cross_validation", verbose=False)
    nar, s, g, f = _score_cv_inputs(m)
    cL, cR = K.kernel_poly(m.spec)

    def kfun(u):
        c = np.where(u < 0, 0, 1)
        out = np.zeros_like(u)
        for side, coef in ((0, cL), (1, cR)):
            v = sum(a * u ** p for p, a in enumerate(coef))
            out = np.where(c == side, v, out)
        return out
    r = K.ridge_param(m.spec)
    for h in (0.004, 0.02, 0.07):
        ours = K.score_criterion([K.frame_score_cv(m.spec, nar, h)], h, 0.12)
        port = kmatch_cvs(s[f], g[f], s[~f], g[~f], h, kfun, None if r < 0 else r, 0.12)
        np.testing.assert_allclose(ours, port, rtol=1e-12)


def test_score_cv_search_and_paths(ref_data):
    kw = dict(exact=["r", "i"], score="ps_true", bandwidth="cross_validation", verbose=False)
    m = match_kernel("d ~ x1", ref_data, **kw)
    info = m.bandwidth_info
    assert info.rule == "cross_validation" and info.params["criterion"] == "score balance"
    t = info.cv
    assert m.bandwidth == t["bandwidth"][int(np.argmin(t["criterion"].to_numpy()))]
    # the same bandwidth a stratum batch at a time
    mb = match_kernel("d ~ x1", ref_data, memory="by_stratum", **kw)
    assert mb.bandwidth == m.bandwidth
    # kmatch's: 15 evaluations, with the limit factor
    mk = match_kernel("d ~ x1", ref_data, **{**kw, "bandwidth": CrossValidation(
        penalty="support_loss")})
    assert mk.bandwidth_info.cv.height == 15
    assert mk.bandwidth_info.params["penalty"] == "support_loss"
    # a grid: every point evaluated
    mg = match_kernel("d ~ x1", ref_data, **{**kw, "bandwidth": CrossValidation(
        grid=[0.01, 0.03, 0.09])})
    assert mg.bandwidth_info.cv["bandwidth"].to_list() == [0.01, 0.03, 0.09]


KMCV = DATA / "ref_kmatch_cv.json"
KMCV_CASES = {   # name: (match_stream options, estimand)
    "kernel_att_cvs": (dict(kernel="epan"), "att"),
    "ridge_att_cvs": (dict(ridge=True, kernel="epan"), "att"),
    "kernel_atc_cvs": (dict(kernel="epan"), "atc"),
    "kernel_att_cvs_tri": (dict(kernel="triangle"), "att"),
    "kernel_att_cvs_grid": (dict(kernel="epan", grid=[0.01, 0.02, 0.04, 0.08, 0.16]), "att"),
}


@pytest.mark.skipif(not KMCV.exists(), reason="run tests/reference/stata/kmatch_cv.do")
@pytest.mark.parametrize("name", list(KMCV_CASES))
def test_kmatch_score_cv_reference(ref_data, name):
    ref = json.loads(KMCV.read_text())[name]
    if ref["rc"]:
        pytest.fail(f"kmatch stopped with return code {ref['rc']} ({ref['options']})")
    kw, estimand = KMCV_CASES[name]
    kw = dict(kw)
    grid = kw.pop("grid", None)
    m = match_kernel("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], score="ps_true",
                     focal="control" if estimand == "atc" else "treated", id="id",
                     bandwidth=CrossValidation(grid=grid, penalty="support_loss"),
                     verbose=False, **kw)
    # kmatch's search: the same bandwidths in the same order, the same
    # criteria (first run, kmatch 1.1.6: to 3e-12; bandwidths identical)
    path = np.array(ref.get("cv") or ref.get("cv_treated"))
    ours = m.bandwidth_info.cv
    np.testing.assert_allclose(ours["bandwidth"].to_numpy(), path[:, 0], rtol=1e-14)
    np.testing.assert_allclose(ours["criterion"].to_numpy(), path[:, 1], rtol=1e-10)
    assert m.bandwidth == ref["bwidth"][0]
    fit = m.effect("y")
    np.testing.assert_allclose(fit.beta[0], ref["est"], rtol=1e-12)
    np.testing.assert_allclose(kmatch_se(fit, ref_data.height), ref["se"], rtol=1e-12)
    # the other arm's weights, each normalized to sum 1
    km = pl.read_csv(DATA / f"ref_kmatch_cv_{name}_weights.csv.gz")
    other = 0 if estimand == "att" else 1
    km = km.filter(pl.col("d") == other).select("id", (pl.col("_W") / pl.col("_W").sum())
                                                .alias("ref"))
    got = m.weights().collect().filter(pl.col("treated") == bool(other))
    got = got.select("id", (pl.col("weight") / pl.col("weight").sum()).alias("w"))
    both = km.join(got, on="id", how="full", coalesce=True)
    assert both["w"].null_count() == 0 and both["ref"].null_count() == 0
    np.testing.assert_allclose(both["w"].to_numpy(), both["ref"].to_numpy(), rtol=1e-10,
                               atol=1e-15)
