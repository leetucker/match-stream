"""Arguments, errors, sources and the ATC."""

import polars as pl
import pytest

from match_stream import (
    CrossValidation,
    PairDistance,
    match_cem,
    match_kernel,
    match_nn,
    simulate_matching,
)


@pytest.fixture(scope="module")
def sim():
    return simulate_matching(4_000, seed=3).rename({"ps": "ps_true"})


def test_pipe_in_formula_points_to_exact(sim):
    with pytest.raises(ValueError, match="exact="):
        match_nn("d ~ x1 | r", sim)


def test_missing_column(sim):
    with pytest.raises(ValueError, match="not in the data"):
        match_nn("d ~ x1", sim, exact="nope", score="ps_true")


def test_treatment_must_be_binary(sim):
    with pytest.raises(ValueError, match="0/1"):
        match_nn("r ~ x1", sim, score="ps_true")


def test_caliper_units(sim):
    # a number or a dict value takes the key's default units; (width, unit) overrides
    base = dict(exact=["r", "i"], score="ps_true", replace=True)
    a = match_nn("d ~ x1", sim, caliper={"ps_true": 0.2, "x1": 0.5}, **base)
    assert (a.spec.caliper, a.spec.caliper_std, a.spec.cov_calipers_std) == (
        0.2, True, {"x1": False})
    b = match_nn("d ~ x1", sim, caliper={"ps_true": (0.02, "raw"), "x1": (0.5, "sd")}, **base)
    assert (b.spec.caliper, b.spec.caliper_std, b.spec.cov_calipers_std) == (
        0.02, False, {"x1": True})
    # a caliper in SDs of x1 is the raw caliper times its standard deviation
    sd = sim["x1"].std()
    c = match_nn("d ~ x1", sim, caliper={"x1": (0.5, "sd")}, **base)
    d = match_nn("d ~ x1", sim, caliper={"x1": 0.5 * sd}, **base)
    assert c.matches().collect().equals(d.matches().collect())


@pytest.mark.parametrize("bad", [(0.2, "sds"), (0.2,), (0.2, "sd", 1), (-1.0, "sd")])
def test_bad_caliper_value(sim, bad):
    with pytest.raises(ValueError, match="caliper"):
        match_nn("d ~ x1", sim, score="ps_true", caliper={"x1": bad})


def test_score_and_distance(sim):
    with pytest.raises(ValueError, match="score_model"):
        match_nn("d ~ x1", sim, score="ps_true", score_model="d ~ x1")
    with pytest.raises(ValueError, match="score="):
        match_nn("d ~ x1", sim, score="ps_true", distance="mahalanobis")
    with pytest.raises(ValueError, match="score="):          # a column name is not a distance
        match_nn("d ~ x1", sim, distance="ps_true")
    with pytest.raises(ValueError, match="distance must be"):
        match_kernel("d ~ x1", sim, distance="mahalanobis")


def test_focal(sim):
    with pytest.raises(ValueError, match="focal"):
        match_nn("d ~ x1", sim, score="ps_true", focal="att")
    assert match_nn("d ~ x1", sim, score="ps_true", focal="control").spec.estimand == "atc"
    assert match_kernel("d ~ x1", sim, score="ps_true", bandwidth=0.05,
                        focal="control").spec.estimand == "atc"


def test_kernel_ridge_option(sim):
    kw = dict(score="ps_true", bandwidth=0.05)
    assert match_kernel("d ~ x1", sim, **kw).spec.method == "kernel"
    assert match_kernel("d ~ x1", sim, ridge=False, **kw).spec.method == "kernel"
    r = match_kernel("d ~ x1", sim, ridge=True, **kw)
    assert r.spec.method == "ridge" and r.spec.ridge is None
    z = match_kernel("d ~ x1", sim, ridge=0, **kw)               # local-linear
    assert z.spec.method == "ridge" and z.spec.ridge == 0.0


def test_bandwidth_rules(sim):
    kw = dict(score="ps_true", exact=["r", "i"])
    d = match_kernel("d ~ x1", sim, **kw)
    assert d.spec.bandwidth == "pair_distance" and d.bandwidth_info.rule == "pair_distance"
    assert match_kernel("d ~ x1", sim, bandwidth="pair_distance", **kw).bandwidth == d.bandwidth
    p = match_kernel("d ~ x1", sim, bandwidth=PairDistance(factor=3.0), **kw)
    assert p.bandwidth == pytest.approx(2 * d.bandwidth)
    cv = match_kernel("d ~ x1", sim, bandwidth=CrossValidation(outcome="y", grid=[0.02, 0.05]),
                      **kw)
    assert cv.bandwidth_info.rule == "cross_validation" and cv.bandwidth in (0.02, 0.05)
    assert match_kernel("d ~ x1", sim, bandwidth=0.04, **kw).bandwidth == 0.04
    for bad in ("pm", "cv", 0, -1.0, "wide"):
        with pytest.raises(ValueError, match="bandwidth"):
            match_kernel("d ~ x1", sim, bandwidth=bad, **kw)


def test_bandwidth_objects_validate():
    with pytest.raises(ValueError, match="factor"):
        PairDistance(factor=0)
    with pytest.raises(ValueError, match="quantile"):
        PairDistance(quantile=1.5)
    with pytest.raises(ValueError, match="penalty"):
        CrossValidation(penalty="kmatch")
    with pytest.raises(ValueError, match="outcome"):
        CrossValidation(weighted=True)


def test_cem_needs_cutpoints(sim):
    with pytest.raises(TypeError):
        match_cem("d ~ x1", sim)


def test_atc_is_att_with_roles_swapped(sim):
    # matching controls to treated units in ascending score order is matching
    # "treated" = controls in descending order of the negated score
    a = match_nn("d ~ x1", sim, exact=["r", "i"], score="ps_true", focal="control",
                 caliper={"ps_true": 0.05, "x1": 0.5})
    flipped = sim.with_columns((1 - pl.col("d")).alias("d"), (-pl.col("ps_true")).alias("ps_true"))
    b = match_nn("d ~ x1", flipped, exact=["r", "i"], score="ps_true", match_order="largest",
                 caliper={"ps_true": 0.05, "x1": 0.5})
    pa = set(a.matches().select("treated_row", "control_row").collect().iter_rows())
    pb = {(c, t) for t, c in b.matches().select("treated_row", "control_row").collect().iter_rows()}
    assert pa == pb and len(pa) > 0
    assert a.spec.match_order == "smallest"


def test_parquet_source_and_fitted_score(sim, tmp_path):
    path = tmp_path / "units.parquet"
    sim.with_columns(pl.when(pl.col("id") == 5).then(None).otherwise(pl.col("x1")).alias("x1")) \
        .write_parquet(path)
    m = match_nn("d ~ x1 + x2 + x3", str(path), exact=["r", "i"], caliper=0.2,
                 score_model="d ~ x1 + x2 + x3 | r^i", id="id")
    assert m.score_fit is not None and m.n_dropped == 1
    c = m.counts()
    assert c["matched_treated"] == c["matched_control"] == c["pairs"] > 0
    pairs = m.matches().collect()
    assert pairs["control_id"].n_unique() == pairs.height       # without replacement


def test_summary_prints(ref_data, capsys):
    # summary() reads the spec's fields by name; a renamed option once broke it unseen
    m = match_nn("d ~ x1 + x2", ref_data, n_matches=2, replace=True, verbose=False)
    m.summary()
    assert "2:1" in capsys.readouterr().out
