"""Common support (discard=, reestimate_score=) against MatchIt 4.8.2
(tests/reference/reference_r_discard.R -> tests/data/ref_r_discard.json and
ref_matchit_discard_*): the units discarded, the matches, the weights, the
balance tables and the sample sizes."""

import json

import numpy as np
import polars as pl
import pytest
from conftest import DATA
from test_balance import _compare

from match_stream import match_kernel, match_nn

REF = json.loads((DATA / "ref_r_discard.json").read_text())
SPECS = {   # MatchIt's spec -> match_stream options
    "both": dict(score="ps_true", discard="both", caliper=0.2, exact=["r", "i"]),
    "treated_rep": dict(score="ps_true", discard="treated", replace=True, exact=["r", "i"]),
    "control_k2": dict(score="ps_true", discard="control", n_matches=2, exact=["r", "i"]),
    "atc_both": dict(score="ps_true", discard="both", replace=True, focal="control"),
    "glm_both": dict(distance="ps", discard="both", replace=True, caliper=0.1),
    "glm_reest": dict(distance="ps", discard="both", reestimate_score=True, replace=True,
                      caliper=0.1),
}


@pytest.fixture(scope="module")
def fits(ref_data):
    return {name: match_nn("d ~ x1 + x2 + x3", ref_data, id="id", verbose=False, **kw)
            for name, kw in SPECS.items()}


@pytest.mark.parametrize("name", list(SPECS))
def test_discarded_units(fits, name):
    c = fits[name].counts()
    assert c["discarded_treated"] == REF[name]["discarded_treated"]
    assert c["discarded_control"] == REF[name]["discarded_control"]


@pytest.mark.parametrize("name", list(SPECS))
def test_matches(fits, name):
    m = fits[name]
    ref = pl.read_csv(DATA / f"ref_matchit_discard_{name}.csv.gz", null_values="NA")
    ref = {r[0]: {x for x in r[1:] if x is not None} for r in ref.iter_rows()}
    ref = {f: c for f, c in ref.items() if c}
    t = m.matches().collect()
    foc, oth = ("treated_id", "control_id") if m.focal_is_treated else \
        ("control_id", "treated_id")
    ours = {a: set(b) for a, b in t.group_by(foc).agg(pl.col(oth)).iter_rows()}
    assert ours == ref


@pytest.mark.parametrize("name", list(SPECS))
def test_weights(fits, name):
    ref = pl.read_csv(DATA / f"ref_matchit_discard_{name}_weights.csv.gz")
    got = fits[name].weights().collect().select("id", "weight")
    both = ref.join(got, on="id", how="full", coalesce=True)
    assert both["weight"].null_count() == 0 and both["weight_right"].null_count() == 0
    np.testing.assert_allclose(both["weight_right"].to_numpy(), both["weight"].to_numpy(),
                               rtol=1e-10)


@pytest.mark.parametrize("name", list(SPECS))
def test_balance_and_sizes(fits, name):
    b = fits[name].balance()
    _compare(b.all, REF[name]["balance_all"])
    _compare(b.matched, REF[name]["balance_matched"])
    control, treated = REF[name]["nn"]
    np.testing.assert_allclose(b.sizes["Control"].to_numpy(), control, rtol=1e-10)
    np.testing.assert_allclose(b.sizes["Treated"].to_numpy(), treated, rtol=1e-10)


def test_checks(ref_data):
    with pytest.raises(ValueError, match="compares propensity scores"):
        match_nn("d ~ x1 + x2", ref_data, distance="mahalanobis", discard="both")
    with pytest.raises(ValueError, match="reestimate_score"):
        match_nn("d ~ x1", ref_data, score="ps_true", discard="both", reestimate_score=True)
    with pytest.raises(TypeError, match="discard"):       # kernel matching has no discard=
        match_kernel("d ~ x1", ref_data, score="ps_true", discard="both")
    # a column of units to drop, with any distance
    drop = ref_data.with_columns((pl.col("id") <= 100).alias("drop"))
    m = match_nn("d ~ x1 + x2", drop, distance="mahalanobis", discard="drop", replace=True)
    c = m.counts()
    assert c["discarded_treated"] + c["discarded_control"] == 100
    assert not set(range(100)) & set(m.matches().collect()["treated_row"].to_list())
