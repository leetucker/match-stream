"""Nearest-neighbor matching against Stata's kmatch
(tests/reference/stata/kmatch_nn.do; skipped until its outputs are in
tests/data/).

Conventions compared (from kmatch 1.1.6's Mata code):
with replacement kmatch keeps every control tied at the k-th distance
(match-stream's keep_ties=True); its caliper bwidth() is in score units and
inclusive (raw score units here); without replacement it is greedy in
ascending pair distance (match_order="closest"). Two differences by design are
handled: with k > 1 and a caliper kmatch drops a focal unit with fewer
than k matches inside it (match-stream's min_matches=k; kmatch's keepall
and match-stream's default keep it, as MatchIt); without replacement and
k > 1 kmatch fills each treated unit in one pass in distance order
(match-stream's match_in_rounds=False), MatchIt and match-stream's default in
rounds."""

import json

import numpy as np
import polars as pl
import pytest
from conftest import DATA
from test_kernel import kmatch_se

from match_stream import match_nn

REF = DATA / "ref_kmatch_nn.json"
pytestmark = pytest.mark.skipif(not REF.exists(), reason="run tests/reference/stata/kmatch_nn.do")

CAL = dict(caliper={"ps_true": (0.01, "raw")})
CASES = {   # name: (match_stream options, estimand)
    "nn_att_m1": (dict(replace=True, keep_ties=True), "att"),
    "nn_att_m3": (dict(replace=True, keep_ties=True, n_matches=3), "att"),
    "nn_atc_m2": (dict(replace=True, keep_ties=True, n_matches=2), "atc"),
    "nn_att_m1_cal": (dict(replace=True, keep_ties=True, **CAL), "att"),
    "nn_att_m3_cal_keepall": (dict(replace=True, keep_ties=True, n_matches=3, **CAL), "att"),
    "nn_att_m3_cal": (dict(replace=True, keep_ties=True, n_matches=3, min_matches=3, **CAL), "att"),
    "wor_att_m1": (dict(match_order="closest"), "att"),
    "wor_att_m1_cal": (dict(match_order="closest", **CAL), "att"),
    "wor_att_m2": (dict(match_order="closest", n_matches=2, match_in_rounds=False), "att"),
    # kmatch's one pass
}


@pytest.fixture(scope="module")
def km():
    return json.loads(REF.read_text())


def km_pairs(name) -> dict[int, set[int]]:
    t = pl.read_csv(DATA / f"ref_kmatch_nn_{name}_ids.csv.gz")
    ids = [c for c in t.columns if c.startswith("_ID_")]
    return {r[0]: {x for x in r[1:] if x is not None} for r in t.select("id", *ids).iter_rows()}


def our_pairs(m) -> dict[int, set[int]]:
    t = m.matches().collect()
    foc, oth = ("treated_id", "control_id") if m.focal_is_treated else \
        ("control_id", "treated_id")
    return {a: set(b) for a, b in t.group_by(foc).agg(pl.col(oth)).iter_rows()}


def _match(ref_data, kw, estimand):
    return match_nn("d ~ x1", ref_data, exact=["r", "i"], score="ps_true", id="id",
                    focal="control" if estimand == "atc" else "treated", **kw)


@pytest.mark.parametrize("name", list(CASES))
def test_same_matches_estimate_and_weights(ref_data, km, name):
    kw, estimand = CASES[name]
    m = _match(ref_data, kw, estimand)
    assert our_pairs(m) == km_pairs(name)
    fit = m.effect("y", vcov="if")
    np.testing.assert_allclose(fit.beta[0], km[name]["est"], rtol=1e-10)
    np.testing.assert_allclose(kmatch_se(fit, ref_data.height), km[name]["se"], rtol=1e-8)
    ref = pl.read_csv(DATA / f"ref_kmatch_nn_{name}_weights.csv.gz")
    other = 0 if estimand == "att" else 1
    ref = ref.filter(pl.col("d") == other).select("id", (pl.col("_W") / pl.col("_W").sum())
                                                  .alias("ref"))
    got = m.weights().collect().filter(pl.col("treated") == bool(other))
    got = got.select(pl.col("id"), (pl.col("weight") / pl.col("weight").sum()).alias("w"))
    both = ref.join(got, on="id", how="full", coalesce=True)
    assert both["w"].null_count() == 0 and both["ref"].null_count() == 0
    np.testing.assert_allclose(both["w"].to_numpy(), both["ref"].to_numpy(), rtol=1e-10)


def test_ate(ref_data, km):
    m = _match(ref_data, dict(replace=True, keep_ties=True), "att")
    fit = m.effect("y", estimand="ate", vcov="if")
    np.testing.assert_allclose(fit.beta[0], km["nn_ate_m1"]["est"], rtol=1e-10)
    # kmatch's ATE variance: that of p ATT + (1 - p) ATC with the share p estimated
    np.testing.assert_allclose(kmatch_se(fit, ref_data.height), km["nn_ate_m1"]["se"], rtol=1e-8)


def test_wor_k2_rounds_differ_from_kmatch(ref_data):
    # the default (MatchIt's rounds) differs from kmatch's one pass for k > 1
    ref = km_pairs("wor_att_m2")
    ours_ = our_pairs(_match(ref_data, dict(match_order="closest", n_matches=2), "att"))
    same = sum(ours_.get(t) == c for t, c in ref.items()) / len(ref)
    print(f"wor n_matches=2: {same:.1%} of kmatch's treated units have the same matches in rounds")


def test_mahalanobis_is_teffects(km):
    # kmatch md and teffects' Mahalanobis agree (both: the full-sample
    # covariance); match-stream follows MatchIt's pooled within-group one
    te = json.loads((DATA / "ref_teffects_match.json").read_text())
    np.testing.assert_allclose(km["md_att_m1"]["est"], te["nnm_atet_maha_m1_iid"]["est"],
                               rtol=1e-8)
