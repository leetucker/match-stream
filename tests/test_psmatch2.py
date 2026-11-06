"""Nearest-neighbor matching against Stata's psmatch2 (confirmed on the first
run, Stata 18 and psmatch2 4.0.12: every case below agrees, the port of its
search without replacement included)
(tests/reference/stata/psmatch2.do; the Stata comparisons skip until its
outputs are in tests/data/).

psmatch2's conventions, from its source (psmatch2.ado 4.0.12):
one global match (no exact strata); the caliper is in score units and strict
(caliper in raw score units here; the data have no distance equal to a caliper);
with replacement each treated unit takes its k nearest controls,
`keep_ties` adds every control as near as the first (only k = 1 here);
weights are 1 / (the unit's matches); `common` drops treated units strictly
outside the controls' score range, MatchIt's discard="treated"; Mahalanobis
uses the pooled within-arm covariance, as MatchIt. Without replacement
(1:1 only) it is not greedy nearest-available matching: see
tests/psmatch2_port.py, whose port of the search is checked against
psmatch2's own matches here, and printed beside match-stream's
match_order="smallest" ("largest" with `descending`).

Standard errors: psmatch2's default ignores that the matches are
estimated, sqrt(var1 / N1 + var0 * sum(w^2) / N1^2); it is rebuilt here from
match-stream's matches. ai(1) population is Abadie and Imbens (2006) with
one own-arm neighbor: match-stream's vcov="ai", n_variance_matches=1 (R Matching's
Var.calc = 1). ai(1) alone is their conditional (sample) variance, which
match-stream does not offer: printed."""

import json

import numpy as np
import polars as pl
import pytest
from conftest import DATA
from psmatch2_port import psmatch2_pairs

from match_stream import match_nn

REF = DATA / "ref_psmatch2.json"

def score_caliper(width):      # psmatch2's caliper is in score units
    return {"ps_true": (width, "raw")}

REP = {     # name: match_stream options, with replacement (exact comparisons)
    "rep_n1": dict(replace=True),
    "rep_n1_ties": dict(replace=True, keep_ties=True),
    "rep_n3_cal": dict(replace=True, n_matches=3, caliper=score_caliper(0.01)),
    "rep_n1_common": dict(replace=True, discard="treated"),
    "rep_n1_ate": dict(replace=True),
    "rep_n1_ai1": dict(replace=True),
    "rep_n1_ai1_sample": dict(replace=True),
    "mahal": dict(replace=True, distance="mahalanobis"),
}
WOR = {     # name: (port options, the nearest match-stream order)
    "wor_asc": (dict(noreplacement=True), dict(match_order="smallest")),
    "wor_desc": (dict(noreplacement=True, descending=True), dict(match_order="largest")),
    "wor_cal": (dict(noreplacement=True, caliper=0.005),
                dict(match_order="smallest", caliper=score_caliper(0.005))),
}
ONE_ID = {"rep_n1_ties"}            # psmatch2's _n1 holds one tied control only


@pytest.fixture(scope="module")
def p2():
    if not REF.exists():
        pytest.skip("run tests/reference/stata/psmatch2.do")
    out = json.loads(REF.read_text())
    failed = {k: v for k, v in out.items() if isinstance(v, dict) and v.get("rc")}
    if failed:
        print(f"psmatch2 cases that failed: {failed}")
    return out


def _case(p2, name):
    if p2.get(name, {}).get("rc", 1):
        pytest.skip(f"psmatch2 case {name} did not run")
    return p2[name]


def _match(ref_data, kw):
    if kw.get("distance") == "mahalanobis":
        return match_nn("d ~ x1 + x2 + x3", ref_data, id="id", **kw)
    return match_nn("d ~ x1", ref_data, score="ps_true", id="id", **kw)


def p2_pairs(name) -> dict[int, set[int]]:
    t = pl.read_csv(DATA / f"ref_psmatch2_{name}.csv.gz").filter(pl.col("d") == 1)
    cols = [c for c in t.columns if c.startswith("match")]
    return {r[0]: {x for x in r[1:] if x is not None}
            for r in t.select("id", *cols).iter_rows() if any(x is not None for x in r[1:])}


def our_pairs(m) -> dict[int, list[int]]:
    t = m.matches().collect()
    return {a: list(b) for a, b in t.group_by("treated_id").agg(pl.col("control_id")).iter_rows()}


def as_sets(pairs):
    return {t: set(c) for t, c in pairs.items()}


def att_and_se(ref_data, pairs):
    """psmatch2's ATT and default SE from a treated -> controls map."""
    y = dict(zip(ref_data["id"], ref_data["y"]))
    yt = np.array([y[t] for t in pairs])
    ym = np.array([np.mean([y[c] for c in cs]) for cs in pairs.values()])
    w = {}
    for cs in pairs.values():
        for c in cs:
            w[c] = w.get(c, 0.0) + 1 / len(cs)
    n1 = len(pairs)
    var0 = np.var([y[c] for c in w], ddof=1)
    se = np.sqrt(np.var(yt, ddof=1) / n1 + var0 * sum(v * v for v in w.values()) / n1**2)
    return yt.mean() - ym.mean(), se


# without Stata: the port against match-stream

@pytest.mark.parametrize("port_kw, ours", [
    (dict(), dict(replace=True)),
    (dict(n_matches=3, caliper=0.01), REP["rep_n3_cal"]),
])
def test_port_with_replacement_is_match_stream(ref_data, port_kw, ours):
    port = psmatch2_pairs(ref_data, score="ps_true", **port_kw)
    assert as_sets(port) == as_sets(our_pairs(_match(ref_data, ours)))


def test_port_without_replacement_is_not_greedy(ref_data):
    # the forward-only pointer: some treated units get a control while a
    # nearer one is free below the pointer
    port = psmatch2_pairs(ref_data, score="ps_true", noreplacement=True)
    ours_ = our_pairs(_match(ref_data, WOR["wor_asc"][1]))
    same = sum(set(port[t]) == set(ours_[t]) for t in port) / len(port)
    print(f"psmatch2 noreplacement: {same:.1%} of treated units as match_order='smallest'")
    assert 0.5 < same < 1
    ps = dict(zip(ref_data["id"], ref_data["ps_true"]))

    def gap(pairs):
        return np.mean([abs(ps[t] - ps[c[0]]) for t, c in pairs.items()])
    assert gap(ours_) < gap(port)


# against psmatch2's outputs

@pytest.mark.parametrize("name", list(REP))
def test_with_replacement(ref_data, p2, name):
    ref = _case(p2, name)
    m = _match(ref_data, REP[name])
    pairs = our_pairs(m)
    if name not in ONE_ID:
        assert as_sets(pairs) == p2_pairs(name)
    fit = m.effect("y", vcov="if")
    np.testing.assert_allclose(fit.beta[0], ref["est"], rtol=1e-10)
    if "ai1" not in name:
        # psmatch2 squares the weights into a float variable (gen w2 =
        # _weight^2): exact for whole weights, ~1e-9 off for k = 3's thirds
        np.testing.assert_allclose(att_and_se(ref_data, pairs)[1], ref["se"], rtol=1e-8)
    w = pl.read_csv(DATA / f"ref_psmatch2_{name}.csv.gz")
    w = w.filter((pl.col("d") == 0) & pl.col("_weight").is_not_null() & (pl.col("_weight") > 0))
    w = w.select("id", (pl.col("_weight") / pl.col("_weight").sum()).alias("ref"))
    got = m.weights().collect().filter(~pl.col("treated") & (pl.col("weight") > 0))
    got = got.select("id", (pl.col("weight") / pl.col("weight").sum()).alias("w"))
    both = w.join(got, on="id", how="full", coalesce=True)
    assert both["w"].null_count() == 0 and both["ref"].null_count() == 0
    np.testing.assert_allclose(both["w"].to_numpy(), both["ref"].to_numpy(), rtol=1e-10)


@pytest.mark.parametrize("name", list(WOR))
def test_without_replacement_is_the_port(ref_data, p2, name):
    ref = _case(p2, name)
    port_kw, ours = WOR[name]
    port = psmatch2_pairs(ref_data, score="ps_true", **port_kw)
    assert as_sets(port) == p2_pairs(name)
    est, se = att_and_se(ref_data, port)
    np.testing.assert_allclose(est, ref["est"], rtol=1e-10)
    np.testing.assert_allclose(se, ref["se"], rtol=1e-10)
    ours_ = our_pairs(_match(ref_data, ours))
    same = sum(set(port.get(t, ())) == set(ours_.get(t, ())) for t in port | ours_)
    print(f"{name}: {same} of {len(port | ours_)} treated units as match-stream {ours}")


def test_common_support(ref_data, p2):
    ref = _case(p2, "rep_n1_common")
    m = _match(ref_data, REP["rep_n1_common"])
    t = pl.read_csv(DATA / "ref_psmatch2_rep_n1_common.csv.gz")
    # _support as 0/1, or as its value labels (round 3 was exported with them)
    off = t.filter((pl.col("d") == 1)
                   & pl.col("_support").cast(pl.Utf8).is_in(["0", "Off support"])).height
    assert len(our_pairs(m)) == ref["matched_treated"]
    assert m.counts()["discarded_treated"] == off


def test_ate(ref_data, p2):
    ref = _case(p2, "rep_n1_ate")
    m = _match(ref_data, REP["rep_n1_ate"])
    np.testing.assert_allclose(m.effect("y", estimand="ate", vcov="if").beta[0], ref["ate"],
                               rtol=1e-10)
    np.testing.assert_allclose(m.effect("y", estimand="atc", vcov="if").beta[0], ref["atu"],
                               rtol=1e-10)


def test_ai_se(ref_data, p2):
    ref = _case(p2, "rep_n1_ai1")
    m = _match(ref_data, REP["rep_n1_ai1"])
    se = m.effect("y", vcov="ai", n_variance_matches=1).se[0]
    np.testing.assert_allclose(se, ref["se"], rtol=1e-8)
    sample = _case(p2, "rep_n1_ai1_sample")["se"]
    print(f"AI SE: population {se} (= psmatch2), psmatch2's sample (conditional) {sample}")
