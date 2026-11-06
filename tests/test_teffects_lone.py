"""How Stata's teffects nnmatch treats a unit alone in its arm within its
exact cell (ematch) when the robust variance needs that unit's conditional
variance (tests/reference/stata/teffects_lone.do; skipped until its outputs
are in tests/data/).

The candidates (match-stream's `effect(lone_unit_variance=)` and R Matching's):
an error (match-stream's default, a NaN SE with a warning), the unit's
variance set to 0 (R Matching's), from its nearest same-arm units in any
cell (``lone_unit_variance="nearest"``), or its arm's mean variance
(``"pooled"``). The test says which teffects does, and fails if none.

Found on the first run (Stata 18): teffects refuses. vce(robust, nn(2))
stops with return code 459 in both variants; nn(1) with 198 (an invalid
option: its robust variance takes at least two neighbors). vce(iid) agrees
with match-stream, matches included."""

import json
import warnings

import numpy as np
import polars as pl
import pytest
from conftest import DATA

from match_stream import effects as E
from match_stream import match_nn

REF = DATA / "ref_teffects_lone.json"
pytestmark = pytest.mark.skipif(not REF.exists(),
                                reason="run tests/reference/stata/teffects_lone.do")

# case: (variant, estimand, n_variance_matches)
CASES = {f"lone_{v}_{s}_{tag}": (v, s.replace("atet", "att"), vc)
         for v, s in (("treated", "ate"), ("control", "atet"))
         for tag, vc in (("vc1", 1), ("vc2", 2), ("iid", 0))}


@pytest.fixture(scope="module")
def te():
    return json.loads(REF.read_text())


def lone_data(ref_data, variant):
    """teffects_lone.do's data: the cell r = 0, i = 0 keeps only its first
    (lowest id) unit of the arm."""
    arm = 1 if variant == "treated" else 0
    cell = (pl.col("r") == 0) & (pl.col("i") == 0) & (pl.col("d") == arm)
    first = ref_data.filter(cell)["id"].min()
    return ref_data.filter(~cell | (pl.col("id") == first))


def te_pairs(name) -> dict[int, set[int]]:
    t = pl.read_csv(DATA / f"ref_teffects_lone_{name}_matches.csv.gz")
    mcols = [c for c in t.columns if c not in ("id", "d")]
    return {r[0]: {x for x in r[2:] if x is not None}
            for r in t.select("id", "d", *mcols).iter_rows()}


def our_pairs(m, estimand) -> dict[int, set[int]]:
    out = {}
    for x in E._sets(m, estimand):
        t = x.matches().collect()
        foc, oth = ("treated_id", "control_id") if x.focal_is_treated else \
            ("control_id", "treated_id")
        out.update({a: set(b) for a, b in t.group_by(foc).agg(pl.col(oth)).iter_rows()})
    return out


def zero_se(m, estimand, n_variance_matches):
    """The AI SE with the lone units' conditional variances set to 0 (R Matching's)."""
    df = E._outcomes(m, "y", [])
    s2 = np.nan_to_num(E._sigma2(m, df, n_variance_matches), nan=0.0)
    rows = df["__ms_row"].to_list()
    sets = [(x._pairs, x.focal_is_treated) for x in E._sets(m, estimand)]
    _, var, _ = E.ai_estimate(sets, dict(zip(rows, df["__y"].to_list())), estimand,
                              dict(zip(rows, s2.tolist())))
    return float(np.sqrt(var))


@pytest.mark.parametrize("name", list(CASES))
def test_lone_unit(ref_data, te, name):
    variant, estimand, n_variance_matches = CASES[name]
    ref = te[name]
    m = match_nn("d ~ x1", lone_data(ref_data, variant), exact=["r", "i"],
                 score="ps_true", replace=True, id="id", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = m.effect("y", estimand=estimand, n_variance_matches=n_variance_matches)
    if ref["rc"]:
        # teffects refuses: match-stream's default must not give a number either
        print(f"{name}: teffects stops with return code {ref['rc']}")
        assert n_variance_matches > 0 and np.isnan(default.se[0])
        return
    assert our_pairs(m, estimand) == te_pairs(name)
    np.testing.assert_allclose(default.beta[0], ref["est"], rtol=1e-10)
    if n_variance_matches == 0:
        np.testing.assert_allclose(default.se[0], ref["se"], rtol=1e-8)
        return
    cands = {"zero (R Matching)": zero_se(m, estimand, n_variance_matches)}
    for fb in ("nearest", "pooled"):
        cands[f'lone_unit_variance="{fb}"'] = m.effect(
            "y", estimand=estimand, n_variance_matches=n_variance_matches,
            lone_unit_variance=fb).se[0]
    hits = [k for k, v in cands.items() if np.isclose(v, ref["se"], rtol=1e-7, atol=0)]
    print(f"{name}: teffects SE {ref['se']}; candidates {cands}; teffects is {hits}")
    assert hits, f"teffects' SE {ref['se']} matches no candidate rule: {cands}"
