"""Balance tables against MatchIt's summary(m, un = TRUE) (tests/data/ref_r.json)."""

import numpy as np
import pytest

from match_stream import match_cem, match_nn

CAL3 = {"ps_true": 0.2, "x1": 0.25, "x2": 0.25}
SPECS = {
    "nn_ps_rep": dict(replace=True, caliper=0.2),
    "nn_ps_rep_k3_cal": dict(replace=True, n_matches=3, caliper=CAL3),
    "nn_ps_wor_largest": dict(match_order="largest", caliper=CAL3),
    "cem5": dict(cutpoints=5),
}
COLS = ["Means Treated", "Means Control", "Std. Mean Diff.", "Var. Ratio", "eCDF Mean",
        "eCDF Max", "Std. Pair Dist."]


@pytest.fixture(scope="module")
def tables(ref_data):
    out = {}
    for name, kw in SPECS.items():
        if "cutpoints" in kw:
            m = match_cem("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], **kw)
        else:
            m = match_nn("d ~ x1 + x2 + x3", ref_data, exact=["r", "i"], score="ps_true", **kw)
        out[name] = m.balance()
    return out


def _compare(ours, ref):
    got = {r["term"]: r for r in ours.iter_rows(named=True)}
    # MatchIt also lists the exact-matching variables; compare the shared terms
    shared = [row for row in ref if row["term"] in got]
    assert len(shared) == len(got)
    for row in shared:
        mine = got[row["term"]]
        for c in COLS:
            if c not in row:
                continue
            a, b = mine[c], row[c]
            if b is None or (isinstance(b, float) and np.isnan(b)) or b == "NA":
                assert a is None or np.isnan(a), (row["term"], c, a)
            else:
                np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12,
                                           err_msg=f"{row['term']} {c}")


@pytest.mark.parametrize("spec", list(SPECS))
def test_balance_before_matching(tables, ref_r, spec):
    _compare(tables[spec].all, ref_r["matchit"][spec]["balance_all"])


@pytest.mark.parametrize("spec", list(SPECS))
def test_balance_after_matching(tables, ref_r, spec):
    _compare(tables[spec].matched, ref_r["matchit"][spec]["balance_matched"])


@pytest.mark.parametrize("spec", list(SPECS))
def test_sample_sizes(tables, ref_r, spec):
    # MatchIt's nn table: All (ESS), All, Matched (ESS), Matched, Unmatched, Discarded
    control, treated = ref_r["matchit"][spec]["nn"]
    s = tables[spec].sizes
    np.testing.assert_allclose(s["Control"].to_numpy(), control, rtol=1e-10)
    np.testing.assert_allclose(s["Treated"].to_numpy(), treated, rtol=1e-10)
