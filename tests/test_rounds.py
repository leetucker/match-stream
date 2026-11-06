"""Without replacement and k > 1: MatchIt's rounds (the default) or one
pass (match_in_rounds=False; with match_order="closest", kmatch's wor). The one-pass
orders are checked against direct implementations: moremata's mm_greedy
for "closest" (every allowed pair in ascending distance, a treated unit
taking controls until it has k) and, for the other orders, each unit
taking its k nearest free controls at its turn."""

import numpy as np
import polars as pl
import pytest

from match_stream import match_nn


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(8)
    n = 1500
    s = rng.beta(2, 4, n)                       # continuous: no tied distances
    d = rng.random(n) < 0.15 + 0.6 * s
    return pl.DataFrame({"d": d.astype(np.int64), "s": s, "g": rng.integers(0, 3, n),
                         "x": rng.standard_normal(n), "id": np.arange(n)})


def mm_greedy(df, k, cal):
    out = {}
    for _, g in df.group_by("g"):
        t, c = g.filter(pl.col("d") == 1), g.filter(pl.col("d") == 0)
        ts, cs = t["s"].to_numpy(), c["s"].to_numpy()
        D = np.abs(ts[:, None] - cs[None, :])
        ii, jj = np.nonzero(D <= cal)
        order = np.argsort(D[ii, jj], kind="stable")
        used, got = set(), {}
        for p in order:
            i, j = ii[p], jj[p]
            if j in used or len(got.get(i, [])) >= k:
                continue
            got.setdefault(i, []).append(j)
            used.add(j)
        tid, cid = t["id"].to_list(), c["id"].to_list()
        out |= {tid[i]: {cid[j] for j in js} for i, js in got.items()}
    return out


def unit_pass(df, k, cal, order):
    out = {}
    for _, g in df.group_by("g"):
        t = g.filter(pl.col("d") == 1)
        t = (t.sort("id") if order == "data"
             else t.sort(["s", "id"], descending=[order == "largest", False]))
        c = g.filter(pl.col("d") == 0)
        cs, cid = c["s"].to_numpy(), c["id"].to_numpy()
        free = np.ones(len(cs), bool)
        for tid, s in zip(t["id"].to_list(), t["s"].to_list()):
            dist = np.where(free, np.abs(cs - s), np.inf)
            pick = [j for j in np.argsort(dist, kind="stable")[:k]
                    if np.isfinite(dist[j]) and dist[j] <= cal]     # free ones only
            free[pick] = False
            if pick:
                out[tid] = set(cid[pick].tolist())
    return out


def ours(m):
    t = m.matches().collect()
    return {a: set(b) for a, b in t.group_by("treated_id").agg(pl.col("control_id")).iter_rows()}


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("cal", [np.inf, 0.01])
def test_one_pass_closest_is_mm_greedy(sim, k, cal):
    kw = {} if np.isinf(cal) else dict(caliper={"s": (cal, "raw")})
    m = match_nn("d ~ x", sim, exact="g", score="s", n_matches=k, match_order="closest",
                 match_in_rounds=False, id="id", **kw)
    assert ours(m) == mm_greedy(sim, k, cal)


@pytest.mark.parametrize("order", ["largest", "smallest"])
def test_one_pass_in_unit_order(sim, order):
    m = match_nn("d ~ x", sim, exact="g", score="s", n_matches=3, match_order=order,
                 match_in_rounds=False,
                 id="id")
    assert ours(m) == unit_pass(sim, 3, np.inf, order)


@pytest.mark.parametrize("order", ["closest", "largest"])
def test_rounds_differ_for_k_above_1_and_agree_for_k_1(sim, order):
    kw = dict(exact="g", score="s", match_order=order, id="id")
    a = ours(match_nn("d ~ x", sim, n_matches=2, **kw))
    b = ours(match_nn("d ~ x", sim, n_matches=2, match_in_rounds=False, **kw))
    assert a != b
    assert ours(match_nn("d ~ x", sim, **kw)) == \
        ours(match_nn("d ~ x", sim, match_in_rounds=False, **kw))


def test_metric_one_pass(sim):
    kw = dict(exact="g", distance="euclidean", id="id")
    a = match_nn("d ~ s", sim, n_matches=2, **kw)
    b = match_nn("d ~ s", sim, n_matches=2, match_in_rounds=False, **kw)
    # on one covariate the Euclidean distance is the gap in it; the default
    # order for a metric distance is data order
    assert ours(b) == unit_pass(sim, 2, np.inf, "data")
    assert ours(a) != ours(b)


def test_rounds_checks(sim):
    with pytest.raises(ValueError, match="without replacement"):
        match_nn("d ~ x", sim, score="s", replace=True, match_in_rounds=False)


@pytest.mark.parametrize("order", ["largest", "closest", "data"])
@pytest.mark.parametrize("match_in_rounds", [True, False])
def test_more_focal_units_than_controls(order, match_in_rounds):
    # regression: with no caliper, a stratum whose controls run out used to
    # accept a position past the end of the controls (memory corruption in
    # the compiled kernel); now the remaining focal units go unmatched
    rng = np.random.default_rng(2)
    n = 200
    df = pl.DataFrame({"d": (rng.random(n) < 0.8).astype(np.int64), "s": rng.random(n),
                       "x": np.zeros(n), "g": rng.integers(0, 4, n), "id": np.arange(n)})
    m = match_nn("d ~ x", df, exact="g", score="s", n_matches=2, match_order=order,
                 match_in_rounds=match_in_rounds,
                 id="id")
    t = m.matches().collect()
    controls = set(df.filter(pl.col("d") == 0)["id"].to_list())
    assert set(t["control_id"].to_list()) <= controls
    assert t["control_id"].n_unique() == t.height             # each control once
    assert t.height == len(controls)                           # every control used


@pytest.mark.parametrize("keep_ties", [False, True])
def test_k_above_a_stratums_controls_with_replacement(keep_ties):
    # regression: with no caliper, k larger than a stratum's controls used to
    # take positions past the stratum (a control of the next stratum, or out
    # of bounds); now a focal unit gets every control of its stratum
    rng = np.random.default_rng(4)
    n = 120
    g = rng.integers(0, 6, n)
    d = (rng.random(n) < 0.85).astype(np.int64)
    df = pl.DataFrame({"d": d, "s": rng.random(n), "x": np.zeros(n), "g": g, "id": np.arange(n)})
    m = match_nn("d ~ x", df, exact="g", score="s", n_matches=5, replace=True,
                 keep_ties=keep_ties,
                 id="id")
    t = m.matches().collect()
    gid = dict(zip(df["id"].to_list(), df["g"].to_list()))
    assert all(gid[a] == gid[b] for a, b in zip(t["treated_id"].to_list(),
                                                   t["control_id"].to_list()))
    n_c = df.filter(pl.col("d") == 0).group_by("g").len()
    cap = dict(zip(n_c["g"].to_list(), n_c["len"].to_list()))
    per = t.group_by("treated_id").len()
    assert all(c == min(5, cap[gid[f]]) for f, c in per.iter_rows())
