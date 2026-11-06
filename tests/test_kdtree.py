"""The k-d tree path (match_stream/_kdtree.py) gives the brute-force
kernel's matches (kernels/metric.py), ties included."""

import numpy as np
import polars as pl
import pytest

import match_stream._kdtree as kd
from match_stream import match_nn
from match_stream._kdtree import knn_kdtree
from match_stream.kernels.metric import knn_replace


def _blocks(rng, sizes_t, sizes_c, p, discrete):
    tz = rng.standard_normal((sum(sizes_t), p))
    cz = rng.standard_normal((sum(sizes_c), p))
    if discrete:                       # few distinct values: many exact ties
        tz, cz = np.round(tz), np.round(cz)
    return (np.concatenate([[0], np.cumsum(sizes_t)]), tz,
            np.concatenate([[0], np.cumsum(sizes_c)]), cz)


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("discrete", [False, True])
def test_same_as_brute_force(k, discrete):
    rng = np.random.default_rng(k + 10 * discrete)
    sizes_t = [300, 2, 150, 40, 0, 7]
    sizes_c = [900, 3, 400, 1, 50, 0]          # strata with fewer controls than k
    tstarts, tz, starts, cz = _blocks(rng, sizes_t, sizes_c, 3, discrete)
    none = np.zeros((tz.shape[0], 0)), np.zeros((cz.shape[0], 0)), np.zeros(0)
    brute = knn_replace(tstarts, tz, none[0], starts, cz, none[1], none[2], k)
    tree = knn_kdtree(tstarts, tz, starts, cz, k)
    np.testing.assert_array_equal(tree, brute)


@pytest.mark.parametrize("distance", ["mahalanobis", "euclidean"])
def test_end_to_end_with_and_without_the_tree(monkeypatch, distance):
    rng = np.random.default_rng(5)
    n = 6000
    df = pl.DataFrame({"d": (rng.random(n) < 0.3).astype(np.int64),
                       "x1": np.round(rng.standard_normal(n), 1),
                       "x2": rng.standard_normal(n), "g": rng.integers(0, 7, n)})
    kw = dict(exact="g", distance=distance, replace=True, n_matches=2)
    tree = match_nn("d ~ x1 + x2", df, **kw).matches().collect()
    monkeypatch.setattr(kd, "KDTREE_MAX_DIM", 0)          # force the brute force
    brute = match_nn("d ~ x1 + x2", df, **kw).matches().collect()
    assert tree.equals(brute)


# --------------------------------------------------- without replacement
from match_stream._kdtree import match_without_replacement  # noqa: E402
from match_stream.kernels.metric import greedy_no_replace  # noqa: E402


def _case(rng, discrete, scarce, p=3):
    sizes_t = [120, 3, 60, 30, 0, 8] if not scarce else [150, 3, 80, 40, 0, 8]
    sizes_c = [400, 5, 200, 2, 30, 0] if not scarce else [100, 2, 50, 20, 30, 0]
    tstarts, tz, starts, cz = _blocks(rng, sizes_t, sizes_c, p, discrete)
    tx = np.ascontiguousarray(tz[:, :1])            # a caliper on the first covariate
    cx = np.ascontiguousarray(cz[:, :1])
    return tstarts, tz, tx, starts, cz, cx


@pytest.mark.parametrize("k,rounds", [(1, True), (2, True), (3, True), (2, False), (3, False)])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("scarce", [False, True])
@pytest.mark.parametrize("caliper", [None, 0.5])
def test_without_replacement_same_as_brute_force(k, rounds, discrete, scarce, caliper):
    rng = np.random.default_rng(100 * k + 10 * discrete + scarce + int(rounds))
    tstarts, tz, tx, starts, cz, cx = _case(rng, discrete, scarce)
    calx = np.array([caliper]) if caliper else np.zeros(0)
    if not caliper:
        tx, cx = np.zeros((tz.shape[0], 0)), np.zeros((cz.shape[0], 0))
    brute = greedy_no_replace(tstarts, tz, tx, starts, cz, cx, calx, k, rounds)
    tree = match_without_replacement(tstarts, tz, tx, starts, cz, cx, calx, k, rounds, False)
    np.testing.assert_array_equal(tree, brute)


def _closest_reference(tstarts, tz, tx, starts, cz, cx, calx, k, rounds):
    """Repeatedly the globally closest (focal, free control) pair passing the
    calipers; ties to the earlier focal unit, then the earlier control."""
    out = np.full((tz.shape[0], k), -1, np.int64)
    for g in range(starts.size - 1):
        units = list(range(tstarts[g], tstarts[g + 1]))
        ctrls = np.arange(starts[g], starts[g + 1])
        free = np.ones(cz.shape[0], bool)
        got = {i: 0 for i in units}
        for rnd in range(k if rounds else 1):
            active = [i for i in units if rnd == 0 or out[i, rnd - 1] >= 0]
            while active:
                best = None
                for i in active:
                    ok = ctrls[free[ctrls]]
                    if calx.size:
                        ok = ok[np.all(np.abs(cx[ok] - tx[i]) <= calx, axis=1)]
                    if ok.size == 0:
                        continue
                    d = ((cz[ok] - tz[i]) ** 2).sum(1)
                    j = ok[np.lexsort((ok, d))[0]]
                    cand = (((cz[j] - tz[i]) ** 2).sum(), i, j)
                    if best is None or cand < best:
                        best = cand
                if best is None:
                    break
                _, i, j = best
                out[i, rnd if rounds else got[i]] = j
                got[i] += 1
                free[j] = False
                if rounds or got[i] >= k:
                    active.remove(i)
    return out


@pytest.mark.parametrize("k,rounds", [(1, True), (2, True), (2, False)])
@pytest.mark.parametrize("scarce", [False, True])
@pytest.mark.parametrize("caliper", [None, 0.5])
def test_closest_on_a_metric(k, rounds, scarce, caliper):
    rng = np.random.default_rng(7 * k + scarce + int(rounds))
    tstarts, tz, tx, starts, cz, cx = _case(rng, False, scarce)
    calx = np.array([caliper]) if caliper else np.zeros(0)
    if not caliper:
        tx, cx = np.zeros((tz.shape[0], 0)), np.zeros((cz.shape[0], 0))
    tree = match_without_replacement(tstarts, tz, tx, starts, cz, cx, calx, k, rounds, True)
    ref = _closest_reference(tstarts, tz, tx, starts, cz, cx, calx, k, rounds)
    np.testing.assert_array_equal(tree, ref)


@pytest.mark.parametrize("order", ["data", "random"])
def test_without_replacement_end_to_end(monkeypatch, order):
    rng = np.random.default_rng(9)
    n = 5000
    df = pl.DataFrame({"d": (rng.random(n) < 0.35).astype(np.int64),
                       "x1": np.round(rng.standard_normal(n), 1),
                       "x2": rng.standard_normal(n), "g": rng.integers(0, 6, n)})
    kw = dict(exact="g", distance="mahalanobis", n_matches=2, match_order=order, seed=3,
              caliper={"x1": 0.3})
    tree = match_nn("d ~ x1 + x2", df, **kw).matches().collect()
    monkeypatch.setattr(kd, "KDTREE_MAX_DIM", 0)
    brute = match_nn("d ~ x1 + x2", df, **kw).matches().collect()
    assert tree.equals(brute)
    # "closest" on a metric only exists on the tree path
    with pytest.raises(NotImplementedError, match="closest"):
        match_nn("d ~ x1 + x2", df, exact="g", distance="mahalanobis", match_order="closest")
    monkeypatch.undo()
    m = match_nn("d ~ x1 + x2", df, exact="g", distance="mahalanobis", match_order="closest")
    used = m.matches().collect()["control_row"]
    assert used.n_unique() == used.len()                  # without replacement


# ------------------------------------------- with replacement, with calipers
from match_stream._kdtree import knn_with_calipers  # noqa: E402


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("caliper", [2.0, 0.5, 0.05])     # loose to very tight
def test_with_replacement_and_calipers_same_as_brute_force(k, discrete, caliper):
    rng = np.random.default_rng(31 * k + 7 * discrete + int(caliper * 100))
    tstarts, tz, tx, starts, cz, cx = _case(rng, discrete, False)
    calx = np.array([caliper])
    brute = knn_replace(tstarts, tz, tx, starts, cz, cx, calx, k)
    tree = knn_with_calipers(tstarts, tz, tx, starts, cz, cx, calx, k)
    np.testing.assert_array_equal(tree, brute)


def test_with_replacement_and_calipers_end_to_end(monkeypatch):
    rng = np.random.default_rng(12)
    n = 5000
    df = pl.DataFrame({"d": (rng.random(n) < 0.3).astype(np.int64),
                       "x1": np.round(rng.standard_normal(n), 1),
                       "x2": rng.standard_normal(n), "g": rng.integers(0, 6, n)})
    kw = dict(exact="g", distance="mahalanobis", replace=True, n_matches=3, caliper={"x2": 0.2})
    tree = match_nn("d ~ x1 + x2", df, **kw).matches().collect()
    monkeypatch.setattr(kd, "KDTREE_MAX_DIM", 0)
    brute = match_nn("d ~ x1 + x2", df, **kw).matches().collect()
    assert tree.equals(brute)
