"""k nearest neighbors: k = 3 with replacement, ties kept, a caliper on a
covariate; min_matches drops units left with fewer than k; without
replacement, matching in rounds against one pass."""

import polars as pl

from match_stream import match_nn, simulate_matching

pl.Config.set_tbl_hide_dataframe_shape(True)

df = simulate_matching(100_000)
kw = dict(exact=["r", "i"], distance="ps", id="id", verbose=False)

# WITH replacement (replace=True): a control can be the match of many treated units
m = match_nn("d ~ x1 + x2 + x3", df, n_matches=3, replace=True, keep_ties=True,
             caliper={"ps": 0.1, "x1": 0.1}, **kw)
per = m.matches().group_by("treated_id").len().collect()["len"]
print("matches per treated unit:", per.value_counts().sort("len").rows())

m3 = match_nn("d ~ x1 + x2 + x3", df, n_matches=3, replace=True, keep_ties=True,
              caliper={"ps": 0.1, "x1": 0.1}, min_matches=3, **kw)
print("matched treated, keeping short units:", m.counts()["matched_treated"],
      " with min_matches=3:", m3.counts()["matched_treated"])

# WITHOUT replacement (replace=False, the default; written out here), k = 2:
# each control is used once. In rounds (every treated unit takes its first
# match, then its second) or in one pass (pairs taken in order of distance)
for rounds in (True, False):
    w = match_nn("d ~ x1 + x2 + x3", df, n_matches=2, replace=False,
                 match_order="closest", match_in_rounds=rounds, **kw)
    d = w.matches().select(pl.col("distance").mean()).collect().item()
    two = (w.matches().group_by("treated_id").len().collect()["len"] == 2).sum()
    print(f"match_in_rounds={rounds}: {two:,} treated with 2 controls, mean distance {d:.5f}")
