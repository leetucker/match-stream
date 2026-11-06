"""Coarsened exact matching: five equal-width bins per covariate, exact on
r and i; the cells, balance and the ATT."""

import polars as pl

from match_stream import match_cem, simulate_matching

pl.Config.set_tbl_hide_dataframe_shape(True)

df = simulate_matching(100_000)
m = match_cem("d ~ x1 + x2 + x3", df, cutpoints=5, exact=["r", "i"],
              id="id")
m.summary()
print(m.weights().head(4).collect())
print()
m.balance().summary()
print()
m.effect("y").summary()
