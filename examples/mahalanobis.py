"""Mahalanobis matching on the covariates (the covariance pooled
within treatment groups), within exact strata, with a caliper on one
covariate; balance and the ATT."""

import polars as pl

from match_stream import match_nn, simulate_matching

pl.Config.set_tbl_hide_dataframe_shape(True)

df = simulate_matching(100_000)
m = match_nn("d ~ x1 + x2 + x3", df, distance="mahalanobis", exact=["r", "i"],
             replace=True, caliper={"x1": 0.2}, id="id")
m.summary()
print()
b = m.balance()
print(b.matched.select("term", "Std. Mean Diff.", "Var. Ratio", "eCDF Max"))
print()
m.effect("y").summary()
