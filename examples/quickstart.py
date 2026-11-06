"""Quickstart: 1:1 nearest-neighbor matching on a fitted propensity score,
with replacement, within exact strata, with a caliper; then balance and the
ATT with Abadie-Imbens standard errors. The true effect is 1."""

import polars as pl

from match_stream import match_nn, simulate_matching

pl.Config.set_tbl_hide_dataframe_shape(True)
df = simulate_matching(100_000)

# distance="ps" (the default) fits a logit of d on the formula's covariates
m = match_nn("d ~ x1 + x2 + x3", df, exact=["r", "i"], replace=True, caliper=0.2,
             id="id", verbose=False)
m.summary()

print("\nThe first pairs:")
print(m.matches().head(4).collect())
print("\nThe weights:")
print(m.weights().head(4).collect())

print()
m.balance().summary()

print()
m.effect("y").summary()
