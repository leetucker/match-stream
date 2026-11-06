"""Kernel and ridge matching on the score: the default bandwidth (from the
1:1 nearest-neighbor distances), a cross-validated one, and the ATT with
influence-function standard errors."""

import polars as pl

from match_stream import CrossValidation, match_kernel, simulate_matching

pl.Config.set_tbl_hide_dataframe_shape(True)

df = simulate_matching(100_000)
kw = dict(exact=["r", "i"], verbose=False)

m = match_kernel("d ~ x1 + x2 + x3", df, **kw)
m.summary()
m.effect("y").summary()

r = match_kernel("d ~ x1 + x2 + x3", df, kernel="epan", ridge=True, **kw)
print(f"\nridge, bandwidth {r.bandwidth:.5f}:")
r.effect("y").summary()

grid = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
cv = match_kernel("d ~ x1 + x2 + x3", df, ridge=True,
                  bandwidth=CrossValidation(outcome="y", grid=grid), **kw)
print("\ncross-validation:")
print(cv.bandwidth_info.cv)
print(f"chosen: {cv.bandwidth}")
cv.effect("y").summary()
