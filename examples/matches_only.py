"""Matches only, no outcome: greedy 1:1 matching without replacement within
exact strata, with calipers on the score and on two covariates, written to
Parquet. The score model absorbs a fixed effect (after "|")."""

import tempfile
from pathlib import Path

import polars as pl

from match_stream import match_nn, simulate_matching

pl.Config.set_tbl_hide_dataframe_shape(True)
df = simulate_matching(100_000)

m = match_nn(
    "d ~ x1 + x2 + x3", df,
    exact=["r"],                                  # match within r
    score_model="d ~ x1 + x2 + x3 | i",                # logit with i absorbed
    caliper={"ps": 0.2, "x1": 0.25, "x2": 0.25},  # ps in SDs of the score; x1, x2 raw
    match_order="largest",                            # the default for a score
    id="id", verbose=False)
m.summary()
print(m.counts())

out = Path(tempfile.mkdtemp()) / "pairs.parquet"
m.matches().sink_parquet(out)                     # lazy: streams to disk
print(f"\n{pl.scan_parquet(out).select(pl.len()).collect().item():,} pairs written")
print(pl.read_parquet(out).head(3))
