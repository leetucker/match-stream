"""Out of core: the data as a set of Parquet files, matched by stratum
(memory bounded by the largest stratum), the score fitted by hdfe-stream
without loading the data, the pairs streamed back to Parquet."""

import tempfile
from pathlib import Path

import polars as pl

from match_stream import match_nn, simulate_matching

pl.Config.set_tbl_hide_dataframe_shape(True)

tmp = Path(tempfile.mkdtemp())
data, out = tmp / "extract", tmp / "output"
data.mkdir()
out.mkdir()
for part in range(4):                                     # four files, as an extract would be
    simulate_matching(250_000, seed=part).with_columns(
        (pl.col("id") + part * 250_000).alias("id")).write_parquet(data / f"part{part}.parquet")

# effect() and balance() read the source again (lazily): keep outputs out of
# the glob, and do not change the files between matching and those calls
m = match_nn("d ~ x1 + x2 + x3", str(data / "*.parquet"), exact=["r", "i"],
             replace=False, caliper=0.2, id="id", memory="by_stratum", workdir=tmp,
             verbose=False)
print("memory path:", m.memory)
m.summary()
m.matches().sink_parquet(out / "pairs.parquet")
print(f"{pl.scan_parquet(out / 'pairs.parquet').select(pl.len()).collect().item():,} pairs")
m.effect("y").summary()
