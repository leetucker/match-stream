# Memory, disk and threads

## Memory

`match_nn(..., memory=)` (and `match_kernel`) chooses how the narrow columns (the row's
position, treatment, score, strata, caliper covariates and id: tens of
bytes a row) are held while matching:

- **`"in_memory"`**: all strata at once. The fastest.
- **`"by_stratum"`**: whole strata at a time, read back from a Parquet copy
  of the narrow columns. Memory is bounded by the largest stratum plus a
  batch of about two million rows. The copy costs one pass to disk, under
  `workdir`.
- **`"auto"`** (the default): `"in_memory"` when the estimated memory fits
  the headroom that stream-limits' `MemoryBudget` reports, else
  `"by_stratum"`. Pass `max_memory_gb=` and `memory_reserve=` to plan with
  other limits. Without exact strata there is one stratum, so the path is
  in-memory.

Both paths give the same matches. A single stratum too large for memory
cannot be matched: use finer strata.

The score model is fitted by hdfe-stream's `feglm_stream`, which has its
own memory handling (see hdfe-stream's documentation); its scratch files
go under `workdir` too.

The results hold only the pair table (or the matched units' weights), a
few tens of bytes per pair. `matches()` and `weights()` are LazyFrames, so
they can be sunk to Parquet. `effect()` and `balance()` read the source
again, as streaming queries.

## Disk

`workdir=` (default: `$HDFE_STREAM_WORKDIR`, else the system temporary
directory) holds the score model's run directory and the by-stratum copy.
Both are removed when no longer needed.

## Threads

`n_threads=` in `match_nn`, `match_kernel` and `match_cem` holds the matching kernels (numba), BLAS
and the k-d trees to that many threads, through stream-limits'
`thread_scope`, for the match and its `effect()`. The default leaves them
as they are: every core numba and scipy see. Polars' pool is fixed when
polars is imported, so limit it in the job script
(`STREAM_LIMITS_MAX_THREADS`) or call `stream_limits.limit_threads()`
before importing polars, as for the sibling packages.

## Profiling a match: `profile=True`

`profile=True` in the three matching functions returns hdfe-stream's `Profile` without
matching:

```python
p = match_nn("treat ~ age + tenure", "extract/*.parquet", exact=["state"],
             caliper=0.2, profile=True)
print(p)              # memory and disk, expected and upper bound, by component
p.recommend().pbs     # 'select=1:ncpus=8:mem=14gb'
p.notes               # strata, the largest stratum, the path memory="auto" takes, ...
```

What it reads: one scan of the columns the match uses, a group-by of
units by arm per stratum, and, when the score is a column with a caliper
(or a fixed kernel bandwidth), a 1,000-bin histogram of the score per
stratum. A fitted score adds the score model's own profile
(`feglm_stream(..., profile=True)`) as the phase "score".

What it reports:

- **Memory** for the match phase: the narrow columns (as a frame and as
  arrays), the sort keys, the pair table, and for a multivariate distance
  the k-d tree and its candidate lists; for `by_stratum`, the largest batch
  of strata. Checked against measured peaks at 2 million rows: the
  expected figure within about 10%, the upper bound above the measurement.
- **Disk** for `by_stratum`: the narrow copy and its partitions.
- **Notes:** rows, strata, units by arm, the largest stratum, the path
  `memory="auto"` takes (the same rule the match uses), and the work of the
  search instead of a time: `candidate_pairs` (the treated-control pairs
  within the caliper, expected and an upper bound; within 2% of the actual
  count in tests) and `brute_force_pairs` (all pairs within strata).
- **Warnings:** a largest stratum that alone does not fit in memory (use
  finer strata), more than 15 covariates (the brute-force search).

Not counted: `effect()` and `balance()`, which stream the source again and
hold the units used and their outcome (about 16 bytes a unit) on top; the
rows a score model drops (fixed-effect singletons) are known only after
fitting, so the figures count every row with its values.
