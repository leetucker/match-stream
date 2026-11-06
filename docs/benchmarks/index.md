# Benchmarks

match-stream is compared with MatchIt and Matching (R), pymatchit, fastmatch,
and with the bare operations underneath it (a nearest-neighbor search, a
group-by, an as-of join), on simulated data from 10,000 to 100 million units.
The suites are in `benchmarks/`; its README says how to run them.

The short version:

- **Against the matching packages**, match-stream is 6 to 35 times faster
  than MatchIt at 10 million units where MatchIt finishes, and it finishes
  where the others run out of memory or time. pymatchit and R's Matching
  are slower still.
- **Against the bare search**, matching costs little on top of it: with no
  strata (design G) match-stream takes 81.6 s at 100 million units, and a
  bare SciPy k-d tree search takes 71.8 s.
- **Against a simple group-by or join**, the specialized shortcuts win:
  Polars' `group_by` does coarsened exact matching about 5 times faster
  than match-stream, and its `join_asof` does design A about 9 times faster.
  They answer a narrower question, with no calipers, no choice of controls
  and no weights.
- **The in-memory path has a limit.** Design H (1:1 Mahalanobis matching
  without replacement, with covariate calipers) at 100 million units needs
  more than the 32 GB of the test machine. Use `memory="by_stratum"` there:
  187 s and 9.2 GB.

## What is measured

Two suites, on simulated units in 500 exact strata:

- **Scale** (`scale_benchmark.py`): eight designs, from 1:1 matching on the
  score with a caliper to coarsened exact matching, kernel matching and
  Mahalanobis matching, with and without replacement:

  | design | what it matches |
  |---|---|
  | A | 1:1 on the score, with replacement, caliper 0.2 SD |
  | B | 1:1 Mahalanobis, with replacement, exact strata |
  | C | coarsened exact matching, 5 bins |
  | D | kernel matching on the score |
  | E | 3:1 on the score, with replacement, three calipers |
  | F | 1:1 on the score, greedy, without replacement, three calipers |
  | G | 1:1 Mahalanobis, with replacement, no strata |
  | H | 1:1 Mahalanobis, without replacement, exact strata, calipers on two covariates |

  match-stream runs each in memory and by stratum. The others are MatchIt
  and Matching (R), pymatchit, fastmatch (faiss, scikit-learn and voyager
  backends), Polars' `join_asof` (design A) and `group_by` (design C), and,
  as a floor for the Mahalanobis designs B and G, a bare SciPy k-d tree and
  an exact faiss search.
- **Effects** (`effects_benchmark.py`): matching followed by the effect and
  its standard error. It covers influence-function, clustered and
  Abadie-Imbens standard errors, the bias adjustment, and the Abadie-Imbens
  (2016) correction for a fitted score, against R Matching's Abadie-Imbens
  standard errors and fastmatch's bias-corrected standard error.

For each configuration: wall time, peak memory, peak disk, and the
estimate's largest difference from match-stream's, which checks that every
library made the same design.

The runs were on one AWS EC2 spot instance (a c8id.4xlarge: an Intel Xeon
6975P-C with 16 vCPUs and 30.8 GB of memory, local NVMe disk, Linux; the
benchmark process capped at 96% of memory).
`benchmarks/results/machine.json` has the details and the versions. Each
point is a single run, and a spot instance can be interrupted and restarted;
an interrupted configuration is run again from the start. Read the
differences between libraries, which are several-fold, rather than the kinks
in a line. A configuration that fails, runs out of memory or
exceeds 30 minutes is not run at larger sizes.

## Results: matching

Wall time and peak memory at 10 million and 100 million units. "Memory" and
"time" mean the library ran out of memory or exceeded 30 minutes.

| design | match-stream | MatchIt | pymatchit | other |
|---|---|---|---|---|
| **10 million units** | | | | |
| A | 3.6 s, 1.4 GB | 20 s, 3.0 GB | 120 s, 4.7 GB | `join_asof` 0.5 s, 1.8 GB |
| B | 6.2 s, 2.7 GB | 171 s, 4.5 GB | 171 s, 4.4 GB | k-d tree 5.6 s, 2.5 GB |
| C | 0.7 s, 1.9 GB | 16 s, 3.2 GB | memory | `group_by` 0.5 s, 1.1 GB |
| D | 7.2 s, 1.7 GB | | | |
| E | 5.0 s, 2.0 GB | 61 s, 3.4 GB | 1,003 s, 5.8 GB | |
| F | 4.4 s, 1.9 GB | 35 s, 3.9 GB | 861 s, 5.6 GB | |
| G | 6.1 s, 2.2 GB | time | 614 s, 4.4 GB | k-d tree 6.7 s, 2.2 GB |
| H | 9.5 s, 3.8 GB | 334 s, 5.3 GB | time | |
| **100 million units** | | | | |
| A | 58 s, 8.2 GB | 284 s, 27.9 GB | memory | `join_asof` 6.2 s, 15.2 GB |
| B | 89 s, 21.3 GB | memory | memory | k-d tree 65 s, 20.0 GB |
| C | 7.3 s, 13.7 GB | 169 s, 29.6 GB | | `group_by` 1.5 s, 5.7 GB |
| D | 107 s, 10.5 GB | | | |
| E | 75 s, 15.5 GB | 1,176 s, 29.6 GB | memory | |
| F | 57 s, 12.6 GB | 742 s, 29.6 GB | memory | |
| G | 82 s, 19.7 GB | | | k-d tree 72 s, 19.4 GB |
| H | memory (by stratum: 187 s, 9.2 GB) | memory | | |

- **MatchIt** gives the same matches as match-stream (the estimates agree to
  about 1e-14), 5 to 35 times more slowly where it finishes, and it is close
  to the machine's memory by 100 million units: 28 to 30 GB in A, C, E and F.
  In F it completed once and was killed for memory on a second run (shown:
  the completed run). In G it takes 177 s at 1 million units and exceeds 30
  minutes at 10 million, because it has no strata to split the search.
- **pymatchit** runs out of memory or time at 10 to 100 million units in most
  designs, and was left out of G and H at 100 million. At 100,000 units it
  disagrees with the others in B, G and H by up to 0.003 in the estimate (see
  [Implementations](../implementations.md)).
- **R Matching** finishes only up to 100,000 units (44 to 97 s), and exceeds
  30 minutes at 1 million in A, B, E, F and H. Its `Matchby` function, which
  matches within strata, handles design A to 10 million units (1,042 s). Its
  calipers and covariance differ from MatchIt's in several designs, so its
  estimates differ by design.
- **fastmatch** (design G) agrees exactly with a pre-whitened Euclidean
  search. With scikit-learn it takes 16.6 s at 10 million units and 176 s at
  100 million, against match-stream's 6.1 s and 82 s. Its faiss backend takes
  219 s at 10 million and exceeds 30 minutes at 100 million; its voyager
  backend is approximate (estimates off by up to 0.01) and runs out of memory
  at 100 million.
- **The bare search.** In B and G, match-stream is within about 1.4 times of
  a SciPy k-d tree search on the same data, and is faster than it at 10
  million units in G (6.1 s against 6.7 s). Exact faiss (CPU) is much slower
  here: 13 s at 10 million units in B and 632 s at 100 million, against 6.2 s
  and 89 s.
- **Memory.** Where it matters, `memory="by_stratum"` cuts the peak: B at 100
  million units takes 9.1 GB instead of 21.3 GB, for 15% more time. In D, E
  and F the peak is about 9 to 16 GB either way.

## Results: effects

Matching followed by the effect and its standard error, for design A
(1:1 on the score, exact strata), with design G for the bias adjustment:

| 10 million units | time | memory |
|---|---|---|
| influence-function or clustered standard error | 4.4 s | 1.3 GB |
| Abadie-Imbens standard error | 4.6 s | 1.8 GB |
| with `n_variance_matches=1`, with or without the bias adjustment | 9 s | 2.1 to 2.3 GB |
| with a fitted score and its correction | 27 s | 4.3 GB |
| G with the bias adjustment | 82 s | 3.0 GB |
| fastmatch (faiss), G with its bias-corrected standard error | 197 s | 2.7 GB |

At 100 million units the first three rows take 51 s, 54 s and 113 to 120 s,
and G with the bias adjustment takes 865 s and 27 GB. The fitted score's
correction needs more memory than the 32 GB machine has at that size, and
fastmatch's takes more than 30 minutes.

## Figures

The designs with exact strata (A, B, E, F and H):

```{image} ../figures/scale_nn_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of units for designs A, B, E, F and H, one line per library, log-log
```
```{image} ../figures/scale_nn_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of units for designs A, B, E, F and H, one line per library, log-log
```

```{image} ../figures/scale_nn_memory.light.svg
:class: only-light benchmark
:alt: Peak memory against the number of units for designs A, B, E, F and H, one line per library, log-log
```
```{image} ../figures/scale_nn_memory.dark.svg
:class: only-dark benchmark
:alt: Peak memory against the number of units for designs A, B, E, F and H, one line per library, log-log
```

Coarsened exact matching, kernel matching and the Mahalanobis match without
strata (C, D and G):

```{image} ../figures/scale_other_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of units for designs C, D and G, one line per library, log-log
```
```{image} ../figures/scale_other_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of units for designs C, D and G, one line per library, log-log
```

```{image} ../figures/scale_other_memory.light.svg
:class: only-light benchmark
:alt: Peak memory against the number of units for designs C, D and G, one line per library, log-log
```
```{image} ../figures/scale_other_memory.dark.svg
:class: only-dark benchmark
:alt: Peak memory against the number of units for designs C, D and G, one line per library, log-log
```

Matching and the effect with its standard error:

```{image} ../figures/effects_time.light.svg
:class: only-light benchmark
:alt: Wall time against the number of units for matching with each standard error, one line per configuration, log-log
```
```{image} ../figures/effects_time.dark.svg
:class: only-dark benchmark
:alt: Wall time against the number of units for matching with each standard error, one line per configuration, log-log
```

```{image} ../figures/effects_memory.light.svg
:class: only-light benchmark
:alt: Peak memory against the number of units for matching with each standard error, one line per configuration, log-log
```
```{image} ../figures/effects_memory.dark.svg
:class: only-dark benchmark
:alt: Peak memory against the number of units for matching with each standard error, one line per configuration, log-log
```

Points for runs that failed, ran out of memory or timed out are left off the
lines. Peak disk, which only the by-stratum runs use, is in
`benchmarks/results/*.csv`.

## Agreement

Every library is asked for the same design, and the estimate's largest
difference from match-stream's is recorded (`max_abs_diff` in the CSVs):

- MatchIt, and the SciPy k-d tree, agree with match-stream's estimate to
  about 1e-14 in every design they support, at every size. The by-stratum
  path agrees to about 1e-15.
- faiss, which searches in single precision, agrees to about 5e-5 at 10
  million units.
- fastmatch on pre-whitened covariates agrees exactly. Its default
  Mahalanobis metric does not compute a Mahalanobis distance (see
  [Implementations](../implementations.md)).
- Where a package differs by design (R Matching's calipers and covariance,
  `Matchby`, pymatchit's Mahalanobis, voyager's approximate index), the
  difference is reported, not hidden.
