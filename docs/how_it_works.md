# How it works

Matching looks quadratic: every treated unit against every control. With
exact strata and a scalar score it is not. Within a stratum, sorted by the
score, a treated unit's candidates are a contiguous window of controls,
found by binary search. match-stream is built on that, and on keeping only
the few columns matching needs in memory.

## One pass over the source, then narrow columns

1. **The score.** If it is fitted (`distance="ps"`), hdfe-stream's
   `feglm_stream` fits the logit or probit out of core, absorbing any fixed
   effects after `|`. Its prediction is added lazily, so the source is not
   loaded.
2. **One streaming query** computes what must be global: the counts, the
   score's standard deviation over all units (the unit of a caliper), the
   covariates' standard deviations for standardized calipers, and the
   number of strata.
3. **The narrow projection** keeps only what matching reads: the row's
   position, the treatment, the score, the stratum columns, caliper
   covariates and the id. That is tens of bytes a row.
4. **Matching** runs on those columns. Outcomes and covariates are read
   from the source again only for `effect()` and `balance()`, which are
   streaming sums.

## Nearest neighbors on a score

Within each stratum the controls are sorted by score, and the focal units
are put in the match order.

- **With replacement.** A binary search finds where the focal unit's score
  falls among the controls. The k nearest are then taken by walking
  outward, left or right, whichever is closer.
  - Calipers on covariates are checked as the walk goes. A control that
    fails one is skipped and the walk goes on.
  - With `keep_ties=True`, the walk continues through controls at the k-th
    distance.
  - Focal units are independent, so they run in parallel.
- **Without replacement, in a unit order** (largest, smallest, random,
  data). The same walk, but taken controls must be skipped quickly.
  - Two arrays of "next free control to the right / left" pointers, with
    path compression (a union-find structure), make each skip cost about a
    binary search however many controls are taken.
  - Strata are independent, so they run in parallel.
- **Without replacement, `closest`.** Pairs in ascending distance.
  - A heap holds each focal unit's current nearest free control.
  - The closest pair is popped. If its control was taken meanwhile, the
    unit's next free control is found and pushed back.
  - This is greedy matching on the full pair list without ever forming the
    list.
- **k > 1 without replacement.** The same machinery, in rounds (every unit takes
  its first match, then its second, ...) or in one pass (`match_in_rounds=False`).

## Mahalanobis and other metrics

The covariates are whitened once (multiplied by the inverse Cholesky
factor of the covariance pooled within the treatment groups), so the Mahalanobis distance becomes a
Euclidean one. Then:

- **With replacement and no covariate calipers:** a k-d tree (scipy's
  `cKDTree`) over the controls of every stratum at once, queried on all
  cores.
  - The stratum is an extra coordinate: 0 within a stratum, so distances
    there are exact, and farther than any covariate distance between strata.
  - The result is the brute force's, tie for tie (equal distances go to the
    control earlier in the data). It costs O(N log N): on 400,000 units in
    five strata, 0.5 s against 11 s on one core.
  - Used up to 15 whitened covariates; k-d trees lose their edge beyond.
- **Without replacement** (up to 15 covariates, with or without covariate
  calipers): candidate lists from the same tree.
  - Each focal unit gets its 16 nearest controls (at least 4k), re-sorted
    by the brute force's own distance arithmetic, then position.
  - The greedy pass (strata in parallel, in numba) takes each unit's first
    candidate that is still free and passes the calipers.
  - If a list runs out, or the control found sits at the list's last
    distance, that unit falls back to an exact scan of its stratum. The
    matches are always the brute force's; on 400,000 units in five
    strata, 1 s against 12-30 s.
  - `match_order="closest"` (pairs in ascending distance) uses the same lists
    with a heap, and is only available on this path.
- **With replacement and covariate calipers:** the same candidate lists;
  each unit takes the first k candidates that pass, with the same exact
  fallback. Nothing is taken, so units run in parallel. The tighter the
  caliper, the more candidates fail and the more units fall back: at
  400,000 units, 1 s against 25 s for a loose caliper, and the brute
  force's time for one that rejects almost everything (never slower).
- **More than 15 covariates:** blocked brute force within each stratum,
  in numba, strata (and focal units with replacement) in parallel. This is
  quadratic within a stratum: fine for strata of thousands, slow for
  strata of millions.

## Kernel and ridge matching without pairs

A kernel weight K((s_j − s_i)/h) is a polynomial in the gap on each side
of the focal unit for every kernel offered. So the sums kernel and ridge
matching need are combinations of the window's moments Σ v·u^a, with
u = (s_j − s_i)/h, for a few values v and powers a:

- **The focal unit's side:** its kernel average, or Frölich's ridge fit.
- **Each control's side:** its total weight over the focal units it serves.
- **Cross-validation:** each control's leave-one-out prediction.

The moments come from blocks of 64 consecutive controls (never crossing a
stratum):

- **Inner blocks.** Each block's moments about its own midpoint are
  precomputed. A block wholly inside a window is shifted to the query
  point with the binomial theorem; both offsets are within one bandwidth,
  so no digits are lost.
- **The window's ends** are summed directly.

Global running sums would be simpler, but differences of prefix sums lose
the digits of a narrow window in a long array: the loss grows with the
array's length and with the score's spread relative to the bandwidth, and
on large data it can reach most of them. The block sums are tested to agree
with a direct computation to about 1e-13 relative, at any size,
and the cost per unit is O(64 + window/64). No pair is ever formed, so a
wide bandwidth (thousands of controls per window) costs little more than a
narrow one.

## Coarsened exact matching

A streaming group-by: each covariate is binned, cells with both arms are
kept, and the weights follow from the cell counts.

## Effects and standard errors

- **Abadie-Imbens.** The variance needs each unit's K_i (the times it is
  used, weighted by 1/k) and the focal units' effects, which are sums over
  the pair table.
  - Conditional variances with `n_variance_matches > 0` come from each unit's J
    nearest same-arm neighbors, found the same way as matches.
  - The bias adjustment is a small weighted least-squares fit over the
    matched units.
- **Influence-function SEs** are streamed sums. Each matched row
  contributes w_i(y_i − mean of its arm)/W of its arm. Clustered variances
  need only the per-cluster sums of those contributions. Multi-way
  clustering adds and subtracts the one-way and intersection terms
  (Cameron, Gelbach and Miller).

## Memory paths

- **`in_memory`** collects the narrow columns of all strata at once. It
  is the fastest.
- **`by_stratum`** reads whole strata back from a Parquet copy, a batch at
  a time, through hdfe-stream's `iter_groups`. Memory is bounded by the
  largest stratum plus the batch.
- **The two give identical results.** Strata are independent, also for
  greedy matching without replacement, and every computation within a
  stratum is the same whatever else is in the batch. The kernel moments'
  blocks never cross a stratum for this reason.
- **`auto`** takes `in_memory` when the narrow columns fit the memory
  headroom that stream-limits reports.

See [Memory, disk and threads](resources.md).
