# k nearest neighbors

Three matches per treated unit with replacement, ties kept, a caliper on
the score and one on `x1`. Units the calipers leave short of three matches
are kept by default, or dropped with `min_matches=3`. Without
replacement and k = 2, matching in rounds (the default) against one pass
(`match_in_rounds=False`).

```{literalinclude} ../../examples/knn_calipers.py
:language: python
:start-at: import
```

Output:

```text
matches per treated unit: [(1, 1009), (2, 946), (3, 22334)]
matched treated, keeping short units: 24289  with min_matches=3: 22334
match_in_rounds=True: 25,414 treated with 2 controls, mean distance 0.02845
match_in_rounds=False: 25,447 treated with 2 controls, mean distance 0.02837
```

**Notes.**

- **Ties.** `keep_ties=True` keeps every control at exactly the k-th distance
  (otherwise exactly k are kept). Each treated
  unit's matches share its weight equally.
- **Rounds or one pass.** In rounds, every treated unit takes its first
  match before any takes its second. In one pass, pairs are taken in order
  of distance and a unit keeps taking controls until it has k. Here the
  two differ little; they differ more when treated units compete for few
  controls (see [Conventions](../implementations.md)).
