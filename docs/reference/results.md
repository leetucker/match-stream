# Results

A match returns a `MatchSet` (nearest neighbor), a `KernelSet` (kernel or
ridge) or a `CEMSet` (coarsened exact matching). None of them needs an
outcome. `effect()` turns any of them into a `MatchResult`, and `balance()`
into a `BalanceTable`.

```{eval-rst}
.. currentmodule:: match_stream

.. autoclass:: MatchSet
   :members: matches, weights, effect, balance, counts, summary

.. autoclass:: KernelSet
   :members: weights, effect, balance, counts, summary

.. autoclass:: CEMSet
   :members: weights, effect, balance, counts, summary

.. autoclass:: MatchResult
   :no-members:

.. autoclass:: BalanceTable
   :members: summary
```

`MatchResult` is a subclass of hdfe-stream's `RegressionResult`, so
`summary()`, `tidy()`, `confint()`, `wald_test()` and `etable` work as they
do for hdfe-stream's regressions.
