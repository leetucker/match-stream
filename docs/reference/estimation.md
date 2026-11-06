# Matching

Three functions, one per kind of matching. They share the data and
formula arguments, exact strata, the score, memory and thread options,
and `profile=True`.

```{eval-rst}
.. currentmodule:: match_stream

.. autofunction:: match_nn

.. autofunction:: match_kernel

.. autofunction:: match_cem
```

## Kernel bandwidth rules

`match_kernel(bandwidth=)` takes a number, the name of a rule, or one of
these objects to change a rule's settings.

```{eval-rst}
.. autoclass:: PairDistance

.. autoclass:: CrossValidation
```
