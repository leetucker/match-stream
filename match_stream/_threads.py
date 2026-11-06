"""Thread limits for matching (as in hdfe-stream).

`scope(n)` holds numba's and BLAS's threads to `n` (stream-limits'
`thread_scope`) and makes the k-d tree queries use `n` workers, for the
length of a match or an effect; None leaves everything as it is (numba's
default, all the cores it sees, and every core for scipy's trees). Polars'
pool is fixed when it is imported: limit it in the job script
(``STREAM_LIMITS_MAX_THREADS``) or with `stream_limits.limit_threads`
before importing polars.
"""

from __future__ import annotations

import contextlib

# scipy cKDTree's `workers`: -1 is every core
WORKERS = -1


@contextlib.contextmanager
def scope(n: int | None):
    global WORKERS
    if not n:
        yield
        return
    from stream_limits import thread_scope

    previous = WORKERS
    WORKERS = int(n)
    try:
        with thread_scope(int(n)):
            yield
    finally:
        WORKERS = previous
