"""match-stream: matching on data larger than memory.

Nearest-neighbor matching with or without replacement (on a score or a
multivariate distance), calipers on the score and on covariates, exact
strata, kernel and ridge matching, coarsened exact matching, balance
diagnostics, and treatment effects with Abadie-Imbens or influence-function
standard errors. Conventions follow R's MatchIt. See
https://leetucker.github.io/match-stream/.

Layout
    api.py          match_nn(), match_kernel(), match_cem()
    bandwidth.py    PairDistance, CrossValidation (kernel bandwidth rules)
    _spec.py        argument parsing and checks -> Spec
    _prep.py        pass 0: the score model (feglm_stream), the narrow columns
    _match.py       nearest neighbor: sort, order, run the kernel, both memory paths
    _kernel.py      kernel and ridge matching: bandwidths, weights
    _cem.py         coarsened exact matching (streaming group-by)
    kernels/        numba: nn.py (score), metric.py (Mahalanobis etc.),
                    kernel.py (window moments), var.py (conditional variances)
    matches.py      MatchSet, CEMSet, KernelSet: pairs, weights, counts
    balance.py      BalanceTable: MatchIt's summary(m, un = TRUE)
    effects.py      effect(): estimates, AI and influence-function variances; MatchResult
    simulate.py     simulate_matching()
    _core/          code bound for hdfe_stream.core (influence functions, IPW,
                    the first-stage correction, the weighted balance table)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .api import match_cem, match_kernel, match_nn
from .balance import BalanceTable
from .bandwidth import CrossValidation, PairDistance
from .effects import MatchResult
from .matches import CEMSet, KernelSet, MatchSet
from .simulate import simulate_matching

try:
    __version__ = _version("match-stream")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0.dev0"

__all__ = ["BalanceTable", "CEMSet", "CrossValidation", "KernelSet", "MatchResult", "MatchSet",
           "PairDistance", "__version__", "match_cem", "match_kernel", "match_nn",
           "simulate_matching"]
