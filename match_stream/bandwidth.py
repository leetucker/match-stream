"""The two data-driven bandwidth rules of kernel and ridge matching, as objects.

``match_kernel(..., bandwidth=)`` takes a number (score units), the name of a
rule with its defaults (``"pair_distance"``, ``"cross_validation"``), or one of
these objects when a rule's settings are changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

PENALTIES = (None, "support_loss")


@dataclass(frozen=True)
class PairDistance:
    """The bandwidth from the distances of 1:1 nearest-neighbor matching.

    ``factor`` times the ``quantile`` quantile of the nonzero distances
    between each focal unit and its nearest other-arm unit (with replacement,
    pooled over strata). The default, 1.5 times the 90th percentile, is narrow:
    focal units with no other-arm unit within it are unmatched.
    """

    factor: float = 1.5
    quantile: float = 0.9

    def __post_init__(self):
        if not float(self.factor) > 0:
            raise ValueError("PairDistance factor must be positive")
        if not 0 < float(self.quantile) <= 1:
            raise ValueError("PairDistance quantile must be in (0, 1]")


@dataclass(frozen=True)
class CrossValidation:
    """The bandwidth chosen by leave-one-out cross-validation.

    Parameters
    ----------
    outcome : str, optional
        Predict this outcome column of each other-arm unit from the rest and
        choose the bandwidth with the smallest error. Without it the criterion
        is the balance of the score: how little leaving out any one other-arm
        unit moves the matched mean of the score.
    grid : sequence of float, optional
        The bandwidths tried. Default: the pair-distance bandwidth times
        1.5^k for k = -4..4, extended and refined around the minimum.
    weighted : bool, default False
        Weight the outcome's errors by the units' matching weights. Needs
        `outcome`.
    penalty : None or "support_loss"
        None judges every bandwidth on the same units. "support_loss" judges
        each bandwidth on the units it can evaluate, times a penalty for the
        units it loses to common support (and for bandwidths above the score's
        standard deviation), searching outward from the pair-distance
        bandwidth (or over `grid`).
    """

    outcome: str | None = None
    grid: Sequence[float] | None = None
    weighted: bool = False
    penalty: str | None = None

    def __post_init__(self):
        if self.penalty not in PENALTIES:
            raise ValueError(f"CrossValidation penalty must be None or 'support_loss', "
                             f"not {self.penalty!r}")
        if self.weighted and self.outcome is None:
            raise ValueError("CrossValidation weighted=True weights the outcome's errors: "
                             "give outcome=")
        if self.grid is not None:
            object.__setattr__(self, "grid", tuple(float(x) for x in self.grid))
