"""Staging for code bound for ``hdfe_stream.core``.

Each module here has the API it will have in core and no matching-specific
names, so that other packages (a streaming staggered-DiD package, for one)
can use it once it moves. Until hdfe-stream releases it, match-stream imports
it from here.

Modules:

- ``influence``: influence-function variances from streamed per-cluster
  sums (one-way and multi-way); `weighted_mean_difference`.
- ``mestimation``: a fitted binary logit/probit first stage (scores,
  observed information) and the first-stage correction phi + J H^-1 s.
- ``ipw``: inverse-probability weights (trimming, Hajek scale) and the
  Hajek IPW difference in means with the estimated-score correction.
- ``balance``: the weighted two-group balance table.

All four are now in hdfe-stream's ``hdfe_stream.core`` (unreleased; same
code, docstrings aside). When that hdfe-stream is released, switch the
imports, pin it, and delete them here; until then edit both copies
together.

Not here yet: group tallies for ``core.profile.scan_source`` (they extend a
function that lives in hdfe-stream and serve only ``profile=True``).
"""
