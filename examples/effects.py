"""Treatment effects and standard errors: Abadie-Imbens (homoskedastic and
heteroskedastic, with and without the Abadie-Imbens 2016 correction for the
estimated score), the ATE, the bias adjustment, and influence-function
standard errors clustered on the pair or on the unit."""

import warnings

import polars as pl

from match_stream import match_nn, simulate_matching

pl.Config.set_tbl_hide_dataframe_shape(True)

df = simulate_matching(100_000)
kw = dict(exact=["r", "i"], distance="ps", id="id", verbose=False)

rep = match_nn("d ~ x1 + x2 + x3", df, replace=True, n_matches=4, keep_ties=True, **kw)
# n_variance_matches=0 cannot carry the 2016 correction (teffects' vce(iid) does not
# either): match-stream says so with a warning, silenced here
warnings.filterwarnings("ignore", message="the standard error is not corrected")
rows = []
for label, fit in [
    ("AI, homoskedastic", rep.effect("y")),
    ("AI, n_variance_matches=4, score taken as known",
     rep.effect("y", n_variance_matches=4, score_correction=False)),
    ("AI, n_variance_matches=4, score corrected", rep.effect("y", n_variance_matches=4)),
    ("AI, bias-adjusted", rep.effect("y", bias_adjustment=True)),
    ("AI, ATE", rep.effect("y", estimand="ate")),
    ("IF, by unit", rep.effect("y", vcov="if")),
    ("IF, subclass + id", rep.effect("y", vcov={"CRV1": "subclass+id"})),
]:
    rows.append((label, fit.beta[0], fit.vcov[0, 0] ** 0.5))

wor = match_nn("d ~ x1 + x2 + x3", df, **kw)       # without replacement
fit = wor.effect("y")                                    # default: clustered on the pair
rows.append(("without replacement, CRV1 pair", fit.beta[0], fit.vcov[0, 0] ** 0.5))

print(f"{'':40s}{'estimate':>10s}{'s.e.':>10s}")
for label, b, se in rows:
    print(f"{label:40s}{b:10.4f}{se:10.4f}")
