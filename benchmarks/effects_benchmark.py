"""Time and peak memory of the effect and its standard error as the number of
units grows: match-stream's influence-function and Abadie-Imbens variances
(with and without the bias adjustment, and the Abadie-Imbens 2016 correction
for a fitted score), R Matching's Abadie-Imbens SEs on the same design, and
fastmatch's bias-corrected estimate with its Otsu-Rai SE.

    python benchmarks/effects_benchmark.py                  # the default sizes
    python benchmarks/effects_benchmark.py 10000 100000     # units
    python benchmarks/effects_benchmark.py --only "match-stream ai"

The design is the one Matching reproduces exactly: 1:1 nearest neighbor on
the score column with replacement, exact on (r, i), no caliper (a caliper
would differ by design: Matching counts the focal units it leaves unmatched
in N). Each configuration matches and then estimates in the same process;
t_match and t_effect split its time. max_abs_diff compares estimate and SE
with the match-stream configuration Matching reproduces (e.g. "Matching AI
(R)" with "match-stream ai"); the fitted-score configurations are compared
with nothing (teffects psmatch is the reference for the 2016 correction, in
tests/test_teffects.py).

fastmatch has no exact strata, so it is timed on scenario G's design (1:1
Mahalanobis on x1-x3 with replacement, no strata; see _scenarios.py), beside
match-stream's AI variance with the bias adjustment on the same matches. The
two are different estimators and are not compared: fastmatch fits its
linear bias correction on every control and uses Otsu and Rai's (2017)
variance; match-stream (as R Matching) fits it on the matched controls,
weighted by their use, with the Abadie-Imbens variance. Results go to
benchmarks/results/effects.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _harness import (
    DATA_DIR,
    RESULTS_DIR,
    external_command,
    external_missing,
    run_child,
    run_configs,
    write_machine_info,
)
from _scenarios import data_path, simulate_chunk

SCRIPT = Path(__file__).resolve()
DEFAULT_SIZES = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
TIMEOUT_S = 1800

DESIGN = dict(score="score", replace=True)
FITTED = dict(distance="ps", replace=True)          # a logit score, fitted by match-stream
GLOBAL = dict(distance="mahalanobis", replace=True)  # scenario G: no strata
DESIGNS = {"ms:column": (DESIGN, ["r", "i"]), "ms:fitted": (FITTED, ["r", "i"]),
           "ms:global": (GLOBAL, None)}
# label: (kind, the effect's options); kind "ms:<design>" or "R:<variant>"
CONFIGS = {
    "match-stream if": ("ms:column", dict(vcov="if")),
    "match-stream cluster r": ("ms:column", dict(vcov="CRV1:r")),
    "match-stream ai": ("ms:column", dict(vcov="ai")),
    "match-stream ai n_variance_matches=1": ("ms:column", dict(vcov="ai", n_variance_matches=1)),
    "match-stream ai n_variance_matches=1 bias_adjustment": (
        "ms:column", dict(vcov="ai", n_variance_matches=1, bias_adjustment=True)),
    "match-stream fitted score ai n_variance_matches=1": (
        "ms:fitted", dict(vcov="ai", n_variance_matches=1)),
    "match-stream fitted score ai n_variance_matches=1 score_correction": (
        "ms:fitted", dict(vcov="ai", n_variance_matches=1, score_correction=True)),
    "Matching AI (R)": ("R:0", None),
    "Matching AI n_variance_matches=1 (R)": ("R:1", None),
    "Matching AI n_variance_matches=1 BiasAdjust (R)": ("R:1:bias", None),
    "match-stream G ai n_variance_matches=1 bias_adjustment": (
        "ms:global", dict(vcov="ai", n_variance_matches=1, bias_adjustment=True)),
    "fastmatch G bias-corrected (faiss)": ("fastmatch:faiss", None),
}
REFERENCE = {"Matching AI (R)": "match-stream ai",
             "Matching AI n_variance_matches=1 (R)": "match-stream ai n_variance_matches=1",
             "Matching AI n_variance_matches=1 BiasAdjust (R)":
                 "match-stream ai n_variance_matches=1 bias_adjustment",
             "*": None}


def runner(config, n, path, workdir):
    import polars as pl

    from match_stream import match_nn
    label = config
    kind, options = CONFIGS[label]
    design, exact = DESIGNS[kind]
    # warm up: numba compiles (or loads) its kernels on the first call
    small = pl.from_arrow(simulate_chunk(4_000, 0, 1))
    match_nn("d ~ x1 + x2 + x3", small, exact=exact, verbose=False,
             **design).effect("y", **options)
    t0 = time.perf_counter()
    m = match_nn("d ~ x1 + x2 + x3", str(path), exact=exact, workdir=str(workdir),
                 verbose=False, **design)
    t_match = time.perf_counter() - t0
    fit = m.effect("y", **options)
    t_fit = time.perf_counter() - t0
    return {"t_read": 0.0, "t_fit": t_fit, "t_match": t_match, "t_effect": t_fit - t_match,
            "matched": int(m.counts()["matched_treated"]), "converged": True,
            "coefs": {"att": float(fit.beta[0]), "se": float(fit.se[0])}}


def command_for(path):
    def command(label, kind):
        if kind.startswith("fastmatch:"):
            return (external_command("fastmatch", path=path, scenario="G",
                                     backend=kind.split(":")[1], bias_adjustment=True)
                    or external_missing("fastmatch"))
        if not kind.startswith("R:"):
            return None
        _r, n_variance_matches, *bias = kind.split(":")
        return external_command("R", fit="matching_ai", path=path,
                                n_variance_matches=int(n_variance_matches),
                                bias_adjustment=bool(bias)) or external_missing("R")
    return command


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "effects.csv"
    write_machine_info(RESULTS_DIR / "machine.json")
    # the child is given the label as its "kind" (run_configs passes the kind)
    configs = {label: label if kind.startswith("ms:") else kind
               for label, (kind, _o) in CONFIGS.items()}
    for n in args.sizes or DEFAULT_SIZES:
        path = data_path(n, DATA_DIR)
        run_configs(SCRIPT, out, "effects", n, path, configs, command=command_for(path),
                    timeout=args.timeout, only=args.only, reference=REFERENCE,
                    extra=lambda _l, n=n: {"rows": n})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(runner, *sys.argv[2:])
    else:
        main()
