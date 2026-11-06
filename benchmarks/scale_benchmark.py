"""Time and peak memory as the number of units grows, for the eight designs of
_scenarios.py (A-H), with match-stream (in memory and by stratum), MatchIt and
Matching (R), pymatchit and Polars' join_asof (scenario A).

    python benchmarks/scale_benchmark.py                    # the default sizes
    python benchmarks/scale_benchmark.py 10000 100000       # units
    python benchmarks/scale_benchmark.py --scenario A --only match-stream

The data are _scenarios.data_path's: n units in 500 exact strata, written
once per size. max_abs_diff compares each configuration's ATT with
match-stream's. Results go to benchmarks/results/scale.csv.
"""

from __future__ import annotations

import argparse
import sys
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
from _scenarios import SCENARIOS, commands, configs, data_path, runner

SCRIPT = Path(__file__).resolve()
DEFAULT_SIZES = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
TIMEOUT_S = 1800


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int)
    ap.add_argument("--scenario", choices=SCENARIOS, action="append")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "scale.csv"
    write_machine_info(RESULTS_DIR / "machine.json")
    for n in args.sizes or DEFAULT_SIZES:
        path = data_path(n, DATA_DIR)
        for scenario in args.scenario or SCENARIOS:
            run_configs(SCRIPT, out, scenario, n, path, configs(scenario),
                        command=commands(path, external_command, external_missing),
                        timeout=args.timeout, only=args.only, reference="match-stream",
                        extra=lambda _l, n=n: {"rows": n})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(runner, *sys.argv[2:])
    else:
        main()
