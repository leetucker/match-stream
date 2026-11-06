"""Memory tests: matching and its effects hold a few narrow columns of the
data, not the data.

Run with ``pytest -m resource`` (the default run leaves them out: they take
about a minute). Each design runs in a child process under stream-limits'
``resource_runner``: a watchdog on every platform, or a kernel limit when
``STREAM_LIMITS_TEST_CGROUP_PARENT`` names a cgroup set up for it.

Unlike hdfe-stream's fits, matching holds something for every unit: the
narrow projection (id, treatment, score, strata) and the pairs, and the
effects and the balance table one or two columns more. Its memory therefore
grows with the units, and what these tests check is that it does not grow
with the width of the data, which in a real extract is most of it. The data
are 2.5 million units with 40 columns besides simulate_matching's (0.98 GB
in memory), read as five Parquet files. Measured on Linux (WSL2, 2 threads),
private memory (not counting memory-mapped scratch files, which the kernel
can reclaim), streamed and with the data also held in memory:

- nearest-neighbor matching in exact strata, effect() and balance():
  1.64 GB, and 2.60 GB with the data held; the limit is 2.0 GB;
- kernel matching and effect(): 1.24 GB, and 2.18 GB held; the limit is
  1.7 GB.

The limits are on private memory, which stream-limits measures on Linux and
Windows. On macOS it measures resident memory, which counts the mapped
scratch files, so there they are 1 GB looser.
On Windows, private memory is committed memory: a buffer or an allocator
segment counts in full once allocated, touched or not (an hdfe-stream logit
fit had 1.61 GB committed while 1.14 GB was resident). So Windows also gets
looser limits (see SLACK_GB), until its own CI figures set tighter ones.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl
import pytest

from match_stream.simulate import simulate_matching

pytestmark = pytest.mark.resource

PARTS, UNITS_PER_PART, WIDE = 5, 500_000, 40    # 2.5 million units
BUDGET_GB = 1.0                                 # max_memory_gb
# extra room per platform: macOS counts mapped scratch files; Windows counts
# committed memory. Windows, nearest-neighbor matching: the first CI run was
# stopped at 3.23 GB committed (1.27 GB resident) before it finished, so its
# peak is not known yet; the slack is wide until a run reports it. Kernel
# matching: 1.18 GB committed.
SLACK_GB = {
    "nn": {"darwin": 1.0, "win32": 3.0},
    "kernel": {"darwin": 1.0, "win32": 0.5},
}
SPEC = ("'d ~ x1 + x2 + x3', SRC, exact=['r', 'i'], id='id', workdir=WORK, verbose=False, "
        f"max_memory_gb={BUDGET_GB}")


@pytest.fixture(scope="module")
def wide_units(tmp_path_factory):
    """simulate_matching's units with 40 more columns, as an extract would
    have, in parts, each part's ids numbered after the last part's."""
    folder = tmp_path_factory.mktemp("big")
    size = 0
    for p in range(PARTS):
        d = simulate_matching(UNITS_PER_PART, seed=10 + p)
        d = d.with_columns(pl.col("id") + p * UNITS_PER_PART)
        rng = np.random.default_rng(p)
        d = d.with_columns([pl.Series(f"z{j}", rng.normal(size=UNITS_PER_PART))
                            for j in range(WIDE)])
        d.write_parquet(folder / f"part{p}.parquet")
        size += d.estimated_size()
    assert size > 0.95e9
    return folder / "*.parquet"


def _run_in_child(resource_runner, body: str, path, workdir, limit_gb: float,
                  kind: str) -> dict:
    code = f"""
import json
from match_stream import match_kernel, match_nn
{body}
"""
    code = code.replace("SRC", repr(str(path))).replace("WORK", repr(str(workdir)))
    limit_gb += SLACK_GB[kind].get(sys.platform, 0.0)
    result = resource_runner([sys.executable, "-c", code], memory_gb=limit_gb,
                             threads=2, timeout=900)
    report = {k: result.as_dict()[k] for k in ("status", "mode", "memory_measure",
                                               "peak_private_bytes", "peak_rss_bytes",
                                               "elapsed_s")}
    print(json.dumps(report))
    assert result.status == "ok", (report, result.stderr[-3000:])
    if result.kernel:
        assert result.kernel["events"].get("oom_kill", 0) == 0
    return json.loads(result.stdout.strip().splitlines()[-1])


# The checks against the simulated effect (1) and balance are sanity checks
# only: agreement with MatchIt, Matching, teffects and kmatch is the rest of
# the suite's job.

def test_nearest_neighbor_matching_inside_the_limit(resource_runner, wide_units, tmp_path):
    out = _run_in_child(resource_runner, f"""
m = match_nn({SPEC}, replace=True)
att = float(m.effect("y", score_correction=False).coef()["ATT:y"])
smd = m.balance().matched.filter(term="x1")["Std. Mean Diff."].item()
print(json.dumps(dict(att=att, smd=smd)))
""", wide_units, tmp_path / "work", limit_gb=2.0, kind="nn")
    assert abs(out["att"] - 1.0) < 0.05
    assert abs(out["smd"]) < 0.05


def test_kernel_matching_inside_the_limit(resource_runner, wide_units, tmp_path):
    out = _run_in_child(resource_runner, f"""
m = match_kernel({SPEC})
print(json.dumps(dict(att=float(m.effect("y").coef()["ATT:y"]))))
""", wide_units, tmp_path / "work", limit_gb=1.7, kind="kernel")
    assert abs(out["att"] - 1.0) < 0.05
