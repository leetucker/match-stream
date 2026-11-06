"""Shared fixtures: the reference data set and R's reference outputs
(tests/data, made by tests/reference/)."""

import json
from pathlib import Path

import polars as pl
import pytest

DATA = Path(__file__).parent / "data"

# resource_runner and the `resource` marker (tests/test_resources.py)
pytest_plugins = ["stream_limits.pytest_plugin"]


@pytest.fixture(scope="session")
def ref_data() -> pl.DataFrame:
    # the true score is renamed: distance="ps" means "fit a score" in match_stream
    return pl.read_csv(DATA / "ref_data.csv").rename({"ps": "ps_true"})


@pytest.fixture(scope="session")
def ref_r() -> dict:
    return json.loads((DATA / "ref_r.json").read_text())


def matchit_pairs(spec: str) -> dict[int, set[int]]:
    """MatchIt's match.matrix for `spec`: treated id -> set of control ids
    (empty when unmatched)."""
    ref = pl.read_csv(DATA / f"ref_matchit_{spec}.csv.gz", null_values="NA")
    return {row[0]: {x for x in row[1:] if x is not None} for row in ref.iter_rows()}


def matchit_weights(spec: str) -> pl.DataFrame:
    return pl.read_csv(DATA / f"ref_matchit_{spec}_weights.csv.gz", null_values=["NA", ""])
