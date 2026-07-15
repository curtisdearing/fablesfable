"""Shared fixtures for the prop-shortlister test suite.

Uses a 2-season slice (2019-2020) rather than the full 2019-2023 parquet so
the suite runs quickly; the leakage/reproducibility properties being tested
don't depend on how many seasons are loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.features import load_pbp  # noqa: E402

FAST_SEASONS = [2019, 2020]


@pytest.fixture(scope="session")
def pbp_fast():
    df = load_pbp()
    return df[df["season"].isin(FAST_SEASONS)].copy()


@pytest.fixture(scope="session")
def pbp_tiny():
    """A small (single-season) slice for tests where determinism, not data
    volume, is what's being checked -- keeps the suite fast."""
    df = load_pbp()
    return df[df["season"] == 2019].copy()


@pytest.fixture(scope="session")
def backtest_report_fast(tmp_path_factory):
    """One shared run of the backtest (single season, for speed) -- reused by
    every smoke-test assertion instead of each test re-running the pipeline.
    All generated files live outside the repository checkout."""
    import prop_backtest
    directory = tmp_path_factory.mktemp("prop-backtest")
    report = prop_backtest.run(
        seasons=[2019], output_path=str(directory / "prop_backtest.json"),
        db_path=str(directory / "prop_backtest.db"),
    )
    report["_test_output_path"] = str(directory / "prop_backtest.json")
    return report
