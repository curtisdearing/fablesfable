"""Shared fixtures for the prop-shortlister test suite.

Uses a 2-season slice (2019-2020) rather than the full 2019-2023 parquet so
the suite runs quickly; the leakage/reproducibility properties being tested
don't depend on how many seasons are loaded.

Why there is a committed fixture
--------------------------------
``historical/historical_pbp.parquet`` is a ~19 MB LOCAL CACHE and is gitignored
(``*.parquet``). Before this fixture existed, a fresh clone could not run
``test_leakage``, ``test_reproducibility``, ``test_positions`` or
``test_backtest_smoke`` at all -- they raised FileNotFoundError. Combined with
a CI workflow that runs an explicit allowlist of data-independent files, that
meant **the leakage guards never ran anywhere except a machine that already had
the cache**. For the file its own docstring calls "the #1 kill bug test", that
is not an acceptable place to be.

``tests/fixtures/pbp_2019_2020.parquet`` (1.2 MB, zstd) is a real, unmodified
slice of that cache -- seasons 2019-2020, all 22 projected columns, 90,745
rows. Real data, not synthetic: the leakage semantics under test depend on
genuine rolling histories, so a fabricated frame would prove nothing.

Resolution order is cache-first, so a developer holding the full cache still
tests against the real thing; the fixture is the fallback that makes the
guards runnable everywhere else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.features import load_pbp  # noqa: E402

FAST_SEASONS = [2019, 2020]

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
PBP_FIXTURE = FIXTURE_DIR / "pbp_2019_2020.parquet"
SCHEDULES_FIXTURE = FIXTURE_DIR / "schedules_2019_2020.parquet"

#: Set to 1 in CI. When set, an unavailable fixture is a hard FAILURE rather
#: than a skip -- so the leakage guards can never quietly stop running. A green
#: CI run that silently skipped them is the exact outcome this flag prevents.
STRICT_ENV = "FABLESFABLE_STRICT_FIXTURES"


def _strict() -> bool:
    return os.environ.get(STRICT_ENV, "").strip() not in ("", "0", "false")


def _unavailable(message: str):
    if _strict():
        pytest.fail(f"{message} ({STRICT_ENV} is set, so this is not skippable.)")
    pytest.skip(message)


def _load_pbp_or_fixture():
    """Full local cache if present, else the committed 2019-2020 slice."""
    try:
        return load_pbp(), "historical/historical_pbp.parquet"
    except (FileNotFoundError, OSError):
        pass
    if PBP_FIXTURE.exists():
        import pandas as pd
        return pd.read_parquet(PBP_FIXTURE), str(PBP_FIXTURE)
    _unavailable(
        "no play-by-play available: neither historical/historical_pbp.parquet "
        "(local cache, rebuild with `python3 scripts/bootstrap_history.py`) "
        f"nor {PBP_FIXTURE} (committed fixture) exists.")


@pytest.fixture(scope="session")
def pbp_source():
    """Which source the pbp fixtures resolved to -- surfaced so a run can
    report whether it exercised the full cache or the committed slice."""
    _df, source = _load_pbp_or_fixture()
    return source


@pytest.fixture(scope="session")
def pbp_fast():
    df, _source = _load_pbp_or_fixture()
    return df[df["season"].isin(FAST_SEASONS)].copy()


@pytest.fixture(scope="session")
def pbp_tiny():
    """A small (single-season) slice for tests where determinism, not data
    volume, is what's being checked -- keeps the suite fast."""
    df, _source = _load_pbp_or_fixture()
    return df[df["season"] == 2019].copy()


@pytest.fixture(scope="session")
def schedules_fast():
    """Schedules/lines for the same 2-season window, cache-first."""
    import pandas as pd
    root = Path(__file__).resolve().parents[1]
    cache = root / "historical_lines.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    elif SCHEDULES_FIXTURE.exists():
        df = pd.read_parquet(SCHEDULES_FIXTURE)
    else:
        _unavailable(f"no schedules available: neither {cache} nor "
                     f"{SCHEDULES_FIXTURE} exists.")
    return df[df["season"].isin(FAST_SEASONS)].copy()


@pytest.fixture(scope="session")
def backtest_report_fast(tmp_path_factory):
    """One shared run of the backtest (single season, for speed) -- reused by
    every smoke-test assertion instead of each test re-running the pipeline.
    All generated files live outside the repository checkout.

    This one genuinely needs the FULL caches: ``prop_backtest`` reads them by
    path rather than through a fixture seam. The committed slices cover the
    feature and leakage tests; wiring the end-to-end backtest onto them is a
    separate change and is registered as a follow-up rather than faked here.
    """
    root = Path(__file__).resolve().parents[1]
    have_pbp = (root / "historical" / "historical_pbp.parquet").exists()
    have_lines = (root / "historical_lines.parquet").exists()
    if not (have_pbp and have_lines):
        # A PLAIN skip, deliberately NOT subject to STRICT_FIXTURES. Strict
        # mode exists to catch a missing COMMITTED FIXTURE (a repo defect);
        # the full 19 MB cache is legitimately absent in CI, and failing on it
        # would make strict mode unusable there -- which would end with the
        # flag being turned off and the leakage guards going quiet again.
        pytest.skip(
            "backtest smoke needs the full local caches "
            "(historical/historical_pbp.parquet + historical_lines.parquet); "
            "rebuild with `python3 scripts/bootstrap_history.py`. The committed "
            "fixtures cover the feature/leakage tests, not the end-to-end "
            "backtest.")

    import prop_backtest
    directory = tmp_path_factory.mktemp("prop-backtest")
    report = prop_backtest.run(
        seasons=[2019], output_path=str(directory / "prop_backtest.json"),
        db_path=str(directory / "prop_backtest.db"),
    )
    report["_test_output_path"] = str(directory / "prop_backtest.json")
    return report
