"""Auto-ingest: loaders compose seasons on disk; refresh degrades loudly."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import ingest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: These three tests assert the composition of the LOCAL, GITIGNORED cache
#: (which seasons are on disk, that 2025 has 272 REG games). They are checking
#: the cache itself, so a committed slice cannot stand in for it -- unlike the
#: leakage/feature tests, which only need real rolling histories and now run
#: from tests/fixtures/. Without the cache they must SKIP with an actionable
#: message, not fail: a red suite on a fresh clone trains a reader to ignore
#: red, which is worse than an honest skip.
requires_full_cache = pytest.mark.skipif(
    not (ROOT / "historical" / "historical_pbp.parquet").exists(),
    reason=("needs the full local nflverse cache "
            "(historical/historical_pbp.parquet, ~19 MB, gitignored). "
            "Rebuild with `python3 scripts/bootstrap_history.py`."))



def test_current_season_league_year():
    assert ingest.current_season(dt.date(2026, 7, 1)) == 2026
    assert ingest.current_season(dt.date(2026, 1, 15)) == 2025   # Jan = prior season
    assert ingest.current_season(dt.date(2026, 9, 10)) == 2026


@requires_full_cache
def test_loaders_compose_extra_seasons():
    extra = ingest.extra_seasons_on_disk()
    assert 2024 in extra and 2025 in extra                       # ingested earlier
    pbp = ingest.load_all_pbp()
    assert set(pbp["season"].unique()) >= {2019, 2023, 2024, 2025}
    assert (pbp["season_type"] == "REG").all()
    sched = ingest.load_all_schedules()
    assert sched[(sched["season"] == 2025) & (sched["game_type"] == "REG")].shape[0] == 272
    assert sched["game_id"].is_unique


@requires_full_cache
def test_refresh_degrades_loudly_not_silently(monkeypatch):
    """A dead nflverse pull must report errors + stale, never raise or
    silently serve nothing."""
    # nflreadpy is an OPTIONAL dependency (requirements.txt only needs it to
    # rebuild caches for new seasons). Without this guard the suite reports a
    # FAILURE rather than a skip when it is absent, which makes "262 green"
    # quietly conditional on an optional install. Phase 7.5.
    nfl = pytest.importorskip("nflreadpy")

    def boom(**kw):
        raise RuntimeError("nflverse unreachable (test)")

    monkeypatch.setattr(nfl, "load_pbp", boom)
    monkeypatch.setattr(nfl, "load_schedules", boom)
    res = ingest.refresh(season=2025)     # 2025 file exists -> cache keeps serving
    assert res["errors"]
    assert res["stale"] is False          # cached pbp exists, so features still build
    res2 = ingest.refresh(season=2031)    # no cache for a fake future season
    assert res2["stale"] is True


@requires_full_cache
def test_build_week_inputs_full_history():
    from nflvalue.candidates import build_week_inputs, games_for_week
    import pandas as pd
    # schedules path only (feature build is exercised elsewhere; keep fast)
    sched = ingest.load_all_schedules()
    slate = games_for_week(2025, 14, sched)
    assert len(slate) == 14
    assert build_week_inputs.__defaults__[-1] is True            # full_history default on