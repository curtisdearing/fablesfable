import sys
from types import SimpleNamespace

import pandas as pd

from nflvalue import context_features
from nflvalue.sources import rosters


def test_forced_injury_refresh_preserves_unrequested_seasons(tmp_path, monkeypatch):
    path = tmp_path / "injuries.parquet"
    old = pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "gsis_id": "old",
                         "position": "WR", "report_status": "Out", "full_name": "Old"}])
    old.to_parquet(path, index=False)
    fresh = pd.DataFrame([{"season": 2024, "week": 1, "team": "B", "gsis_id": "new",
                           "position": "RB", "report_status": "Out", "full_name": "New"}])
    fake = SimpleNamespace(load_injuries=lambda seasons: SimpleNamespace(to_pandas=lambda: fresh))
    monkeypatch.setattr(context_features, "INJURIES", str(path))
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)
    context_features.load_injury_history([2024], refresh=True)
    assert set(pd.read_parquet(path)["season"]) == {2023, 2024}


def test_forced_roster_refresh_preserves_unrequested_seasons(tmp_path, monkeypatch):
    path = tmp_path / "rosters.parquet"
    old = pd.DataFrame([{"season": 2023, "week": 1, "team": "A", "position": "WR",
                         "player_id": "old", "full_name": "Old"}])
    old.to_parquet(path, index=False)
    fresh = pd.DataFrame([{"season": 2024, "week": 1, "team": "B", "position": "RB",
                           "gsis_id": "new", "full_name": "New"}])
    fake = SimpleNamespace(load_rosters_weekly=lambda seasons: SimpleNamespace(to_pandas=lambda: fresh))
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)
    rosters.fetch_rosters_weekly([2024], cache_path=str(path), force_refresh=True)
    assert set(pd.read_parquet(path)["season"]) == {2023, 2024}
