"""Live weather features must be the forecast for the real kickoff hour.

nflverse ``gameday``/``gametime`` are Eastern clock time. The live path used
to label that clock ``+00:00``, so every outdoor game read the forecast four
(EDT) or five (EST) hours BEFORE kickoff -- 2026-09-22 run: ATL@GB (20:15 ET
= 00:15Z) was stamped temp 66.6F / wind 8.9mph, the 15:15 CDT forecast,
instead of the kickoff hour's 58.7F / 3.2mph, and those are configured
ranker features (``temp``, ``wind``).
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue.sources import weather as wxmod  # noqa: E402


def _run(monkeypatch, gameday, gametime):
    seen = []

    def fake_forecast(home, commence_iso):
        seen.append(commence_iso)
        return {"dome": False, "temp_f": 58.7, "wind_mph": 3.2}

    monkeypatch.setattr(wxmod, "forecast_for_game", fake_forecast)
    slate = pd.DataFrame([{"game_id": "g1", "gameday": gameday, "gametime": gametime,
                           "home_team": "GB", "away_team": "ATL"}])
    adv = SimpleNamespace(weather={"g1": (None, None)})
    pw._apply_forecast_weather(adv, slate)
    return seen, adv


def test_forecast_requested_for_true_utc_kickoff_edt(monkeypatch):
    seen, adv = _run(monkeypatch, "2026-09-24", "20:15")
    assert dt.datetime.fromisoformat(seen[0]) == dt.datetime(2026, 9, 25, 0, 15, tzinfo=dt.timezone.utc)
    assert adv.weather["g1"] == (58.7, 3.2)


def test_forecast_requested_for_true_utc_kickoff_est(monkeypatch):
    seen, _ = _run(monkeypatch, "2026-11-29", "13:00")
    assert dt.datetime.fromisoformat(seen[0]) == dt.datetime(2026, 11, 29, 18, 0, tzinfo=dt.timezone.utc)
