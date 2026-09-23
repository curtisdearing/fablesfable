"""A T-90 run that arrives shortly BEFORE a game's [T-90, T-0) window waits
for the window (bounded) instead of exiting as a no-op.

GitHub starts this repo's scheduled runs late and unpredictably (2026-09:
the 23:15Z slot started 01:02-01:35Z, the 19:20Z slot 21:26-21:35Z, the
15:55Z slot 15:58-16:09Z), so backup triggers are placed before the window.
One that lands early must not waste itself: it sleeps until the window
opens, at most :data:`T90_MAX_EARLY_WAIT_MINUTES` (the run job has a
60-minute timeout), then processes the game like any in-window run. A run
further out than that is still a no-op -- it never processes a game before
the inactives are expected, and never waits past the bound.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.auto_weekly as aw  # noqa: E402

ET = ZoneInfo("America/New_York")
KICK = dt.datetime(2026, 9, 24, 20, 15, tzinfo=ET)       # Thursday ATL@GB


def _slate():
    return pd.DataFrame([{"game_id": "2026_03_ATL_GB", "season": 2026, "week": 3,
                          "kickoff": KICK}])


def test_wait_is_zero_inside_the_window_and_none_after_kickoff():
    assert aw.t90_wait_seconds(_slate(), KICK - dt.timedelta(minutes=60)) == 0
    assert aw.t90_wait_seconds(_slate(), KICK - dt.timedelta(minutes=90)) == 0
    assert aw.t90_wait_seconds(_slate(), KICK) is None
    assert aw.t90_wait_seconds(_slate(), KICK + dt.timedelta(minutes=5)) is None


def test_wait_until_the_window_opens_only_within_the_bound():
    opens = KICK - dt.timedelta(minutes=aw.T90_DUE_MINUTES)
    assert aw.t90_wait_seconds(_slate(), opens - dt.timedelta(minutes=20)) == 20 * 60
    bound = aw.T90_MAX_EARLY_WAIT_MINUTES
    assert 0 < bound <= 40                       # fits the run job's 60-minute timeout
    assert aw.t90_wait_seconds(_slate(), opens - dt.timedelta(minutes=bound)) == bound * 60
    assert aw.t90_wait_seconds(_slate(), opens - dt.timedelta(minutes=bound + 1)) is None


def _patch_job(monkeypatch, clock, slept, processed):
    monkeypatch.setattr(aw, "ensure_current_inputs", lambda job: {})
    monkeypatch.setattr(aw, "load_slate", _slate)
    monkeypatch.setattr(aw, "now_et", lambda: clock[0])
    monkeypatch.setattr(aw, "write_pipeline_heartbeat", lambda *a, **k: {})
    monkeypatch.setattr(aw, "write_pick_cards", lambda *a, **k: None)

    def fake_sleep(seconds):
        slept.append(seconds)
        clock[0] = clock[0] + dt.timedelta(seconds=seconds)

    monkeypatch.setattr(aw, "_sleep", fake_sleep)
    from nflvalue import config as cfgmod
    monkeypatch.setattr(cfgmod, "load_config", lambda: {"discord_enabled": False})
    from nflvalue import db as dbmod
    monkeypatch.setattr(dbmod, "connect", lambda p=None: _MemConn())
    monkeypatch.setattr(dbmod, "query_df", lambda conn, sql, params=(): pd.DataFrame({"game_id": []}))
    import pipeline_weekly as pw
    monkeypatch.setattr(pw, "run_t90", lambda *a, **k: processed.append((a[2], clock[0])) or {"voided": []})
    import nflvalue.candidates as cand
    monkeypatch.setattr(cand, "build_week_inputs", lambda: object())


class _MemConn:
    def close(self):
        pass


def test_early_run_sleeps_to_the_window_then_processes(monkeypatch):
    opens = KICK - dt.timedelta(minutes=aw.T90_DUE_MINUTES)
    clock, slept, processed = [opens - dt.timedelta(minutes=25)], [], []
    _patch_job(monkeypatch, clock, slept, processed)
    assert aw.job_t90() == 0
    assert slept == [25 * 60]
    assert processed == [("2026_03_ATL_GB", opens)]


def test_run_beyond_the_bound_is_still_a_no_op(monkeypatch):
    opens = KICK - dt.timedelta(minutes=aw.T90_DUE_MINUTES)
    clock, slept, processed = [opens - dt.timedelta(hours=2)], [], []
    _patch_job(monkeypatch, clock, slept, processed)
    assert aw.job_t90() == 0
    assert slept == [] and processed == []
