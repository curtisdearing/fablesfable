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


def _patch_job(monkeypatch, clock, slept, processed, overrun=0, done_after_wait=(),
               resnaps=None):
    monkeypatch.setattr(aw, "ensure_current_inputs", lambda job: {})
    monkeypatch.setattr(aw, "load_slate", _slate)
    monkeypatch.setattr(aw, "now_et", lambda: clock[0])
    monkeypatch.setattr(aw, "write_pipeline_heartbeat", lambda *a, **k: {})
    monkeypatch.setattr(aw, "write_pick_cards", lambda *a, **k: None)

    def fake_sleep(seconds):
        slept.append(seconds)
        clock[0] = clock[0] + dt.timedelta(seconds=seconds + overrun)
        # while this run slept, the serialized concurrency group let no other
        # run publish; a game already processed is visible only via the DB read
        # that job_t90 makes AFTER the wait
        state["done"] = list(done_after_wait)

    monkeypatch.setattr(aw, "_sleep", fake_sleep)
    from nflvalue import config as cfgmod
    monkeypatch.setattr(cfgmod, "load_config", lambda: {"discord_enabled": False})
    from nflvalue import db as dbmod
    import pipeline_weekly as pw
    monkeypatch.setattr(dbmod, "connect", lambda p=None: _MemConn())
    state = {"done": []}

    def query_df(conn, sql, params=()):
        if "FROM leans" in sql:
            return pd.DataFrame({"game_id": state["done"]})
        return pd.DataFrame({"game_id": ["2026_03_ATL_GB"]})     # has stored lines

    monkeypatch.setattr(dbmod, "query_df", query_df)
    if resnaps is not None:
        monkeypatch.setattr(cfgmod, "load_config",
                            lambda: {"discord_enabled": False, "odds_api_key": "k"})
        from nflvalue.sources import oddsapi_props as oap
        monkeypatch.setattr(oap, "resnap_lines", lambda cfg, emap, conn=None: resnaps.append(
            dict(emap)) or {"pulled": list(emap), "empty": [], "rows_written": 0,
                            "credits_spent": 5.0, "credits_billed_measured": 5.0,
                            "credits_estimated": 0.0, "credits_planned": 5.0,
                            "account_usage_delta": 5.0, "budget_remaining": 65.0})
        monkeypatch.setattr(pw, "build_event_map", lambda cfg, s: {g: f"e_{g}" for g in s.game_id})
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


def test_after_the_wait_the_kickoff_is_rechecked(monkeypatch):
    """If the sleep overruns past kickoff, the game is no longer due: nothing
    is processed and no odds are acquired for a game already under way."""
    opens = KICK - dt.timedelta(minutes=aw.T90_DUE_MINUTES)
    clock, slept, processed, resnaps = [opens - dt.timedelta(minutes=10)], [], [], []
    _patch_job(monkeypatch, clock, slept, processed, overrun=95 * 60, resnaps=resnaps)
    assert aw.job_t90() == 0
    assert slept == [10 * 60] and processed == [] and resnaps == []


def test_wait_then_processed_set_is_read_so_no_duplicate_acquisition(monkeypatch):
    """The processed-game set is read AFTER the wait: a game processed by the
    previous serialized run is neither re-snapped (no credit) nor re-run."""
    opens = KICK - dt.timedelta(minutes=aw.T90_DUE_MINUTES)
    clock, slept, processed, resnaps = [opens - dt.timedelta(minutes=10)], [], [], []
    _patch_job(monkeypatch, clock, slept, processed,
               done_after_wait=["2026_03_ATL_GB"], resnaps=resnaps)
    assert aw.job_t90() == 0
    assert slept == [10 * 60] and processed == [] and resnaps == []


def test_single_early_run_acquires_once_after_the_wait(monkeypatch):
    opens = KICK - dt.timedelta(minutes=aw.T90_DUE_MINUTES)
    clock, slept, processed, resnaps = [opens - dt.timedelta(minutes=10)], [], [], []
    _patch_job(monkeypatch, clock, slept, processed, resnaps=resnaps)
    assert aw.job_t90() == 0
    assert resnaps == [{"2026_03_ATL_GB": "e_2026_03_ATL_GB"}]
    assert processed == [("2026_03_ATL_GB", opens)]
