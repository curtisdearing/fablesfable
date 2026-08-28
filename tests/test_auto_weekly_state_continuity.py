"""Scheduled-job state continuity: current-season inputs must be established
BEFORE a job decides which week (if any) it is looking at.

The failure this file exists to prevent
---------------------------------------
A clean GitHub Actions runner starts with only the FROZEN 2019-2023 history.
``scripts/bootstrap_history.py`` writes ``historical_lines.parquet`` and
refuses any cohort other than 2019-2023, so the 2024 -> current-season
schedule file (``historical/lines_extra.parquet``) -- the file that makes a
2026 game visible to ``load_all_schedules()`` at all -- exists only because
``nflvalue.ingest.refresh()`` wrote it. The workflow's rolling cache for that
file is SAVED only on the Tuesday job and may be evicted at any time, so no
later runner may assume Wednesday's files are still there.

Before this module, only ``job_wed`` called ``ingest.refresh()``. ``job_t90``
and ``job_tuesday`` called ``load_slate()`` directly, so on a cold runner they
inspected a 2019-2023-only slate and:

  * T-90 found no kickoff inside its window and reported a healthy no-op while
    a real game was 90 minutes from kickoff;
  * Tuesday resolved "last completed week" to 2023 week 18 and wrote a
    heartbeat claiming it had graded a week -- freshness it never earned.

Both failures are silent: exit code 0, a green workflow, and no picks.

Everything here is data-independent and offline: ``refresh`` and
``load_slate`` are injected, so no test in this file touches nflverse, the
Odds API, the warehouse DB, or any secret.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import auto_weekly as aw  # noqa: E402
from nflvalue import ingest as ingestmod  # noqa: E402

ET = ZoneInfo("America/New_York")

# A cold runner sees only these two rows (the frozen 2019-2023 cohort).
FROZEN_ROWS = [
    dict(game_id="2019_01_AAA_BBB", season=2019, week=1,
         gameday="2019-09-08", gametime="13:00", result=7.0),
    dict(game_id="2023_18_CCC_DDD", season=2023, week=18,
         gameday="2024-01-07", gametime="16:25", result=-3.0),
]
# ingest.refresh() is what puts these on disk (historical/lines_extra.parquet).
CURRENT_ROWS = [
    dict(game_id="2026_01_AAA_BBB", season=2026, week=1,
         gameday="2026-09-13", gametime="13:00", result=3.0),    # completed
    dict(game_id="2026_02_CCC_AAA", season=2026, week=2,
         gameday="2026-09-20", gametime="13:00", result=float("nan")),  # upcoming
]

T90_NOW = dt.datetime(2026, 9, 20, 11, 30, tzinfo=ET)   # 90 min before wk2 kick
TUE_NOW = dt.datetime(2026, 9, 15, 10, 23, tzinfo=ET)   # Tuesday after wk1
WED_NOW = dt.datetime(2026, 9, 16, 10, 17, tzinfo=ET)   # Wednesday before wk2


def _slate(current_season_present: bool, with_results: bool = True) -> pd.DataFrame:
    rows = list(FROZEN_ROWS) + (list(CURRENT_ROWS) if current_season_present else [])
    df = pd.DataFrame(rows)
    df["game_type"] = "REG"
    if not with_results:
        df["result"] = float("nan")
    df["kickoff"] = [
        dt.datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        for d, t in zip(df["gameday"], df["gametime"])
    ]
    return df


class FakeFeed:
    """Stands in for ``historical/`` on a clean runner.

    The slate is 2019-2023 only until ``refresh()`` succeeds -- which is
    exactly the on-disk relationship between ``bootstrap_history.py`` (frozen
    cohort) and ``ingest.refresh()`` (current-season files). ``calls`` records
    the interleaving so a test can assert ordering, not just occurrence.
    """

    def __init__(self, *, stale: bool = False, errors=None, raises: Exception = None,
                 with_results: bool = True):
        self.calls: list[str] = []
        self.stale = stale
        self.errors = list(errors or [])
        self.raises = raises
        self.with_results = with_results
        self.current_season_on_disk = False

    def refresh(self, season=None, force=False) -> dict:
        self.calls.append("refresh")
        if self.raises is not None:
            raise self.raises
        if not self.stale:
            # A refresh that reached nflverse wrote the current-season files,
            # even if some non-schedule sub-feed reported an error.
            self.current_season_on_disk = True
        return {"season": 2026, "pbp_rows": 1234, "sched_rows": 285,
                "stale": self.stale, "errors": list(self.errors)}

    def load_slate(self) -> pd.DataFrame:
        self.calls.append("load_slate")
        return _slate(self.current_season_on_disk, self.with_results)


def _wire(monkeypatch, feed: FakeFeed, now: dt.datetime) -> list:
    """Inject the feed + a fixed clock; capture heartbeats instead of writing."""
    beats: list[dict] = []
    monkeypatch.setattr(ingestmod, "refresh", feed.refresh)
    monkeypatch.setattr(aw, "load_slate", feed.load_slate)
    monkeypatch.setattr(aw, "now_et", lambda: now)
    monkeypatch.setattr(
        aw, "write_pipeline_heartbeat",
        lambda status, detail, job: beats.append(
            {"status": status, "detail": detail, "job": job}) or {})
    return beats


def _stub_t90_downstream(monkeypatch) -> list:
    """Everything T-90 touches after slate selection, stubbed offline.

    No Odds API key is configured, so ``resnap_lines`` must never be reached;
    it is wired to explode if the job ever tries to spend a credit here.
    """
    import pipeline_weekly as pw
    from nflvalue import candidates as candmod, db as dbmod, notify
    from nflvalue.sources import oddsapi_props as oap

    ran: list[str] = []

    class _Conn:
        def close(self):
            pass

    def _boom(*a, **k):
        raise AssertionError("T-90 must not call the Odds API without a key")

    monkeypatch.setattr(dbmod, "connect", lambda *a, **k: _Conn())
    monkeypatch.setattr(dbmod, "query_df",
                        lambda *a, **k: pd.DataFrame({"game_id": []}))
    monkeypatch.setattr(oap, "resnap_lines", _boom)
    monkeypatch.setattr(notify, "resolve_webhook", lambda: None)
    monkeypatch.setattr(candmod, "build_week_inputs", lambda *a, **k: "INPUTS")

    def _run_t90(season, week, game_id, **kwargs):
        ran.append(game_id)
        return {"voided": []}

    monkeypatch.setattr(pw, "run_t90", _run_t90)
    return ran


def _stub_config(monkeypatch, **overrides):
    from nflvalue import config as cfgmod
    cfg = {"odds_api_key": "", "discord_enabled": False}
    cfg.update(overrides)
    monkeypatch.setattr(cfgmod, "load_config", lambda: dict(cfg))
    return cfg


def _schedule_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows).drop(columns=["result"])
    df["game_type"] = "REG"
    return df


# --------------------------------------------------------------------------- #
# The premise, pinned: bootstrap alone can never produce a current-season slate
# --------------------------------------------------------------------------- #
def test_frozen_bootstrap_cohort_cannot_contain_the_current_season():
    """If this ever fails, the reasoning in this file needs revisiting."""
    import bootstrap_history as bh

    assert set(bh.BASE_SEASONS) == {2019, 2020, 2021, 2022, 2023}
    assert ingestmod.current_season(dt.date(2026, 9, 20)) == 2026
    assert 2026 not in set(bh.BASE_SEASONS)
    # The 2024-> file is written by refresh(), not by bootstrap.
    assert ingestmod.LINES_EXTRA.endswith("lines_extra.parquet")


def test_load_slate_cannot_see_the_current_season_until_refresh_writes_it(
        tmp_path, monkeypatch):
    """The REAL loader, not a stand-in.

    ``ingest.load_all_schedules()`` composes the frozen base file with
    ``lines_extra.parquet``, and only ``refresh()`` ever writes the latter.
    Everything else in this file rests on that on-disk fact, so it is checked
    against the actual composition logic rather than assumed.
    """
    base = tmp_path / "historical_lines.parquet"
    extra = tmp_path / "lines_extra.parquet"
    _schedule_frame(FROZEN_ROWS).to_parquet(base, index=False)
    monkeypatch.setattr(ingestmod, "BASE_LINES", str(base))
    monkeypatch.setattr(ingestmod, "LINES_EXTRA", str(extra))

    cold = aw.load_slate()
    assert int(cold["season"].max()) == 2023
    assert aw.current_week(cold, T90_NOW) is None, (
        "a cold runner reads the frozen cohort as 'no current week' in the "
        "middle of September")

    # Exactly what a successful ingest.refresh() puts on disk.
    _schedule_frame(CURRENT_ROWS).to_parquet(extra, index=False)

    warm = aw.load_slate()
    assert int(warm["season"].max()) == 2026
    assert aw.current_week(warm, T90_NOW) == (2026, 2)


# --------------------------------------------------------------------------- #
# T-90
# --------------------------------------------------------------------------- #
def test_t90_refreshes_current_inputs_before_reading_the_slate(monkeypatch):
    """Ordering, stated as ordering: refresh must precede the first slate read."""
    feed = FakeFeed()
    _wire(monkeypatch, feed, T90_NOW)
    _stub_config(monkeypatch)
    _stub_t90_downstream(monkeypatch)

    aw.job_t90()

    assert feed.calls, "job_t90 read no inputs at all"
    assert feed.calls[0] == "refresh", (
        "job_t90 selected a slate before establishing current-season inputs; "
        f"call order was {feed.calls}")
    assert "load_slate" in feed.calls


def test_t90_on_a_cold_runner_sees_the_due_game_instead_of_no_opping(monkeypatch):
    """The consequence: a 2019-2023-only slate must not be the last word."""
    feed = FakeFeed()
    beats = _wire(monkeypatch, feed, T90_NOW)
    _stub_config(monkeypatch)
    ran = _stub_t90_downstream(monkeypatch)

    rc = aw.job_t90()

    assert rc == 0
    assert ran == ["2026_02_CCC_AAA"], (
        "T-90 skipped a game kicking off in 90 minutes because it inspected "
        "the frozen 2019-2023 slate")
    assert beats and beats[-1]["status"] == "active"
    assert "no kickoff" not in beats[-1]["detail"].lower()


def test_t90_shared_helper_is_used(monkeypatch):
    """One shared, testable refresh helper -- not a copy per job."""
    feed = FakeFeed()
    _wire(monkeypatch, feed, T90_NOW)
    _stub_config(monkeypatch)
    _stub_t90_downstream(monkeypatch)

    seen: list[str] = []
    real = aw.ensure_current_inputs
    monkeypatch.setattr(aw, "ensure_current_inputs",
                        lambda job: seen.append(job) or real(job))

    aw.job_t90()
    assert seen == ["t90"]


def test_t90_degrades_loudly_when_refresh_is_stale(monkeypatch):
    """Stale refresh: keep going on cached inputs, but never claim freshness."""
    feed = FakeFeed(stale=True, errors=["schedules: connection reset"])
    beats = _wire(monkeypatch, feed, T90_NOW)
    _stub_config(monkeypatch)
    _stub_t90_downstream(monkeypatch)

    rc = aw.job_t90()

    assert rc == 0                      # cached inputs are still worth running
    assert beats, "a degraded T-90 run still owes the dashboard a heartbeat"
    assert beats[-1]["status"] == "degraded"
    assert "connection reset" in beats[-1]["detail"]


def test_t90_survives_a_refresh_that_raises(monkeypatch):
    """A broken feed must not turn a scheduled job into a crash."""
    feed = FakeFeed(raises=RuntimeError("nflverse 503"))
    beats = _wire(monkeypatch, feed, T90_NOW)
    _stub_config(monkeypatch)
    _stub_t90_downstream(monkeypatch)

    rc = aw.job_t90()

    assert rc == 0
    assert "load_slate" in feed.calls, "the job gave up instead of using cache"
    assert beats[-1]["status"] == "degraded"
    assert "nflverse 503" in beats[-1]["detail"]


def test_t90_offseason_no_op_still_works(monkeypatch):
    """Existing behavior preserved: a real offseason is still a clean no-op."""
    feed = FakeFeed()
    beats = _wire(monkeypatch, feed, dt.datetime(2027, 3, 1, 10, 0, tzinfo=ET))
    _stub_config(monkeypatch)
    ran = _stub_t90_downstream(monkeypatch)

    rc = aw.job_t90()

    assert rc == 0
    assert ran == []
    assert beats[-1]["status"] == "offseason"
    assert "no kickoff" in beats[-1]["detail"].lower()


def _stub_tuesday_downstream(monkeypatch) -> dict:
    """Grade/CLV/retrain, stubbed offline. Nothing here touches the DB, the
    Odds API, or a real ``ml_test.py`` subprocess."""
    import subprocess
    import pipeline_weekly as pw

    seen: dict = {"graded": None, "clv": None, "subprocess": 0}

    def _run_grade(season, week, *a, **k):
        seen["graded"] = (season, week)
        return {"graded": 4, "hit_rate": 0.5, "why": {"recent_miss_reasons": []}}

    def _resolve_clv(season, week, *a, **k):
        seen["clv"] = (season, week)
        return {"resolved": 4, "killcheck": {"verdict": "continue"}}

    class _Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _run(cmd, *a, **k):
        seen["subprocess"] += 1
        return _Completed()

    monkeypatch.setattr(pw, "run_grade", _run_grade)
    monkeypatch.setattr(pw, "resolve_clv", _resolve_clv)
    monkeypatch.setattr(subprocess, "run", _run)
    return seen


def _stub_wed_downstream(monkeypatch) -> dict:
    import pipeline_weekly as pw
    from nflvalue import notify

    seen: dict = {"week": None}

    def _run_week(season, week, *a, **k):
        seen["week"] = (season, week)
        return {"games": ["g1"], "publish": True, "discord": "dry-run"}

    monkeypatch.setattr(pw, "run_week", _run_week)
    monkeypatch.setattr(notify, "resolve_webhook", lambda: None)
    return seen


# --------------------------------------------------------------------------- #
# Tuesday
# --------------------------------------------------------------------------- #
def test_tuesday_refreshes_current_inputs_before_reading_the_slate(monkeypatch):
    feed = FakeFeed()
    _wire(monkeypatch, feed, TUE_NOW)
    _stub_tuesday_downstream(monkeypatch)

    aw.job_tuesday()

    assert feed.calls, "job_tuesday read no inputs at all"
    assert feed.calls[0] == "refresh", (
        "job_tuesday chose a week before establishing current-season inputs; "
        f"call order was {feed.calls}")


def test_tuesday_on_a_cold_runner_does_not_grade_a_2023_week(monkeypatch):
    """The frozen cohort makes 'last completed week' 2023 wk18 -- and Tuesday
    would then write a heartbeat claiming it graded a week."""
    feed = FakeFeed()
    beats = _wire(monkeypatch, feed, TUE_NOW)
    seen = _stub_tuesday_downstream(monkeypatch)

    rc = aw.job_tuesday()

    assert rc == 0
    assert seen["graded"] == (2026, 1), (
        "Tuesday graded a frozen-history week instead of the week that just "
        f"finished: {seen['graded']}")
    assert seen["clv"] == (2026, 1)
    assert beats[-1]["status"] == "active"
    assert "2026 week 1" in beats[-1]["detail"]


def test_tuesday_shared_helper_is_used(monkeypatch):
    feed = FakeFeed()
    _wire(monkeypatch, feed, TUE_NOW)
    _stub_tuesday_downstream(monkeypatch)

    seen: list[str] = []
    real = aw.ensure_current_inputs
    monkeypatch.setattr(aw, "ensure_current_inputs",
                        lambda job: seen.append(job) or real(job))

    aw.job_tuesday()
    assert seen == ["tuesday"]


def test_tuesday_degrades_loudly_when_refresh_is_stale(monkeypatch):
    feed = FakeFeed(stale=True, errors=["schedules: HTTPError 502"])
    beats = _wire(monkeypatch, feed, TUE_NOW)
    _stub_tuesday_downstream(monkeypatch)

    rc = aw.job_tuesday()

    assert rc == 0
    assert beats[-1]["status"] == "degraded"
    assert "HTTPError 502" in beats[-1]["detail"]


def test_tuesday_no_completed_week_no_op_still_works(monkeypatch):
    """Existing behavior preserved: nothing graded yet is still a clean no-op."""
    feed = FakeFeed(with_results=False)
    beats = _wire(monkeypatch, feed, TUE_NOW)
    seen = _stub_tuesday_downstream(monkeypatch)

    rc = aw.job_tuesday()

    assert rc == 0
    assert seen["graded"] is None
    assert seen["subprocess"] == 0, "no retraining without a graded week"
    assert "no week is ready" in beats[-1]["detail"]


# --------------------------------------------------------------------------- #
# Wednesday shares the same helper (one refresh path, not three)
# --------------------------------------------------------------------------- #
def test_wed_uses_the_shared_helper(monkeypatch):
    feed = FakeFeed()
    _wire(monkeypatch, feed, WED_NOW)
    _stub_config(monkeypatch)
    seen_week = _stub_wed_downstream(monkeypatch)

    seen: list[str] = []
    real = aw.ensure_current_inputs
    monkeypatch.setattr(aw, "ensure_current_inputs",
                        lambda job: seen.append(job) or real(job))

    rc = aw.job_wed()

    assert rc == 0
    assert seen == ["wed"], "job_wed still has its own copy of the refresh call"
    assert feed.calls[0] == "refresh"
    assert seen_week["week"] == (2026, 2)


def test_wed_offseason_conclusion_is_degraded_when_refresh_failed(monkeypatch):
    """'No REG week within eight days' is exactly the conclusion a failed
    refresh cannot support, so it must not be reported as healthy."""
    feed = FakeFeed(stale=True, errors=["schedules: name resolution failed"])
    beats = _wire(monkeypatch, feed, WED_NOW)
    _stub_config(monkeypatch)
    _stub_wed_downstream(monkeypatch)

    rc = aw.job_wed()

    assert rc == 0
    assert beats[-1]["status"] == "degraded"
    assert "name resolution failed" in beats[-1]["detail"]


# --------------------------------------------------------------------------- #
# Two-tier honesty, and the deploy heartbeat left alone
# --------------------------------------------------------------------------- #
def test_auxiliary_ingest_warnings_are_reported_without_flipping_status(monkeypatch):
    """A failed NGS pull does not invalidate week selection -- but it is still
    said out loud rather than swallowed."""
    feed = FakeFeed(errors=["ngs/contracts 2026: 404 Not Found"])
    beats = _wire(monkeypatch, feed, T90_NOW)
    _stub_config(monkeypatch)
    _stub_t90_downstream(monkeypatch)

    rc = aw.job_t90()

    assert rc == 0
    assert beats[-1]["status"] == "active"
    assert "404 Not Found" in beats[-1]["detail"]


def test_deploy_heartbeat_behavior_is_unchanged(monkeypatch):
    """A push-triggered deploy must stay a cheap metadata refresh: no ingest,
    same status and wording as before."""
    feed = FakeFeed()
    beats = _wire(monkeypatch, feed, WED_NOW)

    rc = aw.job_deploy()

    assert rc == 0
    assert feed.calls == ["load_slate"], (
        f"job_deploy should not refresh feeds; it called {feed.calls}")
    assert beats == [{"status": "offseason", "job": "deploy",
                      "detail": "Deployment completed without running the betting model."}]
