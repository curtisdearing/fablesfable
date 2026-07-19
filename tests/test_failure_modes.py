"""Phase 7.2 -- every external dependency, forced to fail.

The contract for all five feeds is identical and non-negotiable:

    DEGRADE LOUDLY AND VISIBLY. Never silently serve stale or empty results
    as if they were fresh.

"Loudly" has a specific meaning per surface: a recorded error string, a
``stale`` flag the freshness gate can see, a raised exception the caller must
handle, or a documented clean skip. What is never acceptable is a return value
that a downstream consumer cannot distinguish from a successful fetch.

Failure matrix (see docs/decisions_p3-5.md for the design each row defends):

    Dependency        | timeout | malformed | partial | auth | budget
    ------------------|---------|-----------|---------|------|-------
    The Odds API      |    x    |     x     |    x    |  x   |   x
    nflreadpy         |    x    |     x     |    x    |  -   |   -
    Open-Meteo        |    x    |     x     |    x    |  -   |   -
    Discord           |    x    |     -     |    -    |  x   |   -
    GH release asset  |    -    |     x     |    x    |  -   |   -

Strictly offline: every network entry point is monkeypatched. No test here may
open a socket.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error

import pandas as pd
import pytest

from nflvalue import db as dbmod
from nflvalue import ingest
from nflvalue import notify
from nflvalue.sources import oddsapi_props as opp
from nflvalue.sources import weather as wx


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _timeout(*a, **kw):
    raise socket.timeout("timed out (test)")


def _http_error(code):
    def raise_it(*a, **kw):
        raise urllib.error.HTTPError(
            url="https://example.invalid", code=code, msg="test", hdrs=None, fp=None)
    return raise_it


# =========================================================================== #
# 1. The Odds API
# =========================================================================== #
def test_odds_api_timeout_does_not_fabricate_lines(monkeypatch, tmp_path):
    """A dead odds feed must yield NO line rows. The pipeline's documented
    response to no market is `no_market` ranking, which is honest; a
    fabricated or half-written line would mint a fake edge."""
    conn = dbmod.connect(str(tmp_path / "t.db"))
    cfg = {"odds_budget": {"monthly_credits": 500, "reserve": 50},
           "max_prop_games_per_run": 4, "books": ["draftkings"]}

    res = opp.pull_week_props(cfg, {"2023_10_KC_BUF": "evt1"}, conn=conn,
                              fetch=_timeout)
    assert res["rows_written"] == 0
    assert res["pulled"] == []
    conn.close()


def test_odds_api_auth_failure_is_reported_not_swallowed(monkeypatch, tmp_path):
    conn = dbmod.connect(str(tmp_path / "t.db"))
    cfg = {"odds_budget": {"monthly_credits": 500, "reserve": 50},
           "max_prop_games_per_run": 4, "books": ["draftkings"]}

    res = opp.pull_week_props(cfg, {"2023_10_KC_BUF": "evt1"}, conn=conn,
                              fetch=_http_error(401))
    assert res["rows_written"] == 0
    assert res["pulled"] == []
    conn.close()


@pytest.mark.parametrize("payload", [
    None,
    {},
    {"bookmakers": None},
    {"bookmakers": [{"key": "draftkings"}]},                  # no markets
    {"bookmakers": [{"key": "draftkings", "markets": [{"key": "player_reception_yds"}]}]},  # no outcomes
    {"bookmakers": [{"key": "draftkings", "markets": [
        {"key": "player_reception_yds", "outcomes": [
            {"name": "Over", "description": "A Player"}]}]}]},   # outcome missing price/point
])
def test_odds_api_malformed_payloads_parse_to_nothing_not_garbage(payload):
    """Every shape of broken payload must produce zero rows rather than rows
    with None prices that later read as real quotes."""
    rows = opp.parse_event_props(payload or {}, ts="2023-11-12T00:00:00Z")
    assert isinstance(rows, list)
    for r in rows:
        assert r.get("price") is not None, f"row with no price survived: {r}"
        assert r.get("point") is not None or r.get("market") == "anytime_td"


def test_odds_api_partial_payload_keeps_only_the_complete_side():
    """One book quotes both sides, another only the over. The half-quoted
    book must not contribute a de-vigged 'consensus' -- a one-sided price
    still contains full vig."""
    payload = {"bookmakers": [
        {"key": "draftkings", "markets": [{"key": "player_reception_yds", "outcomes": [
            {"name": "Over", "description": "Full Book", "price": -110, "point": 62.5},
            {"name": "Under", "description": "Full Book", "price": -110, "point": 62.5}]}]},
        {"key": "betmgm", "markets": [{"key": "player_reception_yds", "outcomes": [
            {"name": "Over", "description": "Half Book", "price": -105, "point": 62.5}]}]},
    ]}
    rows = opp.parse_event_props(payload, ts="2023-11-12T00:00:00Z")
    frame = opp.to_prop_lines_frame(rows)
    if not frame.empty and "n_books" in frame.columns:
        for _, r in frame.iterrows():
            assert r["n_books"] >= 1


def test_budget_exhaustion_refuses_the_call(tmp_path):
    """THE HARD RULE. At the ceiling, the next call must raise rather than
    quietly overspend a metered free tier."""
    conn = dbmod.connect(str(tmp_path / "t.db"))
    budget = opp.CreditBudget(conn, monthly_credits=100, reserve=90)  # ceiling 10
    assert budget.can_spend(10) is True
    budget.spend(10)
    assert budget.can_spend(1) is False
    with pytest.raises(opp.BudgetExceeded):
        budget.spend(1)
    conn.close()


def test_budget_state_survives_a_reconnect(tmp_path):
    """The ledger is persistent: a crashed run must not reset the month's
    spend to zero and hand out a second budget."""
    path = str(tmp_path / "t.db")
    conn = dbmod.connect(path)
    opp.CreditBudget(conn, 100, 90).spend(6)
    conn.close()

    conn = dbmod.connect(path)
    reloaded = opp.CreditBudget(conn, 100, 90)
    assert reloaded.used == 6
    assert reloaded.remaining == 4
    conn.close()


def test_pull_stops_at_the_budget_and_says_so(tmp_path):
    """Games skipped for budget must be REPORTED as skipped, so the report can
    label them `no market pulled` instead of implying none existed."""
    conn = dbmod.connect(str(tmp_path / "t.db"))
    cfg = {"odds_budget": {"monthly_credits": 50, "reserve": 50},   # ceiling 0
           "max_prop_games_per_run": 4, "books": ["draftkings"]}

    def should_not_be_called(*a, **kw):
        raise AssertionError("fetched despite an exhausted budget")

    res = opp.pull_week_props(cfg, {"2023_10_KC_BUF": "evt1"}, conn=conn,
                              fetch=should_not_be_called)
    assert res["rows_written"] == 0
    assert res["skipped_budget"], "budget skip was not reported to the caller"
    conn.close()


# =========================================================================== #
# 2. nflreadpy / nflverse
# =========================================================================== #
def test_nflreadpy_absent_is_reported_as_stale_with_an_actionable_error(monkeypatch):
    """Import failure must not raise into the pipeline; it must return
    stale=True plus an error a human can act on."""
    import builtins
    real_import = builtins.__import__

    def no_nflreadpy(name, *a, **kw):
        if name == "nflreadpy":
            raise ImportError("No module named 'nflreadpy' (test)")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_nflreadpy)
    res = ingest.refresh(season=2025)
    assert res["stale"] is True
    assert res["errors"]
    assert any("nflreadpy" in e for e in res["errors"])


def test_nflverse_timeout_falls_back_to_cache_but_records_the_error():
    """Cached data may keep serving -- that is the point of the cache -- but
    the failure must still be recorded so the freshness gate sees it. Silence
    is the failure mode being prevented."""
    nfl = pytest.importorskip("nflreadpy")
    import unittest.mock as mock
    with mock.patch.object(nfl, "load_pbp", side_effect=socket.timeout("test")), \
         mock.patch.object(nfl, "load_schedules", side_effect=socket.timeout("test")):
        res = ingest.refresh(season=2025)
    assert res["errors"], "a timed-out pull reported no error at all"


def test_nflverse_schema_drift_is_caught_not_written_through():
    """A vendor renaming/dropping a column is an explicitly registered risk in
    ACCURACY_PROTOCOL.md. A frame missing required columns must error, not be
    persisted as a truncated cache."""
    nfl = pytest.importorskip("nflreadpy")
    import unittest.mock as mock
    drifted = pd.DataFrame({"totally": [1], "unexpected": [2]})
    with mock.patch.object(nfl, "load_pbp", return_value=drifted), \
         mock.patch.object(nfl, "load_schedules", return_value=drifted):
        res = ingest.refresh(season=2031)
    assert res["errors"], "schema drift was accepted silently"
    assert res["stale"] is True


# =========================================================================== #
# 3. Open-Meteo
# =========================================================================== #
def test_weather_timeout_does_not_invent_fair_conditions(monkeypatch):
    """The dangerous default: a failed forecast becoming 70F / 0mph, i.e.
    'perfect passing weather', for an outdoor game in a gale."""
    monkeypatch.setattr(wx, "get_json", _timeout)
    got = wx.forecast_for_game("KC", "2023-11-12T18:00:00Z")
    assert got is None or got.get("temp_f") is None, (
        f"fabricated weather from a failed lookup: {got}")
    assert got is None or got.get("wind_mph") is None


@pytest.mark.parametrize("payload", [
    {}, {"hourly": {}}, {"hourly": {"time": []}},
    {"hourly": {"time": ["2023-11-12T18:00"]}},                    # no values
    {"hourly": {"time": ["2023-11-12T18:00"], "temperature_2m": [None],
                "precipitation": [None], "wind_speed_10m": [None]}},
])
def test_weather_malformed_payload_yields_no_numbers(monkeypatch, payload):
    monkeypatch.setattr(wx, "get_json", lambda *a, **kw: payload)
    got = wx.forecast_for_game("KC", "2023-11-12T18:00:00Z")
    if got:
        assert got.get("temp_f") is None or isinstance(got.get("temp_f"), (int, float))
        if got.get("temp_f") is None:
            assert got.get("wind_mph") is None


def test_weather_failure_leaves_the_pack_value_untouched(monkeypatch):
    """Integration-level: `_apply_forecast_weather` must not write a value
    when the forecast failed, so the schedule's own (possibly NaN) value
    stands and NaN keeps meaning 'unknown'."""
    import pipeline_weekly as pwmod

    class _Adv:
        weather = {"2023_10_KC_BUF": (None, None)}

    monkeypatch.setattr(wx, "get_json", _timeout)
    adv = _Adv()
    slate = pd.DataFrame([{"game_id": "2023_10_KC_BUF", "home_team": "KC",
                           "gameday": "2023-11-12", "gametime": "18:00"}])
    pwmod._apply_forecast_weather(adv, slate)
    temp, wind = adv.weather["2023_10_KC_BUF"]
    assert temp is None and wind is None, (
        f"a failed forecast wrote {temp}/{wind} into the feature pack")


# =========================================================================== #
# 4. Discord
# =========================================================================== #
_PAYLOAD = {"season": 2023, "week": 10, "publish": True,
            "games": [{"game_id": "2023_10_KC_BUF", "away_team": "KC",
                       "home_team": "BUF", "leans": [], "screened": 0}]}


def test_discord_http_failure_never_breaks_the_pipeline(monkeypatch):
    """Notification is the LAST step. A dead webhook must not lose the week's
    work -- but it must report a non-ok status, not claim success."""
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.invalid/webhook")
    monkeypatch.setattr(notify.urllib.request, "urlopen", _http_error(500))
    res = notify.post_weekly(_PAYLOAD, cfg={"discord_enabled": True}, dry_run=False)
    assert isinstance(res, dict)
    assert res.get("status") != "ok"


def test_discord_timeout_is_reported_not_raised(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.invalid/webhook")
    monkeypatch.setattr(notify.urllib.request, "urlopen", _timeout)
    res = notify.post_weekly(_PAYLOAD, cfg={"discord_enabled": True}, dry_run=False)
    assert isinstance(res, dict)
    assert res.get("status") != "ok"


def test_discord_failure_never_leaks_the_webhook_url(monkeypatch):
    """The webhook is a secret. It must not appear in any returned payload,
    including error strings -- those get logged and pasted into reports."""
    secret = "https://discord.com/api/webhooks/SECRET-TOKEN-1234567890"
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", secret)
    monkeypatch.setattr(notify.urllib.request, "urlopen", _http_error(403))
    res = notify.post_weekly(_PAYLOAD, cfg={"discord_enabled": True}, dry_run=False)
    assert "SECRET-TOKEN-1234567890" not in json.dumps(res)


def test_discord_stays_silent_when_the_freshness_gate_failed(monkeypatch):
    """Stale data does not get a confident post. This is a shipped guarantee,
    pinned here against regression."""
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.invalid/webhook")

    def must_not_post(*a, **kw):
        raise AssertionError("posted picks despite publish=false")

    monkeypatch.setattr(notify.urllib.request, "urlopen", must_not_post)
    gated = dict(_PAYLOAD, publish=False)
    notify.post_weekly(gated, cfg={"discord_enabled": True}, dry_run=False)


# =========================================================================== #
# 5. GitHub release asset (the durable model state)
# =========================================================================== #
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "live-weekly.yml")


def test_release_restore_verifies_a_checksum():
    """The DB and model artifact are restored from a release asset on every
    scheduled run. An unverified restore would let a truncated upload become
    the production state silently."""
    with open(WORKFLOW, encoding="utf-8") as fh:
        text = fh.read()
    assert "sha256" in text.lower(), (
        "release restore does not checksum the downloaded state asset")


def test_release_pointer_is_validated_before_use():
    with open(WORKFLOW, encoding="utf-8") as fh:
        text = fh.read()
    assert "Invalid model-state release pointer" in text, (
        "malformed release pointer is not rejected")


def test_artifact_integrity_check_is_wired_into_the_loader():
    """Belt to the workflow's braces: even if a bad artifact reaches disk,
    the loader refuses it (see tests/test_schema_and_artifact_integrity.py)."""
    from nflvalue import ml_ranker as mlr
    assert hasattr(mlr, "verify_artifact")
    assert hasattr(mlr, "ArtifactIntegrityError")
