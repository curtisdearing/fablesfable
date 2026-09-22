"""2026 Week 3: every scheduled game gets its sportsbook lines, and the T-90
board publishes when ESPN's inactives are simply not out yet.

RED against 2a9b763 (main before this change) and, for (c)/(d)/(e), against
the `agent/ff-kickoff-aware-odds` branch it builds on:

(a) the Wednesday run requests lines for EVERY scheduled game, soonest
    kickoff first (config cap was 4 of 16; the future tier was pull-clock
    ordered, so rationing priced the wrong games);
(b) the board prices from the newest stored quote per game, whichever run
    pulled it (``load_recent_lines``), never ``WHERE ts = <this run>``;
(c) T-90 refreshes a game's lines exactly once: the scheduled job's close
    re-snap is honoured, and ``run_t90`` spends its own event-call only when
    no quote that fresh exists;
(d) the credit arithmetic is computed BEFORE the first metered call, logged,
    returned, and enforced -- a Wednesday pull holds the credits for its own
    pre-kick close in the same ledger month;
(e) #27 owner decision (2026-09-22, option 2): an UNPOPULATED ESPN event
    roster no longer holds the board; it publishes with a visible banner.
    A roster that could not be fetched at all still holds it.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue import notify  # noqa: E402
from nflvalue import report as rptmod  # noqa: E402
from nflvalue.sources import availability as av  # noqa: E402
from nflvalue.sources import oddsapi_props as oap  # noqa: E402
from tests.test_pipeline_weekly import GAME_ID, _fresh_feeds, env  # noqa: E402,F401
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
ET = dt.timezone(dt.timedelta(hours=-4))          # EDT, the 2026 Week 3 offset
#: The real Wednesday run: cron "17 14 * 9,10 3" -> 2026-09-23 14:17Z.
WED = dt.datetime(2026, 9, 23, 14, 17, tzinfo=UTC)

#: The real 2026 Week 3 slate (historical/lines_extra.parquet, nflverse ET).
WEEK3 = {
    "2026_03_ATL_GB": ("2026-09-24", "20:15"),
    "2026_03_LAC_BUF": ("2026-09-27", "13:00"), "2026_03_CAR_CLE": ("2026-09-27", "13:00"),
    "2026_03_NYJ_DET": ("2026-09-27", "13:00"), "2026_03_HOU_IND": ("2026-09-27", "13:00"),
    "2026_03_NE_JAX": ("2026-09-27", "13:00"), "2026_03_KC_MIA": ("2026-09-27", "13:00"),
    "2026_03_TEN_NYG": ("2026-09-27", "13:00"), "2026_03_CIN_PIT": ("2026-09-27", "13:00"),
    "2026_03_SEA_WAS": ("2026-09-27", "13:00"),
    "2026_03_ARI_SF": ("2026-09-27", "16:05"), "2026_03_MIN_TB": ("2026-09-27", "16:05"),
    "2026_03_BAL_DAL": ("2026-09-27", "16:25"), "2026_03_LV_NO": ("2026-09-27", "16:25"),
    "2026_03_LA_DEN": ("2026-09-27", "20:20"),
    "2026_03_PHI_CHI": ("2026-09-28", "20:15"),
}
COST = 5.0     # 5 markets x 1 region (config.json prop_markets_internal, books=)


def _kickoffs():
    return {g: dt.datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
            for g, (d, t) in WEEK3.items()}


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


def _cfg(**kw):
    cfg = {"odds_api_key": "k", "regions": "us", "max_prop_games_per_run": 16,
           "odds_budget": {"monthly_credits": 500, "reserve": 50},
           "prop_markets_internal": ["receiving_yards", "receptions", "rushing_yards",
                                     "passing_yards", "anytime_td"]}
    cfg.update(kw)
    return cfg


def _seed(conn, rows):
    dbmod.upsert(conn, "lines", rows,
                 ["ts", "game_id", "book", "market", "player_name", "side"])


def _quote(ts, game_id, side="over", player="J.Smith-Njigba", point=6.5):
    return {"ts": ts, "game_id": game_id, "book": "draftkings", "market": "receptions",
            "player_id": None, "player_name": player, "side": side,
            "point": point, "price": 1.91}


# --------------------------------------------------------------------------- #
# (a) every scheduled game, soonest kickoff first
# --------------------------------------------------------------------------- #
def test_config_cap_admits_a_full_slate():
    cfg = json.loads((ROOT / "config.json").read_text())
    assert int(cfg["max_prop_games_per_run"]) >= 16, (
        "a 16-game week must not be rationed by the per-run cap (was 4: "
        "12 of 16 games rendered NO_MARKET by construction, #26)")


def test_wednesday_order_is_kickoff_order_for_the_real_week3_slate(conn):
    """Pull-clock state from Week 2 must not reorder the week: whatever was
    pulled last, TNF is first and MNF is last."""
    _seed(conn, [_quote("2026-09-20T15:55:00Z", "2026_03_ATL_GB"),      # pulled most recently
                 _quote("2026-09-16T14:17:00Z", "2026_03_PHI_CHI")])    # pulled longest ago
    order = oap.rotation_order(conn, list(WEEK3), kickoffs=_kickoffs(), now=WED)
    assert order[0] == "2026_03_ATL_GB", "Thursday night is pulled first"
    assert order[-1] == "2026_03_PHI_CHI", "Monday night is pulled last"
    assert order[-2] == "2026_03_LA_DEN"
    kos = _kickoffs()
    assert [kos[g] for g in order] == sorted(kos[g] for g in order), "monotone in kickoff"


def test_wednesday_pull_requests_all_sixteen_games(conn):
    calls = []

    def fetch(url, params=None):
        calls.append(url)
        return {"bookmakers": []}

    event_map = {g: f"evt-{i}" for i, g in enumerate(WEEK3)}
    res = oap.pull_week_props(_cfg(), event_map, conn=conn, fetch=fetch,
                              kickoffs=_kickoffs(), now=WED, reserve_close=True)
    assert len(res["pulled"]) == 16 and len(calls) == 16
    assert res["skipped_cap"] == [] and res["skipped_budget"] == []
    assert res["credits_spent"] == 16 * COST


def test_run_week_asks_for_every_game_and_holds_the_close():
    src = inspect.getsource(pw.run_week)
    assert "reserve_close=True" in src
    assert "kickoffs=kickoffs" in src
    assert "WHERE ts=?" not in src, "(b) the board must not read only this run's rows"
    assert "load_recent_lines(conn, game_ids=list(slate[\"game_id\"])" in src


# --------------------------------------------------------------------------- #
# (d) credit arithmetic: computed first, logged, returned, enforced
# --------------------------------------------------------------------------- #
def test_credit_plan_for_a_full_week_within_budget(conn):
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    budget.spend(225.0)                                   # provider-side usage so far
    plan = oap.credit_plan(budget, COST, list(WEEK3), kickoffs=_kickoffs(),
                           reserve_close=True, cap=16)
    assert plan == {
        "month": "2026-09", "cost_per_event": 5.0, "ceiling": 450.0, "used": 225.0,
        "spendable": 225.0, "n_games": 16, "pull_cost": 80.0, "close_reserve": 80.0,
        "needed": 160.0, "affordable_games": 16, "rationed_games": 0,
        "affordable": list(WEEK3), "rationed": [],
    }
    text = oap.plan_text(plan)
    assert "16 game(s) x 5 = 80 to pull + 80 held for pre-kick closes = 160 needed" in text
    assert "225 spendable (225 used of 450) -> 16 affordable, 0 rationed" in text


def test_rationing_prices_the_soonest_kickoffs_and_is_enforced(conn, capsys):
    """100 spendable credits, 16 games at 5 + 5 held each: exactly ten games
    are pulled, they are the ten that kick off first, and the six Sunday-late
    /Monday games are reported -- never fetched."""
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    budget.spend(350.0)
    calls = []

    def fetch(url, params=None):
        calls.append(url)
        return {"bookmakers": []}

    order = oap.rotation_order(conn, list(WEEK3), kickoffs=_kickoffs(), now=WED)
    event_map = {g: f"evt-{g}" for g in order}
    res = oap.pull_week_props(_cfg(), event_map, conn=conn, fetch=fetch, budget=budget,
                              kickoffs=_kickoffs(), now=WED, reserve_close=True)
    assert res["plan"]["spendable"] == 100.0
    assert res["plan"]["affordable_games"] == 10 and res["plan"]["rationed_games"] == 6
    assert res["pulled"] == order[:10] and len(calls) == 10
    assert res["skipped_budget"] == order[10:]
    assert res["credits_spent"] == 50.0 and res["close_reserved"] == 50.0
    assert budget.remaining == 50.0, "the held closes are not spent, only reserved"
    out = capsys.readouterr().out
    assert "[oddsapi] credit plan 2026-09: 16 game(s) x 5 = 80 to pull + 80 held" in out
    assert "-> 10 affordable, 6 rationed" in out


def test_a_close_in_the_next_ledger_month_is_not_held(conn):
    """The Sep 30 Wednesday (Week 4): its closes fall in October's quota."""
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    kos = {"2026_04_A_B": dt.datetime(2026, 10, 4, 13, 0, tzinfo=ET)}
    assert oap.close_reserve_for("2026_04_A_B", COST, kos, "2026-09") == 0.0
    assert oap.close_reserve_for("2026_03_ATL_GB", COST, _kickoffs(), "2026-09") == COST
    assert oap.close_reserve_for("unknown", COST, kos, "2026-09") == COST
    plan = oap.credit_plan(budget, COST, ["2026_04_A_B"], kickoffs=kos, reserve_close=True)
    assert plan["close_reserve"] == 0.0 and plan["needed"] == COST


def test_no_reserve_reproduces_the_plain_budget_check(conn):
    """T-90 and every other caller: no hold, the original per-game check."""
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    budget.spend(443.0)                                   # 7 left: one event, not two
    res = oap.pull_week_props(_cfg(), {"a": "e1", "b": "e2"}, conn=conn, budget=budget,
                              fetch=lambda u, p=None: {"bookmakers": []},
                              kickoffs={"a": WED + dt.timedelta(days=4),
                                        "b": WED + dt.timedelta(days=5)}, now=WED)
    assert res["pulled"] == ["a"] and res["skipped_budget"] == ["b"]
    assert res["plan"]["close_reserve"] == 0.0


# --------------------------------------------------------------------------- #
# (c) T-90 refreshes each game's lines once
# --------------------------------------------------------------------------- #
def _t90_odds_env(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setitem(av.DISPLAY_TO_ABBR, "Team AAA", "AAA")
    monkeypatch.setitem(av.DISPLAY_TO_ABBR, "Team BBB", "BBB")
    # the synthetic slate is dated 2023; a real kickoff would read as started
    monkeypatch.setattr(pw, "slate_kickoffs", lambda slate: {})
    calls = []

    def odds_fetch(url, params=None):
        calls.append(url)
        return {"bookmakers": []}

    events = lambda cfg: [{"id": "evt-1", "home_team": "Team BBB", "away_team": "Team AAA"}]  # noqa: E731
    return calls, odds_fetch, events


def test_t90_does_not_respend_on_a_game_the_job_just_resnapped(env, monkeypatch):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    calls, odds_fetch, events = _t90_odds_env(monkeypatch)
    conn = dbmod.connect()
    _seed(conn, [_quote(now, GAME_ID)])                   # the close, minutes old
    conn.close()
    feeds = dict(_fresh_feeds(now))
    feeds["inactive_rows"] = []
    feeds["inactives_fetched_at"] = now
    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=feeds, odds_fetch=odds_fetch, list_events_fn=events)
    assert calls == [], "a second event-call on the same game buys nothing"
    assert "no credit spent" in (res["line_note"] or "")


def test_t90_pulls_when_no_fresh_quote_exists(env, monkeypatch):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    calls, odds_fetch, events = _t90_odds_env(monkeypatch)
    conn = dbmod.connect()
    _seed(conn, [_quote("2026-09-16T14:17:00Z", GAME_ID)])   # last week's entry only
    conn.close()
    feeds = dict(_fresh_feeds(now))
    feeds["inactive_rows"] = []
    feeds["inactives_fetched_at"] = now
    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=feeds, odds_fetch=odds_fetch, list_events_fn=events)
    assert len(calls) == 1 and "/events/evt-1/odds" in calls[0]
    assert "T-90 odds pull: 1 game(s)" in (res["line_note"] or "")


def test_t90_fresh_window_is_shorter_than_the_stored_quote_window():
    assert 0 < pw.T90_LINE_FRESH_HOURS < oap.MAX_LINE_AGE_HOURS


# --------------------------------------------------------------------------- #
# (e) #27: unpopulated inactives -> banner, not a hold
# --------------------------------------------------------------------------- #
UNPOPULATED_REASON = ("SEA: no entry is marked active (period=0); ESPN has not populated "
                      "this event roster yet")


def _t90_with_inactives(state, reason=""):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                inject_feeds=_fresh_feeds(now))
    feeds = dict(_fresh_feeds(now))
    feeds["inactive_rows"] = []
    feeds["inactives_fetched_at"] = now
    feeds["inactives_state"] = state
    feeds["inactives_reason"] = reason
    return pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=feeds)


def test_unpopulated_event_roster_publishes_with_a_banner(env):
    res = _t90_with_inactives("unpopulated", UNPOPULATED_REASON)
    assert res["publish"] is True, res["publish_reasons"]
    assert res["inactives_state"] == "unpopulated"
    banner = res["inactives_banner"]
    assert banner and banner in res["publish_reasons"]
    assert banner.startswith("inactives: source has not published yet")
    assert UNPOPULATED_REASON in banner
    assert "WITHOUT a game-day inactives check" in banner
    assert res["voided"] == [], "an unpopulated roster must not read as anyone OUT"
    assert any(g["leans"] for g in res["games"]), "the board still carries leans"
    # visible on every surface
    assert "source has not published yet" in Path(res["drop_path"]).read_text()
    md = Path(res["md_path"]).read_text()
    assert "PUBLISHED" in md and "NOT PUBLISHED" not in md
    assert "Feed warnings (published)" in md and "source has not published yet" in md
    msgs = notify.build_messages(res)
    assert "Feed warnings" in msgs[0]["content"] and "source has not published yet" in msgs[0]["content"]


def test_a_roster_that_could_not_be_fetched_still_holds_the_board(env):
    res = _t90_with_inactives("not_fetched", "no ESPN event id resolved for this game")
    assert res["publish"] is False
    assert any(r.startswith("inactives: source could not be fetched") for r in res["publish_reasons"])
    assert res["inactives_banner"] is None


def test_a_populated_roster_is_unchanged(env):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    feeds = dict(_fresh_feeds(now))
    feeds["inactive_rows"] = [
        {"espn_id": "1", "name": "Alpha Wideout", "active": True, "did_not_play": False,
         "starter": True, "team": "AAA"}]
    feeds["inactives_fetched_at"] = now
    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=feeds)
    assert res["publish"] is True and res["inactives_banner"] is None
    assert not [r for r in res["publish_reasons"] if r.startswith("inactives:")]


def test_run_t90_never_looks_up_an_event_id_when_the_feed_is_injected(env, monkeypatch):
    """An injected EMPTY list is still an injected feed: the offline suite
    must never reach the ESPN scoreboard."""
    monkeypatch.setattr(av, "find_event_ids",
                        lambda games: (_ for _ in ()).throw(AssertionError("live ESPN call")))
    _t90_with_inactives("unpopulated", UNPOPULATED_REASON)


# --------------------------------------------------------------------------- #
# ESPN LAR/WSH vs nflverse LA/WAS: the Rams and Commanders resolve an event id
# --------------------------------------------------------------------------- #
def test_espn_abbreviations_bridge_to_nflverse():
    assert av.canonical_abbr("LAR") == "LA"
    assert av.canonical_abbr("WSH") == "WAS"
    assert av.canonical_abbr("LA") == "LA" and av.canonical_abbr("WAS") == "WAS"
    assert av.canonical_abbr("den") == "DEN"


def test_find_event_ids_matches_week3_rams_and_commanders(monkeypatch):
    board = {"events": [
        {"id": "401", "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"abbreviation": "DEN"}},
            {"homeAway": "away", "team": {"abbreviation": "LAR"}}]}]},
        {"id": "402", "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"abbreviation": "WSH"}},
            {"homeAway": "away", "team": {"abbreviation": "SEA"}}]}]},
    ]}
    monkeypatch.setattr(av, "get_json", lambda url, params=None: board)
    got = av.find_event_ids([
        {"game_id": "2026_03_LA_DEN", "gameday": "2026-09-27", "home_team": "DEN", "away_team": "LA"},
        {"game_id": "2026_03_SEA_WAS", "gameday": "2026-09-27", "home_team": "WAS", "away_team": "SEA"},
    ])
    assert got == {"2026_03_LA_DEN": "401", "2026_03_SEA_WAS": "402"}


# --------------------------------------------------------------------------- #
# Published-with-warnings is printed in the markdown too
# --------------------------------------------------------------------------- #
def test_markdown_prints_warnings_on_a_published_board():
    md = rptmod.render_markdown(2026, 3, [], {}, "2026-09-27T16:00:00Z", "t90",
                                publish=True, publish_reasons=["inactives: source has not published yet (x)"])
    assert "Feed warnings (published)" in md and "NOT PUBLISHED" not in md
    quiet = rptmod.render_markdown(2026, 3, [], {}, "2026-09-27T16:00:00Z", "t90")
    assert "Feed warnings" not in quiet
