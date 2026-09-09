"""The odds rotation must not let a game reach kickoff unpriced.

RED against f6a0ff7 (the 2026-09-08 accuracy release):
``rotation_order`` sorted purely by ``(last_pull_ts, game_id)`` and the
pipeline priced a board from ``SELECT * FROM lines WHERE ts = <this run>``.
Together those produced the 2026-09-09 failure: the Wednesday run spent all
four of its event-calls on Sunday games, the Wednesday-night opener was never
pulled, and the quotes it DID have (from the run 29 hours earlier) were
invisible. The opener published with `market_state: NO_MARKET` on all five
of its leans.
"""

from __future__ import annotations

import datetime as dt

import pytest

from nflvalue import db as dbmod
from nflvalue.sources import oddsapi_props as oap

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 9, 9, 14, 17, tzinfo=UTC)      # the real wed run, 10:17 ET

#: The real 2026 Week 1 slate, in the order the old rotation produced.
SUNDAY = dt.datetime(2026, 9, 13, 17, 0, tzinfo=UTC)
OPENER = dt.datetime(2026, 9, 10, 0, 15, tzinfo=UTC)   # Wed 20:15 ET


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


def _seed(conn, rows):
    dbmod.upsert(conn, "lines", rows,
                 ["ts", "game_id", "book", "market", "player_name", "side"])


def _quote(ts, game_id, side="over", book="draftkings", player="J.Smith-Njigba",
           market="receptions", point=6.5, price=1.91):
    return {"ts": ts, "game_id": game_id, "book": book, "market": market,
            "player_id": None, "player_name": player, "side": side,
            "point": point, "price": price}


# --------------------------------------------------------------------------- #
# rotation_order
# --------------------------------------------------------------------------- #
def test_imminent_kickoff_outranks_the_rotation_clock(conn):
    """The defect, reproduced exactly: four Sunday games pulled longest ago,
    one game kicking off tonight, a cap of four. The opener must come first."""
    _seed(conn, [_quote("2026-09-07T22:16:00Z", g)
                 for g in ["2026_01_ARI_LAC", "2026_01_ATL_PIT",
                           "2026_01_BAL_IND", "2026_01_BUF_HOU"]])
    _seed(conn, [_quote("2026-09-08T17:56:00Z", "2026_01_NE_SEA")])

    game_ids = ["2026_01_ARI_LAC", "2026_01_ATL_PIT", "2026_01_BAL_IND",
                "2026_01_BUF_HOU", "2026_01_NE_SEA"]
    kickoffs = {g: SUNDAY for g in game_ids}
    kickoffs["2026_01_NE_SEA"] = OPENER

    order = oap.rotation_order(conn, game_ids, kickoffs=kickoffs, now=NOW)
    assert order[0] == "2026_01_NE_SEA", (
        "the game kicking off tonight must be pulled first; the old order put "
        "it last because it was pulled most recently")
    # and the rest keep the original round-robin among themselves
    assert order[1:] == ["2026_01_ARI_LAC", "2026_01_ATL_PIT",
                         "2026_01_BAL_IND", "2026_01_BUF_HOU"]


def test_two_imminent_games_are_ordered_soonest_first(conn):
    game_ids = ["a", "b", "c"]
    kickoffs = {"a": NOW + dt.timedelta(hours=10),
                "b": NOW + dt.timedelta(hours=3),
                "c": NOW + dt.timedelta(days=4)}
    assert oap.rotation_order(conn, game_ids, kickoffs=kickoffs, now=NOW)[:2] == ["b", "a"]


def test_started_games_sort_last_and_are_never_pulled(conn):
    game_ids = ["done", "soon"]
    kickoffs = {"done": NOW - dt.timedelta(minutes=1),
                "soon": NOW + dt.timedelta(hours=2)}
    assert oap.rotation_order(conn, game_ids, kickoffs=kickoffs, now=NOW) == ["soon", "done"]
    assert oap.started_games(game_ids, kickoffs, now=NOW) == ["done"]


def test_unknown_kickoff_falls_back_to_the_rotation_clock(conn):
    _seed(conn, [_quote("2026-09-08T00:00:00Z", "known")])
    order = oap.rotation_order(conn, ["known", "unknown"],
                               kickoffs={"known": SUNDAY}, now=NOW)
    assert order == ["unknown", "known"]      # never-pulled still sorts first


def test_no_kickoffs_reproduces_the_original_order(conn):
    """Back-compat: every existing caller passes no kickoffs and must be
    byte-identical to the pre-fix behaviour."""
    _seed(conn, [_quote("2026-09-08T00:00:00Z", "b")])
    assert oap.rotation_order(conn, ["c", "b", "a"]) == ["a", "c", "b"]


# --------------------------------------------------------------------------- #
# pull_week_props: the cap must not be spent on a game already under way
# --------------------------------------------------------------------------- #
def test_pull_skips_games_already_under_way(conn, monkeypatch):
    calls = []

    def fake_fetch(url, params):
        calls.append(url)
        return {"bookmakers": []}

    cfg = {"odds_api_key": "k", "max_prop_games_per_run": 4,
           "prop_markets_internal": ["receptions"], "regions": "us"}
    event_map = {"started": "e1", "live": "e2"}
    kickoffs = {"started": NOW - dt.timedelta(hours=1),
                "live": NOW + dt.timedelta(hours=2)}

    res = oap.pull_week_props(cfg, event_map, conn=conn, fetch=fake_fetch,
                              kickoffs=kickoffs, now=NOW)
    assert res["pulled"] == ["live"]
    assert res["skipped_started"] == ["started"]
    assert len(calls) == 1, "a credit must not be spent on a game in progress"


# --------------------------------------------------------------------------- #
# load_recent_lines: an earlier pull must still price the board
# --------------------------------------------------------------------------- #
def test_earlier_pull_is_still_visible_to_a_later_run(conn):
    """The 2026-09-09 case: NE@SEA quotes stored 29 hours earlier."""
    _seed(conn, [_quote("2026-09-08T17:56:00Z", "2026_01_NE_SEA", side="over"),
                 _quote("2026-09-08T17:56:00Z", "2026_01_NE_SEA", side="under")])
    rows = oap.load_recent_lines(conn, game_ids=["2026_01_NE_SEA"], now=NOW)
    assert len(rows) == 2, "quotes from an earlier run must still price the board"


def test_only_the_newest_quote_per_key_is_returned(conn):
    _seed(conn, [_quote("2026-09-08T17:56:00Z", "g", point=6.5),
                 _quote("2026-09-09T14:00:00Z", "g", point=7.5)])
    rows = oap.load_recent_lines(conn, game_ids=["g"], now=NOW)
    assert [r["point"] for r in rows] == [7.5], "a superseded quote must not resurface"


def test_quotes_older_than_the_window_are_dropped(conn):
    _seed(conn, [_quote("2026-09-01T00:00:00Z", "stale"),
                 _quote("2026-09-09T00:00:00Z", "fresh")])
    got = {r["game_id"] for r in oap.load_recent_lines(conn, now=NOW)}
    assert got == {"fresh"}, "a week-old quote must not be priced as current"
