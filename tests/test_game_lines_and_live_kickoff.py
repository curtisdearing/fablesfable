"""Follow-up to 847590e (2026-10-06): every game priced with REAL book GAME lines,
live cards refused after kickoff, and an unknown provider billing cycle.

Reproduced on 847590e:
  (A) the deployed weekly path (auto_weekly -> run_week/run_t90) never acquired
      game lines: ``oddsapi.fetch_game_odds`` is reached only by the legacy
      ``live.build_live_slate`` (weekly.py, nflvalue/pipeline.py), unbudgeted,
      with no quota preflight and no book clocks kept;
  (B) a live T-90 run after kickoff turned a stored pregame quote into an
      executable (``watch``) card: no layer compared kickoff to the decision clock;
  (C) a close in the next CALENDAR month was not held, and holds lapsed at the
      month boundary -- an invented quota renewal.
Payloads are SYNTHETIC fixtures, not empirical odds.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import config as cfgmod  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue import pick_cards as pc  # noqa: E402
from nflvalue.config import prop_markets_external  # noqa: E402
from nflvalue.sources import availability as avmod  # noqa: E402
from nflvalue.sources import oddsapi_games as og  # noqa: E402
from nflvalue.sources import oddsapi_props as oap  # noqa: E402
from tests.test_odds_latest_board_coverage import (  # noqa: E402
    ABBR_TO_DISPLAY, CFG, NOW, UTC, WEEK5, _fetch, _payload, _slate, _ts)
from tests.test_qb_starter_gate_t90 import _feeds as _qb_feeds, _inputs as _qb_inputs  # noqa: E402
from tests.test_qb_starter_gate_t90 import LINES as QB_LINES, QB_MARKETS  # noqa: E402
from tests.test_eval_fixes import _feeds, env  # noqa: E402,F401
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402

GAME = f"{SEASON}_09_AAA_BBB"
PROP_COST = float(len(prop_markets_external(CFG)))


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "games.db"))
    yield c
    c.close()


def _week5():
    slate = _slate([(f"2026_05_{a}_{h}", d, t, h, a) for a, h, d, t in WEEK5])
    return slate, pw.slate_kickoffs(slate)


def _event(gid, home, away, ko, markets=("h2h", "spreads", "totals")):
    H, A = ABBR_TO_DISPLAY.get(home, home), ABBR_TO_DISPLAY.get(away, away)
    mk = {"h2h": {"key": "h2h", "last_update": "2026-10-07T15:58:00Z",
                  "outcomes": [{"name": H, "price": 1.80}, {"name": A, "price": 2.05}]},
          "spreads": {"key": "spreads", "last_update": "2026-10-07T15:57:00Z",
                      "outcomes": [{"name": H, "price": 1.91, "point": -2.5},
                                   {"name": A, "price": 1.91, "point": 2.5}]},
          "totals": {"key": "totals", "last_update": "2026-10-07T15:56:00Z",
                     "outcomes": [{"name": "Over", "price": 1.91, "point": 44.5},
                                  {"name": "Under", "price": 1.91, "point": 44.5}]}}
    return {"id": f"ev_{home}", "home_team": H, "away_team": A,
            "commence_time": ko.astimezone(UTC).isoformat(),
            "bookmakers": [{"key": b, "markets": [mk[m] for m in markets]} for b in CFG["books"]]}


def _identity(slate, kick):
    listing = [{k: v for k, v in _event(g.game_id, g.home_team, g.away_team, kick[g.game_id]).items()
                if k != "bookmakers"} for g in slate.itertuples(index=False)]
    identity: dict = {}
    pw.build_event_map(CFG, slate, list_events_fn=lambda cfg: listing, details=identity)
    return identity


def _bulk(events, calls):
    def fetch(url, params=None):
        calls.append((url, dict(params or {})))
        return json.loads(json.dumps(events)), {"x-requests-last": "3"}
    return fetch


# --------------------------------------------------------------------------- #
# (A) one budgeted bulk call puts real game lines on every listed game
# --------------------------------------------------------------------------- #
def test_one_bulk_call_prices_every_listed_game_and_keeps_both_clocks(conn):
    slate, kick = _week5()
    events = [_event(g.game_id, g.home_team, g.away_team, kick[g.game_id],
                     markets=("h2h",) if g.home_team == "ARI" else ("h2h", "spreads", "totals"))
              for g in slate.itertuples(index=False) if g.home_team != "WAS"]
    events.append({"id": "ev_next_week", "home_team": "x", "away_team": "y", "bookmakers": []})
    calls: list = []
    res = og.pull_game_lines(CFG, _identity(slate, kick), conn=conn, fetch=_bulk(events, calls),
                             kickoffs=kick, now=NOW, ts=_ts(NOW))
    assert len(calls) == 1 and calls[0][0].endswith("/sports/americanfootball_nfl/odds")
    assert calls[0][1]["markets"] == "h2h,spreads,totals" and res["credits_planned"] == 3.0
    assert res["credits_billed"] == 3.0 and res["unmatched_events"] == 1
    cov = og.game_line_coverage(list(slate["game_id"]), CFG, pull=res, conn=conn, now=NOW)
    g = cov["games"]
    assert set(g) == set(slate["game_id"]) and cov["summary"]["priced"] == 14
    assert g["2026_05_NYG_WAS"]["state"] == "omitted_by_provider"
    ari = g["2026_05_DET_ARI"]
    assert ari["state"] == "priced" and ari["markets_missing"] == ["spreads", "totals"]
    dal = g["2026_05_TB_DAL"]
    assert dal["capture_clock"] == _ts(NOW) and dal["book_clock_max"] == "2026-10-07T15:58:00Z"
    assert dal["books_by_market"]["totals"] == sorted(CFG["books"])
    # a re-run inside the fresh window reuses the answer: no second metered call
    again = og.pull_game_lines(CFG, _identity(slate, kick), conn=conn, fetch=_bulk(events, calls),
                               kickoffs=kick, now=NOW + dt.timedelta(minutes=20))
    assert again["reused"] and len(calls) == 1


def test_a_game_under_way_is_never_stored_as_a_pregame_line(conn):
    slate, kick = _week5()
    sunday = dt.datetime(2026, 10, 11, 17, 30, tzinfo=UTC)          # 1pm ET games under way
    events = [_event(g.game_id, g.home_team, g.away_team, kick[g.game_id])
              for g in slate.itertuples(index=False)]
    res = og.pull_game_lines(CFG, _identity(slate, kick), conn=conn, fetch=_bulk(events, []),
                             kickoffs=kick, now=sunday, ts=_ts(sunday))
    assert "2026_05_CHI_GB" in res["skipped_started"] and "2026_05_TB_DAL" in res["skipped_started"]
    assert og.load_game_lines(conn, ["2026_05_CHI_GB"], now=sunday) == []
    cov = og.game_line_coverage(["2026_05_CHI_GB", "2026_05_BUF_LA"], CFG, pull=res, conn=conn, now=sunday)
    assert cov["games"]["2026_05_CHI_GB"]["state"] == "started"
    assert cov["games"]["2026_05_BUF_LA"]["state"] == "priced"


@pytest.mark.parametrize("headers", [
    {}, {"x-requests-used": "nan", "x-requests-remaining": "100", "x-requests-last": "0"},
    {"x-requests-used": "-1", "x-requests-remaining": "100", "x-requests-last": "0"},
    {"x-requests-used": "10", "x-requests-remaining": "inf", "x-requests-last": "0"}])
def test_an_unverified_quota_refuses_with_zero_metered_calls(conn, headers):
    slate, kick = _week5()
    calls: list = []
    res = og.pull_game_lines(CFG, _identity(slate, kick), conn=conn, fetch=_bulk([], calls),
                             quota_fetch=lambda url, params=None: ([], headers), kickoffs=kick, now=NOW)
    assert calls == [] and res["refused"].startswith("provider quota unverified")
    assert not res["called"] and res["credits_planned"] == 0.0


def test_credits_held_for_prop_closes_are_never_spent_on_game_lines(conn):
    slate, kick = _week5()
    budget = oap.CreditBudget(conn, 500, 50, month=NOW.strftime("%Y-%m"))
    budget.used = budget.ceiling - 2 * PROP_COST                 # one prop entry + its close hold
    oap.pull_week_props(CFG, {"2026_05_BUF_LA": "ev_LA"}, conn=conn, budget=budget,
                        fetch=_fetch({"ev_LA": _payload(["draftkings"])}), kickoffs=kick, now=NOW,
                        reserve_close=True, ts=_ts(NOW))
    assert budget.remaining == PROP_COST                        # exactly the held close is left
    calls: list = []
    res = og.pull_game_lines(CFG, _identity(slate, kick), conn=conn, budget=budget,
                             fetch=_bulk([], calls), kickoffs=kick, now=NOW)
    assert calls == [] and res["credits_held_for_closes"] == PROP_COST and res["skipped_budget"]


# --------------------------------------------------------------------------- #
# (A) native caller: run_week acquires, receipts and reports game lines
# --------------------------------------------------------------------------- #
def _run_week_with_game_lines(monkeypatch, price):
    monkeypatch.setitem(cfgmod.DEFAULT_CONFIG, "odds_api_key", "test")
    monkeypatch.setenv("ODDS_API_KEY", "test")
    monkeypatch.setitem(avmod.DISPLAY_TO_ABBR, "Alpha Ants", "AAA")
    monkeypatch.setitem(avmod.DISPLAY_TO_ABBR, "Bravo Bees", "BBB")
    ko = dt.datetime.now(UTC) + dt.timedelta(days=2)       # the 2023 slate, scheduled ahead
    monkeypatch.setattr(pw, "slate_kickoffs", lambda slate: {g: ko for g in slate["game_id"]})
    ev = _event(GAME, "Bravo Bees", "Alpha Ants", ko)
    ev["id"] = "e1"
    for bk in ev["bookmakers"]:
        bk["markets"][0]["outcomes"][0]["price"] = price
    calls: list = []
    from nflvalue.freshness import stamp_now
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_feeds(stamp_now()), live_odds=True,
                      odds_fetch=_fetch({"e1": {"bookmakers": []}}),
                      game_odds_fetch=_bulk([ev], calls),
                      list_events_fn=lambda cfg: [{"id": "e1", "home_team": "Bravo Bees",
                                                   "away_team": "Alpha Ants"}])
    return res, calls


def test_run_week_prices_every_game_with_game_lines_and_the_forecast_ignores_them(env, monkeypatch):
    a, calls_a = _run_week_with_game_lines(monkeypatch, 1.50)
    row = a["game_lines"]["games"][GAME]
    assert len(calls_a) == 1 and row["state"] == "priced"
    assert a["factor_receipt"]["game_lines"]["games"][GAME]["state"] == "priced"
    assert "Game lines: 1/1 game(s)" in Path(a["md_path"]).read_text()
    conn = dbmod.connect()
    conn.execute("DELETE FROM game_line_pulls")              # force a second, different answer
    conn.commit()
    conn.close()
    b, _ = _run_week_with_game_lines(monkeypatch, 3.10)
    proj = lambda r: sorted((l["player_id"], l["market"], l["mean"], l["sd"])  # noqa: E731
                            for g in r["games"] for l in g["leans"])
    assert proj(a) and proj(a) == proj(b), "changing the book's game odds moved the forecast"


# --------------------------------------------------------------------------- #
# (B) the exact live path: T-90 after kickoff -> no executable card
# --------------------------------------------------------------------------- #
def _qb_t90(monkeypatch, minutes_to_kickoff):
    real = cfgmod.load_config
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {**real(*a, **k),
                                                                 "odds_api_key": "TEST-DUMMY"})
    now = dt.datetime.now(UTC)
    ko = now + dt.timedelta(minutes=minutes_to_kickoff)
    monkeypatch.setattr(pw, "slate_kickoffs", lambda slate: {g: ko for g in slate["game_id"]})
    quote_ts = (now - dt.timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = dbmod.connect()
    conn.execute("DELETE FROM lines")
    for m, pt in QB_LINES.items():
        for side in ("over", "under"):
            conn.execute("INSERT INTO lines VALUES (?,?,?,?,?,?,?,?,?)",
                         (quote_ts, GAME, "draftkings", m, None, "Bravo Quarterback", side, pt, 1.91))
    conn.commit()
    conn.close()

    def refuse(*a, **k):
        raise AssertionError("no odds acquisition in this test")
    res = pw.run_t90(SEASON, WEEK, GAME, mode="live", inputs=_qb_inputs(True),
                     inject_feeds=_qb_feeds(None), odds_fetch=refuse, list_events_fn=lambda cfg: [])
    conn = dbmod.connect()
    cards = [c for c in pc.week_cards(conn, SEASON, WEEK)["cards"]
             if c["player_id"] == "QB_B" and c["market"] in QB_MARKETS]
    kept = conn.execute("SELECT COUNT(*) FROM lines WHERE game_id=?", (GAME,)).fetchone()[0]
    conn.close()
    return res, cards, kept


def test_live_t90_after_kickoff_never_issues_an_executable_card(env, monkeypatch):
    res, cards, kept = _qb_t90(monkeypatch, -30)
    assert cards, "the QB rows still render"
    assert not [c for c in cards if c["status"] in ("watch", "actionable")], (
        [(c["market"], c["status"], c["status_reasons"]) for c in cards])
    assert res["odds_coverage"]["games"][GAME]["state"] == "started"
    assert kept == 2 * len(QB_LINES), "the pregame quotes stay stored as research/grading evidence"


def test_live_t90_before_kickoff_is_still_executable(env, monkeypatch):
    _, cards, _ = _qb_t90(monkeypatch, 90)
    assert cards and all(c["status"] == "watch" for c in cards)


# --------------------------------------------------------------------------- #
# (C) the provider billing cycle is unknown: holds never lapse at a month edge
# --------------------------------------------------------------------------- #
def test_a_close_after_a_calendar_month_boundary_is_still_held(conn):
    kos = {"g": dt.datetime(2026, 10, 4, 17, 0, tzinfo=UTC)}          # pulled Sep 30
    assert oap.close_reserve_for("g", PROP_COST, kos, "2026-09") == PROP_COST
    sep30 = dt.datetime(2026, 9, 30, 16, 0, tzinfo=UTC)
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    oap.pull_week_props(CFG, {"g": "e"}, conn=conn, budget=budget, kickoffs=kos, now=sep30,
                        fetch=_fetch({"e": _payload(["draftkings"])}), reserve_close=True,
                        ts=_ts(sep30))
    oct2 = dt.datetime(2026, 10, 2, 16, 0, tzinfo=UTC)
    assert oap.outstanding_holds(conn, "2026-10", now=oct2) == {"g": PROP_COST}, (
        "no evidence the provider quota renewed on October 1")
