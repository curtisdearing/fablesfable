"""Every scheduled game is priced from the provider's LATEST answer, or is
reported with honest per-game missingness (workstream B, 2026-10-06).

Defects reproduced against 76deda9 (each test below failed on that code):

  (a) latest board -- ``load_recent_lines`` took the newest row per
      (game, book, market, player, side) ACROSS pulls, so a book the provider
      no longer returned kept pricing the board from an older pull, and an
      over from one pull paired with an under from another at another point;
  (b) an answered pull with NO quotes left no trace, so the earlier quotes of
      a game whose props the books had taken down still priced it;
  (c) a future-dated quote row outranked the real current quote;
  (d) the Wednesday close hold lived only inside one call: a later call (the
      T-90 pull of a rationed game) spent the credits held for closes;
  (e) event identity ignored the clock: the LAST listed event for a team pair
      won, even one days away from the slate kickoff;
  (f) no per-game coverage: a game absent from the listing, rationed, started
      or answered empty was a count in a sentence, never a row.

All payloads here are SYNTHETIC v4-shaped fixtures, not empirical odds.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue.config import prop_markets_external  # noqa: E402
from nflvalue.sources import availability as avmod  # noqa: E402
from nflvalue.sources import oddsapi_props as oap  # noqa: E402

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 7, 16, 0, tzinfo=UTC)          # Wednesday 12:00 ET
CFG = {"odds_api_key": "test", "books": ["draftkings", "betmgm", "hardrockbet"],
       "regions": "us", "max_prop_games_per_run": 16,
       "odds_budget": {"monthly_credits": 500, "reserve": 50}}
COST = float(len(prop_markets_external(CFG)))


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "odds.db"))
    yield c
    c.close()


def _ts(d: dt.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _payload(books, player="Ace Receiver", point=61.5, market="player_reception_yds"):
    return {"bookmakers": [{"key": b, "markets": [{"key": market, "outcomes": [
        {"name": "Over", "description": player, "price": 1.91, "point": point},
        {"name": "Under", "description": player, "price": 1.91, "point": point}]}]}
        for b in books]}


def _fetch(by_event, calls=None):
    def fetch(url, params=None):
        eid = url.rstrip("/").split("/")[-2]
        if calls is not None:
            calls.append(eid)
        answer = by_event[eid]
        if isinstance(answer, Exception):
            raise answer
        return json.loads(json.dumps(answer))
    return fetch


def _pull(conn, at, payload, game="g", event="e", **kw):
    return oap.pull_week_props(CFG, {game: event}, conn=conn, ts=_ts(at), now=at,
                               fetch=_fetch({event: payload}), **kw)


# --------------------------------------------------------------------------- #
# (a)-(c) the board is the latest answer, never a collage of older pulls
# --------------------------------------------------------------------------- #
def test_a_book_absent_from_the_latest_answer_does_not_price_the_board(conn):
    t1, t2 = NOW - dt.timedelta(hours=20), NOW - dt.timedelta(hours=1)
    _pull(conn, t1, _payload(["draftkings", "betmgm"]))
    _pull(conn, t2, _payload(["draftkings"], point=64.5))
    rows = oap.load_recent_lines(conn, game_ids=["g"], now=NOW)
    assert {r["book"] for r in rows} == {"draftkings"}, (
        "BetMGM did not answer the latest pull; its older quote is not on offer now")
    assert {r["ts"] for r in rows} == {_ts(t2)}


def test_an_over_and_an_under_from_different_pulls_never_pair(conn):
    t1, t2 = NOW - dt.timedelta(hours=20), NOW - dt.timedelta(hours=1)
    _pull(conn, t1, _payload(["draftkings"], player="Ace Passer", point=250.5,
                             market="player_pass_yds"))
    over_only = {"bookmakers": [{"key": "draftkings", "markets": [{
        "key": "player_pass_yds", "outcomes": [
            {"name": "Over", "description": "Ace Passer", "price": 1.87, "point": 255.5}]}]}]}
    _pull(conn, t2, over_only)
    rows = oap.load_recent_lines(conn, game_ids=["g"], now=NOW)
    for r in rows:
        r["player_id"] = "P1"
    frame = oap.to_prop_lines_frame(rows)
    assert frame.empty or (frame["over_ts"] == frame["under_ts"]).all(), (
        "an under quoted at 250.5 a day earlier was paired with today's 255.5 over")


def test_an_answer_with_no_quotes_supersedes_earlier_quotes(conn):
    t1, t2 = NOW - dt.timedelta(hours=20), NOW - dt.timedelta(hours=1)
    _pull(conn, t1, _payload(["draftkings", "betmgm"]))
    res = _pull(conn, t2, {"bookmakers": []})
    assert res["empty"] == ["g"]
    assert oap.load_recent_lines(conn, game_ids=["g"], now=NOW) == [], (
        "the provider answered with no props; yesterday's quotes are not on offer")
    snap = oap.latest_snapshots(conn, ["g"], now=NOW)["g"]
    assert snap["ts"] == _ts(t2) and snap["n_rows"] == 0 and snap["fresh"]


def test_a_future_dated_quote_never_masks_the_current_board(conn):
    _pull(conn, NOW - dt.timedelta(hours=2), _payload(["draftkings"]))
    dbmod.upsert(conn, "lines", [{
        "ts": _ts(NOW + dt.timedelta(hours=3)), "game_id": "g", "book": "draftkings",
        "market": "receiving_yards", "player_id": None, "player_name": "Ace Receiver",
        "side": side, "point": 99.5, "price": 1.91} for side in ("over", "under")],
        ["ts", "game_id", "book", "market", "player_name", "side"])
    rows = oap.load_recent_lines(conn, game_ids=["g"], now=NOW)
    assert {r["point"] for r in rows} == {61.5}, "a quote stamped in the future is not current"


# --------------------------------------------------------------------------- #
# (d) a close held on Wednesday stays held across calls
# --------------------------------------------------------------------------- #
def test_a_close_held_on_wednesday_is_not_spent_by_a_later_call(conn):
    kick = {"a": NOW + dt.timedelta(days=4, hours=1), "b": NOW + dt.timedelta(days=4, hours=4),
            "c": NOW + dt.timedelta(days=4, hours=1)}
    budget = oap.CreditBudget(conn, 500, 50, month=NOW.strftime("%Y-%m"))
    budget.used = budget.ceiling - 4 * COST        # two entries + their two closes, exactly
    payloads = {"ea": _payload(["draftkings"]), "eb": _payload(["draftkings"]),
                "ec": _payload(["draftkings"])}
    wed = oap.pull_week_props(CFG, {"a": "ea", "b": "eb"}, conn=conn, budget=budget,
                              fetch=_fetch(payloads), kickoffs=kick, now=NOW,
                              reserve_close=True, ts=_ts(NOW))
    assert wed["pulled"] == ["a", "b"] and budget.remaining == 2 * COST

    # Sunday 11:30 ET: run_t90 pulls the game Wednesday could not afford
    sunday = kick["c"] - dt.timedelta(minutes=90)
    calls: list = []
    t90 = oap.pull_week_props(CFG, {"c": "ec"}, conn=conn, budget=budget,
                              fetch=_fetch(payloads, calls), kickoffs={"c": kick["c"]},
                              now=sunday, ts=_ts(sunday))
    assert t90["skipped_budget"] == ["c"] and calls == [], (
        "the credits left were held for the closes of a and b")
    assert t90["credits_held_for_closes"] == 2 * COST

    close = oap.resnap_lines(CFG, {"a": "ea", "b": "eb"}, conn=conn, budget=budget,
                             fetch=_fetch(payloads), ts=_ts(sunday), now=sunday)
    assert close["pulled"] == ["a", "b"], "both held closes must still be affordable"
    assert oap.outstanding_holds(conn, budget.month, now=sunday) == {}


def test_a_hold_lapses_at_kickoff(conn):
    kick = {"a": NOW + dt.timedelta(hours=30)}
    _pull(conn, NOW, _payload(["draftkings"]), game="a", event="ea", kickoffs=kick,
          reserve_close=True)
    month = NOW.strftime("%Y-%m")
    assert oap.outstanding_holds(conn, month, now=NOW) == {"a": COST}
    assert oap.outstanding_holds(conn, month, now=kick["a"] + dt.timedelta(minutes=1)) == {}


# --------------------------------------------------------------------------- #
# (e) event identity: team pair AND clock
# --------------------------------------------------------------------------- #
def _slate(rows):
    return pd.DataFrame(rows, columns=["game_id", "gameday", "gametime", "home_team", "away_team"])


def test_event_identity_takes_the_listed_event_nearest_the_slate_kickoff():
    """Week 18 and a Wild Card rematch at the same venue can both be listed."""
    slate = _slate([("2026_18_PIT_BAL", "2027-01-03", "16:25", "BAL", "PIT")])
    events = [{"id": "w18", "home_team": "Baltimore Ravens", "away_team": "Pittsburgh Steelers",
               "commence_time": "2027-01-03T21:25:00Z"},
              {"id": "wc", "home_team": "Baltimore Ravens", "away_team": "Pittsburgh Steelers",
               "commence_time": "2027-01-10T01:15:00Z"}]
    assert pw.build_event_map(CFG, slate, list_events_fn=lambda cfg: events) == {
        "2026_18_PIT_BAL": "w18"}


def test_a_listed_event_days_from_the_slate_kickoff_is_not_this_game():
    slate = _slate([("2026_05_TB_DAL", "2026-10-08", "20:15", "DAL", "TB")])
    events = [{"id": "far", "home_team": "Dallas Cowboys", "away_team": "Tampa Bay Buccaneers",
               "commence_time": "2026-12-20T18:00:00Z"}]
    assert pw.build_event_map(CFG, slate, list_events_fn=lambda cfg: events) == {}


def test_a_postponed_game_keeps_its_event_and_reports_the_moved_kickoff():
    slate = _slate([("2026_05_NYG_WAS", "2026-10-11", "13:00", "WAS", "NYG")])
    events = [{"id": "moved", "home_team": "Washington Commanders",
               "away_team": "New York Giants", "commence_time": "2026-10-13T23:00:00Z"}]
    details: dict = {}
    emap = pw.build_event_map(CFG, slate, list_events_fn=lambda cfg: events, details=details)
    assert emap == {"2026_05_NYG_WAS": "moved"}
    g = details["games"]["2026_05_NYG_WAS"]
    assert g["kickoff_delta_minutes"] == 54 * 60 and g["kickoff_moved"] is True


# --------------------------------------------------------------------------- #
# (f) a full week, Thursday to Monday: every game priced or honestly missing
# --------------------------------------------------------------------------- #
WEEK5 = [  # (away, home, gameday, ET gametime) -- nflverse 2026 Week 5, as listed 2026-10-05
    ("TB", "DAL", "2026-10-08", "20:15"), ("PHI", "JAX", "2026-10-11", "09:30"),
    ("CHI", "GB", "2026-10-11", "13:00"), ("CIN", "MIA", "2026-10-11", "13:00"),
    ("CLE", "NYJ", "2026-10-11", "13:00"), ("HOU", "TEN", "2026-10-11", "13:00"),
    ("IND", "PIT", "2026-10-11", "13:00"), ("LV", "NE", "2026-10-11", "13:00"),
    ("MIN", "NO", "2026-10-11", "13:00"), ("NYG", "WAS", "2026-10-11", "13:00"),
    ("DEN", "LAC", "2026-10-11", "16:05"), ("DET", "ARI", "2026-10-11", "16:25"),
    ("SF", "SEA", "2026-10-11", "16:25"), ("BAL", "ATL", "2026-10-11", "20:20"),
    ("BUF", "LA", "2026-10-12", "20:15")]
ABBR_TO_DISPLAY = {v: k for k, v in avmod.DISPLAY_TO_ABBR.items()}


def _week5():
    slate = _slate([(f"2026_05_{a}_{h}", d, t, h, a) for a, h, d, t in WEEK5])
    kick = pw.slate_kickoffs(slate)
    events, payloads = [], {}
    for g in slate.itertuples(index=False):
        if g.game_id == "2026_05_NYG_WAS":
            continue                                   # not posted / postponed: absent
        eid = f"ev_{g.home_team}"
        events.append({"id": eid, "home_team": ABBR_TO_DISPLAY[g.home_team],
                       "away_team": ABBR_TO_DISPLAY[g.away_team],
                       "commence_time": kick[g.game_id].astimezone(UTC).isoformat()})
        books = ["draftkings"] if g.home_team == "LAC" else CFG["books"]   # sparse
        payloads[eid] = _payload(books, player=f"{g.home_team} Receiver")
    payloads["ev_ARI"] = {"bookmakers": []}            # answered: no props posted
    payloads["ev_SEA"] = RuntimeError("HTTP 502")      # provider failed this event
    cands = pd.DataFrame([{"player_id": f"P_{g.home_team}", "name": f"{g.home_team} Receiver",
                           "team": g.home_team} for g in slate.itertuples(index=False)])
    return slate, events, payloads, cands


def _price(conn, slate, events, payloads, cands, now, calls=None):
    """The run_week pricing sequence, step for step."""
    identity: dict = {}
    emap = pw.build_event_map(CFG, slate, list_events_fn=lambda cfg: events, details=identity)
    kickoffs = pw.slate_kickoffs(slate)
    pull = oap.pull_week_props(CFG, emap, conn=conn, fetch=_fetch(payloads, calls),
                               kickoffs=kickoffs, reserve_close=True, now=now, ts=_ts(now))
    rows = oap.load_recent_lines(conn, game_ids=list(slate["game_id"]), now=now)
    rows = oap.match_player_ids(rows, cands, game_teams=pw._game_teams(slate))
    frame = oap.to_prop_lines_frame(rows)
    cov = oap.slate_coverage(list(slate["game_id"]), CFG, identity=identity, pull=pull,
                             board_rows=rows, prop_lines=frame, now=now, conn=conn)
    return pull, frame, cov


def test_full_week_every_game_is_priced_or_honestly_missing(conn):
    slate, events, payloads, cands = _week5()
    calls: list = []
    pull, frame, cov = _price(conn, slate, events, payloads, cands, NOW, calls)
    games = cov["games"]
    assert set(games) == set(slate["game_id"]), "every scheduled game gets a coverage row"
    assert games["2026_05_NYG_WAS"]["state"] == "not_in_events_listing"
    assert games["2026_05_DET_ARI"]["state"] == "answered_no_quotes"
    assert games["2026_05_SF_SEA"]["state"] == "pull_error"
    priced = {g for g, r in games.items() if r["state"] == "priced"}
    assert len(priced) == 12 and {"2026_05_TB_DAL", "2026_05_PHI_JAX",
                                  "2026_05_BUF_LA"} <= priced, "Thursday, London and Monday"
    assert set(frame["game_id"]) == priced, "no line exists for a game that is not priced"
    sparse = games["2026_05_DEN_LAC"]
    assert sparse["books_offered"] == ["draftkings"]
    assert sparse["books_missing"] == ["betmgm", "hardrockbet"]
    dal = games["2026_05_TB_DAL"]
    assert dal["markets_offered"] == ["receiving_yards"]
    assert "passing_yards" in dal["markets_missing"], "an absent market is reported absent"
    assert cov["summary"]["priced"] == 12 and cov["summary"]["n_games"] == 15
    assert len(calls) == 14, "one event-call per listed game, none for the absent one"


def test_monday_after_the_weekend_is_stale_not_priced_from_wednesday(conn):
    slate, events, payloads, cands = _week5()
    _price(conn, slate, events, payloads, cands, NOW)
    mnf_t90 = pw.slate_kickoffs(slate)["2026_05_BUF_LA"].astimezone(UTC) - dt.timedelta(minutes=90)
    rows = oap.load_recent_lines(conn, game_ids=["2026_05_BUF_LA"], now=mnf_t90)
    assert rows == [], "Wednesday's quote is >60h old at the Monday T-90"
    cov = oap.slate_coverage(["2026_05_BUF_LA"], CFG, board_rows=rows,
                             prop_lines=oap.to_prop_lines_frame(rows), now=mnf_t90, conn=conn)
    row = cov["games"]["2026_05_BUF_LA"]
    assert row["state"] == "stale_quotes_only" and row["quote_clock"] == _ts(NOW)


def test_a_rerun_bills_nothing_for_games_answered_moments_ago(conn):
    """The T-90 job resnaps a game, then run_t90 asks again: no second call."""
    slate, events, payloads, cands = _week5()
    _price(conn, slate, events, payloads, cands, NOW)
    later = NOW + dt.timedelta(minutes=20)
    assert oap.answered_since(conn, ["2026_05_DET_ARI", "2026_05_TB_DAL", "2026_05_SF_SEA"],
                              now=later, max_age_hours=1.0) == {
        "2026_05_DET_ARI": _ts(NOW), "2026_05_TB_DAL": _ts(NOW)}, (
        "an empty answer is still an answer; a failed call is not")


# --------------------------------------------------------------------------- #
# the caller: run_week reports and receipts coverage for every game
# --------------------------------------------------------------------------- #
from tests.test_eval_fixes import _feeds, env  # noqa: E402,F401  (fixture reuse)
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402


GAME = f"{SEASON}_09_AAA_BBB"


def _live_odds_env(monkeypatch):
    from nflvalue import config as cfgmod
    monkeypatch.setitem(cfgmod.DEFAULT_CONFIG, "odds_api_key", "test")
    monkeypatch.setenv("ODDS_API_KEY", "test")
    # the shipped test_eval_fixes live-odds test never priced anything: these
    # synthetic names were absent from DISPLAY_TO_ABBR, so no event ever matched
    monkeypatch.setitem(avmod.DISPLAY_TO_ABBR, "Alpha Ants", "AAA")
    monkeypatch.setitem(avmod.DISPLAY_TO_ABBR, "Bravo Bees", "BBB")
    conn = dbmod.connect()
    stored = (dt.datetime.now(UTC) - dt.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    dbmod.upsert(conn, "lines", [{
        "ts": stored, "game_id": GAME, "book": book, "market": "receiving_yards",
        "player_id": None, "player_name": "Alpha Wideout", "side": side, "point": 61.5,
        "price": price} for book in ("draftkings", "betmgm")
        for side, price in (("over", 1.87), ("under", 1.95))],
        ["ts", "game_id", "book", "market", "player_name", "side"])
    conn.close()
    return stored


def _run_live(calls):
    from nflvalue.freshness import stamp_now
    return pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                       inject_feeds=_feeds(stamp_now()), live_odds=True,
                       odds_fetch=_fetch({"e1": _payload(["draftkings"], player="Alpha Wideout",
                                                         point=63.5)}, calls),
                       list_events_fn=lambda cfg: [
                           {"id": "e1", "home_team": "Bravo Bees", "away_team": "Alpha Ants"}])


def test_run_week_prices_the_latest_answer_and_receipts_every_games_coverage(env, monkeypatch):
    stored = _live_odds_env(monkeypatch)
    # the synthetic slate is dated 2023; a real kickoff would read as started
    monkeypatch.setattr(pw, "slate_kickoffs", lambda slate: {})
    calls: list = []
    res = _run_live(calls)
    cov = res["odds_coverage"]
    row = cov["games"][GAME]
    assert set(cov["games"]) == {GAME} and calls == ["e1"] and row["pulled_this_run"]
    assert row["state"] == "priced" and row["event_id"] == "e1"
    assert row["books_offered"] == ["draftkings"], (
        "BetMGM answered two hours ago but not now; the board is this run's answer")
    assert row["quote_clock"] > stored and row["books_missing"] == ["betmgm", "hardrockbet"]
    assert res["factor_receipt"]["odds_coverage"]["games"][GAME]["state"] == "priced"
    assert "Coverage: 1/1 game(s) priced" in Path(res["md_path"]).read_text()
    priced = [l for g in res["games"] for l in g["leans"] if l.get("line_source") == "odds_api"]
    assert priced and {l["line"] for l in priced} == {63.5}


def test_run_week_flags_a_game_under_way_and_spends_nothing_on_it(env, monkeypatch):
    stored = _live_odds_env(monkeypatch)         # 2023 kickoff: the game is under way
    calls: list = []
    res = _run_live(calls)
    row = res["odds_coverage"]["games"][GAME]
    assert calls == [], "no credit on a game in progress"
    # the live decision clock refuses the stored pregame quote; the row keeps its clock
    assert row["started"] is True and row["state"] == "started" and row["quote_clock"] == stored
    assert not [l for g in res["games"] for l in g["leans"] if l.get("line_source") == "odds_api"]


def test_run_week_without_odds_still_lists_every_game(env):
    from nflvalue.freshness import stamp_now
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_feeds(stamp_now()))
    cov = res["odds_coverage"]
    assert {r["state"] for r in cov["games"].values()} == {"odds_not_requested"}
    assert cov["summary"]["n_games"] == len(cov["games"]) >= 1


def test_the_line_never_moves_the_performance_forecast():
    """Pricing joins AFTER the forecast: two different boards, same mean/sd."""
    from nflvalue import candidates as candmod
    inputs = synthetic_inputs()
    base = candmod.enumerate_candidates(SEASON, WEEK, inputs=inputs)
    pick = base.iloc[0]
    boards = []
    for point, over, under in ((float(pick["line"]) - 3.0, 1.80, 2.02),
                               (float(pick["line"]) + 9.0, 2.40, 1.55)):
        boards.append(pd.DataFrame([{
            "game_id": pick["game_id"], "market": pick["market"], "player_id": pick["player_id"],
            "point": point, "over_price": over, "under_price": under, "book": "x/y",
            "consensus_p_over": 0.5, "n_books": 2, "over_book": "x", "under_book": "y",
            "over_ts": None, "under_ts": None}], columns=oap.PROP_LINE_COLS))
    a, b = (candmod.enumerate_candidates(SEASON, WEEK, inputs=inputs, prop_lines=f)
            for f in boards)
    key = ["player_id", "market"]
    m = a[key + ["mean", "sd"]].merge(b[key + ["mean", "sd"]], on=key, suffixes=("_a", "_b"))
    assert len(m) == len(base)
    assert (m["mean_a"] == m["mean_b"]).all() and (m["sd_a"] == m["sd_b"]).all()
