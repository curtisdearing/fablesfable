"""A pick whose own issuing run held publication (e.g. a T-90 refresh whose inactives feed
could not be fetched) is never an executable card, even with a fresh verified quote.

RED on 47f1432: week_cards preferred the held T-90 lean and build_card never consulted the
run's publish decision, so a held board's pick could be 'watch'."""
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import pick_cards as pc  # noqa: E402
from nflvalue.freshness import stamp_now  # noqa: E402
from tests.test_pipeline_weekly import GAME_ID, _fresh_feeds, env  # noqa: E402,F401
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402
from tests.test_t90_factor_receipts import _t90_feeds  # noqa: E402

NOW = dt.datetime(2026, 9, 24, 20, 0, tzinfo=dt.timezone.utc)
AVAIL = json.dumps({"team": "ATL", "stages": {}, "availability": {
    "status": "OK", "eligibility": "eligible", "availability_state": "not_listed"}})


def _db(receipts):
    conn = sqlite3.connect(":memory:")
    cols = ("season", "week", "clock", "game_id", "player_id", "name", "market", "side", "line",
            "line_source", "price", "book", "mean", "sd", "p_side", "status", "as_of", "created_at",
            "quote_book", "quote_ts", "run_id", "stage_json")
    conn.execute(f"CREATE TABLE leans ({', '.join(cols)})")
    conn.execute("CREATE TABLE lines (ts, game_id, book, market, player_id, player_name, side, point, price)")
    conn.execute("CREATE TABLE run_receipts (run_id, season, week, clock, as_of, receipt_json, created_at)")
    for run, clock, pub in receipts:
        conn.execute("INSERT INTO leans VALUES (" + ",".join("?" * len(cols)) + ")",
                     (2026, 3, clock, "2026_03_ATL_GB", "p1", "D.London", "receptions", "over", 5.5,
                      "odds_api", 1.87, "draftkings", 6.1, 2.4, 0.55, "active", "2026-09-24T18:35:00Z",
                      f"2026-09-24T18:3{5 if clock == 'wed' else 9}:00Z", "draftkings",
                      "2026-09-24T18:30:00Z", run, AVAIL))
        if pub != "absent":
            conn.execute("INSERT INTO run_receipts VALUES (?,?,?,?,?,?,?)",
                         (run, 2026, 3, clock, "2026-09-24T18:35:00Z",
                          json.dumps({"run_id": run, "publish": pub, "publish_reasons":
                                      ["inactives: source could not be fetched (404)"] if not pub else []}),
                          "2026-09-24T18:35:00Z"))
    conn.execute("INSERT INTO lines VALUES ('2026-09-24T18:30:00Z','2026_03_ATL_GB','draftkings',"
                 "'receptions','','Drake London','over',5.5,1.87)")
    return conn


def _card(receipts):
    return pc.week_cards(_db(receipts), 2026, 3, now=NOW)["cards"][0]


def test_permitted_run_is_watch():
    assert _card([("r-wed", "wed", True)])["status"] == "watch"


def test_held_t90_run_is_not_executable_even_with_a_verified_fresh_quote():
    c = _card([("r-wed", "wed", True), ("r-wed:t90:2026_03_ATL_GB", "t90", False)])
    assert c["status"] == "research" and c["quote"] is None
    assert "held publication" in " ".join(c["status_reasons"]) and "404" in " ".join(c["status_reasons"])


def test_unrecorded_publish_decision_is_not_treated_as_permitted():
    c = _card([("r-legacy", "wed", "absent")])
    assert c["status"] == "research" and "not recorded" in " ".join(c["status_reasons"])


def test_wed_and_t90_receipts_record_their_publish_decision(env):
    wed = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_fresh_feeds(stamp_now()))
    assert wed["factor_receipt"]["publish"] is wed["publish"] is True
    feeds = dict(_t90_feeds(stamp_now()), inactives_state="not_fetched", inactive_rows=[],
                 inactives_reason="404")
    t90 = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(), inject_feeds=feeds)
    rc = t90["factor_receipt"]
    assert rc["publish"] is False and any("could not be fetched" in r for r in rc["publish_reasons"])
