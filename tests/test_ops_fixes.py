"""Operational fixes shipped with the picks selector: kickoff timezone
handling, same-book-same-point odds pairing, and the T-90 path working
without test-only injection."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue.sources import availability as avmod  # noqa: E402
from nflvalue.sources import oddsapi_props as oap  # noqa: E402
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402

GAME_ID = f"{SEASON}_09_AAA_BBB"


# --------------------------------------------------------------------------- #
# Kickoff timezone: one tz-aware ET -> UTC conversion
# --------------------------------------------------------------------------- #
def test_kickoffs_convert_eastern_to_utc_with_dst():
    slate = pd.DataFrame([
        {"game_id": "g_edt", "gameday": "2025-09-14", "gametime": "13:00"},  # EDT (UTC-4)
        {"game_id": "g_est", "gameday": "2026-01-04", "gametime": "13:00"},  # EST (UTC-5)
        {"game_id": "g_snf", "gameday": "2025-09-14", "gametime": "20:20"},
    ])
    k = pw.kickoffs_for(slate)
    assert k["g_edt"] == "2025-09-14T17:00:00Z"   # 1pm EDT = 17:00 UTC
    assert k["g_est"] == "2026-01-04T18:00:00Z"   # 1pm EST = 18:00 UTC
    assert k["g_snf"] == "2025-09-15T00:20:00Z"   # SNF crosses the UTC date line
    # the old bug stamped "13:00Z" -- ordering vs UTC snapshots was hours off
    assert not k["g_edt"].startswith("2025-09-14T13:00")


# --------------------------------------------------------------------------- #
# Odds pairing: over/under only from the same book AND same point
# --------------------------------------------------------------------------- #
def _row(book, side, point, price, name="Player X"):
    return {"ts": "t1", "game_id": "g1", "book": book, "market": "receiving_yards",
            "player_id": "P1", "player_name": name, "side": side,
            "point": point, "price": price}


def test_prop_lines_frame_pairs_same_book_same_point():
    rows = [
        # book A quotes the main line AND an alternate line
        _row("alpha", "over", 82.5, 1.91), _row("alpha", "under", 82.5, 1.91),
        _row("alpha", "over", 74.5, 1.50), _row("alpha", "under", 74.5, 2.60),
        # book B quotes only the main line
        _row("beta", "over", 82.5, 1.95), _row("beta", "under", 82.5, 1.87),
    ]
    frame = oap.to_prop_lines_frame(rows)
    assert len(frame) == 1
    r = frame.iloc[0]
    assert r["point"] == 82.5          # the point quoted two-sided by MOST books
    assert r["n_books"] == 2
    # symmetric 1.91/1.91 + near-symmetric book B -> fair prob ~0.5; a
    # cross-point pairing (1.91 over @82.5 vs 2.60 under @74.5) would skew this
    assert abs(r["consensus_p_over"] - 0.5) < 0.02


def test_snapshot_prob_uses_matching_points_only(tmp_path):
    conn = dbmod.connect(str(tmp_path / "clv.db"))
    rows = [
        # book alpha's over and under sit at DIFFERENT points (a moved line
        # captured mid-move): the old iloc[0]-per-book pairing de-vigged
        # 1.60@82.5 against 2.60@74.5 into a nonsense "fair" prob; the fix
        # refuses cross-point pairs, so only beta's clean two-sided quote counts
        _row("alpha", "over", 82.5, 1.60), _row("alpha", "under", 74.5, 2.60),
        _row("beta", "over", 82.5, 1.91), _row("beta", "under", 82.5, 1.91),
    ]
    dbmod.upsert(conn, "lines", rows,
                 ["ts", "game_id", "book", "market", "player_name", "side", "point"])
    from nflvalue.clv import snapshot_prob
    snap = snapshot_prob(conn, "g1", "receiving_yards", "P1", "over")
    assert snap is not None
    assert snap["point"] == 82.5
    assert snap["n_books"] == 1                # alpha's mismatched pair refused
    assert abs(snap["prob"] - 0.5) < 1e-6      # beta symmetric at 82.5
    conn.close()


# --------------------------------------------------------------------------- #
# T-90 without injection: event ids fetched/passed in the normal path
# --------------------------------------------------------------------------- #
@pytest.fixture()
def env(tmp_path, monkeypatch):
    real_connect = dbmod.connect
    db_path = str(tmp_path / "pipe.db")
    monkeypatch.setattr(dbmod, "connect", lambda p=None: real_connect(db_path))
    from nflvalue import config as cfgmod
    from nflvalue import report as rptmod
    monkeypatch.setattr(rptmod, "REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setattr(rptmod, "WEEKLY_PROPS_JSON", str(tmp_path / "weekly_props.json"))
    monkeypatch.setattr(cfgmod, "LATEST_PATH", str(tmp_path / "latest.json"))
    monkeypatch.setattr(cfgmod, "DASHBOARD_PATH", str(tmp_path / "dashboard.html"))
    return {"tmp": tmp_path}


def test_freshness_blocked_run_persists_as_non_active(env):
    """A publish-gate failure must not leave ACTIVE betting evidence behind:
    leans and picks persist with status='blocked', and CLV / grading /
    kill-check (all filtered on status='active') see nothing."""
    stale = {"injury_rows": [], "injuries_fetched_at": "2020-01-01T00:00:00Z",
             "sleeper_df": None, "sleeper_fetched_at": "2020-01-01T00:00:00Z",
             "news_items": [], "news_fetched_at": "2020-01-01T00:00:00Z"}
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=stale)
    assert res["publish"] is False
    conn = dbmod.connect()
    leans = dbmod.query_df(conn, "SELECT status FROM leans WHERE season=? AND week=?",
                           (SEASON, WEEK))
    assert len(leans) and (leans["status"] == "blocked").all()
    picks = dbmod.query_df(conn, "SELECT status FROM picks WHERE season=? AND week=?",
                           (SEASON, WEEK))
    assert (picks["status"] == "blocked").all() if len(picks) else True
    from nflvalue import killcheck as kcmod
    assert kcmod.report(conn)["leans_logged"] == 0
    from nflvalue import clv as clvmod
    assert clvmod.log_close_for_week(conn, SEASON, WEEK,
                                     {GAME_ID: "2025-11-09T18:00:00Z"}).empty
    conn.close()


def test_t90_fetches_event_ids_without_injection(env, monkeypatch):
    """run_t90 with NO injected feeds must discover the ESPN event id itself,
    fetch that event's roster, and act on the inactives -- proving the
    normal CLI/scheduler path works, not just test injection."""
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    calls = {}

    monkeypatch.setattr(avmod, "fetch_team_injuries",
                        lambda: {"rows": [], "fetched_at": now})
    def fake_find(gameday, home, away):
        calls["find"] = (str(gameday), home, away)
        return "401999999"
    monkeypatch.setattr(avmod, "find_espn_event_id", fake_find)
    def fake_rosters(event_id):
        calls["rosters"] = event_id
        return {"rows": [{"espn_id": "1", "name": "Alpha Wideout", "active": False,
                          "did_not_play": True, "starter": True, "team": "AAA"}],
                "fetched_at": now}
    monkeypatch.setattr(avmod, "fetch_event_rosters", fake_rosters)
    from nflvalue.sources import espn_news
    monkeypatch.setattr(espn_news, "fetch_news",
                        lambda: {"items": [], "fetched_at": now})
    from nflvalue.sources import sleeper as slpmod
    monkeypatch.setattr(slpmod, "fetch_projections",
                        lambda s, w: (_ for _ in ()).throw(RuntimeError("offline")))

    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=None)

    assert set(calls["find"][1:]) == {"AAA", "BBB"}   # slate teams passed
    assert calls["rosters"] == "401999999"            # discovered id was used
    # and the fetched inactive actually acted: WR_A never in the t90 leans
    assert all(l["player_id"] != "WR_A"
               for g in res["games"] for l in g["leans"])
