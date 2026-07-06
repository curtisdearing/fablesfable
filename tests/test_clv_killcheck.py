"""CLV math + kill-check verdicts (Block A)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import clv as clvmod  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue import killcheck  # noqa: E402
from nflvalue.oddsmath import devig_multiplicative  # noqa: E402


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


def _line(ts, side, price, point=52.5, book="draftkings", market="receiving_yards",
          game_id="2023_10_CLE_BAL", player_id="00-A1", name="Mark Andrews"):
    return {"ts": ts, "game_id": game_id, "book": book, "market": market,
            "player_id": player_id, "player_name": name, "side": side,
            "point": point, "price": price}


def _seed_lean(conn, side="over", as_of="2023-11-08T12:00:00Z"):
    dbmod.upsert(conn, "leans", [{
        "season": 2023, "week": 10, "clock": "wed", "game_id": "2023_10_CLE_BAL",
        "player_id": "00-A1", "name": "M.Andrews", "market": "receiving_yards",
        "side": side, "line": 52.5, "line_source": "odds_api", "price": 1.87,
        "book": "draftkings", "mean": 60.0, "sd": 20.0, "p_side": 0.62,
        "composite": 70.0, "edge": 0.06, "confidence_comp": 0.4, "matchup_comp": 0.6,
        "screened_n": 40, "reason": "test", "status": "active", "void_reason": None,
        "as_of": as_of, "created_at": as_of,
    }], ["season", "week", "clock", "game_id", "player_id", "market"])


def test_devig_and_snapshot_prob(conn):
    dbmod.upsert(conn, "lines", [
        _line("2023-11-08T10:00:00Z", "over", 1.87),
        _line("2023-11-08T10:00:00Z", "under", 1.95),
    ], ["ts", "game_id", "book", "market", "player_name", "side"])
    snap = clvmod.snapshot_prob(conn, "2023_10_CLE_BAL", "receiving_yards", "00-A1", "over")
    expected_over, _ = devig_multiplicative([1.87, 1.95])
    assert snap["prob"] == pytest.approx(expected_over, abs=1e-9)
    assert snap["prob_kind"] == "devig"
    assert snap["n_books"] == 1


def test_clv_entry_vs_close_math(conn):
    # entry snapshot: over 1.87/1.95 -> devig p_over ~= .5105
    # close snapshot: over 1.72/2.10 -> devig p_over ~= .5497 (market moved toward us)
    dbmod.upsert(conn, "lines", [
        _line("2023-11-08T10:00:00Z", "over", 1.87),
        _line("2023-11-08T10:00:00Z", "under", 1.95),
        _line("2023-11-12T17:00:00Z", "over", 1.72),
        _line("2023-11-12T17:00:00Z", "under", 2.10),
        _line("2023-11-12T19:00:00Z", "over", 3.00),  # post-kick junk: must be ignored
        _line("2023-11-12T19:00:00Z", "under", 1.30),
    ], ["ts", "game_id", "book", "market", "player_name", "side"])
    _seed_lean(conn)

    out = clvmod.log_close_for_week(conn, 2023, 10,
                                    kickoffs={"2023_10_CLE_BAL": "2023-11-12T18:00:00Z"})
    assert len(out) == 1
    row = out.iloc[0]
    p_entry, _ = devig_multiplicative([1.87, 1.95])
    p_close, _ = devig_multiplicative([1.72, 2.10])
    assert row["entry_prob"] == pytest.approx(p_entry, abs=1e-4)
    assert row["close_prob"] == pytest.approx(p_close, abs=1e-4)
    assert row["clv_prob"] == pytest.approx(p_close - p_entry, abs=1e-4)
    assert row["clv_prob"] > 0                       # we beat the close in this scenario

    stats = clvmod.rolling_clv(conn)
    assert stats["n"] == 1 and stats["positive_rate"] == 1.0


def test_clv_needs_two_distinct_snapshots(conn):
    dbmod.upsert(conn, "lines", [
        _line("2023-11-08T10:00:00Z", "over", 1.87),
        _line("2023-11-08T10:00:00Z", "under", 1.95),
    ], ["ts", "game_id", "book", "market", "player_name", "side"])
    _seed_lean(conn)
    out = clvmod.log_close_for_week(conn, 2023, 10,
                                    kickoffs={"2023_10_CLE_BAL": "2023-11-12T18:00:00Z"})
    assert out.empty                                  # entry == close -> no claim made


# --------------------------------------------------------------------------- #
# Phase 7.3 GAP #1: close-window floor -- a snapshot outside
# [kickoff - CLOSE_WINDOW_H, kickoff] must NOT resolve as a "close" (that
# would fake CLV ~= 0 against a stale entry-era snapshot).
# --------------------------------------------------------------------------- #
def test_close_window_floor_rejects_stale_snapshot(conn):
    dbmod.upsert(conn, "lines", [
        _line("2023-11-08T10:00:00Z", "over", 1.87),
        _line("2023-11-08T10:00:00Z", "under", 1.95),
        # only a stale snapshot exists before kickoff -- no line in the last
        # 6h before kickoff (2023-11-12T18:00:00Z), so nothing should resolve
        _line("2023-11-09T09:00:00Z", "over", 1.80),
        _line("2023-11-09T09:00:00Z", "under", 2.05),
    ], ["ts", "game_id", "book", "market", "player_name", "side"])
    _seed_lean(conn)
    out = clvmod.log_close_for_week(conn, 2023, 10,
                                    kickoffs={"2023_10_CLE_BAL": "2023-11-12T18:00:00Z"},
                                    close_window_hours=6.0)
    assert out.empty, "a stale (>6h pre-kickoff) snapshot must not count as a close"


def test_close_window_floor_accepts_in_window_snapshot(conn):
    dbmod.upsert(conn, "lines", [
        _line("2023-11-08T10:00:00Z", "over", 1.87),
        _line("2023-11-08T10:00:00Z", "under", 1.95),
        _line("2023-11-12T15:00:00Z", "over", 1.72),   # 3h pre-kickoff -- inside a 6h window
        _line("2023-11-12T15:00:00Z", "under", 2.10),
    ], ["ts", "game_id", "book", "market", "player_name", "side"])
    _seed_lean(conn)
    out = clvmod.log_close_for_week(conn, 2023, 10,
                                    kickoffs={"2023_10_CLE_BAL": "2023-11-12T18:00:00Z"},
                                    close_window_hours=6.0)
    assert len(out) == 1
    # a narrower window that excludes the same snapshot must reject it
    out_narrow = clvmod.log_close_for_week(conn, 2023, 10,
                                           kickoffs={"2023_10_CLE_BAL": "2023-11-12T18:00:00Z"},
                                           close_window_hours=2.0)
    assert out_narrow.empty


# --------------------------------------------------------------------------- #
# Phase 7.4 scheduling guard: has_close_snapshot lets the T-90 job avoid
# re-paying for a resnap of a game it already caught inside its close window
# (the T-90 job can fire multiple times while a game sits in the window,
# since cron granularity is coarser than the window; resnap_lines itself has
# no dedup, so this guard belongs to the CALLER).
# --------------------------------------------------------------------------- #
def test_has_close_snapshot_false_when_nothing_in_window(conn):
    dbmod.upsert(conn, "lines", [
        _line("2023-11-08T10:00:00Z", "over", 1.87),  # entry-era, way outside window
        _line("2023-11-08T10:00:00Z", "under", 1.95),
    ], ["ts", "game_id", "book", "market", "player_name", "side"])
    assert clvmod.has_close_snapshot(
        conn, "2023_10_CLE_BAL", "2023-11-12T18:00:00Z", close_window_hours=6.0) is False


def test_has_close_snapshot_true_once_in_window(conn):
    dbmod.upsert(conn, "lines", [
        _line("2023-11-08T10:00:00Z", "over", 1.87),
        _line("2023-11-08T10:00:00Z", "under", 1.95),
        _line("2023-11-12T15:00:00Z", "over", 1.72),   # 3h pre-kickoff -- in a 6h window
        _line("2023-11-12T15:00:00Z", "under", 2.10),
    ], ["ts", "game_id", "book", "market", "player_name", "side"])
    assert clvmod.has_close_snapshot(
        conn, "2023_10_CLE_BAL", "2023-11-12T18:00:00Z", close_window_hours=6.0) is True
    # a narrower window that excludes the same snapshot must say False
    assert clvmod.has_close_snapshot(
        conn, "2023_10_CLE_BAL", "2023-11-12T18:00:00Z", close_window_hours=2.0) is False


# --------------------------------------------------------------------------- #
# Phase 7.3 GAP #3: clock dedup -- a wed + t90 lean for the same key must
# resolve to exactly ONE clv row, against the earliest active as_of.
# --------------------------------------------------------------------------- #
def test_clock_dedup_resolves_one_row_against_earliest_as_of(conn):
    dbmod.upsert(conn, "lines", [
        _line("2023-11-08T10:00:00Z", "over", 1.87),
        _line("2023-11-08T10:00:00Z", "under", 1.95),
        _line("2023-11-12T15:00:00Z", "over", 1.72),
        _line("2023-11-12T15:00:00Z", "under", 2.10),
    ], ["ts", "game_id", "book", "market", "player_name", "side"])
    # wed entry (earlier as_of) + a t90 lean for the SAME key (later as_of)
    dbmod.upsert(conn, "leans", [{
        "season": 2023, "week": 10, "clock": "wed", "game_id": "2023_10_CLE_BAL",
        "player_id": "00-A1", "name": "M.Andrews", "market": "receiving_yards",
        "side": "over", "line": 52.5, "line_source": "odds_api", "price": 1.87,
        "book": "draftkings", "mean": 60.0, "sd": 20.0, "p_side": 0.62,
        "composite": 70.0, "edge": 0.06, "confidence_comp": 0.4, "matchup_comp": 0.6,
        "screened_n": 40, "reason": "test", "status": "active", "void_reason": None,
        "as_of": "2023-11-08T12:00:00Z", "created_at": "2023-11-08T12:00:00Z",
    }, {
        "season": 2023, "week": 10, "clock": "t90", "game_id": "2023_10_CLE_BAL",
        "player_id": "00-A1", "name": "M.Andrews", "market": "receiving_yards",
        "side": "over", "line": 52.5, "line_source": "odds_api", "price": 1.72,
        "book": "draftkings", "mean": 60.0, "sd": 20.0, "p_side": 0.65,
        "composite": 72.0, "edge": 0.08, "confidence_comp": 0.4, "matchup_comp": 0.6,
        "screened_n": 40, "reason": "test", "status": "active", "void_reason": None,
        "as_of": "2023-11-12T15:30:00Z", "created_at": "2023-11-12T15:30:00Z",
    }], ["season", "week", "clock", "game_id", "player_id", "market"])
    out = clvmod.log_close_for_week(conn, 2023, 10,
                                    kickoffs={"2023_10_CLE_BAL": "2023-11-12T18:00:00Z"})
    assert len(out) == 1, "wed+t90 leans for the same key must dedup to one clv row"
    p_entry, _ = devig_multiplicative([1.87, 1.95])   # the WED (earliest as_of) snapshot
    assert out.iloc[0]["entry_prob"] == pytest.approx(p_entry, abs=1e-4)


# --------------------------------------------------------------------------- #
# Kill-check verdicts
# --------------------------------------------------------------------------- #
def _seed_clv(conn, n, clv_value):
    rows = [{
        "season": 2023, "week": 10, "game_id": f"G{i}", "player_id": f"P{i}",
        "market": "receiving_yards", "side": "over",
        "entry_ts": "t0", "entry_point": 50.5, "entry_price": 1.9,
        "entry_prob": 0.5, "close_ts": f"t{i+1}", "close_point": 50.5,
        "close_price": None, "close_prob": 0.5 + clv_value,
        "clv_prob": clv_value, "point_moved": 0.0,
    } for i in range(n)]
    dbmod.upsert(conn, "clv", rows, ["season", "week", "game_id", "player_id", "market", "side"])


def test_killcheck_insufficient_sample(conn):
    _seed_clv(conn, 10, 0.02)
    r = killcheck.report(conn)
    assert r["verdict"] == "INSUFFICIENT_SAMPLE"
    assert "no conclusion" in r["detail"]


def test_killcheck_go(conn):
    _seed_clv(conn, 160, 0.015)
    r = killcheck.report(conn)
    assert r["verdict"] == "GO"
    assert "SHRUNK edge" in r["detail"]              # staking guardrails restated even on GO


def test_killcheck_no_go_pre_committed(conn):
    _seed_clv(conn, 160, -0.01)
    r = killcheck.report(conn)
    assert r["verdict"] == "NO_GO"
    assert "KILL CRITERION MET" in r["detail"]
    assert "stop staking" in r["detail"]


# --------------------------------------------------------------------------- #
# Phase 7.3 monitorability: coverage = resolved_n / logged_n (a low value
# warns the close-snapshot budget is too thin to resolve the log).
# --------------------------------------------------------------------------- #
def test_killcheck_coverage_field(conn):
    # 10 resolved CLV rows, but only some fraction of active leans have
    # resolved -- seed 25 active leans total so coverage = 10/25 = 0.4
    _seed_clv(conn, 10, 0.02)
    for i in range(25):
        dbmod.upsert(conn, "leans", [{
            "season": 2023, "week": 10, "clock": "wed", "game_id": f"G{i}",
            "player_id": f"P{i}", "name": f"Player {i}", "market": "receiving_yards",
            "side": "over", "line": 50.5, "line_source": "odds_api", "price": 1.9,
            "book": "draftkings", "mean": 55.0, "sd": 15.0, "p_side": 0.55,
            "composite": 65.0, "edge": 0.03, "confidence_comp": 0.4, "matchup_comp": 0.6,
            "screened_n": 30, "reason": "test", "status": "active", "void_reason": None,
            "as_of": "2023-11-08T12:00:00Z", "created_at": "2023-11-08T12:00:00Z",
        }], ["season", "week", "clock", "game_id", "player_id", "market"])
    r = killcheck.report(conn)
    assert r["leans_logged"] == 25
    assert r["coverage"] == pytest.approx(10 / 25, abs=1e-6)


def test_killcheck_coverage_none_when_no_leans_logged(conn):
    r = killcheck.report(conn)
    assert r["leans_logged"] == 0
    assert r["coverage"] is None
