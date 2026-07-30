"""Real-line backtest harness: fail-closed states, movement stats, reliability
join (synthetic leans excluded), and the populated path."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.real_line_backtest import (build_report, MIN_MOVEMENT_ROWS,
                                         MIN_RELIABILITY_N)
from nflvalue import db as dbmod


@pytest.fixture
def tmp_db(tmp_path):
    path = str(tmp_path / "props.db")
    conn = dbmod.connect(path)
    conn.close()
    return path


def _seed_open_close(path, n, market="receiving_yards", moved=0.02):
    conn = dbmod.connect(path)
    rows = [{
        "season": 2026, "week": 1 + i % 4, "game_id": f"g{i}", "player_id": f"p{i}",
        "market": market, "side": "over",
        "open_ts": "2026-09-01T10:00:00Z", "open_point": 50.5, "open_prob": 0.5,
        "open_n_books": 3,
        "close_ts": "2026-09-06T16:00:00Z", "close_point": 51.5,
        "close_prob": 0.5 + moved, "close_n_books": 3,
        "prob_kind": "devig", "point_moved": 1.0, "prob_moved": moved,
    } for i in range(n)]
    dbmod.upsert(conn, "line_open_close", rows,
                 ["season", "week", "game_id", "player_id", "market", "side"])
    conn.commit(); conn.close()


def _seed_graded_leans(path, n, line_source="oddsapi", p=0.6, hit_rate=0.6):
    conn = dbmod.connect(path)
    leans, outcomes = [], []
    for i in range(n):
        key = {"season": 2026, "week": 1 + i % 4, "clock": "wed",
               "game_id": f"g{i}", "player_id": f"p{i}",
               "market": "receiving_yards", "side": "over"}
        leans.append({**key, "name": f"P{i}", "line": 50.5,
                      "line_source": line_source, "p_side": p,
                      "status": "graded"})
        outcomes.append({**key, "name": f"P{i}", "line": 50.5,
                         "hit": 1 if i < int(n * hit_rate) else 0})
    dbmod.upsert(conn, "leans", leans,
                 ["season", "week", "clock", "game_id", "player_id", "market", "side"])
    dbmod.upsert(conn, "lean_outcomes", outcomes,
                 ["season", "week", "clock", "game_id", "player_id", "market", "side"])
    conn.commit(); conn.close()


def test_empty_db_fails_closed_everywhere(tmp_db):
    r = build_report(tmp_db)
    assert r["coverage"]["status"] == "insufficient_data"
    assert r["movement"]["status"] == "insufficient_data"
    assert r["reliability"]["status"] == "insufficient_data"
    assert r["clv"]["gate_state"]["status"] == "insufficient_data"
    assert "ACCRUING" in r["summary"]["headline"]


def test_movement_stats_withheld_below_floor(tmp_db):
    _seed_open_close(tmp_db, MIN_MOVEMENT_ROWS - 1)
    r = build_report(tmp_db)
    m = r["movement"]["per_market"]["receiving_yards"]
    assert "mean_abs_prob_move" not in m
    assert m["n"] == MIN_MOVEMENT_ROWS - 1


def test_movement_stats_present_at_floor(tmp_db):
    _seed_open_close(tmp_db, MIN_MOVEMENT_ROWS, moved=0.03)
    r = build_report(tmp_db)
    m = r["movement"]["per_market"]["receiving_yards"]
    assert m["mean_abs_prob_move"] == pytest.approx(0.03)
    assert m["share_unmoved"] == 0.0
    assert r["coverage"]["share_with_open_and_close"] == 1.0


def test_reliability_excludes_synthetic_leans(tmp_db):
    _seed_graded_leans(tmp_db, MIN_RELIABILITY_N + 20, line_source="synthetic")
    r = build_report(tmp_db)
    assert r["reliability"]["status"] == "insufficient_data"
    assert r["reliability"]["n_resolved_real_line_leans"] == 0


def test_reliability_computes_on_real_lines(tmp_db):
    _seed_graded_leans(tmp_db, MIN_RELIABILITY_N, line_source="oddsapi",
                       p=0.6, hit_rate=0.6)
    r = build_report(tmp_db)
    assert r["reliability"]["status"] == "ok"
    cal = r["reliability"]["calibration"]
    # perfectly calibrated seed: predicted 0.6, observed 0.6
    assert cal["ece"] == pytest.approx(0.0, abs=1e-6)


def test_summary_counts_ready_sections(tmp_db):
    _seed_open_close(tmp_db, MIN_MOVEMENT_ROWS)
    r = build_report(tmp_db)
    assert r["summary"]["sections_ready"] == "2/4"     # coverage + movement
