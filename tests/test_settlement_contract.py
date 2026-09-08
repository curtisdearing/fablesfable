"""Settlement contract: win / loss / push / void / unresolved.

Before this contract ``prop_learning.grade_week`` wrote ``hit=0`` for BOTH
sides when the actual equalled an integer line, and wrote ``actual=0.0,
hit=0`` for a player with no stat row -- a settled loss invented from an
absence.  These tests pin the contract end to end: storage (migration v3),
grading, learning updates, the why-report, and the EV fields.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from nflvalue import db as dbmod
from nflvalue import prop_learning as pl
from nflvalue import settlement as st

FIX = Path(__file__).resolve().parent / "fixtures"


def _pw():
    return pd.DataFrame([
        {"season": 2026, "week": 1, "player_id": "P_PUSH", "receptions": 5.0, "rec_yards": 40.0,
         "rush_yards": 0.0, "carries": 0.0, "pass_yards": 0.0, "pass_attempts": 0.0, "targets": 7.0,
         "rush_tds": 0.0, "rec_tds": 0.0},
        {"season": 2026, "week": 1, "player_id": "P_WIN", "receptions": 7.0, "rec_yards": 90.0,
         "rush_yards": 0.0, "carries": 0.0, "pass_yards": 0.0, "pass_attempts": 0.0, "targets": 9.0,
         "rush_tds": 0.0, "rec_tds": 1.0},
        {"season": 2026, "week": 1, "player_id": "P_TD0", "receptions": 3.0, "rec_yards": 20.0,
         "rush_yards": 0.0, "carries": 0.0, "pass_yards": 0.0, "pass_attempts": 0.0, "targets": 4.0,
         "rush_tds": 0.0, "rec_tds": 0.0},
    ])


def _lean(pid, market, side, line, status="active", clock="wed"):
    return {"season": 2026, "week": 1, "clock": clock, "game_id": "2026_01_AAA_BBB", "player_id": pid,
            "name": pid, "market": market, "side": side, "line": line, "mean": line + 0.4,
            "composite": 0.6, "status": status, "proj_components": None}


def _grade(tmp_path, leans):
    conn = dbmod.connect(str(tmp_path / "t.db"))
    out = pl.grade_week(conn, 2026, 1, _pw(), leans=pd.DataFrame(leans))
    stored = dbmod.query_df(conn, "SELECT * FROM lean_outcomes")
    return out, stored, conn


# ---------------------------------------------------------------- settle() --
@pytest.mark.parametrize("side", ["over", "under"])
def test_equality_to_an_integer_line_is_a_push_for_both_sides(side):
    verdict = st.settle("receptions", side, 5.0, actual=5.0, has_stat_row=True)
    assert verdict.settlement == st.PUSH
    assert verdict.hit is None
    assert verdict.actual == 5.0


def test_two_sided_win_and_loss():
    assert st.settle("receptions", "over", 5.5, actual=7.0, has_stat_row=True).settlement == st.WIN
    assert st.settle("receptions", "under", 5.5, actual=7.0, has_stat_row=True).settlement == st.LOSS
    assert st.settle("receptions", "over", 5.5, actual=7.0, has_stat_row=True).hit == 1
    assert st.settle("receptions", "under", 5.5, actual=7.0, has_stat_row=True).hit == 0


def test_missing_stat_row_is_unresolved_never_a_settled_zero():
    v = st.settle("receiving_yards", "under", 40.5, actual=None, has_stat_row=False)
    assert v.settlement == st.UNRESOLVED
    assert v.hit is None and v.actual is None
    # yes-only market too: absence is not a "no"
    v = st.settle("anytime_td", "over", 0.5, actual=None, has_stat_row=False)
    assert v.settlement == st.UNRESOLVED


def test_anytime_td_is_yes_only_and_settles_on_one_or_more():
    assert st.settle("anytime_td", "over", 0.5, actual=1.0, has_stat_row=True).settlement == st.WIN
    assert st.settle("anytime_td", "over", 0.5, actual=0.0, has_stat_row=True).settlement == st.LOSS
    assert st.settle("anytime_td", "over", 0.5, actual=2.0, has_stat_row=True).hit == 1


def test_voided_lean_settles_as_void():
    v = st.settle("receptions", "over", 5.5, actual=7.0, has_stat_row=True, lean_status="voided")
    assert v.settlement == st.VOID and v.hit is None


def test_non_finite_or_missing_actual_with_a_stat_row_is_unresolved_not_a_guess():
    assert st.settle("receptions", "over", 5.5, actual=float("nan"), has_stat_row=True).settlement == st.UNRESOLVED
    assert st.settle("receptions", "over", 5.5, actual=None, has_stat_row=True).settlement == st.UNRESOLVED


# ------------------------------------------------------------ grade_week() --
def test_grade_week_writes_settlement_and_null_hit_for_push_and_unresolved(tmp_path):
    leans = [_lean("P_PUSH", "receptions", "over", 5.0),
             _lean("P_WIN", "receptions", "over", 5.5), _lean("P_GHOST", "receiving_yards", "under", 40.5),
             _lean("P_WIN", "anytime_td", "over", 0.5), _lean("P_TD0", "anytime_td", "over", 0.5)]
    out, stored, conn = _grade(tmp_path, leans)
    # grade_week grades one clock per call; the T-90 under side of the same push lands on its own clock
    t90 = pd.DataFrame([_lean("P_PUSH", "receptions", "under", 5.0, clock="t90")])
    pl.grade_week(conn, 2026, 1, _pw(), leans=t90, clock="t90")
    stored = dbmod.query_df(conn, "SELECT * FROM lean_outcomes")
    by = {(r["player_id"], r["market"], r["clock"]): r for r in stored.to_dict("records")}
    assert by[("P_PUSH", "receptions", "wed")]["settlement"] == st.PUSH
    assert by[("P_PUSH", "receptions", "t90")]["settlement"] == st.PUSH
    assert pd.isna(by[("P_PUSH", "receptions", "wed")]["hit"]) and pd.isna(by[("P_PUSH", "receptions", "t90")]["hit"])
    assert by[("P_WIN", "receptions", "wed")]["settlement"] == st.WIN and by[("P_WIN", "receptions", "wed")]["hit"] == 1
    ghost = by[("P_GHOST", "receiving_yards", "wed")]
    assert ghost["settlement"] == st.UNRESOLVED and pd.isna(ghost["hit"]) and pd.isna(ghost["actual"])
    assert ghost["primary_reason"] == "unresolved"
    assert by[("P_WIN", "anytime_td", "wed")]["settlement"] == st.WIN
    td0 = by[("P_TD0", "anytime_td", "wed")]
    assert td0["settlement"] == st.LOSS and td0["hit"] == 0


def test_grade_week_marks_voided_leans_void_when_passed_explicitly(tmp_path):
    out, stored, _ = _grade(tmp_path, [_lean("P_WIN", "receptions", "over", 5.5, status="voided")])
    assert stored.iloc[0]["settlement"] == st.VOID and pd.isna(stored.iloc[0]["hit"])


# --------------------------------------------------- learning + reports --
def test_learning_reliability_ignores_pushes_and_unresolved_but_keeps_legacy_binary_rows(tmp_path):
    conn = dbmod.connect(str(tmp_path / "l.db"))
    dbmod.upsert(conn, "candidate_aggregates", [
        {"season": 2026, "week": 1, "market": "receptions", "n": 200, "sum_pred": 1000.0, "sum_actual": 1000.0,
         "created_at": "x"}], ["season", "week", "market"])
    rows = [
        {"season": 2026, "week": 1, "clock": "wed", "game_id": "g", "player_id": f"p{i}", "name": "n",
         "market": "receptions", "side": "over", "line": 5.0, "mean": 5.4, "composite": 0.5,
         "actual": a, "hit": h, "settlement": s, "primary_reason": "x", "graded_at": "t"}
        for i, (a, h, s) in enumerate([(7.0, 1, st.WIN), (3.0, 0, st.LOSS), (5.0, None, st.PUSH),
                                       (None, None, st.UNRESOLVED), (6.0, 1, None)])  # last = legacy row
    ]
    dbmod.upsert(conn, "lean_outcomes", rows, ["season", "week", "clock", "game_id", "player_id", "market"])
    state = pl.rebuild_state(conn)
    assert state.lean_history["receptions"] == [1, 0, 1]
    why = pl.why_report(conn, 2026)
    assert why["n"] == 5
    assert why["settled"] == 3 and why["push"] == 1 and why["unresolved"] == 1
    assert why["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)


# ------------------------------------------------------------- migration --
def test_migration_v3_on_the_populated_production_database_is_additive_and_idempotent(tmp_path):
    src = FIX / "nfl_props_state_2026wk1_user_version1.db"
    if not src.exists():
        pytest.skip("populated production-state fixture not present")
    db = tmp_path / "prod.db"
    shutil.copy(src, db)
    raw = sqlite3.connect(db)
    before = {t: raw.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("leans", "lines", "lean_outcomes")}
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 1
    raw.close()
    conn = dbmod.connect(str(db))
    assert dbmod.user_version(conn) == dbmod.SCHEMA_VERSION >= 3
    cols = [r[1] for r in conn.execute("PRAGMA table_info(lean_outcomes)")]
    assert "settlement" in cols
    after = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("leans", "lines", "lean_outcomes")}
    assert after == before
    assert before["leans"] == 80 and before["lines"] == 605
    conn.close()
    conn = dbmod.connect(str(db))
    assert dbmod.user_version(conn) == dbmod.SCHEMA_VERSION
    assert dbmod.migrate(conn) == dbmod.SCHEMA_VERSION


# --------------------------------------------------------- EV per stake --
def test_ev_per_stake_is_distinguished_from_ev_conditional_on_settlement():
    from nflvalue.composite import score_candidate
    from nflvalue.projection import p_over as _po
    prices = {"over": 1.95, "under": 1.95, "book": "b", "n_books": 2}
    integer = {"player_id": "P", "name": "P", "pos": "WR", "team": "AAA", "defteam": "BBB",
               "game_id": "2026_01_AAA_BBB", "matchup": "AAA @ BBB", "market": "receptions",
               "mean": 5.6, "sd": 2.4, "dist": "negbinom", "line": 5.0, "line_source": "odds_api",
               "p_over": round(_po(5.6, 2.4, 5.0, "negbinom"), 4), "components": {}, "prices": prices}
    integer["p_under"] = round(1 - integer["p_over"], 4)
    out = score_candidate(integer, params={"calibration_passed": True})
    c = out["components"]
    assert c["p_push"] > 0
    assert c["ev_best_price"] is not None and c["ev_per_stake"] is not None
    assert c["ev_per_stake"] == pytest.approx((1 - c["p_push"]) * c["ev_best_price"], abs=2e-4)
    assert c["ev_basis"] == {"ev_best_price": "conditional_on_settlement", "ev_per_stake": "per_original_stake"}
    half = {**integer, "line": 5.5, "p_over": round(_po(5.6, 2.4, 5.5, "negbinom"), 4)}
    half["p_under"] = round(1 - half["p_over"], 4)
    c2 = score_candidate(half, params={"calibration_passed": True})["components"]
    assert c2["p_push"] == 0.0
    assert c2["ev_per_stake"] == pytest.approx(c2["ev_best_price"], abs=1e-9)
