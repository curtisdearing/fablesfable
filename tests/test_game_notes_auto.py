"""Game-notes writeup + the self-scheduling wrapper's week detection."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import game_notes as gn  # noqa: E402

ET = ZoneInfo("America/New_York")


def _sched():
    rows = [
        dict(game_id="2025_01_A_B", season=2025, week=1, game_type="REG",
             gameday="2025-09-07", gametime="13:00", home_team="BBB", away_team="AAA",
             result=-3.0, spread_line=2.0, total_line=44.0),   # away AAA wins
        dict(game_id="2025_01_C_D", season=2025, week=1, game_type="REG",
             gameday="2025-09-07", gametime="16:25", home_team="DDD", away_team="CCC",
             result=7.0, spread_line=3.0, total_line=47.0),    # home DDD wins
        dict(game_id="2025_02_A_D", season=2025, week=2, game_type="REG",
             gameday="2025-09-14", gametime="13:00", home_team="DDD", away_team="AAA",
             result=np.nan, spread_line=1.0, total_line=45.0),
    ]
    return pd.DataFrame(rows)


def test_records_walk_forward():
    rec = gn.build_records(_sched())
    assert gn.record_for(rec, 2025, 1, "AAA") == "0-0"      # entering week 1
    assert gn.record_for(rec, 2025, 2, "AAA") == "1-0"      # won week 1
    assert gn.record_for(rec, 2025, 2, "DDD") == "1-0"
    assert gn.record_for(rec, 2025, 2, "BBB") == "0-1"


def test_game_notes_collects_stories():
    cands = [
        {"name": "A.One", "team": "AAA", "defteam": "DDD", "is_birthday_week": 1,
         "revenge_game": 0, "is_contract_year": 1, "def_out_total": 3.0,
         "def_out_db": 2.0, "oline_outs": 2.0, "qb_continuity": 0.3,
         "wind": 18.0, "temp": 40.0},
        {"name": "B.Two", "team": "DDD", "defteam": "AAA", "is_birthday_week": 0,
         "revenge_game": 1, "is_contract_year": 0, "def_out_total": 0.0,
         "def_out_db": 0.0, "oline_outs": 0.0, "qb_continuity": 0.9,
         "wind": 18.0, "temp": 40.0},
    ]
    rec = gn.build_records(_sched())
    notes = gn.build_game_notes("2025_02_A_D", cands, rec, 2025, 2, "DDD", "AAA")
    joined = " | ".join(notes)
    assert "Records: AAA 1-0 @ DDD 1-0" in joined
    assert "Birthday week: A.One (AAA)" in joined
    assert "Revenge game: B.Two (DDD — ex-AAA)" in joined
    assert "Contract year" in joined and "A.One" in joined
    assert "DDD defense lists 3 Out/Doubtful (2 in the secondary)" in joined
    assert "AAA O-line lists 2" in joined
    assert "18 mph wind" in joined
    assert "AAA: projected starting QB threw only 30%" in joined


def test_attach_notes_stamps_games():
    games = [{"game_id": "2025_02_A_D", "leans": []}]
    cands = pd.DataFrame([{"game_id": "2025_02_A_D", "name": "A.One", "team": "AAA",
                           "defteam": "DDD", "is_birthday_week": 1, "revenge_game": 0,
                           "is_contract_year": 0, "def_out_total": 0.0, "def_out_db": 0.0,
                           "oline_outs": 0.0, "qb_continuity": 0.9, "wind": 2.0,
                           "temp": 70.0}])
    gn.attach_notes(games, cands, _sched(), 2025, 2)
    assert any("Records:" in n for n in games[0]["notes"])
    assert any("Birthday" in n for n in games[0]["notes"])


# --------------------------------------------------------------------------- #
# Phase 7.6 -- correlation flags surfaced in the same display-only notes list
# --------------------------------------------------------------------------- #
def _corr_stub(rho):
    from nflvalue.correlation import CorrelationStructure
    return CorrelationStructure({
        "pair_types": {"sameplayer|WR.rec~WR.rec": {"rho_shrunk": rho, "verdict": "real"}},
        "walk_forward": {},
    })


def _two_correlated_leans():
    return [
        {"player_id": "P1", "name": "A.One", "pos": "WR", "team": "AAA", "market": "receiving_yards"},
        {"player_id": "P1", "name": "A.One", "pos": "WR", "team": "AAA", "market": "receptions"},
    ]


def test_correlation_notes_for_game_flags_real_pair():
    notes = gn.correlation_notes_for_game(_two_correlated_leans(), _corr_stub(0.76))
    assert len(notes) == 1
    assert "Correlated legs" in notes[0] and "0.76" in notes[0] and "move together" in notes[0]


def test_correlation_notes_for_game_negative_rho_says_hedge():
    notes = gn.correlation_notes_for_game(_two_correlated_leans(), _corr_stub(-0.30))
    assert "hedge against each other" in notes[0]


def test_correlation_notes_for_game_empty_without_corr():
    assert gn.correlation_notes_for_game(_two_correlated_leans(), None) == []


def test_attach_notes_appends_correlation_flags_when_corr_given():
    games = [{"game_id": "2025_02_A_D", "leans": _two_correlated_leans()}]
    cands = pd.DataFrame([{"game_id": "2025_02_A_D", "name": "A.One", "team": "AAA",
                           "defteam": "DDD", "is_birthday_week": 0, "revenge_game": 0,
                           "is_contract_year": 0, "def_out_total": 0.0, "def_out_db": 0.0,
                           "oline_outs": 0.0, "qb_continuity": 0.9, "wind": 2.0,
                           "temp": 70.0}])
    gn.attach_notes(games, cands, _sched(), 2025, 2, corr=_corr_stub(0.76))
    assert any("Correlated legs" in n for n in games[0]["notes"])


def test_attach_notes_unaffected_without_corr():
    """Backward-compatible default: no corr -> no correlation notes, and the
    existing story notes are untouched."""
    games = [{"game_id": "2025_02_A_D", "leans": _two_correlated_leans()}]
    cands = pd.DataFrame([{"game_id": "2025_02_A_D", "name": "A.One", "team": "AAA",
                           "defteam": "DDD", "is_birthday_week": 0, "revenge_game": 0,
                           "is_contract_year": 0, "def_out_total": 0.0, "def_out_db": 0.0,
                           "oline_outs": 0.0, "qb_continuity": 0.9, "wind": 2.0,
                           "temp": 70.0}])
    gn.attach_notes(games, cands, _sched(), 2025, 2)
    assert not any("Correlated legs" in n for n in games[0]["notes"])


# --------------------------------------------------------------------------- #
# auto_weekly week detection (pure functions, injected clock)
# --------------------------------------------------------------------------- #
def _wrapper_slate():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import auto_weekly as aw
    s = _sched()
    s["kickoff"] = [dt.datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
                    for d, t in zip(s["gameday"], s["gametime"])]
    return aw, s


def test_current_week_detection():
    aw, s = _wrapper_slate()
    wed_before_wk2 = dt.datetime(2025, 9, 10, 10, 0, tzinfo=ET)
    assert aw.current_week(s, wed_before_wk2) == (2025, 2)
    sunday_wk1_morning = dt.datetime(2025, 9, 7, 9, 0, tzinfo=ET)
    assert aw.current_week(s, sunday_wk1_morning) == (2025, 1)
    offseason = dt.datetime(2026, 3, 1, 10, 0, tzinfo=ET)
    assert aw.current_week(s, offseason) is None


def test_last_completed_week():
    aw, s = _wrapper_slate()
    tue_after_wk1 = dt.datetime(2025, 9, 9, 9, 0, tzinfo=ET)
    assert aw.last_completed_week(s, tue_after_wk1) == (2025, 1)
    before_season = dt.datetime(2025, 9, 1, 9, 0, tzinfo=ET)
    assert aw.last_completed_week(s, before_season) is None
