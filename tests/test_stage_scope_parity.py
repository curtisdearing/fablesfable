"""Stage disclosure scope == the executing consumer's scope.

The card label for a primary stage must come from what the stage can actually touch.
These tests run the real ``candidates`` stage functions on every market and compare
the rows they changed with the per-row stamp states -- no copied scope constants.
"""
import numpy as np
import pandas as pd

from nflvalue import candidates as candmod
from nflvalue import factor_integration as fimod

MARKETS = ["receiving_yards", "receptions", "passing_yards", "pass_attempts", "completions",
           "passing_tds", "rushing_yards", "rush_attempts"]


def _cands(**extra):
    rows = [{"player_id": f"P{i}", "team": "ATL", "market": m, "mean": 50.0, "sd": 10.0,
             "line": 45.5, "dist": "normal", "p_over": 0.6, "p_under": 0.4, **extra}
            for i, m in enumerate(MARKETS)]
    return pd.DataFrame(rows)


def _stamp_states(cands, stage):
    ran = {s: True for s in fimod.STAGES}
    stamps = fimod.build_stamps(cands, ran, {})
    return {m: stamps[(pid, m)]["stages"][stage]["state"]
            for pid, m in zip(cands["player_id"], cands["market"])}


def test_backup_qb_disclosure_matches_the_markets_the_stage_changes():
    before = _cands(qb_continuity=0.2)
    after = candmod.apply_backup_qb_adjustment(before)
    changed = set(after.loc[after["mean"] != before["mean"], "market"])
    # the actual consumer: receivers, receptions and QB passing yards -- not attempts/completions/TDs
    assert changed == {"receiving_yards", "receptions", "passing_yards"}
    states = _stamp_states(after, "backup_qb")
    assert {m for m, s in states.items() if s == "applied"} == changed
    assert {m for m, s in states.items() if s == "not_applicable"} == set(MARKETS) - changed


def test_backup_qb_missing_input_is_not_evaluated_on_every_market_the_stage_covers():
    cands = _cands(qb_continuity=np.nan)
    after = candmod.apply_backup_qb_adjustment(cands)
    assert after["mean"].tolist() == cands["mean"].tolist()
    states = _stamp_states(after, "backup_qb")
    for m in ("receiving_yards", "receptions", "passing_yards"):
        assert states[m] == "not_evaluated", m
    for m in ("pass_attempts", "completions", "passing_tds", "rushing_yards"):
        assert states[m] == "not_applicable", m


def test_absence_qb_disclosure_matches_the_markets_the_stage_changes():
    pw = pd.DataFrame([{"season": 2026, "week": 1, "team": "ATL", "player_id": "WR1",
                        "role": "WR", "targets": 40, "carries": 0}])
    before = _cands()
    after = candmod.apply_absence_qb_adjustment(before, pw, 2026, 3, {"WR1"})
    changed = set(after.loc[after["mean"] != before["mean"], "market"])
    assert changed == {"passing_yards", "pass_attempts"}
    states = _stamp_states(after, "absence_qb")
    assert {m for m, s in states.items() if s == "applied"} == changed
    assert {m for m, s in states.items() if s == "not_applicable"} == set(MARKETS) - changed
