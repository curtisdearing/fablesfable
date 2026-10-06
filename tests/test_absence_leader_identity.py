"""Live absence-QB leader identity: native caller regression coverage.

The historical usage leader is a *measurement* anchored before the forecast
week.  Its membership for a live forecast must nevertheless be verified from
the same as-of active-roster snapshot that seated candidates.  A traded player
must not remain the old team's absent leader merely because his historical
usage is still there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue.candidates import WeekInputs  # noqa: E402
from tests.test_pipeline_weekly import _fresh_feeds, _roster, env  # noqa: E402,F401
from tests.test_report_phase2 import SEASON, WEEK, _pw_row, synthetic_inputs  # noqa: E402


def _inputs_with_aaa_qb():
    """AAA has a live QB; WR_A remains its trailing historical WR leader."""
    base = synthetic_inputs()
    qba = [_pw_row(wk, "QB_A", "Alpha Quarterback", "AAA", "BBB", "QB",
                   pass_attempts=31.0, completions=20.0, pass_yards=230.0,
                   roll_pass_attempts=31.0, roll_completions=20.0, roll_ypa=7.4)
           for wk in range(1, WEEK + 1)]
    return WeekInputs(pw=pd.concat([base.pw, pd.DataFrame(qba)], ignore_index=True),
                      opd=base.opd, tw=base.tw, schedules=base.schedules)


def _trade_feeds(now):
    """WR_A's pregame identity is BBB, while his usage history is all AAA."""
    feeds = dict(_fresh_feeds(now, wr_a_status="Out"))
    roster = _roster(now, extra=[{"player_id": "QB_A", "team": "AAA", "status": "ACT", "week": WEEK}])
    roster["rows"] = [({**r, "team": "BBB"} if r["player_id"] == "WR_A" else r)
                      for r in roster["rows"]]
    feeds["active_roster"] = roster
    # The injury source correctly identifies the receiver on his current team.
    feeds["injury_rows"] = [{**r, "team": "BBB"} if r["name"] == "Alpha Wideout" else r
                            for r in feeds["injury_rows"]]
    return feeds


def test_live_run_rejects_traded_historical_leader_from_old_team_absence(env, monkeypatch):
    """Native ``run_week`` must not haircut AAA QB markets for a WR now on BBB."""
    from nflvalue.freshness import stamp_now

    seen = {}
    real_generate = pw.rptmod.generate

    def capture_candidates(*args, **kwargs):
        seen["candidates"] = kwargs["candidates_df"].copy()
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(pw.rptmod, "generate", capture_candidates)
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=_inputs_with_aaa_qb(),
                      inject_feeds=_trade_feeds(stamp_now()))

    qba = seen["candidates"][(seen["candidates"]["player_id"] == "QB_A")
                             & seen["candidates"]["market"].isin(["passing_yards", "pass_attempts"])]
    assert len(qba) >= 1, "test premise: live AAA QB passing candidate reached the native caller"
    multipliers = qba.get("absence_qb_mult", pd.Series(index=qba.index, dtype=float))
    assert not multipliers.notna().any(), (
        "WR_A is OUT for BBB, but his trailing AAA usage must not affect AAA QB after the trade"
    )
    aaa_identity = res["factor_receipt"]["absence_leader_identity"]["teams"]["AAA"]
    # AAA also has independently verified leaders in this fixture.  The stale WR
    # blocks only its own role; it must not suppress legitimate role effects.
    assert aaa_identity["state"] == "partial"
    assert aaa_identity["rejected"] == [{"role": "WR", "player_id": "WR_A",
                                         "reason": "current_team_mismatch", "roster_team": "BBB"}]


def _leader_history(*, include_target_week=False):
    """AAA's valid pre-week WR leader is W1; a W3 row is deliberately future data."""
    rows = [
        {"season": 2026, "week": 1, "team": "AAA", "player_id": "WR_OLD", "role": "WR",
         "targets": 40, "carries": 0},
        {"season": 2026, "week": 1, "team": "AAA", "player_id": "RB_A", "role": "RB",
         "targets": 0, "carries": 35},
    ]
    if include_target_week:
        rows.append({"season": 2026, "week": 3, "team": "AAA", "player_id": "WR_FUTURE",
                     "role": "WR", "targets": 500, "carries": 0})
    return pd.DataFrame(rows)


def _qb_candidates():
    return pd.DataFrame([
        {"player_id": "QB_A", "team": "AAA", "market": "passing_yards", "mean": 100.0,
         "sd": 10.0, "line": 95.5, "dist": "normal", "p_over": 0.6, "p_under": 0.4},
        {"player_id": "QB_A", "team": "AAA", "market": "pass_attempts", "mean": 30.0,
         "sd": 5.0, "line": 29.5, "dist": "normal", "p_over": 0.6, "p_under": 0.4},
    ])


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ([], "missing_current_identity"),
        ([{"player_id": "WR_OLD", "team": "AAA", "status": "ACT"},
          {"player_id": "WR_OLD", "team": "BBB", "status": "ACT"}],
         "ambiguous_current_identity"),
    ],
    ids=["sparse_snapshot", "same_id_conflicting_teams"],
)
def test_live_leader_identity_fails_closed_for_sparse_or_ambiguous_roster(rows, reason):
    from nflvalue import candidates as candmod

    after = candmod.apply_absence_qb_adjustment(
        _qb_candidates(), _leader_history(), 2026, 3, {"WR_OLD"}, active_roster_rows=rows)
    assert after["mean"].tolist() == [100.0, 30.0]
    identity = after.attrs["absence_leader_identity"]["teams"]["AAA"]
    assert identity["state"] == "blocked"
    assert any(r["player_id"] == "WR_OLD" and r["reason"] == reason for r in identity["rejected"])
    assert set(after["absence_qb_identity_state"]) == {"blocked"}


def test_live_identity_keeps_verified_current_leader_adjustment_and_historical_path():
    from nflvalue import candidates as candmod

    history = _leader_history()
    roster = [{"player_id": "WR_OLD", "team": "AAA", "status": "ACT"},
              {"player_id": "RB_A", "team": "AAA", "status": "ACT"}]
    live = candmod.apply_absence_qb_adjustment(
        _qb_candidates(), history, 2026, 3, {"WR_OLD"}, active_roster_rows=roster)
    historical = candmod.apply_absence_qb_adjustment(_qb_candidates(), history, 2026, 3, {"WR_OLD"})
    assert live["mean"].tolist() == historical["mean"].tolist() == [92.1, 27.63]
    assert live.attrs["absence_leader_identity"]["teams"]["AAA"]["state"] == "verified"
    assert historical.attrs["absence_leader_identity"]["mode"] == "historical_asof"


def test_invalid_historical_role_does_not_suppress_verified_current_role():
    """A stale WR must not erase a valid, independently identified RB effect."""
    from nflvalue import candidates as candmod

    roster = [{"player_id": "WR_OLD", "team": "BBB", "status": "ACT"},
              {"player_id": "RB_A", "team": "AAA", "status": "ACT"}]
    after = candmod.apply_absence_qb_adjustment(
        _qb_candidates(), _leader_history(), 2026, 3, {"RB_A"}, active_roster_rows=roster)
    assert after["mean"].tolist() == [97.1, 29.13]
    identity = after.attrs["absence_leader_identity"]["teams"]["AAA"]
    assert identity["state"] == "partial"
    assert identity["accepted"] == [{"role": "RB", "player_id": "RB_A"}]
    assert identity["rejected"] == [{"role": "WR", "player_id": "WR_OLD",
                                     "reason": "current_team_mismatch", "roster_team": "BBB"}]


def test_leader_selection_remains_preweek_safe_with_mixed_played_unplayed_rows():
    from nflvalue import candidates as candmod

    roster = [{"player_id": "WR_OLD", "team": "AAA", "status": "ACT"},
              {"player_id": "WR_FUTURE", "team": "AAA", "status": "ACT"},
              {"player_id": "RB_A", "team": "AAA", "status": "ACT"}]
    identity = candmod.team_leader_identity(_leader_history(include_target_week=True), 2026, 3, roster)
    assert {tuple(r[k] for k in ("team", "role", "player_id")) for r in identity["leaders"]} >= {
        ("AAA", "WR", "WR_OLD")}
    assert identity["teams"]["AAA"]["state"] == "verified"
