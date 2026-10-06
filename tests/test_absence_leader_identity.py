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
from nflvalue import candidates as candmod  # noqa: E402
from nflvalue import factor_integration as fimod  # noqa: E402
from nflvalue.candidates import WeekInputs  # noqa: E402
from nflvalue.freshness import stamp_now  # noqa: E402
from tests.test_pipeline_weekly import GAME_ID, _fresh_feeds, _roster, env  # noqa: E402,F401
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


# All feeds below are intentionally SYNTHETIC fixtures.  They exercise only
# caller ordering and receipt behavior, not NFL personnel accuracy.
ABSENCE_WR = candmod.ABSENCE_QB_MULT["WR"]


def _t90_feeds(feeds, now, *, inactive_rows=None):
    """Add a frozen populated T-90 inactives response to a synthetic feed set."""
    result = dict(feeds)
    result["inactive_rows"] = list(inactive_rows or [
        {"espn_id": "9", "name": "Nobody Inactive", "active": False,
         "did_not_play": True, "starter": False, "team": "BBB"},
    ])
    result["inactives_fetched_at"] = now
    return result


def _current_out_leader_feeds(now):
    """WR_A remains AAA's current rostered historical WR leader and is OUT."""
    feeds = dict(_fresh_feeds(now, wr_a_status="Out"))
    feeds["active_roster"] = _roster(
        now, extra=[{"player_id": "QB_A", "team": "AAA", "status": "ACT", "week": WEEK}])
    return _t90_feeds(feeds, now, inactive_rows=[
        {"espn_id": "1", "name": "Alpha Wideout", "active": False,
         "did_not_play": True, "starter": True, "team": "AAA"},
    ])


def _capture_stamped_runs(monkeypatch):
    """Capture post-adjustment frames that each native caller hands to stamps."""
    seen = []
    real_build = fimod.build_stamps

    def capture(cands, ran, why, **kwargs):
        stamps = real_build(cands, ran, why, **kwargs)
        seen.append({"cands": cands.copy(), "ran": dict(ran), "stamps": stamps})
        return stamps

    monkeypatch.setattr(fimod, "build_stamps", capture)
    return seen


def _qb_markets(run):
    rows = run["cands"]
    rows = rows[(rows["player_id"] == "QB_A") & rows["market"].isin(candmod._QB_MARKETS)]
    return {r["market"]: r for r in rows.to_dict("records")}


def _factor(row):
    value = row.get("absence_qb_mult")
    return None if value is None or pd.isna(value) else float(value)


def test_t90_rejects_moved_historical_leader_with_same_identity_receipt_as_wednesday(env, monkeypatch):
    """A former AAA WR cannot become an AAA QB absence at the refresh clock."""
    seen = _capture_stamped_runs(monkeypatch)
    now = stamp_now()
    feeds = _t90_feeds(_trade_feeds(now), now)
    wed = pw.run_week(SEASON, WEEK, mode="live", inputs=_inputs_with_aaa_qb(), inject_feeds=feeds)
    t90 = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=_inputs_with_aaa_qb(),
                     inject_feeds=feeds)

    wed_rows, t90_rows = _qb_markets(seen[0]), _qb_markets(seen[1])
    assert wed_rows and set(wed_rows) == set(t90_rows)
    for market in wed_rows:
        assert _factor(wed_rows[market]) is None
        assert _factor(t90_rows[market]) is None
        assert t90_rows[market]["absence_qb_identity_state"] == "partial"
        assert t90_rows[market]["mean"] == pytest.approx(wed_rows[market]["mean"])
        assert seen[1]["stamps"][("QB_A", market)]["stages"]["absence_qb"] == \
            seen[0]["stamps"][("QB_A", market)]["stages"]["absence_qb"]
    assert t90["factor_receipt"]["absence_leader_identity"] == wed["factor_receipt"]["absence_leader_identity"]
    assert t90["factor_receipt"]["absence_leader_identity"]["teams"]["AAA"]["rejected"] == [{
        "role": "WR", "player_id": "WR_A", "reason": "current_team_mismatch", "roster_team": "BBB",
    }]


def test_t90_keeps_confirmed_out_current_leader_factor_and_matches_wednesday_once(env, monkeypatch):
    """The same frozen OUT leader produces one x0.921 adjustment at both clocks."""
    seen = _capture_stamped_runs(monkeypatch)
    now = stamp_now()
    feeds = _current_out_leader_feeds(now)
    wed = pw.run_week(SEASON, WEEK, mode="live", inputs=_inputs_with_aaa_qb(), inject_feeds=feeds)
    t90 = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=_inputs_with_aaa_qb(),
                     inject_feeds=feeds)

    wed_rows, t90_rows = _qb_markets(seen[0]), _qb_markets(seen[1])
    assert wed_rows and set(wed_rows) == set(t90_rows)
    for market in wed_rows:
        assert _factor(wed_rows[market]) == ABSENCE_WR
        assert _factor(t90_rows[market]) == ABSENCE_WR
        assert t90_rows[market]["mean"] == pytest.approx(wed_rows[market]["mean"])
        assert seen[1]["stamps"][("QB_A", market)]["stages"]["absence_qb"] == \
            seen[0]["stamps"][("QB_A", market)]["stages"]["absence_qb"]
    for result in (wed, t90):
        assert "absence_qb" in result["factor_receipt"]["stages_executed"]
        assert result["factor_receipt"]["absence_leader_identity"]["teams"]["AAA"]["state"] == "verified"


def test_t90_applies_absence_factor_once_not_twice(env, monkeypatch):
    """A factor applied at T-90 must be baseline x0.921, never x0.921 squared."""
    seen = _capture_stamped_runs(monkeypatch)
    now = stamp_now()
    base_feeds = dict(_fresh_feeds(now, wr_a_status="Active"))
    base_feeds["active_roster"] = _roster(
        now, extra=[{"player_id": "QB_A", "team": "AAA", "status": "ACT", "week": WEEK}])
    pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=_inputs_with_aaa_qb(),
               inject_feeds=_t90_feeds(base_feeds, now))
    pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=_inputs_with_aaa_qb(),
               inject_feeds=_current_out_leader_feeds(now))

    base_rows, out_rows = _qb_markets(seen[0]), _qb_markets(seen[1])
    assert base_rows and set(base_rows) == set(out_rows)
    for market in base_rows:
        assert _factor(base_rows[market]) is None
        assert _factor(out_rows[market]) == ABSENCE_WR
        once = round(float(base_rows[market]["mean"]) * ABSENCE_WR, 3)
        twice = round(float(base_rows[market]["mean"]) * ABSENCE_WR ** 2, 3)
        assert float(out_rows[market]["mean"]) == pytest.approx(once)
        assert float(out_rows[market]["mean"]) != pytest.approx(twice)


def test_historical_t90_keeps_absence_stage_not_live(env):
    """The live repair must not relabel a historical T-90 receipt as executed."""
    now = stamp_now()
    result = pw.run_t90(SEASON, WEEK, GAME_ID, mode="historical", inputs=_inputs_with_aaa_qb(),
                        inject_feeds=_current_out_leader_feeds(now))
    receipt = result["factor_receipt"]
    assert "absence_qb" not in receipt["stages_executed"]
    assert receipt["stages_not_executed"]["absence_qb"] == "not a live run"
