"""Native live team identity through run_week / run_t90.

RED against a8e30bb: the live WED and T-90 runs enumerated carry-forward
candidates BEFORE the active roster was acquired and without a decision
clock, so a player who changed teams sat on his last played team (and the
roster gate then dropped him); identity provenance was in no run receipt.

Synthetic two-team slate (tests/test_report_phase2.synthetic_inputs) plus WR_X,
who played weeks 1-8 for BBB and whose live roster row puts him on AAA. The
roster payloads and their clocks are hand-built: this checks the wiring only
and is not football evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import candidates as candmod  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue import features as F  # noqa: E402
from nflvalue.candidates import WeekInputs  # noqa: E402
from tests.test_pipeline_weekly import GAME_ID, _fresh_feeds, _roster, env  # noqa: E402,F401
from tests.test_report_phase2 import SEASON, WEEK, _pw_row, synthetic_inputs  # noqa: E402

X_ROW = {"player_id": "WR_X", "team": "AAA", "status": "ACT", "week": WEEK}


def _inputs():
    base = synthetic_inputs()
    extra = [_pw_row(wk, "WR_X", "Xfer Wideout", "BBB", "AAA", "WR",
                     targets=8.0, receptions=5.0, rec_yards=68.0,
                     roll_targets=8.0, roll_target_share=0.25, roll_ypt=8.4, roll_catch_rate=0.64)
             for wk in range(1, 9)]
    pwf = pd.concat([base.pw, pd.DataFrame(extra)], ignore_index=True)
    return WeekInputs(pw=pwf, opd=base.opd, tw=base.tw, schedules=base.schedules)


def _feeds(now, fetched_at="same"):
    f = dict(_fresh_feeds(now))
    ros = _roster(now, extra=[X_ROW])
    if fetched_at is None:
        ros.pop("fetched_at")
    elif fetched_at != "same":
        ros["fetched_at"] = fetched_at
    f["active_roster"] = ros
    return f


@pytest.fixture()
def spy(monkeypatch):
    """Record every enumerate_candidates result the pipeline itself produces."""
    seen = []
    real = candmod.enumerate_candidates

    def wrapped(*a, **kw):
        out = real(*a, **kw)
        seen.append({"kw": kw, "df": out.copy(), "attrs": dict(out.attrs)})
        return out
    monkeypatch.setattr(candmod, "enumerate_candidates", wrapped)
    return seen


@pytest.fixture()
def no_odds(monkeypatch):
    calls = {"odds": 0, "events": 0}

    def odds(*a, **k):
        calls["odds"] += 1
        raise AssertionError("no odds acquisition in these runs")

    def events(*a, **k):
        calls["events"] += 1
        raise AssertionError("no event listing in these runs")
    return calls, odds, events


def _receipt(clock):
    conn = dbmod.connect()
    rows = dbmod.query_df(conn, "SELECT receipt_json FROM run_receipts WHERE clock=? "
                                "ORDER BY created_at DESC", (clock,))
    conn.close()
    assert len(rows), f"no {clock} run receipt persisted"
    return json.loads(rows["receipt_json"].iloc[0])


def _x(df):
    return df[df["player_id"] == "WR_X"]


def test_wed_seats_transfer_in_new_team_game_with_verified_identity(env, spy, no_odds):
    from nflvalue.freshness import stamp_now
    calls, odds, events = no_odds
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=_inputs(), inject_feeds=_feeds(stamp_now()),
                      odds_fetch=odds, list_events_fn=events)
    first = spy[0]
    assert first["kw"]["decision_at"] is not None                      # clock reached the consumer
    x = _x(first["df"])
    assert len(x) and set(x["team"]) == {"AAA"} and set(x["defteam"]) == {"BBB"}
    assert set(x["game_id"]) == {GAME_ID}
    # unchanged stat history: WR_X's history equals WR_A's, so on the same team every market matches
    a = first["df"][first["df"]["player_id"] == "WR_A"].set_index("market")["mean"]
    pd.testing.assert_series_equal(x.set_index("market")["mean"], a, check_names=False)
    # availability/eligibility gates still ran and kept him (roster team == seat)
    assert res["publish"] is True
    assert any(l["player_id"] == "WR_X" for g in res["games"] for l in g["leans"])
    ident = _receipt("wed")["team_identity"]
    assert ident["clock"] == "decision_at" and ident["decision_at"] == ident["identity_at"]
    assert ident["roster_fetched_at"] <= ident["identity_at"] <= res["as_of"]
    assert {"player_id": "WR_X", "from_team": "BBB", "to_team": "AAA",
            "team_source": F.TEAM_SOURCE_VERIFIED, "roster_season": SEASON,
            "roster_week": WEEK} in ident["reseated"]
    assert ident["n_unseated"] == 0
    assert calls == {"odds": 0, "events": 0}


@pytest.mark.parametrize("fetched_at", ["2099-01-01T00:00:00Z", None, "2026-09-24T12:00:00"],
                         ids=["captured_after_decision", "unknown_capture", "naive_capture"])
def test_wed_rejects_late_or_unknown_same_week_identity(env, spy, fetched_at):
    from nflvalue.freshness import stamp_now
    feeds = _feeds(stamp_now(), fetched_at=fetched_at)
    assert all(r["week"] == WEEK for r in feeds["active_roster"]["rows"])   # SAME week rows
    pw.run_week(SEASON, WEEK, mode="live", inputs=_inputs(), inject_feeds=feeds)
    raw = spy[0]["df"]
    assert _x(raw).empty                                  # never seated on BBB as if verified
    assert spy[0]["attrs"]["asof_team_identity"]["team_source_counts"].get(F.TEAM_SOURCE_VERIFIED, 0) == 0
    ident = _receipt("wed")["team_identity"]
    assert ident["reseated"] == []
    unseated = {u["player_id"]: u for u in ident["unseated_on_slate"]}
    assert unseated["WR_X"]["last_played_team"] == "BBB"
    assert unseated["WR_X"]["team_source"] == F.TEAM_SOURCE_ROSTER_REJECTED
    assert ident["n_unseated"] >= 4


def test_roster_acquired_once_before_candidates(env, spy, monkeypatch):
    from nflvalue.freshness import stamp_now
    from nflvalue.sources import active_roster as armod
    order = []
    now = stamp_now()

    def fake_fetch(season, week=None, http=None):
        order.append("roster")
        return _roster(now, extra=[X_ROW])
    monkeypatch.setattr(armod, "fetch_active_roster", fake_fetch)
    real = candmod.enumerate_candidates
    monkeypatch.setattr(candmod, "enumerate_candidates",
                        lambda *a, **k: (order.append("candidates"), real(*a, **k))[1])
    feeds = _feeds(now)
    feeds.pop("active_roster")                               # production path: real acquisition
    pw.run_week(SEASON, WEEK, mode="live", inputs=_inputs(), inject_feeds=feeds)
    assert order[0] == "roster" and order.count("roster") == 1
    ident = _receipt("wed")["team_identity"]
    assert ident["roster_fetched_at"] == now and ident["reseated"][0]["player_id"] == "WR_X"


def test_t90_uses_the_same_live_identity(env, spy):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    feeds = _feeds(now)
    feeds["inactive_rows"] = [{"espn_id": "9", "name": "Bravo Quarterback", "active": True,
                               "did_not_play": False, "starter": True, "team": "BBB"}]
    feeds["inactives_fetched_at"] = now
    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=_inputs(), inject_feeds=feeds)
    x = _x(spy[0]["df"])
    assert set(x["team"]) == {"AAA"} and set(x["game_id"]) == {GAME_ID}
    ident = _receipt("t90")["team_identity"]
    assert ident["clock"] == "decision_at" and ident["identity_at"] <= res["as_of"]
    assert [r["player_id"] for r in ident["reseated"]] == ["WR_X"]


def test_t90_late_identity_is_rejected_not_last_played(env, spy):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    feeds = _feeds(now, fetched_at="2099-01-01T00:00:00Z")
    feeds["inactive_rows"], feeds["inactives_fetched_at"] = [], now
    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=_inputs(), inject_feeds=feeds)
    raw = spy[0]["df"]
    # every rostered player's identity was rejected: none is seated anywhere
    assert not set(raw["player_id"]) & {"WR_X", "WR_A", "RB_A", "QB_B"}
    assert all(l["player_id"] != "WR_X" for g in res.get("games") or [] for l in g.get("leans") or [])
    ident = _receipt("t90")["team_identity"]
    assert {u["player_id"] for u in ident["unseated_on_slate"]} == {"WR_X", "WR_A", "RB_A", "QB_B"}
    assert ident["reseated"] == [] and ident["roster_fetched_at"] > ident["identity_at"]


def test_t90_error_names_identity_when_nobody_is_seated(env, spy):
    from nflvalue.freshness import stamp_now
    now = stamp_now()
    inp = _inputs()
    inp = WeekInputs(pw=inp.pw[inp.pw["player_id"] != "WR_COLD"], opd=inp.opd, tw=inp.tw,
                     schedules=inp.schedules)
    feeds = _feeds(now, fetched_at=None)
    feeds["inactive_rows"], feeds["inactives_fetched_at"] = [], now
    with pytest.raises(ValueError, match=r"4 player\(s\) unseated"):
        pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=inp, inject_feeds=feeds)


def test_historical_mode_unchanged_no_identity_clock(env, spy):
    pw.run_week(SEASON, WEEK, mode="historical", inputs=_inputs())
    assert spy[0]["kw"]["decision_at"] is None and spy[0]["kw"]["roster_mode"] == "as_played"
    assert _receipt("wed")["team_identity"]["clock"].startswith("not a live run")
