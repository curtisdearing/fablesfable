"""Availability resolver (nflvalue/sources/availability.py): two-clock status
resolution + honest usage reallocation (PHASE1_HANDSOFF_DESIGN.md H1/H2/H8)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.sources import availability as av  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

# SYNTHETIC inactives (clearly labeled): ESPN zeroes `active` after a game
# ends, so a REAL pre-kick payload can't be recorded in the offseason. The
# recorded fixture (espn_event_roster_recorded.json) pins the SCHEMA; this
# synthetic list exercises the T-90 BEHAVIOR.
SYNTH_INACTIVES = [
    {"espn_id": "1", "name": "Amon-Ra St. Brown", "active": True, "did_not_play": False, "starter": True},
    {"espn_id": "2", "name": "Sam LaPorta", "active": False, "did_not_play": True, "starter": False},
    {"espn_id": "3", "name": "Jahmyr Gibbs", "active": True, "did_not_play": False, "starter": False},
]


@pytest.fixture(scope="module")
def injuries_fixture():
    return json.loads((FIXTURES / "espn_injuries_recorded.json").read_text())


@pytest.fixture(scope="module")
def roster_fixture():
    return json.loads((FIXTURES / "espn_event_roster_recorded.json").read_text())


def test_parse_team_injuries_recorded(injuries_fixture):
    rows = av.parse_team_injuries(injuries_fixture["payload"])
    assert rows, "recorded fixture parsed to zero rows"
    assert {r["team"] for r in rows} <= {"ARI", "ATL", "BAL", "BUF"}   # display names mapped to abbrs
    assert all(r["status"] in ("OK", "RISK", "OUT") for r in rows)
    assert all(r["name"] for r in rows)


def test_parse_team_injuries_rejects_garbage():
    with pytest.raises(av.EspnSchemaError):
        av.parse_team_injuries({"nope": []})


def test_normalize_status_mapping():
    assert av.normalize_status("Out") == "OUT"
    assert av.normalize_status("Injured Reserve") == "OUT"
    assert av.normalize_status("Doubtful") == "OUT"
    assert av.normalize_status("Questionable") == "RISK"
    assert av.normalize_status("Active") == "OK"
    assert av.normalize_status(None) == "UNKNOWN"      # no designation is not a health claim
    assert av.normalize_status("Some New Label") == "UNKNOWN"


def test_parse_event_roster_recorded_schema(roster_fixture):
    rows = av.parse_event_roster(roster_fixture["payload"])
    assert rows
    assert set(rows[0]) >= {"espn_id", "name", "active", "did_not_play", "starter"}


def test_resolve_wed_matches_by_name_and_team(injuries_fixture):
    inj_rows = av.parse_team_injuries(injuries_fixture["payload"])
    out_row = next((r for r in inj_rows if r["status"] == "OUT"), None) or inj_rows[0]
    risk_row = next((r for r in inj_rows if r["status"] == "RISK"), None)
    players = pd.DataFrame([
        {"player_id": "00-TEST0001", "player_name": out_row["name"], "team": out_row["team"]},
        {"player_id": "00-TEST0002", "player_name": "Nonexistent Player", "team": out_row["team"]},
    ] + ([{"player_id": "00-TEST0003", "player_name": risk_row["name"], "team": risk_row["team"]}]
         if risk_row else []))
    res = av.resolve_statuses(players, inj_rows, clock="wed",
                              injuries_fetched_at="2026-07-01T12:00:00Z")
    s1 = res["statuses"]["00-TEST0001"]
    assert s1["status"] == out_row["status"]
    assert s1["source"] == "espn_team_injuries"
    assert s1["matched_by"] in ("name+team", "name_only")
    assert s1["timestamp"] == "2026-07-01T12:00:00Z"
    # unlisted on a RECEIVED team report -> OK/not_listed (eligible, not medically cleared)
    assert res["statuses"]["00-TEST0002"]["status"] == "OK"
    assert res["statuses"]["00-TEST0002"]["availability_state"] == "not_listed"
    assert res["statuses"]["00-TEST0002"]["matched_by"] == "unmatched"
    if risk_row:
        assert res["statuses"]["00-TEST0003"]["status"] == "RISK"


def test_resolve_t90_requires_inactives():
    players = pd.DataFrame([{"player_id": "x", "player_name": "A B", "team": "DET"}])
    with pytest.raises(ValueError, match="t90"):
        av.resolve_statuses(players, [], inactive_rows=None, clock="t90")


def test_resolve_t90_inactive_overrides_and_active_clears():
    players = pd.DataFrame([
        {"player_id": "gsis-laporta", "player_name": "Sam LaPorta", "team": "DET"},
        {"player_id": "gsis-asb", "player_name": "Amon-Ra St. Brown", "team": "DET"},
    ])
    # Wednesday said: LaPorta merely Questionable, St. Brown Questionable too
    inj_rows = [
        {"team": "DET", "name": "Sam LaPorta", "status_raw": "Questionable", "status": "RISK"},
        {"team": "DET", "name": "Amon-Ra St. Brown", "status_raw": "Questionable", "status": "RISK"},
    ]
    res = av.resolve_statuses(players, inj_rows, inactive_rows=SYNTH_INACTIVES, clock="t90",
                              inactives_fetched_at="2026-07-01T17:00:00Z")
    assert res["statuses"]["gsis-laporta"]["status"] == "OUT"          # inactive at T-90
    assert res["statuses"]["gsis-laporta"]["source"] == "espn_event_roster"
    assert res["statuses"]["gsis-asb"]["status"] == "OK"               # confirmed active
    assert "active_t90" in res["statuses"]["gsis-asb"]["status_raw"]


def test_unmatched_espn_rows_stay_visible(injuries_fixture):
    inj_rows = av.parse_team_injuries(injuries_fixture["payload"])
    players = pd.DataFrame([{"player_id": "p1", "player_name": "Nobody Realname", "team": "ZZZ"}])
    res = av.resolve_statuses(players, inj_rows, clock="wed")
    assert len(res["unmatched_espn_rows"]) == len(inj_rows)            # none matched -> all visible


# --------------------------------------------------------------------------- #
# Usage reallocation (H8) -- synthetic walk-forward player_week frame
# --------------------------------------------------------------------------- #
def _synthetic_pw():
    """Team AAA, weeks 1-8 of 2023. WR1 misses weeks 5+6; WR2's target share
    jumps in exactly those weeks. 30 team pass attempts every week."""
    rows = []
    for wk in range(1, 9):
        wr1_plays = wk not in (5, 6)
        rows.append(dict(season=2023, week=wk, player_id="WR1", player_name="Alpha One",
                         team="AAA", role="WR", targets=9.0 if wr1_plays else 0.0,
                         carries=0.0, team_pass_att=30.0, team_rush_att=25.0))
        rows.append(dict(season=2023, week=wk, player_id="WR2", player_name="Beta Two",
                         team="AAA", role="WR", targets=9.0 if not wr1_plays else 5.0,
                         carries=0.0, team_pass_att=30.0, team_rush_att=25.0))
        rows.append(dict(season=2023, week=wk, player_id="TE1", player_name="Gamma Three",
                         team="AAA", role="TE", targets=4.0, carries=0.0,
                         team_pass_att=30.0, team_rush_att=25.0))
        rows.append(dict(season=2023, week=wk, player_id="QB1", player_name="Delta Four",
                         team="AAA", role="QB", targets=0.0, carries=1.0,
                         team_pass_att=30.0, team_rush_att=25.0))
    return pd.DataFrame(rows)


def test_reallocation_with_without_split():
    pw = _synthetic_pw()
    res = av.reallocate_usage(pw, season=2023, week=9, out_player_id="WR1")
    assert res["basis"] == "with_without"
    assert res["low_confidence"] is False
    assert res["absent_games"] == 2
    b = res["boosts"]["WR2"]
    assert b["share_delta"] == pytest.approx((9 / 30) - (5 / 30), abs=1e-4)   # +13.3% share
    assert res["boosts"]["TE1"]["share_delta"] == pytest.approx(0.0, abs=1e-4)


def test_reallocation_flags_proportional_guess():
    pw = _synthetic_pw()
    res = av.reallocate_usage(pw, season=2023, week=9, out_player_id="WR2")  # WR2 never missed
    assert res["basis"] == "proportional_guess"
    assert res["low_confidence"] is True
    assert res["boosts"], "guess should still propose share targets"


def test_reallocation_is_walk_forward():
    """Rows at/after the projection week must not influence the estimate."""
    pw = _synthetic_pw()
    poisoned = pd.concat([pw, pd.DataFrame([dict(
        season=2023, week=9, player_id="WR2", player_name="Beta Two", team="AAA",
        role="WR", targets=30.0, carries=0.0, team_pass_att=30.0, team_rush_att=25.0)])],
        ignore_index=True)
    a = av.reallocate_usage(pw, season=2023, week=9, out_player_id="WR1")
    b = av.reallocate_usage(poisoned, season=2023, week=9, out_player_id="WR1")
    assert a["boosts"] == b["boosts"]


def test_reallocation_qb_not_supported():
    res = av.reallocate_usage(_synthetic_pw(), season=2023, week=9, out_player_id="QB1")
    assert res["basis"] == "not_supported"
    assert res["low_confidence"] is True
