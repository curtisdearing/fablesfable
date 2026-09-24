"""As-of team identity for players who changed teams.

RED against 53c07cf: ``features.asof_player_week`` took ``team`` from the
player's LAST PLAYED row, so ``enumerate_candidates(roster_mode="carry_forward")``
seated Brady (2020 W1) in NE's game although he played for TB.

Data: the committed REAL play-by-play slice (tests/fixtures/pbp_2019_2020.parquet)
and a REAL extract of the nflverse weekly rosters
(tests/fixtures/rosters_weekly_2019_2020_skill.parquet: 2019-2020 rows, from the
pinned historical/rosters_weekly.parquet, for the players in that slice).
The ambiguous-roster case uses a hand-built frame; it checks the mechanics only
and is not football evidence.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflvalue import features as F
from nflvalue.candidates import WeekInputs, enumerate_candidates

FIX = Path(__file__).resolve().parent / "fixtures"
NO_ROSTERS = pd.DataFrame(columns=["season", "week", "player_id", "position"])
BRADY, RIVERS, BELL, PETTIS = "00-0019596", "00-0022942", "00-0030496", "00-0034860"


@pytest.fixture(scope="module")
def pbp():
    df = pd.read_parquet(FIX / "pbp_2019_2020.parquet")
    return df[df["season_type"] == "REG"].copy()


@pytest.fixture(scope="module")
def rosters():
    return pd.read_parquet(FIX / "rosters_weekly_2019_2020_skill.parquet")


@pytest.fixture(scope="module")
def pw(pbp):
    return F.build_player_week(pbp, rosters=NO_ROSTERS)


def _played(pw, pid, season, week):
    return pw[(pw["player_id"] == pid) & (pw["season"] == season) & (pw["week"] == week)]["team"].iloc[0]


def _asof(pw, rosters, pid, season, week):
    a = F.asof_player_week(pw, season, week, rosters=rosters)
    return a[a["player_id"] == pid].iloc[0]


def test_week1_offseason_transfer_seated_on_new_team(pw, rosters):
    for pid, old, new in [(BRADY, "NE", "TB"), (RIVERS, "LAC", "IND")]:
        assert _played(pw, pid, 2020, 1) == new
        without = F.asof_player_week(pw, 2020, 1)
        assert without[without["player_id"] == pid]["team"].iloc[0] == old   # the defect, unchanged default
        row = _asof(pw, rosters, pid, 2020, 1)
        assert row["team"] == new and row["team_source"] == F.TEAM_SOURCE_RECONSTRUCTED


def test_in_season_before_and_after_transfer(pw, rosters):
    # Bell: last played NYJ W5; roster lists KC from W6; first KC game W7
    assert _asof(pw, rosters, BELL, 2020, 5)["team"] == "NYJ"
    assert _asof(pw, rosters, BELL, 2020, 6)["team"] == "KC"
    assert _asof(pw, rosters, BELL, 2020, 8)["team"] == "KC"      # after the move
    no_roster = _asof(pw, rosters[rosters["player_id"] != BELL], BELL, 2020, 8)
    assert no_roster["team"] == "KC"                                # his own KC games carry it


def test_late_season_transfer(pw, rosters):
    # Pettis: last played SF W1, on NYG rosters from W9, next played W16 for NYG
    assert _played(pw, PETTIS, 2020, 16) == "NYG"
    assert _asof(pw, rosters, PETTIS, 2020, 8)["team"] == "SF"
    row = _asof(pw, rosters, PETTIS, 2020, 16)
    assert row["team"] == "NYG" and row["team_source"] == F.TEAM_SOURCE_RECONSTRUCTED


def test_future_roster_rows_are_never_read(pw, rosters):
    full = F.asof_player_week(pw, 2020, 1, rosters=rosters)
    clocked = rosters[(rosters["season"] < 2020) | (rosters["week"] <= 1)]
    pd.testing.assert_frame_equal(full, F.asof_player_week(pw, 2020, 1, rosters=clocked))
    # only FUTURE evidence of the move: it must not leak back into week 1
    only_future = rosters[(rosters["season"] == 2020) & (rosters["week"] >= 2)]
    row = _asof(pw, only_future, BRADY, 2020, 1)
    assert row["team"] == "NE" and row["team_source"] == F.TEAM_SOURCE_NO_ROSTER
    # prior-season roster only (week 1 before the new season's roster exists)
    row = _asof(pw, rosters[rosters["season"] == 2019], BRADY, 2020, 1)
    assert row["team"] == "NE"


def test_stale_roster_cannot_undo_a_move_already_played(pw, rosters):
    # roster evidence ends 2020 W1; Bell played for KC from W7: the game wins
    stale = rosters[(rosters["season"] == 2019) | (rosters["week"] <= 1)]
    row = _asof(pw, stale, BELL, 2020, 10)
    assert row["team"] == "KC" and row["team_source"] == F.TEAM_SOURCE_LAST_PLAYED


def test_missing_roster_keeps_row_on_last_played_team_flagged(pw, rosters):
    without = rosters[rosters["player_id"] != BRADY]
    a = F.asof_player_week(pw, 2020, 1, rosters=without)
    row = a[a["player_id"] == BRADY].iloc[0]
    assert row["team"] == "NE" and row["team_source"] == F.TEAM_SOURCE_NO_ROSTER
    # no row is dropped, with or without roster evidence
    assert set(a["player_id"]) == set(F.asof_player_week(pw, 2020, 1)["player_id"])


def test_ambiguous_roster_team_is_unknown_not_guessed_and_row_kept(pw):
    # hand-built: two clubs list him in the same week; a TRD row never counts
    amb = pd.DataFrame({"season": [2020, 2020], "week": [1, 1],
                        "player_id": [BRADY, BRADY], "team": ["TB", "NE"]})
    row = _asof(pw, amb, BRADY, 2020, 1)
    assert pd.isna(row["team"]) and row["team_source"] == F.TEAM_SOURCE_AMBIGUOUS
    assert row["roll_games"] > 0                                     # history still attached
    traded = amb.assign(status=["ACT", "TRD"])
    row = _asof(pw, traded, BRADY, 2020, 1)
    assert row["team"] == "TB" and row["team_source"] == F.TEAM_SOURCE_RECONSTRUCTED


def test_stat_history_carries_across_the_transfer(pw, rosters):
    base = F.asof_player_week(pw, 2020, 1)
    fixed = F.asof_player_week(pw, 2020, 1, rosters=rosters)
    assert list(fixed.columns) == list(base.columns) + ["team_source"]
    cols = [c for c in base.columns if c != "team"]
    pd.testing.assert_frame_equal(base[cols], fixed[cols])           # only the seat moves
    row = fixed[fixed["player_id"] == BRADY].iloc[0]
    assert row["roll_pass_attempts"] > 30 and row["roll_games"] >= 8


def test_native_carry_forward_consumer_places_transfer_in_new_teams_game(pbp, rosters):
    hist = pbp[pbp["season"] == 2019]
    sched = pd.read_parquet(FIX / "schedules_2019_2020.parquet")
    kw = dict(pw=F.build_player_week(hist, rosters=rosters), opd=F.build_opp_pos_def(hist, rosters=rosters),
              tw=F.build_team_week(hist), schedules=sched)
    sd = {"passing_yards": 70.0, "pass_attempts": 6.0}
    run = lambda inp: enumerate_candidates(2020, 1, inputs=inp, markets=list(sd),
                                           roster_mode="carry_forward", sd_by_market=sd)
    before = run(WeekInputs(**kw))
    after = run(WeekInputs(**kw, rosters=rosters))
    b = before[before["player_id"] == BRADY]
    a = after[after["player_id"] == BRADY]
    assert set(b["game_id"]) == {"2020_01_MIA_NE"} and set(b["team"]) == {"NE"}   # the defect
    assert set(a["game_id"]) == {"2020_01_TB_NO"} and set(a["team"]) == {"TB"}
    assert set(a["defteam"]) == {"NO"} and len(a) == len(b)
    r = after[after["player_id"] == RIVERS]
    assert set(r["team"]) == {"IND"} and len(r)
    # everyone else keeps the same projection; history is identical across the fix
    ident = F.asof_team_identity(kw["pw"], rosters, 2020, 1)
    moved = set(ident[ident["team"] != ident["last_played_team"]]["player_id"])
    assert {BRADY, RIVERS} <= moved
    keep = lambda d: d[~d["player_id"].isin(moved)].reset_index(drop=True)
    assert len(keep(after)) > 20
    pd.testing.assert_frame_equal(keep(after)[["game_id", "player_id", "market"]],
                                  keep(before)[["game_id", "player_id", "market"]])
    np.testing.assert_allclose(keep(after)["mean"], keep(before)["mean"])


# --------------------------------------------------------------------------- #
# Decision clock: a week label is not proof the roster was known pregame.
# The capture/decision timestamps below are hand-set on the real roster rows;
# they test the clock mechanics only (no real capture log exists for 2020).
# --------------------------------------------------------------------------- #
DECISION = pd.Timestamp("2020-09-13T16:00:00Z")          # before 2020 W1 Sunday kickoffs


def _captured(rosters, when):
    return rosters.assign(captured_at=when)


def test_without_decision_clock_roster_identity_is_reconstructed_never_verified(pw, rosters):
    for frame in (rosters, _captured(rosters, DECISION - pd.Timedelta(hours=1))):
        row = _asof(pw, frame, BRADY, 2020, 1)
        assert row["team"] == "TB" and row["team_source"] == F.TEAM_SOURCE_RECONSTRUCTED
        assert row["team_source"] != F.TEAM_SOURCE_VERIFIED


def test_same_week_roster_captured_before_decision_is_verified(pw, rosters):
    a = F.asof_player_week(pw, 2020, 1, rosters=_captured(rosters, DECISION - pd.Timedelta(hours=1)),
                           decision_at=DECISION)
    row = a[a["player_id"] == BRADY].iloc[0]
    assert row["team"] == "TB" and row["team_source"] == F.TEAM_SOURCE_VERIFIED


def test_same_week_roster_captured_after_decision_is_rejected(pw, rosters):
    late = _captured(rosters, DECISION + pd.Timedelta(minutes=1))
    a = F.asof_player_week(pw, 2020, 1, rosters=late, decision_at=DECISION)
    row = a[a["player_id"] == BRADY].iloc[0]
    assert row["team"] == "NE" and row["team_source"] == F.TEAM_SOURCE_ROSTER_REJECTED
    assert (a["team_source"] != F.TEAM_SOURCE_VERIFIED).all()
    # only the late W1 row is rejected: a 2019 row captured in time still counts
    mixed = pd.concat([_captured(rosters[rosters["season"] == 2019], DECISION - pd.Timedelta(days=200)),
                       late[late["season"] == 2020]])
    ident = F.asof_team_identity(pw, mixed, 2020, 1, decision_at=DECISION).set_index("player_id")
    assert ident.loc[BRADY, "team"] == "NE" and ident.loc[BRADY, "roster_week"] == 18
    assert ident.loc[BRADY, "n_roster_rows_rejected"] >= 1


def test_unknown_capture_never_becomes_verified_pregame(pw, rosters):
    no_col = rosters                                            # no captured_at column at all
    nat = _captured(rosters, pd.NaT)
    garbage = _captured(rosters, "not a timestamp")
    naive = _captured(rosters, "2020-09-13T15:00:00")          # no zone: capture time unknown
    for frame in (no_col, nat, garbage, naive):
        a = F.asof_player_week(pw, 2020, 1, rosters=frame, decision_at=DECISION)
        assert (a["team_source"] != F.TEAM_SOURCE_VERIFIED).all()
        row = a[a["player_id"] == BRADY].iloc[0]
        assert row["team"] == "NE" and row["team_source"] == F.TEAM_SOURCE_ROSTER_REJECTED


def test_later_week_captured_before_decision_is_still_not_read(pw, rosters):
    only_future = _captured(rosters[(rosters["season"] == 2020) & (rosters["week"] >= 2)],
                            DECISION - pd.Timedelta(hours=1))
    row = F.asof_team_identity(pw, only_future, 2020, 1, decision_at=DECISION).set_index("player_id").loc[BRADY]
    assert row["team"] == "NE" and row["team_source"] == F.TEAM_SOURCE_NO_ROSTER


def test_invalid_decision_clock_fails_loud(pw, rosters):
    with pytest.raises(ValueError):
        F.asof_team_identity(pw, rosters, 2020, 1, decision_at="soon")
    with pytest.raises(ValueError):
        F.asof_team_identity(pw, rosters, 2020, 1, decision_at="2020-09-13T16:00:00")   # naive


def test_active_roster_payload_capture_clock_is_fetched_at(pw):
    rows = [{"player_id": BRADY, "team": "TB", "status": "ACT", "week": 1},
            {"player_id": RIVERS, "team": "IND", "status": "ACT", "week": 1}]
    payload = {"season": 2020, "week": 1, "rows": rows,
               "snapshot_at": "2020-09-12T10:00:00Z", "fetched_at": "2020-09-13T15:00:00Z"}
    frame = F.roster_frame_from_active_roster(payload)
    ident = F.asof_team_identity(pw, frame, 2020, 1, decision_at=DECISION).set_index("player_id")
    assert ident.loc[BRADY, "team_source"] == F.TEAM_SOURCE_VERIFIED and ident.loc[RIVERS, "team"] == "IND"
    fetched_late = F.roster_frame_from_active_roster({**payload, "fetched_at": "2020-09-13T16:00:01Z"})
    assert F.asof_team_identity(pw, fetched_late, 2020, 1, decision_at=DECISION) \
        .set_index("player_id").loc[BRADY, "team"] == "NE"
    no_fetch = F.roster_frame_from_active_roster({**payload, "fetched_at": None})
    assert F.asof_team_identity(pw, no_fetch, 2020, 1, decision_at=DECISION) \
        .set_index("player_id").loc[BRADY, "team_source"] == F.TEAM_SOURCE_ROSTER_REJECTED


def test_native_consumer_reports_identity_clock_and_rejects_late_roster(pbp, rosters):
    hist = pbp[pbp["season"] == 2019]
    sched = pd.read_parquet(FIX / "schedules_2019_2020.parquet")
    kw = dict(pw=F.build_player_week(hist, rosters=rosters), opd=F.build_opp_pos_def(hist, rosters=rosters),
              tw=F.build_team_week(hist), schedules=sched)
    sd = {"passing_yards": 70.0, "pass_attempts": 6.0}

    def run(frame, **extra):
        return enumerate_candidates(2020, 1, inputs=WeekInputs(**kw, rosters=frame), markets=list(sd),
                                    roster_mode="carry_forward", sd_by_market=sd, **extra)

    recon = run(rosters)
    assert recon.attrs["asof_team_identity"]["clock"] == "reconstructed_unverified"
    late = run(_captured(rosters, DECISION + pd.Timedelta(hours=2)), decision_at=DECISION)
    # a late roster never re-seats him, and he is not left on NE as if verified: unseated
    assert late.empty or BRADY not in set(late["player_id"])
    info = late.attrs["asof_team_identity"]
    u = next(x for x in info["unseated_on_slate"] if x["player_id"] == BRADY)
    assert u["last_played_team"] == "NE" and u["team_source"] == F.TEAM_SOURCE_ROSTER_REJECTED
    assert u["n_roster_rows_rejected"] >= 1
    assert info["clock"] == "decision_at" and info["decision_at"] == "2020-09-13T16:00:00Z"
    assert info["team_source_counts"].get(F.TEAM_SOURCE_VERIFIED, 0) == 0
    ok = run(_captured(rosters, DECISION - pd.Timedelta(hours=2)), decision_at=DECISION)
    assert set(ok[ok["player_id"] == BRADY]["game_id"]) == {"2020_01_TB_NO"}
    assert ok.attrs["asof_team_identity"]["team_source_counts"][F.TEAM_SOURCE_VERIFIED] > 0
