"""The live board must be built from the same features the backtest scores.

RED against 7179c8e. ``enumerate_candidates(roster_mode="carry_forward")``
reused each player's LAST PLAYED row and relabelled it to the target week.
Every ``roll_*`` feature is ``shift(1)``-before-aggregating, so that row's
features EXCLUDE its own game: the live board was one game staler than
anything the backtest ever measured. Measured consequence on production: the
2026 Week 2 board (as_of 2026-09-16) carried no 2026 Week 1 information for
73 of its 80 leans.

Separately, ``build_team_week`` only emits rows for weeks that have been
PLAYED, so a live ``(season, week)`` had no team row and
``projection.expected_volume`` silently took its ``roll_targets`` /
``roll_carries`` fallback instead of the team-volume x player-share formula
the backtest uses.

The controlling property: an as-of row built with NO knowledge of week W must
carry exactly the features the as_played row for week W carries, because both
are defined as "this player's history strictly before W".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflvalue.features import (
    asof_player_week,
    asof_team_week,
    build_player_week,
    build_team_week,
)
from nflvalue.projection import expected_volume

CUTOFF_SEASON, CUTOFF_WEEK = 2020, 8

ROLL_COLS = [
    "roll_games", "roll_targets", "roll_target_share", "roll_air_yards", "roll_adot",
    "roll_carries", "roll_carry_share", "roll_pass_attempts", "roll_completions",
    "roll_ypt", "roll_catch_rate", "roll_ypc", "roll_ypa",
    "roll_pass_td_rate", "roll_rush_td_rate", "roll_rec_td_rate",
]


def _truncated(pbp: pd.DataFrame) -> pd.DataFrame:
    keep = (pbp["season"] < CUTOFF_SEASON) | (
        (pbp["season"] == CUTOFF_SEASON) & (pbp["week"] < CUTOFF_WEEK))
    return pbp[keep].copy()


@pytest.fixture(scope="module")
def frames(pbp_fast):
    """(full frame, frame built with the cutoff week and later deleted)."""
    full = build_player_week(pbp_fast)
    trunc = build_player_week(_truncated(pbp_fast))
    return full, trunc


def _paired(full: pd.DataFrame, asof: pd.DataFrame) -> pd.DataFrame:
    """as_played rows at the cutoff week joined to their as-of counterparts.

    Restricted to players whose team AND role agree, because a player who
    changed team or whose roster role changed is a genuinely different row --
    the availability/roster layer owns that, not the feature window.
    """
    played = full[(full["season"] == CUTOFF_SEASON) & (full["week"] == CUTOFF_WEEK)]
    merged = played.merge(asof, on="player_id", suffixes=("_played", "_asof"))
    same = (merged["team_played"] == merged["team_asof"]) & \
           (merged["role_played"] == merged["role_asof"])
    return merged[same]


def test_asof_features_equal_the_as_played_features_for_that_week(frames, pbp_fast):
    """The controlling property. Exact equality, not approximate."""
    full, trunc = frames
    assert trunc[(trunc["season"] == CUTOFF_SEASON) &
                 (trunc["week"] == CUTOFF_WEEK)].empty, \
        "the truncated frame must not already contain the target week"

    asof = asof_player_week(trunc, CUTOFF_SEASON, CUTOFF_WEEK)
    assert not asof.empty
    assert list(asof.columns) == list(trunc.columns)
    assert (asof["season"] == CUTOFF_SEASON).all() and (asof["week"] == CUTOFF_WEEK).all()
    assert asof["player_id"].is_unique

    pairs = _paired(full, asof)
    assert len(pairs) >= 100, f"too few comparable players ({len(pairs)}) to prove anything"

    for col in ROLL_COLS:
        a = pairs[f"{col}_played"].to_numpy(dtype=float)
        b = pairs[f"{col}_asof"].to_numpy(dtype=float)
        assert np.array_equal(a, b, equal_nan=True), (
            f"{col} differs between the as-of row and the as_played row it must match; "
            f"first mismatch at {np.flatnonzero(~((a == b) | (np.isnan(a) & np.isnan(b))))[:3]}")


def test_asof_row_actually_uses_the_most_recent_completed_game(frames):
    """The bug this file exists for: the old path's features were one game old.

    Reproduce the old behaviour (the player's last played row, relabelled) and
    require it to DIFFER from the as-of row for players who played the week
    before the cutoff. If these agreed, the fix would be inert.
    """
    _, trunc = frames
    asof = asof_player_week(trunc, CUTOFF_SEASON, CUTOFF_WEEK)

    hist = trunc.sort_values(["player_id", "season", "week"])
    old = hist.groupby("player_id").tail(1).copy()      # <- pre-fix carry_forward
    last_week = (old["season"] == CUTOFF_SEASON) & (old["week"] == CUTOFF_WEEK - 1)
    old = old[last_week]
    assert len(old) >= 50

    merged = old.merge(asof, on="player_id", suffixes=("_old", "_new"))
    changed = merged["roll_targets_old"].to_numpy(dtype=float) != \
        merged["roll_targets_new"].to_numpy(dtype=float)
    frac = float(np.mean(changed))
    assert frac > 0.5, (
        f"only {frac:.0%} of players who played week {CUTOFF_WEEK - 1} have different "
        "rolling volume under the as-of build -- the stale-window fix is not doing anything")


def test_asof_never_reads_the_target_week(frames, pbp_fast):
    """Leak guard: the as-of rows must not change when the target week and
    everything after it is deleted from the inputs."""
    full, trunc = frames
    from_full = asof_player_week(full[(full["season"] < CUTOFF_SEASON) | (
        (full["season"] == CUTOFF_SEASON) & (full["week"] < CUTOFF_WEEK))],
        CUTOFF_SEASON, CUTOFF_WEEK)
    from_trunc = asof_player_week(trunc, CUTOFF_SEASON, CUTOFF_WEEK)

    a = from_full.sort_values("player_id").reset_index(drop=True)
    b = from_trunc.sort_values("player_id").reset_index(drop=True)
    assert list(a["player_id"]) == list(b["player_id"])
    for col in ROLL_COLS:
        assert np.array_equal(a[col].to_numpy(dtype=float),
                              b[col].to_numpy(dtype=float), equal_nan=True), col


def test_asof_team_week_matches_build_team_week_for_that_week(frames, pbp_fast):
    """Team volume: same property, and it must come out of player_week so the
    live path needs no play-by-play reload."""
    full, trunc = frames
    played = build_team_week(pbp_fast)
    played = played[(played["season"] == CUTOFF_SEASON) & (played["week"] == CUTOFF_WEEK)]
    assert not played.empty

    asof = asof_team_week(trunc, CUTOFF_SEASON, CUTOFF_WEEK,
                          teams=sorted(played["team"].unique()))
    merged = played.merge(asof, on="team", suffixes=("_played", "_asof"))
    assert len(merged) == len(played)
    for col in ("roll_team_pass_att", "roll_team_rush_att"):
        np.testing.assert_allclose(merged[f"{col}_played"].to_numpy(dtype=float),
                                   merged[f"{col}_asof"].to_numpy(dtype=float),
                                   rtol=0, atol=0, err_msg=col)


def test_expected_volume_uses_team_share_not_the_roll_fallback():
    """With a team row present, expected targets are team volume x share.
    Without one, the code falls back to the player's own trailing volume --
    a different formula from the one the backtest scores."""
    player = {"roll_targets": 4.0, "roll_target_share": 0.25, "roll_carries": 8.0,
              "roll_carry_share": 0.40}
    team = {"roll_team_pass_att": 36.0, "roll_team_rush_att": 25.0}
    spec_t = {"opportunity": "targets"}
    spec_c = {"opportunity": "carries"}

    assert expected_volume(player, team, spec_t) == pytest.approx(9.0)
    assert expected_volume(player, None, spec_t) == pytest.approx(4.0)
    assert expected_volume(player, team, spec_c) == pytest.approx(10.0)
    assert expected_volume(player, None, spec_c) == pytest.approx(8.0)


def test_live_week_gets_a_team_row(frames):
    """The live regression: a week that has not been played must still resolve
    a team volume row, so the live board uses the backtest's formula."""
    _, trunc = frames
    teams = sorted(trunc["team"].dropna().unique())[:8]
    asof = asof_team_week(trunc, CUTOFF_SEASON, CUTOFF_WEEK, teams=teams)
    assert sorted(asof["team"]) == teams
    assert asof["roll_team_pass_att"].notna().all()
    assert (asof["roll_team_pass_att"] > 0).all()
    assert asof["roll_team_rush_att"].notna().all()
    assert (asof["roll_team_rush_att"] > 0).all()
