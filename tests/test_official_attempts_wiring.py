"""Production wiring of OFFICIAL QB pass attempts (sacks and two-point tries excluded).

Built on the committed REAL play-by-play slice (tests/fixtures/pbp_2019_2020.parquet),
which has no ``sack``/``down`` columns.  Where a test needs them, SYNTHETIC sack
and two-point flags are stamped onto real pass plays (seeded); that checks the
mechanics only and is not football evidence.  Rosters are passed empty so
positions come from participation inference (no network).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prop_backtest
from nflvalue import features as F
from nflvalue import projection

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pbp_2019_2020.parquet"
NO_ROSTERS = pd.DataFrame(columns=["season", "week", "player_id", "position"])
NEW = ["pass_attempts_official", "roll_pass_attempts_official"]


@pytest.fixture(scope="module")
def pbp_real():
    df = pd.read_parquet(FIXTURE)
    df = df[(df["season_type"] == "REG") & (df["season"] == 2019) & (df["week"] <= 8)].copy()
    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def pbp_flagged(pbp_real):
    df = pbp_real.copy()
    rng = np.random.default_rng(20260923)
    is_pass = (df["pass_attempt"] == 1) & df["passer_player_id"].notna()
    df["sack"] = np.where(is_pass & (rng.random(len(df)) < 0.07), 1.0, 0.0)
    df["down"] = 1.0
    two = is_pass & (df["sack"] == 0) & (rng.random(len(df)) < 0.01)
    df.loc[two, "down"] = np.nan
    return df


@pytest.fixture(scope="module")
def pw_pair(pbp_real, pbp_flagged):
    return (F.build_player_week(pbp_real, rosters=NO_ROSTERS),
            F.build_player_week(pbp_flagged, rosters=NO_ROSTERS))


def test_official_excludes_sacks_and_two_point_tries(pbp_flagged, pw_pair):
    _, pw = pw_pair
    p = pbp_flagged[(pbp_flagged["pass_attempt"] == 1) & pbp_flagged["passer_player_id"].notna()]
    expect = (p[(p["sack"] != 1) & p["down"].notna()]
              .groupby(["season", "week", "passer_player_id"]).size())
    got = pw[pw["pass_attempts"] > 0].set_index(["season", "week", "player_id"])["pass_attempts_official"]
    exp = expect.reindex(got.index).fillna(0.0)
    assert len(got) > 100
    np.testing.assert_array_equal(got.to_numpy(), exp.to_numpy())
    assert (pw["pass_attempts_official"] <= pw["pass_attempts"]).all()
    assert (pw["pass_attempts_official"] < pw["pass_attempts"]).any()


def test_existing_columns_unchanged_team_volume_and_ranker_inputs(pw_pair):
    base, pw = pw_pair
    old_cols = [c for c in base.columns if c not in NEW]
    assert list(pw.columns[:len(old_cols)]) == old_cols and list(pw.columns[len(old_cols):]) == NEW
    pd.testing.assert_frame_equal(base[old_cols], pw[old_cols])   # incl. pass_attempts, roll_pass_attempts, team_pass_att


def test_missing_sack_coverage_stays_unresolved(pw_pair):
    base, _ = pw_pair
    assert base["pass_attempts_official"].isna().all()
    assert base["roll_pass_attempts_official"].isna().all()
    qb = base[base["role"] == "QB"].iloc[-1].to_dict()
    att = projection.project(qb, "pass_attempts", line=30.5, sd=8.0)
    assert not att["eligible_for_shortlist"]
    assert att["mean"] is None or np.isnan(att["mean"])   # unresolved, not 0 or sack-inclusive


def test_attempts_projection_reads_official_roll_passing_yards_unchanged(pw_pair):
    base, pw = pw_pair
    qb = pw[(pw["role"] == "QB") & (pw["roll_games"] >= 3)].iloc[-1].to_dict()
    qb0 = base[(base["player_id"] == qb["player_id"]) & (base["week"] == qb["week"])].iloc[0].to_dict()
    att = projection.project(qb, "pass_attempts", line=30.5, sd=8.0)
    assert att["mean"] == pytest.approx(qb["roll_pass_attempts_official"], abs=1e-3)
    assert qb["roll_pass_attempts_official"] < qb["roll_pass_attempts"]
    y1 = projection.project(qb, "passing_yards", line=240.5, sd=90.0)["mean"]
    y0 = projection.project(qb0, "passing_yards", line=240.5, sd=90.0)["mean"]
    assert y1 == y0


def test_official_roll_is_prior_only_and_asof_equals_as_played(pbp_flagged, pw_pair):
    _, pw = pw_pair
    full = pw[(pw["season"] == 2019) & (pw["week"] == 8)].set_index("player_id")
    asof = F.asof_player_week(pw, 2019, 8).set_index("player_id")
    common = full.index.intersection(asof.index)
    assert len(common) > 20
    np.testing.assert_allclose(asof.loc[common, "roll_pass_attempts_official"],
                               full.loc[common, "roll_pass_attempts_official"])
    assert asof["pass_attempts_official"].isna().all()           # target week is never read
    # a changed week-8 outcome cannot move week-8's pregame feature
    trunc = F.build_player_week(pbp_flagged[pbp_flagged["week"] <= 7], rosters=NO_ROSTERS)
    np.testing.assert_allclose(F.asof_player_week(trunc, 2019, 8).set_index("player_id")
                               .loc[common, "roll_pass_attempts_official"],
                               asof.loc[common, "roll_pass_attempts_official"])


def test_walk_forward_residual_target_is_official():
    assert prop_backtest.ACTUAL_COL["pass_attempts"] == "pass_attempts_official"
    assert prop_backtest.ACTUAL_COL["passing_yards"] == "pass_yards"
    assert projection.MARKETS["passing_yards"].get("volume_col") is None


def test_asof_team_matches_team_actually_played_for_transfers():
    # was a strict xfail: the as-of team came from the LAST PLAYED row (NE).
    # Repaired by roster evidence, as a HISTORICAL RECONSTRUCTION: the weekly
    # roster rows have no capture time, so this is not proof they were known
    # pregame (see tests/test_asof_transfer_identity.py for the decision clock).
    df = pd.read_parquet(FIXTURE)
    df = df[(df["season_type"] == "REG")].copy()
    pw = F.build_player_week(df, rosters=NO_ROSTERS)
    brady = "00-0019596"
    played = pw[(pw["player_id"] == brady) & (pw["season"] == 2020) & (pw["week"] == 1)]["team"].iloc[0]
    rosters = pd.read_parquet(FIXTURE.parent / "rosters_weekly_2019_2020_skill.parquet")
    asof = F.asof_player_week(pw, 2020, 1, rosters=rosters[rosters["season"] <= 2020])
    assert played == "TB"
    assert asof[asof["player_id"] == brady]["team"].iloc[0] == played
    assert asof[asof["player_id"] == brady]["team_source"].iloc[0] == F.TEAM_SOURCE_RECONSTRUCTED


def test_frame_without_official_column_is_unresolved_not_keyerror(pw_pair):
    base, _ = pw_pair
    legacy = base.drop(columns=NEW)
    rows = prop_backtest._predictions_for_market(legacy, "pass_attempts", {}, {})
    assert len(rows) and rows["actual"].isna().all()


def test_candidates_settlement_and_bias_learning_target_is_official(pw_pair):
    from nflvalue import candidates as C
    assert C.ACTUAL_COL["pass_attempts"] == "pass_attempts_official"
    assert C.ACTUAL_COL["rush_attempts"] == "carries" and C.ACTUAL_COL["passing_yards"] == "pass_yards"
    _, pw = pw_pair
    inputs = C.WeekInputs(pw=pw, opd=pd.DataFrame(), tw=pd.DataFrame(), schedules=pd.DataFrame())
    synth = C.synthetic_lines(inputs, "pass_attempts")
    qb = pw[(pw["role"] == "QB") & (pw["roll_games"] >= 4)].index
    assert synth.loc[qb].notna().any()
    # the trailing line is built on OFFICIAL attempts, not the sack-inclusive count
    g = pw.sort_values(["player_id", "season", "week"]).groupby("player_id")
    exp = np.floor(g["pass_attempts_official"].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())) + 0.5
    np.testing.assert_array_equal(synth.to_numpy(), exp.reindex(synth.index).to_numpy())


def test_candidates_frame_without_official_column_has_no_line_not_keyerror(pw_pair):
    from nflvalue import candidates as C
    base, _ = pw_pair
    legacy = base.drop(columns=NEW)
    inputs = C.WeekInputs(pw=legacy, opd=pd.DataFrame(), tw=pd.DataFrame(), schedules=pd.DataFrame())
    assert C.synthetic_lines(inputs, "pass_attempts").isna().all()
    assert C._carry_forward_synth(inputs, "pass_attempts", legacy["player_id"].iloc[0]) is None
