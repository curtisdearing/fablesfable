"""Unit tests for the pure math in nflvalue/projection.py -- no parquet, no
pandas fixtures needed, just the distribution/contract behavior itself."""

from __future__ import annotations

import math

from nflvalue.projection import (
    DEFAULT_SD_FRACTION,
    MARKETS,
    MIN_GAMES_ELIGIBLE,
    SD_FRACTION_BY_MARKET,
    p_over,
    project,
)


def test_p_over_decreases_as_line_increases():
    for dist in ("normal", "gamma", "negbinom", "poisson"):
        mean, sd = 50.0, 15.0
        p_low = p_over(mean, sd, 30.0, dist)
        p_mid = p_over(mean, sd, 50.0, dist)
        p_high = p_over(mean, sd, 70.0, dist)
        assert p_low > p_mid > p_high, dist


def test_p_over_and_p_under_sum_to_one_in_project_contract():
    player_row = {
        "player_id": "00-TEST", "player_name": "Test Player", "team": "TST", "defteam": "OPP", "role": "WR",
        "roll_target_share": 0.2, "roll_targets": 8.0, "roll_ypt": 8.0, "roll_catch_rate": 0.65,
        "roll_carry_share": 0.0, "roll_carries": 0.0, "roll_ypc": 0.0,
    }
    team_row = {"roll_team_pass_att": 34.0, "roll_team_rush_att": 25.0}
    result = project(player_row, "receiving_yards", team_row=team_row, opp_row=None, line=65.5, sd=25.0)
    assert result["p_over"] is not None and result["p_under"] is not None
    assert math.isclose(result["p_over"] + result["p_under"], 1.0, abs_tol=1e-6)


def test_project_without_a_line_leaves_p_over_none():
    player_row = {"player_id": "00-TEST", "player_name": "Test QB", "team": "TST", "defteam": "OPP", "role": "QB",
                   "roll_pass_attempts": 34.0, "roll_ypa": 7.2}
    result = project(player_row, "passing_yards", line=None, sd=50.0)
    assert result["p_over"] is None and result["p_under"] is None
    assert result["mean"] > 0


def test_pass_completions_mean_is_attempts_times_completion_rate():
    """pass_completions mean = projected pass_attempts x trailing completion
    rate (roll_comp_rate), the same volume-x-efficiency shape as the other
    non-anytime_td markets. No opp factor, no game script here -> the mean is
    exactly attempts * comp_rate and the volume/efficiency components echo the
    inputs."""
    player_row = {"player_id": "00-QB", "player_name": "Test QB", "team": "TST",
                  "defteam": "OPP", "role": "QB", "roll_games": 8,
                  "roll_pass_attempts": 34.0, "roll_comp_rate": 0.66}
    result = project(player_row, "pass_completions", line=21.5)
    assert result["dist"] == "negbinom"          # bounded count, like receptions/pass_attempts
    assert math.isclose(result["mean"], 34.0 * 0.66, abs_tol=1e-6)
    assert math.isclose(result["components"]["volume"], 34.0, abs_tol=1e-6)
    assert math.isclose(result["components"]["efficiency"], 0.66, abs_tol=1e-6)
    assert result["p_over"] is not None and result["p_under"] is not None
    assert math.isclose(result["p_over"] + result["p_under"], 1.0, abs_tol=1e-6)


def test_pass_completions_fallback_sd_is_tighter_than_generic():
    """A bounded count is far less dispersed than the generic 0.45 yardage
    prior; the fallback SD (when no measured sd is supplied) must use the
    per-market override, not DEFAULT_SD_FRACTION."""
    assert SD_FRACTION_BY_MARKET["pass_completions"] < DEFAULT_SD_FRACTION
    player_row = {"player_id": "00-QB", "player_name": "Test QB", "team": "TST",
                  "defteam": "OPP", "role": "QB", "roll_games": 8,
                  "roll_pass_attempts": 34.0, "roll_comp_rate": 0.66}
    result = project(player_row, "pass_completions", line=21.5)  # no sd passed
    expected_sd = max(result["mean"] * SD_FRACTION_BY_MARKET["pass_completions"], 0.75)
    assert math.isclose(result["sd"], round(expected_sd, 3), abs_tol=1e-3)
    # and it is genuinely tighter than the generic prior would have produced
    assert result["sd"] < result["mean"] * DEFAULT_SD_FRACTION


def test_pass_completions_measured_sd_overrides_fallback():
    """A supplied walk-forward sd still wins over the per-market fallback."""
    player_row = {"player_id": "00-QB", "player_name": "Test QB", "team": "TST",
                  "defteam": "OPP", "role": "QB", "roll_games": 8,
                  "roll_pass_attempts": 34.0, "roll_comp_rate": 0.66}
    result = project(player_row, "pass_completions", line=21.5, sd=3.1)
    assert math.isclose(result["sd"], 3.1, abs_tol=1e-6)


def test_low_confidence_flag_matches_market_registry():
    for market, spec in MARKETS.items():
        assert spec["low_confidence"] == (market == "anytime_td")


def test_anytime_td_probability_bounded_and_low_confidence():
    player_row = {"player_id": "00-TEST", "player_name": "Test RB", "team": "TST", "defteam": "OPP", "role": "RB",
                  "roll_carries": 18.0, "roll_rush_td_rate": 0.06, "roll_targets": 2.0, "roll_rec_td_rate": 0.02}
    result = project(player_row, "anytime_td", line=0.5)
    assert result["low_confidence"] is True
    assert 0.0 <= result["p_over"] <= 1.0


def test_cold_start_player_is_ineligible_and_forced_low_confidence():
    """Checkpoint 1B cold-start gate: fewer than MIN_GAMES_ELIGIBLE trailing
    games -> never eligible for a shortlist, regardless of the market."""
    player_row = {"player_id": "00-ROOKIE", "player_name": "Rookie WR", "team": "TST", "defteam": "OPP",
                  "role": "WR", "roll_games": 1, "roll_target_share": 0.15, "roll_targets": 4.0,
                  "roll_ypt": 7.5, "roll_catch_rate": 0.6}
    result = project(player_row, "receiving_yards", line=40.5, sd=20.0)
    assert result["eligible_for_shortlist"] is False
    assert result["low_confidence"] is True


def test_established_player_with_enough_history_is_eligible():
    player_row = {"player_id": "00-VET", "player_name": "Veteran WR", "team": "TST", "defteam": "OPP",
                  "role": "WR", "roll_games": MIN_GAMES_ELIGIBLE + 2, "roll_target_share": 0.22,
                  "roll_targets": 9.0, "roll_ypt": 8.0, "roll_catch_rate": 0.68}
    result = project(player_row, "receiving_yards", line=60.5, sd=25.0)
    assert result["eligible_for_shortlist"] is True
    assert result["low_confidence"] is False


def test_missing_roll_games_defaults_to_ineligible():
    """No roll_games at all (e.g. a malformed row) should fail closed, not open."""
    player_row = {"player_id": "00-X", "player_name": "Unknown", "team": "TST", "defteam": "OPP", "role": "RB"}
    result = project(player_row, "rushing_yards", line=40.5, sd=20.0)
    assert result["eligible_for_shortlist"] is False


def test_p_over_is_finite_and_bounded_for_nan_mean_or_sd():
    """A NaN mean/sd must not slip through the clip as a certain OVER (1.0);
    it must return a finite p in [0, 1]. Real calls floor mean/sd upstream, so
    this only guards a malformed edge case."""
    nan = float("nan")
    for dist in ("normal", "gamma", "negbinom", "poisson"):
        for mean, sd in ((nan, 15.0), (50.0, nan), (nan, nan)):
            p = p_over(mean, sd, 50.0, dist)
            assert p == p, (dist, mean, sd)          # not NaN
            assert 0.0 <= p <= 1.0, (dist, mean, sd)
            assert p != 1.0, (dist, mean, sd)        # not a spurious certain OVER
