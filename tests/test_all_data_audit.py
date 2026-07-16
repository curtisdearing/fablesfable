import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.all_data_factor_audit import beta_difference, evaluate_pattern, run_audit
from analysis.build_factor_frame import (
    long_term_incumbent_vacancies,
    official_absence_flags,
    prior_depth,
    team_schedule,
)
from analysis.nested_factor_selection import nested_season_forward
from nflvalue.reproducibility import canonical_csv_sha256


ROOT = Path(__file__).resolve().parents[1]


def _frame():
    rows = []
    for i in range(200):
        exposed = int(i < 80)
        # 60/80 exposed hits versus 40/120 controls.
        outcome = int(i < 60 or (80 <= i < 120))
        rows.append({"official_te1_out": exposed, "over": outcome, "eligible": 1,
                     "team_season": f"T{i % 20}-2024"})
    return pd.DataFrame(rows)


def test_joint_beta_posterior_samples_uncertainty_in_both_groups():
    result = beta_difference(60, 80, 40, 120, draws=30_000)
    assert result["credible_interval_95_low"] > 0
    assert result["posterior_probability_positive"] > 0.999


def test_audit_reports_exact_exposed_and_control_denominators():
    spec = {"name": "TE1 officially out", "exposure": "official_te1_out",
            "outcome": "over", "eligible": "eligible", "cohort": "pregame",
            "definition": "official Out/Doubtful at the prediction clock"}
    result = evaluate_pattern(_frame(), spec, draws=5_000)
    assert (result["exposed_hits"], result["exposed_n"]) == (60, 80)
    assert (result["control_hits"], result["control_n"]) == (40, 120)
    assert result["cluster_bootstrap"]["clusters"] == 20


def test_postgame_usage_absence_is_rejected_from_pregame_audit():
    frame = _frame().rename(columns={"official_te1_out": "usage_absence"})
    with pytest.raises(ValueError, match="postgame leakage"):
        run_audit(frame, [{"name": "bad", "exposure": "usage_absence",
                           "outcome": "over", "cohort": "pregame"}], draws=100)


def test_outer_test_season_cannot_change_factor_selection():
    rows = []
    for season in (2019, 2020, 2021, 2022):
        for i in range(400):
            a, b = int(i % 4 == 0), int(i % 5 == 0)
            if season < 2022:
                outcome = a
            else:
                outcome = b
            rows.append({"season": season, "a": a, "b": b, "over": outcome})
    frame = pd.DataFrame(rows)
    first = nested_season_forward(frame, ["a", "b"], "over", max_order=1,
                                  min_group_n=50, min_train_seasons=3)
    changed = frame.copy()
    changed.loc[changed["season"] == 2022, "over"] ^= 1
    second = nested_season_forward(changed, ["a", "b"], "over", max_order=1,
                                   min_group_n=50, min_train_seasons=3)
    assert first["folds"][0]["selected_factors"] == ["a"]
    assert second["folds"][0]["selected_factors"] == ["a"]
    assert first["folds"][0]["test_accuracy"] != second["folds"][0]["test_accuracy"]


def test_published_audit_is_reproducible_and_retracts_old_counts():
    audit = json.loads((ROOT / "data" / "all_data_factor_audit.json").read_text())
    assert audit["status"] == "research_only_summary"
    assert audit["frame_rows"] == 116554
    assert audit["method"]["family_size"] == 38
    assert len(audit["reproduction"]["frame_canonical_csv_sha256"]) == 64
    assert audit["reproduction"]["canonical_csv_version"] == 1
    assert "withdrawn" in audit["retraction"]
    assert audit["live_scoring_impact"].startswith("none")


def test_nested_projection_does_not_crown_tiny_or_underperforming_result():
    projection = json.loads((ROOT / "data" / "nested_factor_projection.json").read_text())
    assert projection["highest_eligible_accuracy"]["outer_n"] >= 100
    assert projection["highest_eligible_accuracy"]["selective_lift_pp"] <= 0
    assert "no combination scored 100%" in projection["conclusion"]


def test_depth_rank_uses_prior_games_not_current_game_usage():
    pw = pd.DataFrame([
        {"season": 2024, "week": 1, "team": "A", "player_id": "rb-a", "role": "RB",
         "carries": 12, "targets": 0, "pass_attempts": 0},
        {"season": 2024, "week": 1, "team": "A", "player_id": "rb-b", "role": "RB",
         "carries": 4, "targets": 0, "pass_attempts": 0},
        {"season": 2024, "week": 2, "team": "A", "player_id": "rb-a", "role": "RB",
         "carries": 10, "targets": 0, "pass_attempts": 0},
        {"season": 2024, "week": 2, "team": "A", "player_id": "rb-b", "role": "RB",
         "carries": 6, "targets": 0, "pass_attempts": 0},
        # This huge week-three workload must not alter the week-three label.
        {"season": 2024, "week": 3, "team": "A", "player_id": "rb-b", "role": "RB",
         "carries": 30, "targets": 0, "pass_attempts": 0},
    ])
    roster = pd.DataFrame([
        {"season": 2024, "week": 3, "team": "A", "player_id": "rb-a", "position": "RB"},
        {"season": 2024, "week": 3, "team": "A", "player_id": "rb-b", "position": "RB"},
    ])
    depth = prior_depth(roster, pw)
    ranks = dict(zip(depth["player_id"], depth["depth_rank"]))
    assert ranks == {"rb-a": 1, "rb-b": 2}
    injuries = pd.DataFrame([{"season": 2024, "week": 3, "team": "A", "gsis_id": "rb-a",
                              "position": "RB", "report_status": "Out"}])
    flags = official_absence_flags(depth, injuries)
    assert flags.iloc[0]["official_rb1_out"] == 1


def test_canonical_csv_hash_ignores_parquet_layout_and_row_column_order():
    first = pd.DataFrame([
        {"season": 2024, "week": 2, "game_id": "g2", "team": "A", "player_id": "p2", "market": "x", "value": 1.0},
        {"season": 2024, "week": 1, "game_id": "g1", "team": "A", "player_id": "p1", "market": "x", "value": 2.0},
    ])
    second = first.loc[[1, 0], list(reversed(first.columns))]
    keys = ["season", "week", "game_id", "team", "player_id", "market"]
    assert canonical_csv_sha256(first, row_keys=keys) == canonical_csv_sha256(second, row_keys=keys)
    changed = first.copy()
    changed.loc[0, "value"] = 1.1
    assert canonical_csv_sha256(first, row_keys=keys) != canonical_csv_sha256(changed, row_keys=keys)


def test_long_term_incumbent_cohort_tracks_reserve_player_outside_current_depth():
    pw = pd.DataFrame([
        {"season": 2024, "week": week, "team": "A", "player_id": "rb-old", "role": "RB",
         "carries": 12, "targets": 0, "pass_attempts": 0}
        for week in (1, 2, 3)
    ] + [
        {"season": 2024, "week": week, "team": "A", "player_id": "rb-new", "role": "RB",
         "carries": 3, "targets": 0, "pass_attempts": 0}
        for week in (1, 2, 3, 4, 5)
    ])
    roster = pd.DataFrame([
        {"season": 2024, "week": week, "team": "A", "player_id": player,
         "position": "RB", "status": status}
        for week in (1, 2, 3, 4, 5)
        for player, status in (("rb-old", "ACT" if week < 4 else "RES"), ("rb-new", "ACT"))
    ])
    # The helper uses nflverse's ``gsis_id`` convention in production.
    roster = roster.rename(columns={"player_id": "gsis_id"})
    long_term = long_term_incumbent_vacancies(roster, pw)
    week_four = long_term[long_term.week.eq(4)].iloc[0]
    week_five = long_term[long_term.week.eq(5)].iloc[0]
    assert week_four["long_term_rb1_unavailable"] == 1
    assert week_four["long_term_rb1_reserve_status"] == 1
    assert week_five["long_term_rb1_absence_weeks"] == 2


def test_schedule_context_is_computed_strictly_from_scheduled_games():
    schedules = pd.DataFrame([
        {"season": 2024, "week": 1, "game_type": "REG", "game_id": "g1",
         "gameday": "2024-09-01", "home_team": "A", "away_team": "B",
         "spread_line": -3.0, "overtime": True, "div_game": True},
        {"season": 2024, "week": 2, "game_type": "REG", "game_id": "g2",
         "gameday": "2024-09-08", "home_team": "B", "away_team": "A",
         "spread_line": 2.0, "overtime": False, "div_game": True},
    ])
    context = team_schedule(schedules)
    a2 = context[(context["team"] == "A") & (context["week"] == 2)].iloc[0]
    assert a2["rest_days"] == 7
    assert bool(a2["after_overtime"])
    assert bool(a2["division_rematch"])
    assert a2["team_spread"] == -2.0
