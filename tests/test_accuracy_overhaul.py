"""Regression gates for issues #21–#23 and the prop probability audit."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import prop_decision  # noqa: E402
from nflvalue.composite import score_candidate  # noqa: E402
from analysis.prop_probability_grade import grade_rows  # noqa: E402


def _candidate(**overrides):
    row = {
        "player_id": "P1", "market": "receiving_yards", "mean": 60.0,
        "sd": 20.0, "line": 52.5, "dist": "normal", "p_over": 0.60,
        "p_under": 0.40, "components": {"opp_factor": 1.0, "game_script": 1.0},
        "low_confidence": False,
        "prices": {"over": 1.95, "under": 1.95, "n_books": 2,
                   "consensus_p_over": 0.50},
    }
    row.update(overrides)
    return row


def test_missing_or_stale_active_roster_blocks_live_candidate_publication():
    now = dt.datetime(2026, 9, 9, 16, tzinfo=dt.timezone.utc)
    missing = prop_decision.active_roster_gate(None, None, now=now)
    stale = prop_decision.active_roster_gate(
        {"P1"}, "2026-09-01T16:00:00Z", now=now)
    assert missing["publish"] is False
    assert "active roster" in missing["reason"].lower()
    assert stale["publish"] is False
    assert "stale" in stale["reason"].lower()


def test_active_roster_filters_retired_carry_forward_players():
    cands = pd.DataFrame([{"player_id": "active"}, {"player_id": "retired"}])
    filtered = prop_decision.filter_to_active_roster(cands, {"active"})
    assert filtered["player_id"].tolist() == ["active"]


def test_score_uses_distribution_probability_not_uncalibrated_ranker_value():
    cand = _candidate(p_over=0.99, p_under=0.01, ml_score=99.0)
    scored = score_candidate(cand)
    expected = prop_decision.probability_from_projection(cand)
    assert scored["components"]["model_prob"] == pytest.approx(expected, abs=1e-4)
    assert scored["components"]["model_prob_source"] == "mean_sd_line_distribution"
    assert scored["components"]["probability_coherent"] is False


def test_one_book_market_is_context_only_with_no_action_fields():
    scored = score_candidate(_candidate(prices={"over": 1.95, "under": 1.95,
                                                "n_books": 1,
                                                "consensus_p_over": 0.50}))
    assert scored["market_state"] == "ONE_BOOK_CONTEXT_ONLY"
    assert scored["edge"] is None
    assert scored["components"]["ev_best_price"] is None
    assert scored["components"]["kelly_fraction"] is None


def test_two_book_market_is_real_only_when_probability_coheres():
    cand = _candidate()
    p = prop_decision.probability_from_projection(cand)
    cand["p_over"], cand["p_under"] = p, 1 - p
    scored = score_candidate(cand)
    assert scored["market_state"] == "REAL_MARKET"
    assert scored["edge"] is not None


def test_t90_slate_selector_includes_a_wednesday_kickoff():
    from scripts.auto_weekly import games_due_for_t90
    now = dt.datetime(2026, 9, 9, 17, 30, tzinfo=dt.timezone.utc)  # Wednesday
    slate = pd.DataFrame([{"game_id": "wed", "kickoff": now + dt.timedelta(minutes=90)},
                          {"game_id": "sun", "kickoff": now + dt.timedelta(days=4)}])
    assert games_due_for_t90(slate, now)["game_id"].tolist() == ["wed"]


def test_t90_due_window_is_inactives_to_kickoff_only():
    """Due means [T-90, T-0): earlier than the inactives list would read a
    roster without game-day inactives (and a processed game is never
    re-processed); a started game is not due."""
    from scripts.auto_weekly import games_due_for_t90
    now = dt.datetime(2026, 9, 13, 15, 55, tzinfo=dt.timezone.utc)
    slate = pd.DataFrame([
        {"game_id": "too_early", "kickoff": now + dt.timedelta(minutes=91)},
        {"game_id": "due_edge", "kickoff": now + dt.timedelta(minutes=90)},
        {"game_id": "due_late", "kickoff": now + dt.timedelta(minutes=1)},
        {"game_id": "started", "kickoff": now},
        {"game_id": "in_2h", "kickoff": now + dt.timedelta(hours=2)},
    ])
    assert games_due_for_t90(slate, now)["game_id"].tolist() == ["due_edge", "due_late"]


def _graded_rows(n, line_source, weeks=6, market_state=None, p=0.6):
    rows = []
    for i in range(n):
        rows.append({"p_side": p, "hit": int(i % 5 != 0),
                     "line_source": line_source, "market_state": market_state,
                     "season": 2026, "week": 1 + (i % weeks), "game_id": f"g{i % 4}",
                     "player_id": f"p{i}", "market": "receiving_yards", "side": "over",
                     "clock": "wed", "status": "active"})
    return rows


def test_probability_grade_labels_synthetic_evidence_and_reports_bootstrap_interval():
    report = grade_rows(_graded_rows(30, "synthetic_trailing_mean", weeks=6),
                        bootstrap_n=100, seed=7)
    assert report["evidence_kind"] == "synthetic_research_only"
    assert report["coverage"]["synthetic_or_reference"] == 30
    assert report["coverage"]["graded_real_line"] == 0
    assert report["calibration"]["brier"] is not None
    assert report["uncertainty"]["brier_95_interval"][0] is not None
    assert report["uncertainty"]["method"].startswith("cluster bootstrap by season-week")
    assert report["uncertainty"]["n_clusters"] == 6
    assert "ROI" not in report["claim"] or "no" in report["claim"].lower()
