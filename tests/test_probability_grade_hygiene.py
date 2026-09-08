"""Row hygiene, provenance, population, deduplication and guards for the
probability grade (analysis/prop_probability_grade.py).

REAL_MARKET is a market-quality label.  Whether a decision was captured
BEFORE its kickoff is provenance, and only provenance can make a row
prospective.  Non-binary outcomes are rejected before any coercion could
truncate them.  Every row in the recommendation population is counted once.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.prop_probability_grade import (  # noqa: E402
    classify_rows,
    grade_rows,
    parse_binary_outcome,
)


def _row(i, *, hit=1, p=0.6, clock="wed", side="over", line=50.5, status="active",
         line_source="odds_api", market_state="REAL_MARKET", created_at="2026-09-02T17:54:18Z",
         kickoff="2026-09-13T17:00:00Z", settlement=None, week=None, market="receiving_yards"):
    return {"p_side": p, "hit": hit, "line_source": line_source, "market_state": market_state,
            "season": 2026, "week": week if week is not None else 1 + (i % 6), "game_id": f"g{i % 4}",
            "player_id": f"p{i}", "market": market, "side": side, "line": line, "clock": clock,
            "status": status, "created_at": created_at, "kickoff": kickoff, "settlement": settlement}


# ------------------------------------------------------------ binary parse --
@pytest.mark.parametrize("value, expected", [
    (0, 0), (1, 1), (0.0, 0), (1.0, 1), (True, 1), (False, 0), ("0", 0), ("1", 1), (" 1 ", 1),
])
def test_binary_outcomes_are_accepted_exactly(value, expected):
    assert parse_binary_outcome(value) == expected


@pytest.mark.parametrize("value", [0.7, 0.999, 1.5, -1, 2, float("nan"), float("inf"), -float("inf"),
                                   "yes", "true", "1.0x", "", None, [1], {"hit": 1}])
def test_non_binary_outcomes_are_rejected_before_any_truncation(value):
    assert parse_binary_outcome(value) is None


def test_fractional_hit_row_is_invalid_not_truncated_to_a_loss():
    rows = [_row(0, hit=0.7)]
    b = classify_rows(rows)
    assert len(b["invalid"]) == 1 and "binary" in b["invalid"][0]["reject_reason"]
    assert not b["real_line"]


# ---------------------------------------------------------------- settlement --
def test_push_void_unresolved_rows_are_counted_separately_and_never_graded():
    rows = [_row(0, hit=None, settlement="push"), _row(1, hit=None, settlement="void"),
            _row(2, hit=None, settlement="unresolved"), _row(3, hit=1, settlement="win"),
            _row(4, hit=0, settlement="loss"), _row(5, hit=None, settlement=None)]
    b = classify_rows(rows)
    assert len(b["push"]) == 1 and len(b["void"]) == 1 and len(b["unresolved"]) == 2
    assert len(b["real_line"]) == 2
    rep = grade_rows(rows, bootstrap_n=50, seed=1)
    cov = rep["coverage"]
    assert cov["push"] == 1 and cov["void"] == 1 and cov["unresolved"] == 2 and cov["graded_real_line"] == 2


def test_missing_probability_rows_are_counted_not_dropped():
    rows = [_row(0, p=None), _row(1, p=0.55)]
    rep = grade_rows(rows, bootstrap_n=50, seed=1)
    assert rep["coverage"]["missing_probability"] == 1
    assert rep["coverage"]["rows_received"] == 2
    assert rep["coverage"]["eligible"] == 1


# ---------------------------------------------------------------- provenance --
def test_real_market_label_does_not_make_rows_prospective_without_kickoff_provenance():
    rows = [_row(i, kickoff=None) for i in range(150)]
    rep = grade_rows(rows, bootstrap_n=50, seed=1)
    assert rep["evidence_kind"] != "prospective_real_line"
    assert rep["coverage"]["provenance_unknown"] == 150
    assert rep["coverage"]["graded_real_line_prospective"] == 0


def test_retrospective_capture_is_never_promoted_even_at_volume():
    rows = [_row(i, created_at="2026-09-14T03:00:00Z", kickoff="2026-09-13T17:00:00Z") for i in range(150)]
    rep = grade_rows(rows, bootstrap_n=50, seed=1)
    assert rep["evidence_kind"] != "prospective_real_line"
    assert rep["coverage"]["provenance_retrospective"] == 150


def test_prospective_real_line_requires_pre_kickoff_capture_and_the_floor():
    rows = [_row(i) for i in range(150)]
    rep = grade_rows(rows, bootstrap_n=50, seed=1)
    assert rep["evidence_kind"] == "prospective_real_line"
    assert rep["coverage"]["graded_real_line_prospective"] == 150
    few = grade_rows([_row(i) for i in range(20)], bootstrap_n=50, seed=1)
    assert few["evidence_kind"] == "real_line_insufficient"


def test_unparseable_or_naive_timestamps_are_unknown_provenance():
    rows = [_row(0, created_at="2026-09-02 17:54:18"), _row(1, created_at="not a time"), _row(2, created_at=None)]
    b = classify_rows(rows)
    assert all(r["provenance"] == "unknown" for r in b["real_line"])


# ------------------------------------------------------------- deduplication --
def test_t90_supersedes_wednesday_even_when_the_side_flips():
    rows = [_row(0, clock="wed", side="over", week=1), _row(0, clock="t90", side="under", week=1)]
    b = classify_rows(rows)
    assert len(b["real_line"]) == 1 and b["real_line"][0]["clock"] == "t90"
    assert len(b["superseded"]) == 1 and b["superseded"][0]["superseded_reason"] == "clock_supersession+side_change"


def test_line_change_at_t90_is_the_same_decision():
    rows = [_row(0, clock="wed", line=50.5, week=1), _row(0, clock="t90", line=52.5, week=1)]
    b = classify_rows(rows)
    assert len(b["real_line"]) == 1 and b["real_line"][0]["line"] == 52.5
    assert b["superseded"][0]["superseded_reason"] == "clock_supersession+line_change"


def test_repeated_invocation_keeps_the_latest_capture():
    rows = [_row(0, created_at="2026-09-02T10:00:00Z", week=1), _row(0, created_at="2026-09-02T17:54:18Z", week=1)]
    b = classify_rows(rows)
    assert len(b["real_line"]) == 1 and b["real_line"][0]["created_at"] == "2026-09-02T17:54:18Z"
    assert b["superseded"][0]["superseded_reason"] == "repeat_invocation"


def test_dedup_policy_is_declared_in_the_report():
    rep = grade_rows([_row(0)], bootstrap_n=50, seed=1)
    assert "one decision per (season, week, game_id, player_id, market)" in rep["dedup_policy"]


# --------------------------------------------------------------------- guards --
@pytest.mark.parametrize("bad", [0, -5, "x", 1.5, None])
def test_invalid_bootstrap_argument_is_rejected(bad):
    with pytest.raises(ValueError):
        grade_rows([_row(0)], bootstrap_n=bad, seed=1)


def test_invalid_seed_is_rejected():
    with pytest.raises(ValueError):
        grade_rows([_row(0)], bootstrap_n=10, seed="seven")


def test_small_cluster_interval_is_unavailable_not_fabricated():
    rep = grade_rows([_row(i, week=1) for i in range(30)], bootstrap_n=50, seed=1)
    assert rep["uncertainty"]["brier_95_interval"] == [None, None]
    assert rep["uncertainty"]["n_clusters"] == 1


def test_nan_probability_is_missing_and_out_of_range_probability_is_invalid():
    b = classify_rows([_row(0, p=math.nan)])
    assert len(b["missing_probability"]) == 1 and not b["invalid"]
    b = classify_rows([_row(0, p=1.7)])
    assert len(b["invalid"]) == 1 and not b["missing_probability"]
    b = classify_rows([_row(0, p="0.6x")])
    assert len(b["invalid"]) == 1
