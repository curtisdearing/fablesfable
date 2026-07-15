"""End-to-end smoke test for prop_backtest.py: it must run over a real (if
small) walk-forward slice, write data/prop_backtest.json with the metrics the
spec requires, and never claim more than "projection accuracy" (the honest
framing is load-bearing -- PROP_SHORTLISTER_SPEC.md §5 / PREMORTEM.md)."""

from __future__ import annotations

import json
import math
import os

EXPECTED_MARKETS = {
    "receiving_yards", "receptions", "rushing_yards", "passing_yards",
    "pass_attempts", "rush_attempts", "anytime_td",
}


def test_backtest_runs_end_to_end_and_writes_report(backtest_report_fast):
    report = backtest_report_fast

    assert set(report["markets"]) == EXPECTED_MARKETS
    assert "not" in report["note"].lower() and "price" in report["note"].lower(), (
        "the report must keep the honest 'accuracy, not price-beating' framing"
    )

    out_path = report["_test_output_path"]
    assert os.path.exists(out_path)
    on_disk = json.load(open(out_path))
    assert set(on_disk["markets"]) == EXPECTED_MARKETS

    # every market with a non-trivial sample must produce FINITE overall MAE/RMSE
    for market, res in report["markets"].items():
        overall = res["overall"]
        if overall["n"] >= 20:
            assert overall["mae"] is not None and math.isfinite(overall["mae"]), market
            assert overall["rmse"] is not None and math.isfinite(overall["rmse"]), market
        # low_confidence flag must be present and match the projection registry
        assert isinstance(res["low_confidence"], bool)

    assert report["markets"]["anytime_td"]["low_confidence"] is True
    assert report["markets"]["receiving_yards"]["low_confidence"] is False


def test_backtest_by_sample_size_buckets_present(backtest_report_fast):
    buckets = backtest_report_fast["markets"]["receiving_yards"]["by_sample_size"]
    # at minimum, the well-populated buckets should exist for a 2-season slice
    assert "1-3 games" in buckets
    assert "8+ games" in buckets
    for label, m in buckets.items():
        assert m["n"] > 0, label
