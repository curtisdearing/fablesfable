"""Projection sanity-diff contract."""
from analysis.sanity_diff import compare


def test_prop_diff_reports_churn_side_flips_and_rank_changes():
    before = {"games": [{"game_id": "g", "leans": [
        {"player_id": "a", "market": "rush", "composite": 9, "side": "over"},
        {"player_id": "b", "market": "recv", "composite": 8, "side": "under"},
    ]}]}
    after = {"games": [{"game_id": "g", "leans": [
        {"player_id": "b", "market": "recv", "composite": 10, "side": "over"},
        {"player_id": "c", "market": "rush", "composite": 7, "side": "over"},
    ]}]}
    report = compare(before, after, top_n=2)
    assert report["overlap_rate"] == 0.5
    assert report["side_flips"] == ["g|b|recv"]
    assert report["added"] == ["g|c|rush"]
    assert report["removed"] == ["g|a|rush"]


def test_fantasy_summary_mean_shape_is_supported():
    payload = {"players": [{"player_id": "a", "summary": {"mean": 12.0}},
                           {"player_id": "b", "summary": {"mean": 8.0}}]}
    report = compare(payload, payload, top_n=10)
    assert report["overlap_rate"] == 1.0
    assert report["candidate_count"] == 2
