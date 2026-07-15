#!/usr/bin/env python3
"""Run the predeclared nested factor-combination projection family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis.nested_factor_selection import nested_season_forward


COMMON = [
    "birthday_window_5", "revenge", "primetime", "short_rest", "post_bye",
    "big_favorite", "big_underdog", "high_total", "division_rematch",
    "body_clock_1pm_road", "after_overtime", "dome", "turf",
]
SEGMENTS = {
    "QB1 passing yards": {
        "filters": {"role": "QB", "depth_rank": 1, "market": "passing_yards"},
        "factors": COMMON + ["opponent_db_2_plus", "own_ol_2_plus", "official_wr1_out",
                              "official_te1_out", "official_rb1_out"],
    },
    "RB1 rushing yards": {
        "filters": {"role": "RB", "depth_rank": 1, "market": "rushing_yards"},
        "factors": COMMON + ["opponent_front_2_plus", "own_ol_2_plus", "official_te1_out",
                              "official_wr1_out", "official_rb2_out", "heavy_workload_last_game",
                              "rushing_100_last_game"],
    },
    "WR1 receiving yards": {
        "filters": {"role": "WR", "depth_rank": 1, "market": "receiving_yards"},
        "factors": COMMON + ["opponent_db_2_plus", "official_te1_out", "official_rb1_out",
                              "official_wr2_out", "receiving_100_last_game", "target_spike_last_game"],
    },
    "TE1 receiving yards": {
        "filters": {"role": "TE", "depth_rank": 1, "market": "receiving_yards"},
        "factors": COMMON + ["opponent_db_2_plus", "official_wr1_out", "official_rb1_out",
                              "official_te2_out", "receiving_100_last_game", "target_spike_last_game"],
    },
    "RB1 receptions": {
        "filters": {"role": "RB", "depth_rank": 1, "market": "receptions"},
        "factors": COMMON + ["opponent_front_2_plus", "official_wr1_out", "official_te1_out",
                              "official_rb2_out", "target_spike_last_game"],
    },
}


def run(frame: pd.DataFrame, *, max_order: int = 3, min_group_n: int = 100) -> dict:
    outputs = {}
    for name, definition in SEGMENTS.items():
        segment = frame[frame["eligible"].astype(bool)].copy()
        for column, value in definition["filters"].items():
            segment = segment[segment[column] == value]
        outputs[name] = {
            "filters": definition["filters"],
            "candidate_factors": definition["factors"],
            "evaluation": nested_season_forward(
                segment, definition["factors"], "over", max_order=max_order,
                min_group_n=min_group_n, min_train_seasons=3,
            ),
        }
    # Do not crown a tiny outer sample.  This threshold is fixed in code and
    # applied before comparing segment accuracy.
    eligible = [(name, item["evaluation"]) for name, item in outputs.items()
                if item["evaluation"]["outer_test_n"] >= 100]
    descriptive_best = None
    if eligible:
        name, evaluation = max(eligible, key=lambda item: (
            item[1]["outer_test_accuracy"], item[1]["outer_test_n"]))
        descriptive_best = {"segment": name,
                            "outer_test_accuracy": evaluation["outer_test_accuracy"],
                            "outer_test_n": evaluation["outer_test_n"],
                            "outer_reference_accuracy": evaluation["outer_reference_accuracy"],
                            "outer_selective_lift": evaluation["outer_selective_lift"]}
    return {
        "status": "retrospective_nested_diagnostic",
        "target": "actual above strictly-prior trailing mean; not a sportsbook line",
        "selection": "combinations chosen only on seasons before each outer test season",
        "family_warning": "the best segment is descriptive across this predeclared family, not an additional untouched test",
        "best_segment_min_outer_n": 100,
        "prospective_requirement": "freeze before 2026 Week 1 and evaluate real sportsbook lines/CLV",
        "max_order": max_order,
        "min_group_n": min_group_n,
        "segments": outputs,
        "descriptive_best": descriptive_best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--min-group-n", type=int, default=100)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    result = run(frame, max_order=args.max_order, min_group_n=args.min_group_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
