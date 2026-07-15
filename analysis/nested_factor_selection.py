#!/usr/bin/env python3
"""Nested, season-forward selection of predeclared factor conjunctions."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.all_data_factor_audit import RNG_SEED, beta_difference


def conjunctions(factors: list[str], max_order: int = 3):
    for size in range(1, max_order + 1):
        yield from itertools.combinations(factors, size)


def exposure_mask(frame: pd.DataFrame, factors: tuple[str, ...]) -> pd.Series:
    return frame.loc[:, list(factors)].astype(bool).all(axis=1)


def training_score(frame: pd.DataFrame, factors: tuple[str, ...], outcome: str,
                   min_group_n: int, seed: int) -> dict | None:
    exposed = exposure_mask(frame, factors)
    exposed_n, control_n = int(exposed.sum()), int((~exposed).sum())
    if min(exposed_n, control_n) < min_group_n:
        return None
    exposed_hits = int(frame.loc[exposed, outcome].sum())
    control_hits = int(frame.loc[~exposed, outcome].sum())
    rng = np.random.default_rng(seed)
    exposed_rate = rng.beta(exposed_hits + 0.5, exposed_n - exposed_hits + 0.5, 8_000)
    direction = int(exposed_rate.mean() >= 0.5)
    accuracy = exposed_rate if direction else 1 - exposed_rate
    baseline_direction = int(frame[outcome].mean() >= 0.5)
    baseline_accuracy = float((frame[outcome].astype(int) == baseline_direction).mean())
    accuracy_lower = float(np.quantile(accuracy, 0.025))
    effect = beta_difference(exposed_hits, exposed_n, control_hits, control_n,
                             draws=8_000, seed=seed + 1)
    # A condition must credibly improve on the training segment's unconditional
    # majority rule and pay a fixed complexity penalty before outer testing.
    utility = accuracy_lower - baseline_accuracy - 0.005 * (len(factors) - 1)
    return {"factors": factors, "direction": direction,
            "baseline_direction": baseline_direction, "baseline_accuracy": baseline_accuracy,
            "posterior_accuracy_lower_95": accuracy_lower,
            "utility": utility, "training_effect": effect,
            "train_exposed_n": exposed_n, "train_control_n": control_n}


def nested_season_forward(
    frame: pd.DataFrame,
    factors: list[str],
    outcome: str,
    *,
    season_col: str = "season",
    max_order: int = 3,
    min_group_n: int = 100,
    min_train_seasons: int = 3,
) -> dict:
    missing = sorted(set(factors + [outcome, season_col]) - set(frame.columns))
    if missing:
        raise ValueError(f"selection frame lacks columns: {missing}")
    seasons = sorted(int(v) for v in frame[season_col].unique())
    folds = []
    correct = total = 0
    reference_expected_correct = 0.0
    for outer_index, test_season in enumerate(seasons):
        train_seasons = [s for s in seasons if s < test_season]
        if len(train_seasons) < min_train_seasons:
            continue
        train = frame[frame[season_col].isin(train_seasons)]
        candidates = []
        for candidate_index, combo in enumerate(conjunctions(factors, max_order)):
            score = training_score(train, combo, outcome, min_group_n,
                                   RNG_SEED + outer_index * 10_000 + candidate_index)
            if score is not None:
                candidates.append(score)
        if not candidates:
            folds.append({"test_season": test_season, "status": "no_eligible_combination"})
            continue
        selected = max(candidates, key=lambda item: (item["utility"], -len(item["factors"]), item["factors"]))
        if selected["utility"] <= 0:
            folds.append({"test_season": test_season, "status": "no_credible_training_improvement"})
            continue
        test = frame[frame[season_col] == test_season]
        exposed = exposure_mask(test, selected["factors"])
        test_n = int(exposed.sum())
        test_hits = int((test.loc[exposed, outcome].astype(int) == selected["direction"]).sum())
        segment_reference_accuracy = float(
            (test[outcome].astype(int) == selected["baseline_direction"]).mean())
        correct += test_hits
        reference_expected_correct += test_n * segment_reference_accuracy
        total += test_n
        folds.append({
            "test_season": test_season,
            "train_seasons": train_seasons,
            "selected_factors": list(selected["factors"]),
            "selected_direction": "over" if selected["direction"] else "under",
            "selection_utility": selected["utility"],
            "training_reference_accuracy": selected["baseline_accuracy"],
            "training_posterior_accuracy_lower_95": selected["posterior_accuracy_lower_95"],
            "train_exposed_n": selected["train_exposed_n"],
            "test_n": test_n,
            "test_correct": test_hits,
            "test_accuracy": test_hits / test_n if test_n else None,
            "test_segment_reference_accuracy": segment_reference_accuracy,
            "test_selective_lift": (test_hits / test_n - segment_reference_accuracy) if test_n else None,
        })
    posterior = None
    if total:
        rng = np.random.default_rng(RNG_SEED)
        samples = rng.beta(correct + 0.5, total - correct + 0.5, 20_000)
        posterior = {"mean": float(samples.mean()),
                     "credible_interval_95": [float(v) for v in np.quantile(samples, [0.025, 0.975])]}
    return {
        "protocol": "each combination is selected using only seasons earlier than its outer test season",
        "max_order": max_order,
        "min_group_n": min_group_n,
        "folds": folds,
        "outer_test_correct": correct,
        "outer_test_n": total,
        "outer_test_accuracy": correct / total if total else None,
        "outer_reference_expected_correct": reference_expected_correct,
        "outer_reference_accuracy": reference_expected_correct / total if total else None,
        "outer_selective_lift": ((correct - reference_expected_correct) / total) if total else None,
        "accuracy_posterior": posterior,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--factors", nargs="+", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--min-group-n", type=int, default=100)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    result = nested_season_forward(frame, args.factors, args.outcome,
                                   max_order=args.max_order, min_group_n=args.min_group_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
