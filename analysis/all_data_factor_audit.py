#!/usr/bin/env python3
"""Audit predeclared binary factors with joint uncertainty and provenance.

The input is a frozen candidate-level frame: one row per player/market/week,
factor columns known before kickoff, and a binary outcome.  This module never
turns current-week usage into an "absence" and never searches interactions.
Interaction selection belongs in ``nested_factor_selection.py`` so the held-out
season cannot influence which combination gets reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from nflvalue.reproducibility import CANONICAL_CSV_VERSION, canonical_csv_sha256


RNG_SEED = 20260714
DEFAULT_DRAWS = 20_000
PROMOTION_MIN_EXPOSED = 100
PROMOTION_MIN_CONTROL = 100
PROMOTION_Q_MAX = 0.05
FORBIDDEN_PREGAME_COLUMNS = {"usage_absence", "zero_current_usage", "actual_absence"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _binary(series: pd.Series, name: str) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"{name} contains missing values; define exclusions explicitly")
    values = series.astype(int).to_numpy()
    if not set(np.unique(values)).issubset({0, 1}):
        raise ValueError(f"{name} must be binary")
    return values


def beta_difference(
    exposed_hits: int,
    exposed_n: int,
    control_hits: int,
    control_n: int,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = RNG_SEED,
) -> dict[str, float]:
    """Sample both rates under independent Jeffreys beta-binomial posteriors."""
    if min(exposed_n, control_n) <= 0:
        raise ValueError("both exposed and control groups must be non-empty")
    rng = np.random.default_rng(seed)
    exposed = rng.beta(exposed_hits + 0.5, exposed_n - exposed_hits + 0.5, draws)
    control = rng.beta(control_hits + 0.5, control_n - control_hits + 0.5, draws)
    delta = exposed - control
    lo, hi = np.quantile(delta, [0.025, 0.975])
    return {
        "posterior_mean_difference": float(delta.mean()),
        "credible_interval_95_low": float(lo),
        "credible_interval_95_high": float(hi),
        "posterior_probability_positive": float((delta > 0).mean()),
        "posterior_probability_negative": float((delta < 0).mean()),
    }


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    exposure: str,
    outcome: str,
    cluster: str,
    *,
    draws: int = 5_000,
    seed: int = RNG_SEED,
) -> dict[str, float | int | None]:
    """Pairs bootstrap whole clusters, preserving within-team-season dependence."""
    clusters = frame[cluster].dropna().unique()
    if len(clusters) < 2:
        return {"clusters": int(len(clusters)), "interval_95_low": None, "interval_95_high": None}
    grouped = []
    for value in clusters:
        group = frame[frame[cluster] == value]
        exposed = group[exposure].astype(bool)
        grouped.append((
            int(group.loc[exposed, outcome].sum()), int(exposed.sum()),
            int(group.loc[~exposed, outcome].sum()), int((~exposed).sum()),
        ))
    totals = np.asarray(grouped, dtype=float)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        sampled = totals[rng.integers(0, len(totals), len(totals))].sum(axis=0)
        if sampled[1] and sampled[3]:
            differences.append(sampled[0] / sampled[1] - sampled[2] / sampled[3])
    if not differences:
        return {"clusters": int(len(clusters)), "interval_95_low": None, "interval_95_high": None}
    lo, hi = np.quantile(differences, [0.025, 0.975])
    return {"clusters": int(len(clusters)), "interval_95_low": float(lo), "interval_95_high": float(hi)}


def evaluate_pattern(
    frame: pd.DataFrame,
    specification: dict,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = RNG_SEED,
) -> dict:
    exposure = specification["exposure"]
    outcome = specification["outcome"]
    eligible_col = specification.get("eligible")
    filters = specification.get("filters", {})
    cohort = specification.get("cohort", "pregame")
    if cohort == "pregame" and exposure in FORBIDDEN_PREGAME_COLUMNS:
        raise ValueError(f"{exposure} is postgame leakage and cannot define a pregame cohort")
    required = {exposure, outcome, specification.get("cluster", "team_season")}
    if eligible_col:
        required.add(eligible_col)
    required.update(filters)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"pattern {specification['name']} lacks columns: {missing}")

    eligible = frame.copy()
    if eligible_col:
        mask = _binary(eligible[eligible_col], eligible_col).astype(bool)
        eligible = eligible.loc[mask].copy()
    for column, value in filters.items():
        eligible = eligible.loc[eligible[column] == value].copy()
    if eligible.empty:
        raise ValueError(f"pattern {specification['name']} has no rows after filters")
    exp = _binary(eligible[exposure], exposure).astype(bool)
    outcome_values = _binary(eligible[outcome], outcome)
    exposed_hits = int(outcome_values[exp].sum())
    exposed_n = int(exp.sum())
    control_hits = int(outcome_values[~exp].sum())
    control_n = int((~exp).sum())
    if min(exposed_n, control_n) == 0:
        raise ValueError(f"pattern {specification['name']} has an empty exposed/control group")

    table = [[exposed_hits, exposed_n - exposed_hits],
             [control_hits, control_n - control_hits]]
    result = {
        "name": specification["name"],
        "cohort": cohort,
        "exposure": exposure,
        "outcome": outcome,
        "definition": specification.get("definition", ""),
        "exclusions": specification.get("exclusions", []),
        "filters": filters,
        "exposed_n": exposed_n,
        "exposed_hits": exposed_hits,
        "exposed_rate": exposed_hits / exposed_n,
        "control_n": control_n,
        "control_hits": control_hits,
        "control_rate": control_hits / control_n,
        "raw_difference": exposed_hits / exposed_n - control_hits / control_n,
        "control_design": {
            "definition": "unexposed rows in the identical eligible and filtered cohort",
            "exact_n": control_n,
            "matched_control_verified": False,
            "reason": "this audit estimates filtered cohort contrasts; explicit role/game-script matching is a separate required promotion gate",
        },
        "fisher_two_sided_p": float(fisher_exact(table, alternative="two-sided").pvalue),
    }
    result.update(beta_difference(exposed_hits, exposed_n, control_hits, control_n,
                                  draws=draws, seed=seed))
    result["cluster_bootstrap"] = cluster_bootstrap_difference(
        eligible.assign(_exposure=exp, _outcome=outcome_values),
        "_exposure", "_outcome", specification.get("cluster", "team_season"), seed=seed,
    )
    return result


def benjamini_hochberg(results: list[dict]) -> None:
    """Attach monotone BH q-values across exactly the published family."""
    order = sorted(range(len(results)), key=lambda i: results[i]["fisher_two_sided_p"])
    m = len(order)
    running = 1.0
    for rank_from_end, index in enumerate(reversed(order), start=1):
        rank = m - rank_from_end + 1
        running = min(running, results[index]["fisher_two_sided_p"] * m / rank)
        results[index]["bh_q"] = float(running)


def run_audit(frame: pd.DataFrame, specifications: Iterable[dict], **kwargs) -> list[dict]:
    results = [evaluate_pattern(frame, spec, seed=RNG_SEED + i, **kwargs)
               for i, spec in enumerate(specifications)]
    benjamini_hochberg(results)
    for result in results:
        interval = result["cluster_bootstrap"]
        low, high = interval.get("interval_95_low"), interval.get("interval_95_high")
        cluster_excludes_zero = (
            low is not None and high is not None and (low > 0 or high < 0)
        )
        gates = {
            "minimum_exposed_n": result["exposed_n"] >= PROMOTION_MIN_EXPOSED,
            "minimum_control_n": result["control_n"] >= PROMOTION_MIN_CONTROL,
            "bh_q_below_0_05": result["bh_q"] < PROMOTION_Q_MAX,
            "cluster_interval_excludes_zero": cluster_excludes_zero,
            "matched_control_verified": result["control_design"]["matched_control_verified"],
            "season_forward_replication": False,
        }
        result["promotion_gates"] = gates
        result["promotion_status"] = (
            "eligible_for_bounded_shadow_challenger" if all(gates.values())
            else "research_only"
        )
        result["live_scoring_eligible"] = False
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--patterns", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/all_data_factor_audit.json"))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    specs = json.loads(args.patterns.read_text())
    results = run_audit(frame, specs, draws=args.draws)
    root = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": 3,
        "status": "research_only",
        "input": str(args.input),
        "input_file_sha256": file_sha256(args.input),
        "input_canonical_csv_sha256": canonical_csv_sha256(
            frame, row_keys=["season", "week", "game_id", "team", "player_id", "market"]
        ),
        "canonical_csv_version": CANONICAL_CSV_VERSION,
        "patterns": str(args.patterns),
        "patterns_sha256": file_sha256(args.patterns),
        "git_revision": git_revision(root),
        "random_seed": RNG_SEED,
        "posterior": "independent Jeffreys beta-binomial rates; Monte Carlo difference",
        "dependence_check": "pairs bootstrap by team-season",
        "multiple_comparisons": "Benjamini-Hochberg over every result in this file",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
