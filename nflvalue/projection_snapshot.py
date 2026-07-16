"""Scoring-independent projection snapshots shared by NFL consumers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reproducibility import CANONICAL_CSV_VERSION, canonical_csv_sha256


SCHEMA_VERSION = 1
COMPONENT_NAMES = (
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "fumbles_lost",
)
QUANTILES = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90}
SUMMARY_NAMES = {"mean", "sd", *QUANTILES}
VALIDATION_STATUSES = {"approved", "research_only", "unvalidated"}
FORBIDDEN_CONSUMER_KEYS = {
    "bookmaker", "edge", "fantasy_points", "line", "odds", "scoring", "vig",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_components(components: dict[str, pd.DataFrame]) -> tuple[list[str], int]:
    missing = sorted(set(COMPONENT_NAMES) - set(components))
    if missing:
        raise ValueError(f"projection components missing: {missing}")
    first = components[COMPONENT_NAMES[0]]
    ids = [str(value) for value in first.columns]
    if len(ids) != len(set(ids)):
        raise ValueError("projection component player IDs must be unique")
    for name in COMPONENT_NAMES:
        frame = components[name]
        if [str(value) for value in frame.columns] != ids or len(frame) != len(first):
            raise ValueError(f"projection component shape/order mismatch: {name}")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"projection component contains non-finite values: {name}")
    return ids, len(first)


def component_samples_frame(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one canonical row per simulation/player from wide component matrices."""

    ids, simulations = _validate_components(components)
    data: dict[str, Any] = {
        "simulation": np.tile(np.arange(simulations, dtype=int), len(ids)),
        "player_id": np.repeat(ids, simulations),
    }
    for name in COMPONENT_NAMES:
        data[name] = components[name].to_numpy(dtype=float).T.reshape(-1)
    return pd.DataFrame(data)


def write_component_samples(
    components: dict[str, pd.DataFrame], path: str | Path
) -> dict[str, Any]:
    """Persist correlated component draws and return content/integrity metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = component_samples_frame(components)
    samples.to_parquet(path, index=False)
    return {
        "path": path.name,
        "rows": int(len(samples)),
        "simulations": int(samples["simulation"].nunique()),
        "players": int(samples["player_id"].nunique()),
        "canonical_csv_version": CANONICAL_CSV_VERSION,
        "canonical_csv_sha256": canonical_csv_sha256(
            samples, row_keys=["simulation", "player_id"]
        ),
        "parquet_sha256_integrity_only": _file_sha256(path),
    }


def load_component_samples(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load long-form component draws into the simulator's wide dictionary shape."""

    samples = pd.read_parquet(path)
    required = {"simulation", "player_id", *COMPONENT_NAMES}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"component sample artifact missing columns: {missing}")
    if samples.duplicated(["simulation", "player_id"]).any():
        raise ValueError("component sample artifact has duplicate simulation/player rows")
    output = {}
    for name in COMPONENT_NAMES:
        output[name] = samples.pivot(
            index="simulation", columns="player_id", values=name
        ).sort_index().sort_index(axis=1)
    return output


def _component_summary(values: np.ndarray) -> dict[str, float]:
    summary = {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values)),
    }
    summary.update({name: float(np.quantile(values, value)) for name, value in QUANTILES.items()})
    return summary


def build_projection_snapshot(
    players: pd.DataFrame,
    summaries: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    *,
    season: int,
    week: int,
    generated_at: str,
    information_as_of: str,
    model_version: str,
    simulation_metadata: dict[str, Any],
    sample_artifact: dict[str, Any],
    source_manifest: dict[str, Any] | None = None,
    component_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the consumer-neutral JSON half of a projection snapshot."""

    ids, simulations = _validate_components(components)
    identity = players.copy()
    identity["player_id"] = identity["player_id"].astype(str)
    identity = identity.drop_duplicates("player_id").set_index("player_id")
    summary = summaries.copy()
    summary["player_id"] = summary["player_id"].astype(str)
    summary = summary.drop_duplicates("player_id").set_index("player_id")
    missing_identity = sorted(set(ids) - set(identity.index))
    if missing_identity:
        raise ValueError(f"snapshot players missing identity rows: {missing_identity}")

    records = []
    canonical_rows = []
    for column_index, player_id in enumerate(ids):
        row = identity.loc[player_id]
        availability = float(summary.loc[player_id, "availability_probability"])
        component_summary = {
            name: _component_summary(components[name].iloc[:, column_index].to_numpy(dtype=float))
            for name in COMPONENT_NAMES
        }
        record = {
            "player_id": player_id,
            "player_name": str(row.get("player_name", player_id)),
            "position": str(row.get("position", "unknown")),
            "team": str(row.get("team", "unknown")),
            "opponent_team": str(row.get("opponent_team", "unknown")),
            "game_id": str(row.get("game_id", "unknown")),
            "correlation_group": str(row.get("game_id", "unknown")),
            "availability_probability": availability,
            "official_inactive": bool(
                float(row.get("status_inactive", 0) or 0) == 1
                or float(row.get("injury_out", 0) or 0) == 1
            ),
            "components": component_summary,
        }
        records.append(record)
        flat = {
            key: value for key, value in record.items() if key != "components"
        }
        for component, values in component_summary.items():
            for statistic, value in values.items():
                flat[f"{component}__{statistic}"] = value
        canonical_rows.append(flat)

    player_hash = canonical_csv_sha256(pd.DataFrame(canonical_rows), row_keys=["player_id"])
    metadata = {
        key: value for key, value in simulation_metadata.items()
        if key in {"simulations", "random_seed", "players", "games", "calibration"}
    }
    if int(metadata.get("simulations", simulations)) != simulations:
        raise ValueError("simulation metadata count disagrees with component samples")
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "contract": "scoring-independent correlated NFL player event distributions",
        "season": int(season),
        "week": int(week),
        "generated_at": generated_at,
        "information_as_of": information_as_of,
        "model_version": model_version,
        "players_canonical_csv_sha256": player_hash,
        "canonical_csv_version": CANONICAL_CSV_VERSION,
        "simulation": metadata,
        "sample_artifact": sample_artifact,
        "source_manifest": source_manifest or {},
        "component_validation": component_validation or {"status": "unvalidated"},
        "players": records,
    }
    validate_projection_snapshot(snapshot)
    return snapshot


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def validate_projection_snapshot(snapshot: dict[str, Any]) -> None:
    required = {
        "schema_version", "contract", "season", "week", "generated_at",
        "information_as_of", "model_version", "simulation", "sample_artifact", "players",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise ValueError(f"projection snapshot missing fields: {missing}")
    if snapshot["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported projection snapshot schema: {snapshot['schema_version']}")
    validation = snapshot.get("component_validation", {"status": "unvalidated"})
    if not isinstance(validation, dict):
        raise ValueError("projection snapshot component validation must be an object")
    if validation.get("status", "unvalidated") not in VALIDATION_STATUSES:
        raise ValueError("projection snapshot has invalid component validation status")
    markets = validation.get("markets", {})
    if not isinstance(markets, dict) or any(
        status not in VALIDATION_STATUSES for status in markets.values()
    ):
        raise ValueError("projection snapshot has invalid market validation status")
    audit_hash = validation.get("audit_replay_canonical_csv_sha256")
    if audit_hash is not None and len(str(audit_hash)) != 64:
        raise ValueError("projection snapshot has invalid component audit hash")
    forbidden = sorted(FORBIDDEN_CONSUMER_KEYS & set(_walk_keys(snapshot)))
    if forbidden:
        raise ValueError(f"consumer-specific keys leaked into projection snapshot: {forbidden}")
    players = snapshot["players"]
    ids = [str(player.get("player_id", "")) for player in players]
    if not players or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("projection snapshot requires unique nonempty players")
    for player in players:
        components = player.get("components", {})
        if set(components) != set(COMPONENT_NAMES):
            raise ValueError(f"projection snapshot component mismatch for {player['player_id']}")
        for name, values in components.items():
            if not isinstance(values, dict) or set(values) != SUMMARY_NAMES:
                raise ValueError(f"projection snapshot summary mismatch for {name}")
            numeric = {key: float(value) for key, value in values.items()}
            if not all(np.isfinite(value) for value in numeric.values()) or numeric["sd"] < 0:
                raise ValueError(f"projection snapshot summary is non-finite for {name}")
            if not (
                numeric["p10"] <= numeric["p25"] <= numeric["p50"]
                <= numeric["p75"] <= numeric["p90"]
            ):
                raise ValueError(f"projection snapshot quantiles are unordered for {name}")
        probability = float(player["availability_probability"])
        if not 0 <= probability <= 1:
            raise ValueError("availability probability must be between zero and one")
    for key in ("canonical_csv_sha256", "parquet_sha256_integrity_only"):
        if len(str(snapshot["sample_artifact"].get(key, ""))) != 64:
            raise ValueError(f"sample artifact has invalid {key}")


def write_projection_snapshot(snapshot: dict[str, Any], path: str | Path) -> None:
    validate_projection_snapshot(snapshot)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
