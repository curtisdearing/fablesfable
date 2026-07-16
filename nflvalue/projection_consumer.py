"""Fablesfable adapter for scoring-independent projection snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .projection_snapshot import load_component_samples, validate_projection_snapshot
from .reproducibility import canonical_csv_sha256


MARKET_COMPONENTS = {
    "pass_attempts": "attempts",
    "pass_completions": "completions",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "interceptions": "passing_interceptions",
    "rushing_attempts": "carries",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "targets": "targets",
}


@dataclass(frozen=True)
class ProjectionBundle:
    snapshot: dict
    components: dict[str, pd.DataFrame]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_projection_bundle(
    snapshot_path: str | Path, samples_path: str | Path
) -> ProjectionBundle:
    """Load and verify both halves of the shared projection contract."""

    snapshot_path = Path(snapshot_path)
    samples_path = Path(samples_path)
    snapshot = json.loads(snapshot_path.read_text())
    validate_projection_snapshot(snapshot)
    descriptor = snapshot["sample_artifact"]
    actual_file_hash = _file_sha256(samples_path)
    if actual_file_hash != descriptor["parquet_sha256_integrity_only"]:
        raise ValueError("projection sample Parquet integrity hash mismatch")
    samples = pd.read_parquet(samples_path)
    actual_content_hash = canonical_csv_sha256(
        samples, row_keys=["simulation", "player_id"]
    )
    if actual_content_hash != descriptor["canonical_csv_sha256"]:
        raise ValueError("projection sample canonical content hash mismatch")
    components = load_component_samples(samples_path)
    contract_ids = {str(player["player_id"]) for player in snapshot["players"]}
    sample_ids = set(next(iter(components.values())).columns.astype(str))
    if contract_ids != sample_ids:
        raise ValueError("projection snapshot player IDs disagree with component samples")
    return ProjectionBundle(snapshot=snapshot, components=components)


def market_distribution(
    bundle: ProjectionBundle, player_id: str, market: str
) -> np.ndarray:
    """Return simulation draws for a fablesfable stat market."""

    player_id = str(player_id)
    if market == "anytime_td":
        values = (
            bundle.components["rushing_tds"][player_id]
            + bundle.components["receiving_tds"][player_id]
        )
    else:
        component = MARKET_COMPONENTS.get(market)
        if component is None:
            raise ValueError(f"unsupported shared projection market: {market}")
        if player_id not in bundle.components[component]:
            raise KeyError(f"player {player_id} is absent from projection samples")
        values = bundle.components[component][player_id]
    return values.to_numpy(dtype=float)


def market_validation_status(bundle: ProjectionBundle, market: str) -> str:
    """Return the producer's explicit validation status for a component market."""

    validation = bundle.snapshot.get("component_validation", {"status": "unvalidated"})
    return str(
        validation.get("markets", {}).get(market, validation.get("status", "unvalidated"))
    )


def probability_over(
    bundle: ProjectionBundle,
    player_id: str,
    market: str,
    line: float,
    *,
    require_approved: bool = True,
) -> float:
    status = market_validation_status(bundle, market)
    if require_approved and status != "approved":
        raise ValueError(
            f"shared projection market {market} is {status}; approved validation is required"
        )
    values = market_distribution(bundle, player_id, market)
    return float(np.mean(values > float(line)))
