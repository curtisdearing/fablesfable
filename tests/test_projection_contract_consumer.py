import json

import numpy as np
import pandas as pd
import pytest

from nflvalue.projection_consumer import (
    load_projection_bundle,
    market_distribution,
    market_validation_status,
    probability_over,
)
from nflvalue.projection_snapshot import (
    COMPONENT_NAMES,
    build_projection_snapshot,
    write_component_samples,
    write_projection_snapshot,
)


def _bundle(tmp_path):
    ids = ["qb", "wr"]
    components = {
        name: pd.DataFrame(np.zeros((4, 2)), columns=ids) for name in COMPONENT_NAMES
    }
    components["passing_yards"]["qb"] = [220, 240, 260, 280]
    components["rushing_tds"]["wr"] = [0, 0, 1, 0]
    components["receiving_tds"]["wr"] = [0, 1, 0, 1]
    samples_path = tmp_path / "samples.parquet"
    descriptor = write_component_samples(components, samples_path)
    players = pd.DataFrame([
        {"player_id": "qb", "player_name": "Quarterback", "position": "QB",
         "team": "A", "opponent_team": "B", "game_id": "g"},
        {"player_id": "wr", "player_name": "Receiver", "position": "WR",
         "team": "A", "opponent_team": "B", "game_id": "g"},
    ])
    summaries = pd.DataFrame([
        {"player_id": "qb", "availability_probability": 1.0},
        {"player_id": "wr", "availability_probability": 1.0},
    ])
    snapshot = build_projection_snapshot(
        players, summaries, components, season=2026, week=1,
        generated_at="2026-09-09T20:00:00+00:00",
        information_as_of="2026-09-09T19:55:00+00:00", model_version="abc",
        simulation_metadata={"simulations": 4, "random_seed": 8, "players": 2,
                             "games": 1, "calibration": "fixture"},
        sample_artifact=descriptor,
        component_validation={
            "status": "research_only",
            "markets": {"passing_yards": "approved"},
        },
    )
    snapshot_path = tmp_path / "snapshot.json"
    write_projection_snapshot(snapshot, snapshot_path)
    return snapshot_path, samples_path


def test_fablesfable_maps_shared_draws_to_prop_markets(tmp_path):
    snapshot, samples = _bundle(tmp_path)
    bundle = load_projection_bundle(snapshot, samples)
    assert market_distribution(bundle, "qb", "passing_yards").tolist() == [220, 240, 260, 280]
    assert market_distribution(bundle, "wr", "anytime_td").tolist() == [0, 1, 1, 1]
    assert market_validation_status(bundle, "passing_yards") == "approved"
    assert probability_over(bundle, "qb", "passing_yards", 249.5) == 0.5


def test_fablesfable_refuses_unapproved_market_probability(tmp_path):
    snapshot, samples = _bundle(tmp_path)
    bundle = load_projection_bundle(snapshot, samples)
    with pytest.raises(ValueError, match="research_only"):
        probability_over(bundle, "wr", "anytime_td", 0.5)
    assert probability_over(
        bundle, "wr", "anytime_td", 0.5, require_approved=False
    ) == 0.75


def test_fablesfable_rejects_mutated_sample_artifact(tmp_path):
    snapshot, samples = _bundle(tmp_path)
    frame = pd.read_parquet(samples)
    frame.loc[0, "passing_yards"] = 999
    frame.to_parquet(samples, index=False)
    with pytest.raises(ValueError, match="integrity hash mismatch"):
        load_projection_bundle(snapshot, samples)


def test_fablesfable_contract_contains_no_fantasy_package():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert not (root / "nflvalue" / "fantasy").exists()
    assert not (root / ".github" / "workflows" / "fantasy-weekly.yml").exists()


def test_json_schema_names_every_component():
    from pathlib import Path

    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas/player_projection_snapshot.schema.json").read_text()
    )
    assert set(schema["$defs"]["components"]["required"]) == set(COMPONENT_NAMES)
