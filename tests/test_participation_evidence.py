"""Observed snap counts: pinned contract, identity, no fabricated zeros, routes blocked."""

import pandas as pd
import pytest

from nflvalue import factor_evidence as fe
from nflvalue.sources import participation_evidence as pe


def snaps(rows):
    base = dict(season=2026, game_type="REG", position="WR", opponent="X", defense_snaps=0,
                defense_pct=0.0, st_snaps=0, st_pct=0.0, pfr_game_id="g")
    return pd.DataFrame([{**base, **r} for r in rows])


ROWS = [
    dict(game_id="2026_01_GB_DET", week=1, player="A One", pfr_player_id="OneA00", team="GB",
         offense_snaps=50, offense_pct=0.83),
    dict(game_id="2026_01_GB_DET", week=1, player="B Two", pfr_player_id="TwoB00", team="GB",
         offense_snaps=20, offense_pct=0.33),
    dict(game_id="2026_02_BUF_GB", week=2, player="B Two", pfr_player_id="TwoB00", team="GB",
         offense_snaps=40, offense_pct=0.66),
]
PLAYERS = pd.DataFrame([{"pfr_id": "OneA00", "gsis_id": "00-A"}, {"pfr_id": "TwoB00", "gsis_id": "00-B"},
                        {"pfr_id": "Dup00", "gsis_id": "00-X"}, {"pfr_id": "Dup00", "gsis_id": "00-Y"}])


def load(rows=ROWS, **kw):
    kw.setdefault("players", PLAYERS)
    return pe.load_snap_counts(snaps(rows), season=2026, target_week=3,
                               source={"url": pe.SNAP_URL.format(season=2026),
                                       "fetched_at": "2026-09-23T03:00:00Z",
                                       "last_modified": "2026-09-22T11:01:50Z"}, **kw)


def test_future_or_same_week_rows_are_rejected():
    with pytest.raises(pe.ParticipationError, match="target week"):
        load(ROWS + [dict(ROWS[0], week=3, game_id="2026_03_ATL_GB")])


def test_wrong_season_and_missing_pinned_column_rejected():
    with pytest.raises(pe.ParticipationError, match="seasons"):
        pe.load_snap_counts(snaps(ROWS).assign(season=2025), season=2026, target_week=3)
    with pytest.raises(pe.ParticipationError, match="pinned"):
        pe.load_snap_counts(snaps(ROWS).drop(columns=["offense_pct"]), season=2026, target_week=3)


def test_identity_links_only_unique_pfr_ids():
    rows = ROWS + [dict(ROWS[0], player="Dup", pfr_player_id="Dup00")]
    out = load(rows)
    ids = dict(zip(out["rows"]["pfr_player_id"], out["rows"]["player_id"]))
    assert ids["OneA00"] == "00-A" and pd.isna(ids["Dup00"])
    assert out["receipt"]["identity"] == {"linked": 3, "unlinked": 1, "ambiguous_pfr_ids": 1}


def test_missing_week_is_no_row_not_zero_and_record_is_context():
    out = load()
    rec = pe.snap_records(out, player_id="00-A", team="GB", game_id="2026_03_ATL_GB",
                          as_of="2026-09-23T04:00:00Z")[0]
    assert rec["value"] == ["W1: 50 snaps (83%)", "W2: no row"]
    assert "W2: 0" not in rec["observation"]
    norm = fe.normalize_record(rec)
    assert norm["status"] == "context_only" and norm["measurement_kind"] == "observed"


def test_unlinked_player_is_unavailable_not_zero():
    rec = pe.snap_records(load(), player_id="00-Z", team="GB", game_id="g", as_of="2026-09-23T04:00:00Z")[0]
    assert rec["measurement_kind"] == "unavailable" and rec["populated"] is False


def test_game_coverage_stated_against_schedule():
    out = load(schedule_games={1: ["2026_01_GB_DET", "2026_01_A_B"], 2: ["2026_02_BUF_GB"]})
    pw = out["receipt"]["per_week"]
    assert pw[1]["missing_games"] == ["2026_01_A_B"] and pw[2]["missing_games"] == []
    assert load()["receipt"]["per_week"][1]["missing_games"] == "not checked"


def test_snaps_are_not_routes_and_workload_stays_unavailable():
    part = pd.DataFrame({"nflverse_game_id": ["g"], "play_id": [1], "route": ["GO"]})
    ra = pe.route_availability(2026, participation=part,
                               published_assets=["pbp_participation_2025.parquet"])
    assert ra["available"] is False
    assert "pbp_participation_2026.parquet" in ra["blocker"] and "targeted receiver" in ra["blocker"]
    assert "not a route proxy" in ra["not_used"]
    assert pe.expected_workload("00-A")["value"] is None
    rec = pe.snap_records(load(), player_id="00-B", team="GB", game_id="g", as_of="2026-09-23T04:00:00Z")[0]
    assert rec["unit"] == "offense_snaps" and "not routes" in rec["reason_not_applied"]


def test_input_frame_not_mutated_and_hash_stable():
    f = snaps(ROWS)
    before = f.copy()
    a = pe.load_snap_counts(f, season=2026, target_week=3)
    pd.testing.assert_frame_equal(f, before)
    assert a["receipt"]["sha256"] == pe.load_snap_counts(before, season=2026, target_week=3)["receipt"]["sha256"]
