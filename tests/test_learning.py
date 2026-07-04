"""Learning loop guardrails: walk-forward only, bounded, selection-bias-safe;
context tags stay display-only until evidence + human promotion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import context_study, db as dbmod, prop_learning as pl  # noqa: E402
from nflvalue.candidates import apply_reallocation  # noqa: E402
from nflvalue.composite import score_candidate  # noqa: E402
from nflvalue.sources import espn_news  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# Bias / reliability adjustments
# --------------------------------------------------------------------------- #
def test_bias_is_bounded_and_path_independent():
    a = pl.LearningState({"lr": 0.35, "bias_clip": 0.08})
    b = pl.LearningState({"lr": 0.35, "bias_clip": 0.08})
    # model over-projects by 20% for weeks on end
    for _ in range(6):
        a.observe("receiving_yards", 100, 5000.0, 4000.0, [1, 0, 1])
    for _ in range(6):
        b.observe("receiving_yards", 100, 5000.0, 4000.0, [1, 0, 1])
    assert a.bias_mult["receiving_yards"] == b.bias_mult["receiving_yards"]
    assert a.bias_mult["receiving_yards"] >= 0.92            # clipped, never runaway
    assert a.bias_mult["receiving_yards"] < 1.0              # pulls the right direction


def test_bias_needs_a_real_sample():
    s = pl.LearningState()
    s.observe("receptions", 20, 100.0, 60.0, [])             # n=20 < 100
    assert "receptions" not in s.bias_mult                   # no evidence, no adjustment


def test_reliability_shrinks_hard_and_clips():
    s = pl.LearningState({"reliability_k": 50, "reliability_clip": 0.15})
    s.observe("rush_attempts", 200, 2000.0, 2000.0, [0] * 10)   # 0/10: tiny sample
    r10 = s.adjustments()["rush_attempts"]["reliability"]
    assert 0.9 < r10 < 1.0                                   # nudged, not nuked
    s2 = pl.LearningState({"reliability_k": 50, "reliability_clip": 0.15})
    s2.observe("rush_attempts", 200, 2000.0, 2000.0, [0] * 400)
    assert s2.adjustments()["rush_attempts"]["reliability"] == pytest.approx(0.85)  # floor


def test_apply_to_candidates_recomputes_probs_and_is_noop_when_disabled():
    cands = pd.DataFrame([{
        "player_id": "P1", "market": "receiving_yards", "mean": 70.0, "sd": 25.0,
        "line": 65.5, "dist": "gamma", "p_over": 0.55, "p_under": 0.45}])
    adj = {"receiving_yards": {"bias_mult": 0.92, "reliability": 0.9}}
    out = pl.apply_to_candidates(cands, adj, enabled=True)
    assert out.iloc[0]["mean"] == pytest.approx(70 * 0.92, abs=0.01)
    assert out.iloc[0]["p_over"] < 0.55                      # prob follows the mean
    assert out.iloc[0]["reliability_mult"] == 0.9
    noop = pl.apply_to_candidates(cands, adj, enabled=False)
    assert noop.iloc[0]["mean"] == 70.0
    assert noop.iloc[0]["reliability_mult"] == 1.0


def test_reliability_moves_composite_only_when_stamped():
    base = {"player_id": "P1", "name": "A", "pos": "WR", "team": "T", "market": "receiving_yards",
            "mean": 78.0, "sd": 25.0, "line": 68.5, "p_over": 0.62, "p_under": 0.38,
            "components": {"opp_factor": 1.1, "game_script": 1.0}, "prices": None,
            "low_confidence": False}
    plain = score_candidate(base)
    tagged = score_candidate({**base, "reliability_mult": 0.85})
    assert tagged["composite"] == pytest.approx(plain["composite"] * 0.85, abs=0.05)
    assert plain["components"]["reliability_mult"] is None   # absent -> untouched


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def _lean(hit, mean=60.0, actual=30.0, market="receiving_yards", side="under",
          volume=8.0):
    return {"hit": hit, "mean": mean, "actual": actual, "market": market,
            "side": side, "line": 55.5, "proj_components": {"volume": volume}}


def test_attribution_paths():
    hit = pl.attribute(_lean(True), {"targets": 8, "carries": 0}, 3.0, 7.0)
    assert hit["primary_reason"] == "as_projected"

    absent = pl.attribute(_lean(False), None, 3.0, 7.0)
    assert absent["primary_reason"] == "availability_surprise"

    collapsed = pl.attribute(_lean(False, volume=8.0), {"targets": 1.0}, 3.0, 7.0)
    assert collapsed["primary_reason"] == "availability_surprise"

    vol = pl.attribute(_lean(False, mean=60, actual=95, volume=8.0),
                       {"targets": 13.0}, 3.0, 7.0)
    assert vol["primary_reason"] == "volume_miss"

    eff = pl.attribute(_lean(False, mean=60, actual=95, volume=8.0),
                       {"targets": 8.0}, 3.0, 7.0)
    assert eff["primary_reason"] == "efficiency_miss"

    flip = pl.attribute(_lean(False, mean=60, actual=95, volume=8.0),
                        {"targets": 13.0}, 7.0, -14.0)
    assert flip["primary_reason"] == "script_flip"


# --------------------------------------------------------------------------- #
# DB round-trip: aggregates + rebuild = same adjustments
# --------------------------------------------------------------------------- #
def test_rebuild_state_matches_live_state(conn):
    live = pl.LearningState()
    for wk in range(1, 7):
        live.observe("receptions", 120, 600.0, 540.0, [1, 1, 0, 1])
        dbmod.upsert(conn, "candidate_aggregates", [{
            "season": 2025, "week": wk, "market": "receptions", "n": 120,
            "sum_pred": 600.0, "sum_actual": 540.0, "created_at": "t"}],
            ["season", "week", "market"])
        dbmod.upsert(conn, "lean_outcomes", [{
            "season": 2025, "week": wk, "clock": "wed", "game_id": f"G{wk}",
            "player_id": f"P{i}", "name": "x", "market": "receptions", "side": "over",
            "line": 4.5, "mean": 5.0, "composite": 50.0, "actual": 5.0, "hit": h,
            "primary_reason": "as_projected", "volume_log_err": None,
            "efficiency_log_err": None, "detail": "", "graded_at": "t"}
            for i, h in enumerate([1, 1, 0, 1])],
            ["season", "week", "clock", "game_id", "player_id", "market"])
    rebuilt = pl.rebuild_state(conn)
    assert rebuilt.adjustments() == live.adjustments()


# --------------------------------------------------------------------------- #
# Context ledger: measured always, scored never (until promoted)
# --------------------------------------------------------------------------- #
def test_tags_extracted_and_recorded(conn):
    games = [{"game_id": "G1", "leans": [
        {"player_id": "P1", "name": "A", "market": "receiving_yards"}]}]
    contexts = {"G1": {"entries": [
        {"player_id": "P1", "name": "A",
         "items": ["note (espn_news): playing on his birthday against his former team"]}]}}
    n = context_study.record_tags(conn, 2025, 3, games, contexts)
    assert n == 2                                            # birthday + revenge
    ledger = dbmod.query_df(conn, "SELECT tag FROM context_ledger ORDER BY tag")
    assert ledger["tag"].tolist() == ["birthday", "revenge"]


def test_study_requires_n_and_significance(conn):
    # 40 birthday leans at a stellar hit rate -- still below MIN_N
    for i in range(40):
        dbmod.upsert(conn, "context_ledger", [{
            "season": 2025, "week": i % 18 + 1, "player_id": f"P{i}",
            "market": "receiving_yards", "tag": "birthday", "source": "t",
            "note": "", "created_at": "t"}],
            ["season", "week", "player_id", "market", "tag"])
        dbmod.upsert(conn, "lean_outcomes", [{
            "season": 2025, "week": i % 18 + 1, "clock": "wed", "game_id": f"G{i}",
            "player_id": f"P{i}", "name": "x", "market": "receiving_yards",
            "side": "over", "line": 50.5, "mean": 55.0, "composite": 50.0,
            "actual": 60.0, "hit": 1, "primary_reason": "as_projected",
            "volume_log_err": None, "efficiency_log_err": None,
            "detail": "", "graded_at": "t"}],
            ["season", "week", "clock", "game_id", "player_id", "market"])
    s = context_study.study(conn)
    assert s["tags"]["birthday"]["verdict"] == "insufficient_n"
    assert "proposed_mult" not in s["tags"]["birthday"]


def test_context_multipliers_default_noop(conn):
    assert context_study.enabled_multipliers({"context_learning": {"enabled_tags": []}},
                                             conn) == {}
    cands = pd.DataFrame([{"player_id": "P1", "market": "receiving_yards"}])
    out = context_study.apply_context_multipliers(cands, conn, 2025, 3, {})
    assert "context_mult" not in out.columns                 # untouched frame


def test_promotion_requires_config_and_evidence(conn):
    # a genuinely huge sample with a strong effect -> PROMOTABLE
    for i in range(150):
        dbmod.upsert(conn, "context_ledger", [{
            "season": 2025, "week": i % 18 + 1, "player_id": f"Q{i}",
            "market": "receptions", "tag": "revenge", "source": "t",
            "note": "", "created_at": "t"}],
            ["season", "week", "player_id", "market", "tag"])
        dbmod.upsert(conn, "lean_outcomes", [{
            "season": 2025, "week": i % 18 + 1, "clock": "wed", "game_id": f"H{i}",
            "player_id": f"Q{i}", "name": "x", "market": "receptions", "side": "over",
            "line": 4.5, "mean": 5.0, "composite": 50.0, "actual": 6.0,
            "hit": 1 if i % 10 else 0,                        # 90% hit rate
            "primary_reason": "as_projected", "volume_log_err": None,
            "efficiency_log_err": None, "detail": "", "graded_at": "t"}],
            ["season", "week", "clock", "game_id", "player_id", "market"])
    # give the baseline some ordinary leans so 90% is actually anomalous
    for i in range(300):
        dbmod.upsert(conn, "lean_outcomes", [{
            "season": 2025, "week": i % 18 + 1, "clock": "wed", "game_id": f"Z{i}",
            "player_id": f"Z{i}", "name": "x", "market": "receiving_yards",
            "side": "over", "line": 50.5, "mean": 55.0, "composite": 50.0,
            "actual": 60.0, "hit": i % 2, "primary_reason": "as_projected",
            "volume_log_err": None, "efficiency_log_err": None,
            "detail": "", "graded_at": "t"}],
            ["season", "week", "clock", "game_id", "player_id", "market"])
    s = context_study.study(conn)
    assert s["tags"]["revenge"]["verdict"] == "PROMOTABLE"
    assert 1.0 < s["tags"]["revenge"]["proposed_mult"] <= 1.10   # bounded
    # evidence alone changes NOTHING -- config gate still closed
    assert context_study.enabled_multipliers({"context_learning": {"enabled_tags": []}},
                                             conn) == {}
    # human promotes it -> bounded multiplier flows
    mults = context_study.enabled_multipliers(
        {"context_learning": {"enabled_tags": ["revenge"]}}, conn)
    assert set(mults) == {"revenge"}


# --------------------------------------------------------------------------- #
# News + reallocation
# --------------------------------------------------------------------------- #
def test_news_fixture_parses_and_matches():
    raw = json.loads((FIXTURES / "espn_news_recorded.json").read_text())
    items = espn_news.parse_news(raw["payload"])
    assert items and all(i["source"] == "espn_news" for i in items)
    players = pd.DataFrame([
        {"player_id": "00-X1", "player_name": "B.Taylor"},   # matches fixture headline
        {"player_id": "00-X2", "player_name": "Z.Nobody"},
    ])
    mapped = espn_news.news_by_player(items, players)
    assert "00-X2" not in mapped                             # no invented matches


def test_reallocation_prices_vacated_usage_bounded():
    cands = pd.DataFrame([
        {"player_id": "WR2", "market": "receiving_yards", "mean": 50.0, "sd": 20.0,
         "line": 48.5, "dist": "gamma", "p_over": 0.5, "p_under": 0.5},
        {"player_id": "WR2", "market": "rush_attempts", "mean": 1.0, "sd": 1.0,
         "line": 0.5, "dist": "negbinom", "p_over": 0.6, "p_under": 0.4},
    ])
    realloc = [{"out_player_id": "WR1", "role": "WR", "basis": "with_without",
                "low_confidence": False,
                "boosts": {"WR2": {"share_with": 0.20, "share_without": 0.30,
                                   "share_delta": 0.10}}}]
    out = apply_reallocation(cands, realloc)
    rec = out[out["market"] == "receiving_yards"].iloc[0]
    # +50% share capped at x1.35 volume, then the MEASURED second-order
    # efficiency dampening (defense adjusts): eff = 1 - 0.29*(1.35-1) = .8985
    assert rec["realloc_mult"] == pytest.approx(1.35, abs=0.01)
    assert rec["realloc_eff_mult"] == pytest.approx(0.8985, abs=0.001)
    assert rec["mean"] == pytest.approx(50 * 1.35 * 0.8985, abs=0.05)
    assert rec["p_over"] > 0.5                                # prob followed
    ratt = out[out["market"] == "rush_attempts"].iloc[0]
    assert ratt["mean"] == 1.0                                # wrong family: untouched


def test_reallocation_guess_is_dampened():
    cands = pd.DataFrame([{"player_id": "WR2", "market": "receptions", "mean": 4.0,
                           "sd": 2.0, "line": 3.5, "dist": "negbinom",
                           "p_over": 0.55, "p_under": 0.45}])
    realloc = [{"out_player_id": "WR1", "role": "WR", "basis": "proportional_guess",
                "low_confidence": True,
                "boosts": {"WR2": {"share_with": 0.20, "share_without": None,
                                   "share_delta": 0.10}}}]
    out = apply_reallocation(cands, realloc)
    # half the 50% move (guess-dampened) x measured efficiency loss (.9275)
    assert out.iloc[0]["mean"] == pytest.approx(4.0 * 1.25 * 0.9275, abs=0.02)
