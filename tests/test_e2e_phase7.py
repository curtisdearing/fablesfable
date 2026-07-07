"""Phase 7.8 — end-to-end integration: one test that walks a slate through the
WHOLE Phase-7 chain and asserts every hand-off connects, so a regression
ANYWHERE in the chain trips here:

    calibrated ML ranker (7.1/7.2)
      -> correlation-aware selection (7.5/7.6, shortlist.rank_game)
      -> advisory staking readout (7.7, staking.recommend_stakes)
      -> real-line capture on a fixture DB (7.3/7.4, lines/leans/outcomes)
      -> CLV (clv.log_close_for_week)
      -> kill-check (killcheck.report).

Uses each component's REAL interface. The projection/enumerate internals and
each stage's own numeric behavior are covered by their dedicated test modules;
this test is the cross-component wiring.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue import ml_ranker as mlr                        # noqa: E402
from nflvalue import staking as stk                          # noqa: E402
from nflvalue import clv as clvmod                           # noqa: E402
from nflvalue import db as dbmod                             # noqa: E402
from nflvalue import killcheck                               # noqa: E402
from nflvalue.correlation import CorrelationStructure        # noqa: E402
from nflvalue.shortlist import rank_game                     # noqa: E402
import test_ml                                               # noqa: E402

PRICE = 1.0 + 100.0 / 110.0        # -110 decimal
FAIR = 0.5238

# real measured structure (7.5): a QB and his WR move together; a WR's own two
# markets are near-duplicates. Used to drive correlation-aware selection + sizing.
CORR = CorrelationStructure({"pair_types": {
    "sameteam|QB.pass~WR.rec": {"rho_shrunk": 0.297, "verdict": "real"},
    "sameplayer|WR.rec~WR.rec": {"rho_shrunk": 0.764, "verdict": "real"},
}, "walk_forward": {}})


def _cand(pid, name, pos, team, market, p_over, mean, line, game="2023_10_AAA_BBB"):
    """A candidate dict shaped like enumerate_candidates output + a stamped
    ml_score (100 x side prob), forcing the ML ranking path in rank_game."""
    p_side = max(p_over, 1 - p_over) if market != "anytime_td" else p_over
    return {
        "game_id": game, "matchup": "AAA @ BBB", "player_id": pid, "name": name,
        "pos": pos, "team": team, "market": market, "mean": mean, "sd": 20.0,
        "line": line, "p_over": p_over, "p_under": round(1 - p_over, 4),
        "prices": {"over": PRICE, "under": PRICE},
        "components": {"opp_factor": 1.05, "game_script": 1.0},
        "low_confidence": False, "ml_score": round(100 * p_side, 2),
    }


def test_calibrated_ranker_interface_still_holds():
    """Stage 1 wiring: a calibrated MLRanker fits and returns calibrated,
    bounded, walk-forward P(over) — the 7.1/7.2 contract the chain starts from."""
    f = test_ml._frame(n=1200, seasons=(2021, 2022, 2023), weeks=range(1, 10))
    tr, te = f[f["season"] < 2023], f[f["season"] == 2023]
    m = mlr.MLRanker("gbdt", max_iter=60, calibrate="platt_permkt").fit(tr, tr["y_over"])
    p = m.predict_p_over(te)
    assert m.calibrator is not None
    assert p.min() >= 0.0 and p.max() <= 1.0 and len(p) == len(te)


def test_phase7_end_to_end_chain():
    # ---- slate: one game, a near-duplicate pair + a correlated QB/WR --------- #
    slate = [
        _cand("qb1", "QB One", "QB", "AAA", "passing_yards", 0.60, 255, 244.5),
        _cand("wr1", "WR One", "WR", "AAA", "receiving_yards", 0.60, 78, 70.5),
        _cand("wr1", "WR One", "WR", "AAA", "receptions", 0.585, 6.2, 5.5),   # same player -> ~dup
        _cand("rb1", "RB One", "RB", "AAA", "rushing_yards", 0.58, 70, 62.5),
    ]

    # ---- Stage 2: correlation-aware selection vs the pre-7.6 baseline -------- #
    plain = rank_game([dict(c) for c in slate], top_n=3, corr=None)
    aware = rank_game([dict(c) for c in slate], top_n=3, corr=CORR)
    assert len(aware["leans"]) == 3
    # correlation-awareness must CHANGE the selection: the baseline stacks the
    # WR's two near-duplicate markets (ρ≈0.76); the aware run drops one for an
    # independent lean. Proof 7.6 fired end to end.
    sel = lambda res: {(l["player_id"], l["market"]) for l in res["leans"]}
    assert sel(plain) != sel(aware), "correlation-aware selection did not change the shortlist"
    # a discount was actually computed during the greedy walk (on the leg it de-prioritized)
    assert any("corr_discount" in l for l in aware["leans"])
    # every selected lean carries an explainable composite score
    assert all("composite" in l and "side" in l for l in aware["leans"])

    # ---- Stage 3: advisory staking on the selected leans -------------------- #
    staking_leans = [{
        "game_id": l["game_id"], "player_id": l["player_id"], "market": l["market"],
        "pos": l["pos"], "team": l["team"], "side": l["side"],
        "p": l["components"]["model_prob"],
        "market_prob": l["components"]["market_prob"],
        "price": PRICE,
    } for l in aware["leans"]]
    rec = stk.recommend_stakes(staking_leans, bankroll=100.0, corr=CORR)
    assert "ADVISORY ONLY" in rec["disclaimer"]
    assert rec["readout"]["n_staked"] >= 1
    # no single advisory stake breaches the per-bet cap
    assert all(r["stake_frac"] <= stk.StakeConfig().cap_bet + 1e-9
               for r in rec["recommendations"])
    # total slate exposure respects the portfolio cap
    assert rec["readout"]["total_exposure_frac"] <= stk.StakeConfig().max_slate + 1e-9

    # ---- Stage 4: fixture real-line capture -> CLV -> kill-check ------------- #
    dbpath = os.path.join(tempfile.mkdtemp(), "e2e.db")
    conn = dbmod.connect(dbpath)
    gid, pid, market = "2023_10_AAA_BBB", "wr1", "receiving_yards"
    entry_ts, close_ts, kickoff = "2023-11-08T17:00:00Z", "2023-11-12T16:30:00Z", "2023-11-12T18:00:00Z"
    lines = []
    for ts, over_px in ((entry_ts, 1.91), (close_ts, 1.74)):     # over shortens by close -> +CLV
        for book, px in (("dk", over_px), ("fd", over_px + 0.02)):
            lines.append(dict(ts=ts, game_id=gid, book=book, market=market,
                              player_id=pid, player_name="WR One", side="over", point=70.5, price=px))
            lines.append(dict(ts=ts, game_id=gid, book=book, market=market,
                              player_id=pid, player_name="WR One", side="under", point=70.5, price=1.95))
    dbmod.upsert(conn, "lines", lines, ["ts", "game_id", "book", "market", "player_name", "side", "point"])
    dbmod.upsert(conn, "leans", [dict(
        season=2023, week=10, clock="wed", game_id=gid, player_id=pid, name="WR One",
        market=market, side="over", line=70.5, line_source="odds_api", price=1.91,
        book="dk", as_of=entry_ts, status="active")],
        ["season", "week", "clock", "game_id", "player_id", "market"])
    dbmod.upsert(conn, "lean_outcomes", [dict(
        season=2023, week=10, clock="wed", game_id=gid, player_id=pid, name="WR One",
        market=market, side="over", line=70.5, actual=88.0, hit=1)],
        ["season", "week", "clock", "game_id", "player_id", "market"])

    resolved = clvmod.log_close_for_week(conn, 2023, 10, {gid: kickoff})
    assert len(resolved) == 1
    r = resolved.iloc[0]
    assert r["clv_prob"] == pytest.approx(r["close_prob"] - r["entry_prob"], abs=2e-5)
    assert r["clv_prob"] > 0                                   # market moved to our side

    verdict = killcheck.report(conn, min_sample=150)
    assert verdict["verdict"] == "INSUFFICIENT_SAMPLE"        # one resolved lean != a referendum
    assert verdict["n"] == 1 and verdict["leans_logged"] >= 1
