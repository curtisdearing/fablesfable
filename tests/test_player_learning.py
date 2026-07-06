"""Player-level sequential learning guardrails.

Covers ``nflvalue/player_learning.py``: the (player, market) residual ledger
(WHERE/HOW), the shrunk walk-forward player-specific bias, its effective-at
persistence, the apply no-op contract, and the read-side report. Every DB is a
throwaway tmp path (the default data/nfl_props.db can raise a mount 'disk I/O
error'), and the leakage test mirrors the truncation-invariance pattern used by
tests/test_correlation.py / test_leakage.py."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod, player_learning as plr  # noqa: E402


@pytest.fixture()
def conn():
    c = dbmod.connect(os.path.join(tempfile.mkdtemp(), "pl.db"))
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# Synthetic candidate / player-week builders
# --------------------------------------------------------------------------- #
# All tests use market 'receiving_yards' (actual col rec_yards, opportunity col
# targets), so an actual can always be graded and the volume/efficiency split
# is available.
_PW_COLS = ["season", "week", "player_id", "targets", "carries", "pass_attempts",
            "rec_yards", "receptions", "rush_yards", "pass_yards", "rush_tds", "rec_tds"]


def cand(player_id, mean, proj_vol, *, sd=25.0, line=None, home=1,
         team="AAA", defteam="BBB", market="receiving_yards"):
    """One candidate row. ``proj_vol`` is the projected volume (targets)."""
    return {
        "player_id": player_id, "name": player_id, "pos": "WR", "market": market,
        "mean": float(mean), "sd": float(sd),
        "line": float(mean - 5.0) if line is None else float(line),
        "dist": "gamma", "team": team, "defteam": defteam, "home": int(home),
        "components": {"volume": float(proj_vol), "efficiency": 1.0},
    }


def pw(season, week, player_id, *, targets=0.0, rec_yards=0.0):
    row = {c: 0.0 for c in _PW_COLS}
    row.update({"season": season, "week": week, "player_id": player_id,
                "targets": float(targets), "rec_yards": float(rec_yards)})
    return row


def cands_df(rows):
    return pd.DataFrame(rows)


def pw_df(rows):
    return pd.DataFrame(rows, columns=_PW_COLS)


# --------------------------------------------------------------------------- #
# 1. Ledger: count, WHERE (home/opp), HOW (volume / efficiency / on_projection)
# --------------------------------------------------------------------------- #
def test_record_writes_count_and_captures_where_and_how(conn):
    cands = cands_df([
        # volume-driven miss: targets far above proj volume (10 vs 5) -> reason 'volume'
        cand("VOL", mean=60.0, proj_vol=5.0, home=1, defteam="NYG"),
        # efficiency-driven miss: targets on-proj (5==5) but yards way over -> 'efficiency'
        cand("EFF", mean=60.0, proj_vol=5.0, home=0, defteam="DAL"),
        # on-projection: actual lands within 15% log band -> 'on_projection'
        cand("ONP", mean=100.0, proj_vol=7.0, home=1, defteam="PHI"),
    ])
    pws = pw_df([
        pw(2024, 1, "VOL", targets=10.0, rec_yards=120.0),
        pw(2024, 1, "EFF", targets=5.0, rec_yards=120.0),
        pw(2024, 1, "ONP", targets=7.0, rec_yards=103.0),
    ])
    n = plr.record_player_residuals(conn, 2024, 1, cands, pws)
    assert n == 3

    df = dbmod.query_df(conn, "SELECT * FROM player_week_residuals").set_index("player_id")
    assert len(df) == 3

    # WHERE: home flag and opponent are captured verbatim
    assert int(df.loc["VOL", "home"]) == 1 and df.loc["VOL", "opp"] == "NYG"
    assert int(df.loc["EFF", "home"]) == 0 and df.loc["EFF", "opp"] == "DAL"

    # HOW: reason attribution
    assert df.loc["VOL", "primary_reason"] == "volume"
    assert df.loc["EFF", "primary_reason"] == "efficiency"
    assert df.loc["ONP", "primary_reason"] == "on_projection"


def test_record_skips_ungradeable_and_dedups(conn):
    cands = cands_df([
        cand("A", mean=60.0, proj_vol=6.0),
        cand("B", mean=60.0, proj_vol=6.0),   # no matching pw row -> ungradeable, skipped
    ])
    pws = pw_df([pw(2024, 1, "A", targets=6.0, rec_yards=70.0)])
    assert plr.record_player_residuals(conn, 2024, 1, cands, pws) == 1
    # re-recording the same week is an upsert on the PK, not a duplicate
    plr.record_player_residuals(conn, 2024, 1, cands, pws)
    assert len(dbmod.query_df(conn, "SELECT * FROM player_week_residuals")) == 1


# --------------------------------------------------------------------------- #
# 2. Walk-forward / leakage: adjustments use ONLY rows strictly before `before`
# --------------------------------------------------------------------------- #
def _seed_weeks(conn, weeks, players):
    """Seed one residual row per (week, player). ``players`` maps id -> (mean, targets, yards)."""
    for w in weeks:
        cl, pl_ = [], []
        for pid, (mean, tg, yds) in players.items():
            cl.append(cand(pid, mean=mean, proj_vol=6.0))
            pl_.append(pw(2024, w, pid, targets=tg, rec_yards=yds))
        plr.record_player_residuals(conn, 2024, w, cands_df(cl), pw_df(pl_))


def test_compute_is_truncation_invariant(conn):
    """Byte-identical whether or not rows AT/AFTER `before` also exist — no leakage."""
    players = {"UND": (60.0, 6.0, 78.0), "TRK": (60.0, 6.0, 60.0)}
    # weeks 1..5 are the "strictly before (2024,6)" evidence
    _seed_weeks(conn, range(1, 6), players)
    baseline = plr.compute_player_adjustments(conn, before=(2024, 6), params={"min_market_n": 1})

    # now add rows AT and AFTER the boundary (week 6, 7) with wildly different data
    poison = {"UND": (60.0, 6.0, 30.0), "TRK": (60.0, 6.0, 200.0)}
    _seed_weeks(conn, [6, 7], poison)
    after = plr.compute_player_adjustments(conn, before=(2024, 6), params={"min_market_n": 1})

    assert after == baseline
    # sanity: the poison rows ARE visible when the boundary moves forward
    later = plr.compute_player_adjustments(conn, before=(2024, 8), params={"min_market_n": 1})
    assert later != baseline


# --------------------------------------------------------------------------- #
# 3. min_games floor
# --------------------------------------------------------------------------- #
def test_min_games_floor_omits_thin_players(conn):
    # THIN graded 3 weeks (< default min_games=4) -> no entry
    # FULL graded 5 weeks -> entry present
    _seed_weeks(conn, range(1, 4), {"THIN": (60.0, 6.0, 78.0)})
    _seed_weeks(conn, range(1, 6), {"FULL": (60.0, 6.0, 78.0)})
    adj = plr.compute_player_adjustments(conn, before=(2024, 7), params={"min_market_n": 1})
    assert ("THIN", "receiving_yards") not in adj
    assert ("FULL", "receiving_yards") in adj
    assert adj[("FULL", "receiving_yards")]["n_games"] == 5


# --------------------------------------------------------------------------- #
# 4. Shrinkage: systematic under-projection lifts but is heavily muted; a
#    market-tracking player stays ~1.0; clip is never exceeded.
# --------------------------------------------------------------------------- #
def test_shrinkage_isolates_player_specific_part_and_clips(conn):
    # 8 filler players track the market at 1.0x so the pooled market_ratio ~ 1.0,
    # letting the isolation logic show through cleanly.
    players = {"UND": (60.0, 6.0, 78.0),        # actual 1.30x proj, every week
               "TRK": (60.0, 6.0, 60.0)}        # actual 1.00x proj, every week
    for i in range(8):
        players[f"F{i}"] = (60.0, 6.0, 60.0)
    _seed_weeks(conn, range(1, 7), players)     # 6 graded weeks each

    adj = plr.compute_player_adjustments(conn, before=(2024, 7), params={"min_market_n": 1})
    clip = plr.DEFAULTS["bias_clip"]            # 0.12

    und = adj[("UND", "receiving_yards")]["bias_mult"]
    trk = adj[("TRK", "receiving_yards")]["bias_mult"]

    # under-projected: pulls up (>1) but is FAR from the raw 1.30 -- within a few % of 1.0
    assert und > 1.0
    assert und < 1.10
    # market-tracking: the player-specific deviation is ~0, so ~1.0 (near no-op)
    assert trk == pytest.approx(1.0, abs=0.02)
    # clip bound never exceeded for ANY player
    for v in adj.values():
        assert (1.0 - clip) <= v["bias_mult"] <= (1.0 + clip)


def test_shrinkage_clip_binds_on_extreme_bias(conn):
    # a monstrously under-projected player (actual ~3x) must still be clip-bounded.
    players = {"EXT": (60.0, 6.0, 180.0)}
    for i in range(6):
        players[f"F{i}"] = (60.0, 6.0, 60.0)
    _seed_weeks(conn, range(1, 9), players)     # 8 weeks
    adj = plr.compute_player_adjustments(conn, before=(2024, 9), params={"min_market_n": 1})
    clip = plr.DEFAULTS["bias_clip"]
    assert adj[("EXT", "receiving_yards")]["bias_mult"] <= (1.0 + clip) + 1e-9


# --------------------------------------------------------------------------- #
# 4b. effective-at load: persist at w+1, load picks it up at/after, not before
# --------------------------------------------------------------------------- #
def test_persist_and_effective_at_load(conn):
    _seed_weeks(conn, range(1, 6), {"UND": (60.0, 6.0, 78.0)})
    for i in range(8):
        _seed_weeks(conn, range(1, 6), {f"F{i}": (60.0, 6.0, 60.0)})

    # computed from data strictly before week 6, persisted effective-at week 6 (== w+1)
    adj = plr.compute_player_adjustments(conn, before=(2024, 6), params={"min_market_n": 1})
    assert plr.persist_player_adjustments(conn, 2024, 6, adj) == len(adj)

    # load at week 7 (>= 6) picks it up
    loaded = plr.load_player_adjustments(conn, 2024, 7)
    assert ("UND", "receiving_yards") in loaded
    assert loaded[("UND", "receiving_yards")] == adj[("UND", "receiving_yards")]["bias_mult"]

    # load at week 5 (< effective-at 6) does NOT see it
    assert plr.load_player_adjustments(conn, 2024, 5) == {}
    # load exactly AT week 6 sees it (<= semantics)
    assert ("UND", "receiving_yards") in plr.load_player_adjustments(conn, 2024, 6)


# --------------------------------------------------------------------------- #
# 5. apply_player_bias: no-op contract + directional lift + finite probs
# --------------------------------------------------------------------------- #
def _apply_fixture():
    return cands_df([
        cand("UND", mean=60.0, proj_vol=6.0, line=58.5),
        cand("TRK", mean=60.0, proj_vol=6.0, line=58.5),
    ])


def test_apply_disabled_is_byte_identical_noop():
    base = _apply_fixture()
    adj = {("UND", "receiving_yards"): 1.056}
    out = plr.apply_player_bias(base.copy(), adj, enabled=False)
    # mean and p_over untouched vs the input
    pd.testing.assert_series_equal(out["mean"], base["mean"], check_names=False)
    if "p_over" in base:
        pd.testing.assert_series_equal(out["p_over"], base["p_over"], check_names=False)
    assert (out["player_bias_mult"] == 1.0).all()


def test_apply_empty_adj_is_noop():
    base = _apply_fixture()
    out = plr.apply_player_bias(base.copy(), {}, enabled=True)
    pd.testing.assert_series_equal(out["mean"], base["mean"], check_names=False)
    assert (out["player_bias_mult"] == 1.0).all()


def test_apply_enabled_lifts_under_projected_and_recomputes_probs():
    base = _apply_fixture()
    adj = {("UND", "receiving_yards"): 1.08, ("TRK", "receiving_yards"): 1.0}
    out = plr.apply_player_bias(base.copy(), adj, enabled=True).set_index("player_id")
    b = base.set_index("player_id")

    # under-projected player's mean is lifted; market-tracker unchanged
    assert out.loc["UND", "mean"] == pytest.approx(b.loc["UND", "mean"] * 1.08, abs=0.01)
    assert out.loc["TRK", "mean"] == pytest.approx(b.loc["TRK", "mean"], abs=1e-9)
    assert out.loc["UND", "player_bias_mult"] == pytest.approx(1.08)

    # p_over recomputed, finite, in [0, 1], and follows the lifted mean upward
    po = out.loc["UND", "p_over"]
    assert po is not None and 0.0 <= po <= 1.0
    # lifting the mean above the same line must not decrease the over probability
    assert out.loc["UND", "p_over"] >= out.loc["TRK", "p_over"]
    assert out.loc["UND", "p_under"] == pytest.approx(1.0 - out.loc["UND", "p_over"], abs=1e-6)


# --------------------------------------------------------------------------- #
# 6. player_residual_report: coverage + where/how aggregates
# --------------------------------------------------------------------------- #
def test_report_returns_coverage_and_aggregates(conn):
    cands = cands_df([
        cand("VOL", mean=60.0, proj_vol=5.0, home=1),
        cand("EFF", mean=60.0, proj_vol=5.0, home=0),
        cand("ONP", mean=100.0, proj_vol=7.0, home=1),
    ])
    pws = pw_df([
        pw(2024, 1, "VOL", targets=10.0, rec_yards=120.0),
        pw(2024, 1, "EFF", targets=5.0, rec_yards=120.0),
        pw(2024, 1, "ONP", targets=7.0, rec_yards=103.0),
    ])
    plr.record_player_residuals(conn, 2024, 1, cands, pws)

    rep = plr.player_residual_report(conn, season=2024)
    assert rep["n"] == 3
    assert rep["players"] == 3
    # HOW aggregate counts each reason
    assert rep["how"].get("volume") == 1
    assert rep["how"].get("efficiency") == 1
    assert rep["how"].get("on_projection") == 1
    # WHERE aggregate present for both home and away
    assert rep["where_mean_log_resid"]["home"] is not None
    assert rep["where_mean_log_resid"]["away"] is not None
    # per-market coverage present
    assert "receiving_yards" in rep["by_market_mean_log_resid"]


def test_report_empty_is_safe(conn):
    assert plr.player_residual_report(conn, season=2024) == {"n": 0, "players": 0}
