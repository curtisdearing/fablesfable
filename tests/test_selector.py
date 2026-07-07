"""Post-projection best-picks selector: selection order, bet logic, tiers,
writeups, movement-invariance, threshold config, blocked-run hygiene."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue import selector as sel  # noqa: E402
from nflvalue.shortlist import rank_game  # noqa: E402

SEL = sel.selector_config({})   # shipped defaults


def _scored(pid="X", market="receiving_yards", edge=0.05, ev=0.03,
            model_p=0.58, market_p=0.53, real=True, **over):
    """A candidate in the documented POST-SCORING shape classify() consumes
    (what rank_game rows look like after score_candidate ran)."""
    c = {
        "player_id": pid, "name": f"Player {pid}", "pos": "WR", "team": "AAA",
        "market": market, "side": "over", "line": 62.5, "mean": 71.2,
        "line_source": "odds_api" if real else "synthetic_trailing_mean",
        "no_market": not real, "edge": edge if real else None,
        "components": {"model_prob": model_p, "market_prob": market_p,
                       "ev_best_price": ev, "n_books": 3},
        "proj_components": {"volume": 8.1, "efficiency": 8.8, "opp_factor": 1.06},
        "prices": {"over": 1.91, "under": 1.91, "book": "dk", "n_books": 3} if real else None,
    }
    c.update(over)
    return c


def _cand(pid, market="receiving_yards", side="over", edge=0.05, ev=0.03,
          model_p=0.58, market_p=0.53, real=True, composite=60.0, **over):
    c = {
        "player_id": pid, "name": f"Player {pid}", "pos": "WR", "team": "AAA",
        "game_id": "2025_09_AAA_BBB", "matchup": "AAA @ BBB",
        "market": market, "side": side, "line": 62.5, "mean": 71.2, "sd": 22.0,
        "line_source": "odds_api" if real else "synthetic_trailing_mean",
        "no_market": not real,
        "edge": edge if real else None,
        "composite": composite, "confidence": 0.5, "matchup_comp": 0.5,
        "prices": ({"over": 1.91, "under": 1.91, "book": "dk", "n_books": 3}
                   if real else None),
        # like a real candidate from enumerate_candidates: "components" is the
        # PROJECTION breakdown; rank_game moves it to proj_components and
        # score_candidate recomputes edge/EV/model-vs-market from prices
        "components": {"volume": 8.1, "efficiency": 8.8, "opp_factor": 1.06,
                       "game_script": 1.0},
        "p_over": model_p if side == "over" else 1 - model_p,
        "p_under": 1 - model_p if side == "over" else model_p,
    }
    c.update(over)
    return c


# --------------------------------------------------------------------------- #
# 1. picks are selected only AFTER all candidates are evaluated
# --------------------------------------------------------------------------- #
def test_selection_consumes_the_full_scored_pool_not_the_top5():
    """A candidate the RANKER leaves out of the top-5 leans (lowest ml_score
    in the game) MUST still be pickable when it has the best market edge --
    proof the selector reads the FULL evaluated pool, not the shortlist."""
    # six ml-favored candidates with mild probabilities (small edges)...
    cands = [_cand(f"P{i}", model_p=0.54, ml_score=90.0 - i) for i in range(6)]
    # ...and the edge monster the ML ranker likes LEAST (last in ml order)
    monster = _cand("EDGE", model_p=0.62, ml_score=1.0)
    g = rank_game(cands + [monster])
    assert len(g["scored_pool"]) == g["screened_n"] == 7
    assert all(l["player_id"] != "EDGE" for l in g["leans"])  # not a top-5 lean...
    sel.picks_for_games([g], cfg={})
    assert g["picks"], "picks must exist"
    assert g["picks"][0]["player_id"] == "EDGE"               # ...but the #1 pick
    assert "scored_pool" not in g                              # consumed, not persisted


# --------------------------------------------------------------------------- #
# 2. both overs and unders can be selected
# --------------------------------------------------------------------------- #
def test_overs_and_unders_both_selectable():
    over = _cand("O1", side="over", edge=0.06)
    under = _cand("U1", side="under", edge=0.05,
                  p_over=0.42, p_under=0.58)
    g = rank_game([over, under])
    sel.picks_for_games([g], cfg={})
    sides = {p["player_id"]: p["side"] for p in g["picks"]}
    assert sides.get("O1") == "over" and sides.get("U1") == "under"
    # anytime_td never creates an artificial under
    td = _cand("TD1", market="anytime_td", side="over", edge=0.09, ev=0.03,
               model_p=0.45, market_p=0.36)
    g2 = rank_game([td])
    sel.picks_for_games([g2], cfg={})
    assert all(p["side"] == "over" for p in g2["picks"])


# --------------------------------------------------------------------------- #
# 3. synthetic / no-market lines are never labeled best bets
# --------------------------------------------------------------------------- #
def test_synthetic_lines_are_research_only():
    synth = _cand("S1", real=False, composite=99.0)
    real = _cand("R1", edge=0.05)
    g = rank_game([synth, real])
    sel.picks_for_games([g], cfg={})
    assert all(p["line_source"] == "odds_api" for p in g["picks"])
    research = g["research_leans"]
    assert research and research[0]["tier"] == "RESEARCH"
    assert "research" in research[0]["writeup"].lower() or \
           "no-market" in research[0]["writeup"].lower()
    tier, _ = sel.classify(_scored("S1", real=False), SEL)
    assert tier == "RESEARCH"


# --------------------------------------------------------------------------- #
# 4. multiple picks per game, ranked, capped per player
# --------------------------------------------------------------------------- #
def test_multiple_ranked_picks_per_game():
    cands = [_cand(f"P{i}", edge=0.03 + i * 0.01) for i in range(6)]
    cands += [_cand("P0", market="receptions", edge=0.055),
              _cand("P0", market="rushing_yards", edge=0.052),  # 3rd P0 market
              ]
    g = rank_game(cands)
    sel.picks_for_games([g], cfg={})
    assert len(g["picks"]) == SEL["max_picks_per_game"] > 1
    per = {}
    for p in g["picks"]:
        per[p["player_id"]] = per.get(p["player_id"], 0) + 1
    assert max(per.values()) <= SEL["max_per_player"]
    edges = [p["edge"] for p in g["picks"] if p["edge"] is not None]
    assert edges == sorted(edges, reverse=True)   # ranked by edge


# --------------------------------------------------------------------------- #
# 5. tiers: market-specific thresholds + downgrades
# --------------------------------------------------------------------------- #
def test_market_thresholds_change_the_label():
    c = _scored("X", market="receiving_yards", edge=0.045, ev=0.02, model_p=0.56)
    tier_ry, _ = sel.classify(c, SEL)
    assert tier_ry == "PLAYABLE"          # clears the .040 yardage bar
    c_td = _scored("X", market="anytime_td", edge=0.045, ev=0.02, model_p=0.56)
    tier_td, notes = sel.classify(c_td, SEL)
    assert tier_td == "LEAN"              # same edge under the .080 TD bar
    # custom config flips it
    custom = sel.selector_config({"selector": {"thresholds": {
        "anytime_td": {"lean": 0.01, "playable": 0.02, "strong": 0.04, "ev_min": 0.0}}}})
    tier_custom, _ = sel.classify(c_td, custom)
    assert tier_custom == "STRONG"
    # below the lean bar -> PASS
    tier_pass, _ = sel.classify(_scored("X", edge=0.01), SEL)
    assert tier_pass == "PASS"


def test_confidence_downgrades_risk_stale_correlation():
    strong = _scored("X", edge=0.09, ev=0.05, model_p=0.60)
    t0, _ = sel.classify(strong, SEL)
    assert t0 == "STRONG"
    t1, n1 = sel.classify(strong, SEL,
                          availability={"X": {"status": "RISK"}})
    assert t1 == "PLAYABLE" and any("RISK" in n for n in n1)
    t2, n2 = sel.classify(strong, SEL, line_age_hours=48.0)
    assert t2 == "PLAYABLE" and any("stale" in n.lower() or "old" in n for n in n2)
    t3, n3 = sel.classify(strong, SEL, max_corr_rho=0.7)
    assert t3 == "PLAYABLE" and any("correlates" in n for n in n3)
    # calibration floor: model_prob under the strong bar caps at PLAYABLE
    t4, _ = sel.classify(_scored("X", edge=0.09, ev=0.05, model_p=0.53), SEL)
    assert t4 == "PLAYABLE"


# --------------------------------------------------------------------------- #
# 6. line movement is NOT a live input (CLV stays after-the-fact)
# --------------------------------------------------------------------------- #
def test_selector_invariant_to_line_movement_fields():
    base = _cand("X", edge=0.05)
    moved = dict(base)
    moved.update({"entry_prob": 0.50, "close_prob": 0.60, "clv_prob": 0.10,
                  "point_moved": -1.5, "line_moved_toward_us": True})
    g1, g2 = rank_game([base]), rank_game([moved])
    sel.picks_for_games([g1], cfg={})
    sel.picks_for_games([g2], cfg={})
    assert g1["picks"][0]["tier"] == g2["picks"][0]["tier"]
    assert g1["picks"][0]["edge"] == g2["picks"][0]["edge"]


# --------------------------------------------------------------------------- #
# 7. writeups: factual, complete
# --------------------------------------------------------------------------- #
def test_writeup_answers_the_required_questions():
    # rank_game recomputes edge/EV from the REAL prices (1.91/1.91 de-vigs to
    # a 50% fair prob), so the pick shows model 58% vs market 50% = +8.0pt
    g = rank_game([_cand("W1", model_p=0.58)])
    sel.picks_for_games([g], cfg={})
    w = g["picks"][0]["writeup"]
    assert "58%" in w and "50%" in w          # model vs market probability
    assert "+8.0-point" in w                  # edge
    assert "expected value" in w              # EV at best price
    assert "Projection basis" in w            # supporting projection
    assert "risk" in w.lower()                # risks/caveats
    for hype in ("lock", "smash", "hammer", "guarantee"):
        assert hype not in w.lower()


# --------------------------------------------------------------------------- #
# 8. blocked runs never contaminate active evidence
# --------------------------------------------------------------------------- #
def test_blocked_status_excluded_from_evidence(tmp_path):
    conn = dbmod.connect(str(tmp_path / "sel.db"))
    g = rank_game([_cand("B1", edge=0.06)])
    sel.picks_for_games([g], cfg={})
    games = [g]
    from nflvalue.report import persist_leans
    persist_leans(conn, 2025, 9, "wed", games, "2025-11-05T15:00:00Z", status="blocked")
    sel.persist_picks(conn, 2025, 9, "wed", games, "2025-11-05T15:00:00Z", status="blocked")

    # grading skips blocked picks entirely
    pw_frame = pd.DataFrame([{"season": 2025, "week": 9, "player_id": "B1",
                              "rec_yards": 99.0, "receptions": 9, "rush_yards": 0,
                              "pass_yards": 0, "pass_attempts": 0, "completions": 0,
                              "carries": 0, "rush_tds": 0, "rec_tds": 1}])
    assert sel.grade_picks(conn, 2025, 9, pw_frame)["graded"] == 0
    assert sel.picks_record(conn)["n"] == 0
    # CLV resolution sees no active leans
    from nflvalue import clv as clvmod
    resolved = clvmod.log_close_for_week(conn, 2025, 9, {"2025_09_AAA_BBB": "2025-11-09T18:00:00Z"})
    assert resolved.empty
    # kill-check active-lean denominator is zero
    from nflvalue import killcheck as kcmod
    assert kcmod.report(conn)["leans_logged"] == 0
    # the audit trail still exists
    n_rows = dbmod.query_df(conn, "SELECT COUNT(*) AS n FROM picks").iloc[0]["n"]
    assert int(n_rows) == 1
    conn.close()


def test_active_picks_are_graded_and_recorded(tmp_path):
    conn = dbmod.connect(str(tmp_path / "sel2.db"))
    g = rank_game([_cand("A1", edge=0.06, line=62.5)])
    sel.picks_for_games([g], cfg={})
    sel.persist_picks(conn, 2025, 9, "wed", [g], "2025-11-05T15:00:00Z", status="active")
    pw_frame = pd.DataFrame([{"season": 2025, "week": 9, "player_id": "A1",
                              "rec_yards": 88.0, "receptions": 7, "rush_yards": 0,
                              "pass_yards": 0, "pass_attempts": 0, "completions": 0,
                              "carries": 0, "rush_tds": 0, "rec_tds": 1}])
    assert sel.grade_picks(conn, 2025, 9, pw_frame)["graded"] == 1
    rec = sel.picks_record(conn)
    assert rec["n"] == 1
    tier = list(rec["by_tier"])[0]
    assert rec["by_tier"][tier]["hit_rate"] == 1.0    # 88 > 62.5 over
    conn.close()
