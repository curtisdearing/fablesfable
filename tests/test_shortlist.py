"""Phase 2 ranking guardrails: context-neutrality, determinism, screen-count
honesty, edge dominance, no_market degradation."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.composite import score_candidate  # noqa: E402
from nflvalue.shortlist import build_context_panel, rank_game  # noqa: E402


def _cand(pid="P1", name="Alpha One", market="receiving_yards", mean=78.0, sd=25.0,
          line=68.5, p_over=0.62, prices=None, opp_factor=1.12, game_script=1.03):
    return {
        "player_id": pid, "name": name, "pos": "WR", "team": "AAA", "defteam": "BBB",
        "game_id": "2023_10_AAA_BBB", "matchup": "AAA @ BBB",
        "market": market, "mean": mean, "sd": sd, "dist": "gamma",
        "line": line, "line_source": "synthetic_trailing_mean" if prices is None else "odds_api",
        "p_over": p_over, "p_under": round(1 - p_over, 4),
        "components": {"volume": 8.1, "efficiency": 9.6, "opp_factor": opp_factor,
                       "game_script": game_script},
        "prices": prices, "low_confidence": False, "eligible_for_shortlist": True,
        "roll_games": 8.0,
    }


# --------------------------------------------------------------------------- #
# THE non-negotiable: context carries ZERO ranking weight
# --------------------------------------------------------------------------- #
def test_composite_identical_with_and_without_context_notes():
    """score_candidate has no context argument at all -- and stapling context
    onto the candidate dict must not change the number either."""
    plain = _cand()
    with_context = copy.deepcopy(plain)
    with_context["context_notes"] = [{"text": "birthday revenge game vs former team",
                                      "source": "espn_news", "timestamp": "x"}]
    with_context["personal_context"] = "his birthday AND a revenge spot"
    with_context["injury_note"] = "was questionable Wednesday"

    s1 = score_candidate(plain)
    s2 = score_candidate(with_context)
    assert s1 == s2


def test_shortlist_ranking_identical_with_and_without_context():
    cands = [_cand(), _cand(pid="P2", name="Beta", mean=90.0, p_over=0.70),
             _cand(pid="P3", name="Gamma", mean=40.0, line=52.5, p_over=0.31)]
    noisy = copy.deepcopy(cands)
    for c in noisy:
        c["context_notes"] = [{"text": "bereavement; contract-year revenge birthday",
                               "source": "news", "timestamp": "t"}]
    a = rank_game(cands)
    b = rank_game(noisy)
    assert [(l["player_id"], l["market"], l["composite"]) for l in a["leans"]] == \
           [(l["player_id"], l["market"], l["composite"]) for l in b["leans"]]


def test_context_panel_is_built_after_ranking_and_labeled():
    g = rank_game([_cand()])
    panel = build_context_panel(g, synthesis_output={
        "players": [{"player_id": "P1", "status": "OK", "divergence_flag": False,
                     "needs_reallocation": False,
                     "context_notes": [{"text": "birthday Sunday", "source": "espn"}]}]})
    assert "not part of the composite score" in panel["label"]
    assert any("birthday" in i for e in panel["entries"] for i in e["items"])
    # and the leans it decorates are unchanged by construction (already final)
    assert g["leans"][0]["composite"] == rank_game([_cand()])["leans"][0]["composite"]


# --------------------------------------------------------------------------- #
# Determinism + selection honesty
# --------------------------------------------------------------------------- #
def test_rank_deterministic_across_runs_and_input_order():
    cands = [_cand(pid=f"P{i}", name=f"N{i}", mean=50 + i, line=48.5,
                   p_over=0.5 + i * 0.02) for i in range(8)]
    a = rank_game(list(cands))
    b = rank_game(list(reversed(cands)))
    assert [(l["player_id"], l["composite"]) for l in a["leans"]] == \
           [(l["player_id"], l["composite"]) for l in b["leans"]]


def test_screen_count_is_true_denominator():
    cands = [_cand(pid=f"P{i}", name=f"N{i}") for i in range(9)]
    g = rank_game(cands)
    assert g["screened_n"] == 9
    assert g["screened"] == "5 of 9"
    assert len(g["leans"]) == 5


def test_max_per_player_cap():
    cands = [_cand(market="receiving_yards", mean=90, line=60.5, p_over=0.8),
             _cand(market="receptions", mean=8.5, line=5.5, p_over=0.78),
             _cand(market="anytime_td", mean=0.9, line=0.5, p_over=0.72),
             _cand(pid="P2", name="Beta", mean=45, line=44.5, p_over=0.51)]
    g = rank_game(cands, max_per_player=2)
    p1_count = sum(1 for l in g["leans"] if l["player_id"] == "P1")
    assert p1_count == 2
    assert g["screened_n"] == 4                       # cap doesn't hide the denominator


# --------------------------------------------------------------------------- #
# Edge behavior (Phase 3 wiring, testable now)
# --------------------------------------------------------------------------- #
def test_no_market_renormalizes_and_tags():
    s = score_candidate(_cand(prices=None))
    assert s["no_market"] is True
    assert s["edge"] is None
    assert s["components"]["weights_used"] == {"confidence": 0.3, "matchup": 0.2}


def test_edge_dominant_with_real_prices():
    """Same projection: a big model-vs-market edge must outrank a bigger
    z-score with no market disagreement (best VALUE, not best projection)."""
    # candidate A: modest conviction but the market price disagrees with us hard
    a = _cand(pid="A", mean=72.0, line=68.5, sd=25.0, p_over=0.60,
              prices={"over": 2.10, "under": 1.80, "book": "bookx"})  # devig p_over ~.462
    # candidate B: huge z distance but priced dead-on by the market
    b = _cand(pid="B", mean=95.0, line=60.5, sd=15.0, p_over=0.80,
              prices={"over": 1.25, "under": 4.60, "book": "bookx"})  # devig p_over ~.786
    sa, sb = score_candidate(a), score_candidate(b)
    assert sa["edge"] > 0.10                          # ~14 prob-points of edge
    assert sb["edge"] < 0.02
    assert sa["composite"] > sb["composite"]


def test_calibration_gate_forces_no_market():
    priced = _cand(prices={"over": 2.10, "under": 1.80, "book": "bookx"})
    open_gate = score_candidate(priced)
    closed_gate = score_candidate(priced, params={"calibration_passed": False})
    assert open_gate["no_market"] is False
    assert closed_gate["no_market"] is True
    assert closed_gate["edge"] is None


def test_side_flips_to_under_when_market_overprices_over():
    c = _cand(mean=60.0, line=68.5, p_over=0.38,
              prices={"over": 1.75, "under": 2.15, "book": "bookx"})
    s = score_candidate(c)
    assert s["side"] == "under"
    assert s["edge"] is not None and s["edge"] > 0


def test_anytime_td_is_yes_only_and_penalized():
    """'No TD' isn't a purchasable lean: an unlikely-TD player must score LOW,
    not top the list with a degenerate under (found in the first 2023-wk10
    render and fixed here)."""
    scrub = _cand(market="anytime_td", mean=0.07, sd=0.35, line=0.5, p_over=0.068)
    scrub["low_confidence"] = True
    stud = _cand(pid="P9", name="Stud", market="anytime_td", mean=0.95, sd=0.97,
                 line=0.5, p_over=0.61)
    stud["low_confidence"] = True
    receiver = _cand(pid="P5", name="Solid WR", market="receiving_yards",
                     mean=82.0, line=66.5, p_over=0.68)

    s_scrub, s_stud, s_wr = (score_candidate(x) for x in (scrub, stud, receiver))
    assert s_scrub["side"] == "over" and s_stud["side"] == "over"   # rendered YES
    assert s_scrub["confidence"] == 0.0        # model side < 50% -> no conviction credit
    assert s_scrub["composite"] < s_stud["composite"] < s_wr["composite"]


def test_low_confidence_multiplier_applied():
    base = _cand()
    tagged = copy.deepcopy(base)
    tagged["low_confidence"] = True
    assert score_candidate(tagged)["composite"] == pytest.approx(
        score_candidate(base)["composite"] * 0.8, abs=0.02)
