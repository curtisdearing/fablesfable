#!/usr/bin/env python3
"""Generate a SAMPLE picks payload by running the REAL selector on illustrative
candidates, so the transparency site is viewable/deployable before a live week
has been generated. The engine is genuine (nflvalue.shortlist + selector produce
the tiers, writeups and decision chains); only the candidate INPUTS are
hand-authored and clearly labeled SAMPLE. No parquet, no network.

    python3 scripts/make_sample_picks.py    # -> data/weekly_props_sample.json
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import selector as sel  # noqa: E402
from nflvalue.shortlist import rank_game  # noqa: E402


def cand(game_id, matchup, pid, name, pos, team, market, side, mean, line,
         p_over, over_price, under_price, book="dk", n_books=4,
         opp_factor=1.0, game_script=1.0, shape_tilts=None, wx=None,
         real=True, market_p_over=None):
    # market_p_over = the de-vigged CONSENSUS fair probability of the over. It
    # is DISTINCT from p_over (the model's belief); their gap is the edge. When
    # omitted, fall back to the two-sided price de-vig.
    prices = ({"over": over_price, "under": under_price, "book": book,
               "consensus_p_over": market_p_over, "n_books": n_books} if real else None)
    comps = {"volume": round(mean / max(opp_factor, 0.1) / 8.5, 2),
             "efficiency": 8.5, "opp_factor": opp_factor, "game_script": game_script}
    if shape_tilts:
        comps["shape_tilts"] = shape_tilts
    c = {"game_id": game_id, "matchup": matchup, "player_id": pid, "name": name,
         "pos": pos, "team": team, "market": market, "side": side,
         "mean": mean, "sd": max(mean * 0.28, 1.2), "line": line,
         "line_source": "odds_api" if real else "synthetic_trailing_mean",
         "no_market": not real,
         "p_over": p_over, "p_under": round(1 - p_over, 4),
         "prices": prices, "components": comps}
    if wx is not None:
        c["wx_pass_mult"] = wx
    return c


def build_game(game_id, matchup, cands):
    g = rank_game(cands)
    sel.picks_for_games([g], cfg={})
    return g


def main():
    games = []

    # ---- Game 1: a chalky-favorite pass game, one STRONG, one PLAYABLE ------ #
    g1 = build_game("2025_09_MIN_CHI", "MIN @ CHI", [
        cand("2025_09_MIN_CHI", "MIN @ CHI", "JJ", "Justin Jefferson", "WR", "MIN",
             "receiving_yards", "over", 94.1, 82.5, 0.585, 1.95, 1.87,
             market_p_over=0.515, opp_factor=1.08, shape_tilts={"depth": 1.03}),
        cand("2025_09_MIN_CHI", "MIN @ CHI", "JJ", "Justin Jefferson", "WR", "MIN",
             "receptions", "over", 6.8, 5.5, 0.61, 1.98, 1.85,
             market_p_over=0.55, opp_factor=1.05),
        cand("2025_09_MIN_CHI", "MIN @ CHI", "AJ", "Aaron Jones", "RB", "MIN",
             "rushing_yards", "over", 71.0, 64.5, 0.56, 1.93, 1.89,
             market_p_over=0.52, opp_factor=1.10),
        cand("2025_09_MIN_CHI", "MIN @ CHI", "DM", "D.J. Moore", "WR", "CHI",
             "receiving_yards", "under", 55.0, 64.5, 0.58, 1.86, 1.96,
             market_p_over=0.47, opp_factor=0.92),
        cand("2025_09_MIN_CHI", "MIN @ CHI", "SD", "Sam Darnold", "QB", "MIN",
             "passing_yards", "over", 251.0, 244.5, 0.535, 1.91, 1.91,
             market_p_over=0.52, opp_factor=1.04),
    ])
    games.append(g1)

    # ---- Game 2: bad weather game, a weather-driven UNDER + a research lean -- #
    g2 = build_game("2025_09_BUF_MIA", "BUF @ MIA", [
        cand("2025_09_BUF_MIA", "BUF @ MIA", "JA", "Josh Allen", "QB", "BUF",
             "passing_yards", "under", 228.0, 249.5, 0.40, 1.94, 1.88,
             market_p_over=0.515, opp_factor=0.97, wx=0.91),
        cand("2025_09_BUF_MIA", "BUF @ MIA", "TH", "Tyreek Hill", "WR", "MIA",
             "receiving_yards", "under", 61.0, 72.5, 0.435, 1.90, 1.92,
             market_p_over=0.515, opp_factor=0.95, wx=0.90),
        cand("2025_09_BUF_MIA", "BUF @ MIA", "JC", "James Cook", "RB", "BUF",
             "anytime_td", "over", 0.58, 0.5, 0.44, 2.10, None,
             market_p_over=0.455, opp_factor=1.12, n_books=3),
        # a synthetic (no-market) research lean -- must render as RESEARCH only
        cand("2025_09_BUF_MIA", "BUF @ MIA", "RB2", "Raheem Mostert", "RB", "MIA",
             "rushing_yards", "over", 58.0, 49.5, 0.57, 0, 0, real=False),
    ])
    games.append(g2)

    payload = {
        "season": 2025, "week": 9, "clock": "wed",
        "as_of": "2025-11-05T18:30:00Z", "publish": True, "publish_reasons": [],
        "mode": "sample", "sample": True,
        "sample_note": ("ILLUSTRATIVE sample — no live week has been generated in this "
                        "environment. Tiers, writeups and decision chains are produced by the "
                        "REAL nflvalue.selector on hand-authored candidate inputs, to demonstrate "
                        "the transparency UI. Numbers are not live picks."),
        "games": games,
    }
    # strip the internal scored_pool (selector already consumed it)
    for g in payload["games"]:
        g.pop("scored_pool", None)
    out = os.path.join(ROOT, "data", "weekly_props_sample.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    n = sum(len(g.get("picks", [])) + len(g.get("research_leans", [])) for g in games)
    print(f"wrote {out} — {len(games)} games, {n} picks+research")


if __name__ == "__main__":
    main()
