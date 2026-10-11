#!/usr/bin/env python3
"""Attach source-backed game-market and injury frontmatter to Week 5 cards.

This is a publication overlay, not a game model.  Fresh ESPN event summaries
supply the injury report; when their ``odds`` array is empty, every game market
stays unsupported.  Older captured line-only context is retained only as a
clocked historical reference and never becomes a current pick or price.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MARKETS = ("moneyline", "spread", "total")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def _impact(position: str, status: str) -> str:
    group = position.upper()
    if group == "QB":
        return "Quarterback availability is unresolved; no starter or replacement projection is inferred."
    if group in {"RB", "WR", "TE", "FB"}:
        return "Skill-position availability can affect usage, but no beneficiary or workload is assumed."
    if group in {"C", "G", "OG", "OT", "T", "OL"}:
        return "Offensive-line availability is contextual; no protection or rushing adjustment is quantified."
    if status.lower() in {"out", "doubtful", "injured reserve", "reserve/injured"}:
        return "Defensive availability is contextual; no matchup adjustment is quantified."
    return "Availability context only; not a quantified model input."


def _injury_team(summary: dict, abbreviation: str) -> dict:
    for group in summary.get("injuries") or []:
        team = group.get("team") or {}
        if team.get("abbreviation") != abbreviation:
            continue
        items = []
        for row in group.get("injuries") or []:
            athlete = row.get("athlete") or {}
            position = (athlete.get("position") or {}).get("abbreviation") or ""
            status = str(row.get("status") or "No designation")
            items.append({"name": athlete.get("displayName") or athlete.get("fullName") or "Unnamed player",
                          "position": position, "status": status, "impact": _impact(position, status)})
        return {"team": abbreviation, "items": items}
    return {"team": abbreviation, "items": []}


def _prior_lines(card: dict) -> dict:
    game_market = (card.get("context") or {}).get("game_market") or {}
    details = game_market.get("details")
    total = game_market.get("total")
    return {"spread": str(details) if details else None,
            "total": str(total) if total is not None else None,
            "provider": game_market.get("provider")}


def _markets(card: dict, fresh_summary: dict | None) -> list[dict]:
    # Parent-captured fresh summaries currently expose no ESPN ``odds`` offers.
    # Do not manufacture ML, spread prices, or -110 juice from older line-only copy.
    fresh_has_odds = bool((fresh_summary or {}).get("odds"))
    prior = _prior_lines(card)
    records = []
    for market in MARKETS:
        historical_line = prior.get(market)
        records.append({
            "market": market,
            "decision": "No supported pick",
            "line": historical_line,
            "price": None,
            "source_state": ("fresh ESPN market not normalized" if fresh_has_odds else
                             ("previous line only; fresh ESPN market missing" if historical_line else
                              "fresh ESPN market missing")),
            "captured_at": card.get("source_as_of") if historical_line else None,
            "note": ("No current executable price captured; retained line is historical context only."
                     if historical_line else "No current line or price captured."),
        })
    return records


def overlay(cards: list[dict], injury_index: list[dict], injury_dir: str | Path) -> list[dict]:
    index = {str(row["event_id"]): row for row in injury_index}
    result = []
    for original in cards:
        card = dict(original)
        event_id = str(card["event_id"])
        source = index.get(event_id)
        if source:
            summary = read_json(Path(injury_dir) / f"{event_id}.json")
            card["injuries"] = {
                "source": "ESPN event summary injury report",
                "url": source.get("url"),
                "retrieved_at": source.get("retrieved_at"),
                "teams": [_injury_team(summary, card["away"]), _injury_team(summary, card["home"])],
            }
        else:
            summary = None
            card["injuries"] = {
                "source": "No current injury report for archived game",
                "retrieved_at": None,
                "teams": [{"team": card["away"], "items": []}, {"team": card["home"], "items": []}],
            }
        card["game_markets"] = _markets(card, summary)
        result.append(card)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--injury-index", required=True)
    parser.add_argument("--injury-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cards = read_json(args.cards)
    if not isinstance(cards, list) or len(cards) != 15:
        raise SystemExit("expected exactly 15 cards")
    output = overlay(cards, read_json(args.injury_index), args.injury_dir)
    if len(output) != 15 or any(len(card["game_markets"]) != 3 for card in output):
        raise SystemExit("frontmatter coverage is incomplete")
    write_json(args.out, output)
    print(f"wrote {len(output)} cards with {sum(len(c['injuries']['teams']) for c in output)} team injury panels")


if __name__ == "__main__":
    main()
