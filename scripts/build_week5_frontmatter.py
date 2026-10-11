#!/usr/bin/env python3
"""Attach source-backed ESPN game-market and injury frontmatter to Week 5 cards.

This is a publication overlay, not a game model. ESPN's pickcenter close fields
are listed context only: their retrieval time is not a sportsbook quote clock,
and they never become a game recommendation or executable-price assertion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MARKETS = ("moneyline", "spread", "total")
TEAM_ALIASES = {"WSH": "WAS", "WAS": "WSH", "LAR": "LA", "LA": "LAR"}
STATUS_ORDER = {"out": 0, "injured reserve": 1, "reserve/injured": 1,
                "doubtful": 3, "questionable": 4}


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def _impact(position: str, status: str) -> str:
    if position.upper() == "QB":
        return "Quarterback availability reported; no starter or replacement is inferred."
    return ""


def _same_team(left: str | None, right: str) -> bool:
    left = str(left or "").upper()
    right = str(right).upper()
    return left == right or TEAM_ALIASES.get(left) == right or TEAM_ALIASES.get(right) == left


def _injury_description(row: dict) -> tuple[str | None, str | None]:
    details = row.get("details") or {}
    body_part = details.get("type") or details.get("location")
    comment = row.get("shortComment") or row.get("longComment") or details.get("shortComment") or details.get("longComment")
    detail = details.get("detail")
    text = comment or detail
    if text in (None, "", "Not Specified"):
        text = None
    if body_part in (None, "", "Undisclosed", "Not Specified"):
        body_part = None
    return (str(body_part) if body_part is not None else None,
            str(text) if text is not None else None)


def _injury_sort_key(item: dict) -> tuple:
    status = str(item.get("status") or "").lower()
    position = str(item.get("position") or "").upper()
    priority = STATUS_ORDER.get(status, 5)
    if position == "QB" and priority > 1:
        priority = 2
    return priority, str(item.get("report_date") or ""), str(item.get("name") or "")


def _injury_team(summary: dict, abbreviation: str) -> dict:
    for group in summary.get("injuries") or []:
        team = group.get("team") or {}
        if not _same_team(team.get("abbreviation"), abbreviation):
            continue
        items = []
        for row in group.get("injuries") or []:
            athlete = row.get("athlete") or {}
            position = (athlete.get("position") or {}).get("abbreviation") or ""
            status = str(row.get("status") or "No designation")
            body_part, description = _injury_description(row)
            items.append({"name": athlete.get("displayName") or athlete.get("fullName") or "Unnamed player",
                          "position": position, "status": status, "report_date": row.get("date"),
                          "body_part": body_part, "description": description,
                          "impact": _impact(position, status)})
        items.sort(key=_injury_sort_key)
        return {"team": abbreviation, "items": items}
    return {"team": abbreviation, "items": []}


def _close(node: dict | None) -> dict:
    close = node.get("close") if isinstance(node, dict) else None
    return close if isinstance(close, dict) else {}


def _value(value) -> str | None:
    return None if value is None or value == "" else str(value)


def _total_line(value) -> str | None:
    value = _value(value)
    return value[1:] if value and value[:1].lower() in {"o", "u"} else value


def _joined(values: list[str | None]) -> str | None:
    present = [value for value in values if value is not None]
    return " · ".join(present) if present else None


def _market_record(market: str, line: str | None, price: str | None, provider: str | None,
                   captured_at: str | None, source_url: str | None) -> dict:
    listed = line is not None or price is not None
    return {"market": market, "decision": "No supported pick", "line": line, "price": price,
            "provider": provider, "captured_at": captured_at if listed else None, "source_url": source_url,
            "source_state": ("ESPN listed close; not executable verified" if listed else
                             "ESPN pickcenter close unavailable"),
            "note": ("Listed close fields at retrieval; not a sportsbook-verified executable quote."
                     if listed else "No ESPN pickcenter close field was available at retrieval.")}


def _markets(card: dict, fresh_summary: dict | None, captured_at: str | None = None,
             source_url: str | None = None) -> list[dict]:
    pickcenter = (fresh_summary or {}).get("pickcenter") or []
    center = pickcenter[0] if isinstance(pickcenter, list) and pickcenter and isinstance(pickcenter[0], dict) else {}
    provider = _value((center.get("provider") or {}).get("name"))
    away, home = str(card.get("away") or "AWAY"), str(card.get("home") or "HOME")
    moneyline = center.get("moneyline") or {}
    spread = center.get("pointSpread") or {}
    total = center.get("total") or {}
    away_ml, home_ml = _close(moneyline.get("away")), _close(moneyline.get("home"))
    away_spread, home_spread = _close(spread.get("away")), _close(spread.get("home"))
    over, under = _close(total.get("over")), _close(total.get("under"))
    return [
        _market_record("moneyline", _joined([f"{away} {_value(away_ml.get('odds'))}" if _value(away_ml.get("odds")) is not None else None,
                                               f"{home} {_value(home_ml.get('odds'))}" if _value(home_ml.get("odds")) is not None else None]),
                       None, provider, captured_at, source_url),
        _market_record("spread", _joined([f"{away} {_value(away_spread.get('line'))}" if _value(away_spread.get("line")) is not None else None,
                                            f"{home} {_value(home_spread.get('line'))}" if _value(home_spread.get("line")) is not None else None]),
                       _joined([f"{away} {_value(away_spread.get('odds'))}" if _value(away_spread.get("odds")) is not None else None,
                                f"{home} {_value(home_spread.get('odds'))}" if _value(home_spread.get("odds")) is not None else None]),
                       provider, captured_at, source_url),
        _market_record("total", _joined([f"Over {_total_line(over.get('line'))}" if _total_line(over.get("line")) is not None else None,
                                           f"Under {_total_line(under.get('line'))}" if _total_line(under.get("line")) is not None else None]),
                       _joined([f"Over {_value(over.get('odds'))}" if _value(over.get("odds")) is not None else None,
                                f"Under {_value(under.get('odds'))}" if _value(under.get("odds")) is not None else None]),
                       provider, captured_at, source_url),
    ]


def overlay(cards: list[dict], injury_index: list[dict], injury_dir: str | Path) -> list[dict]:
    if isinstance(injury_index, dict):
        injury_index = injury_index.get("events") or []
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
        card["game_markets"] = _markets(card, summary, source.get("retrieved_at") if source else None,
                                        source.get("url") if source else None)
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
