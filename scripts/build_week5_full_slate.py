#!/usr/bin/env python3
"""Join audited Week 5 inputs and build a read-only, full-slate public board.

The command is intentionally input-bound: it fails if its scoreboard, saved source
cards, or captured late-market file is absent. It never calls a sportsbook or a
model endpoint, so a publish rebuild cannot erase the captured analyst snapshot.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from nflvalue import all_props


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"required input not found: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"required input unreadable: {path}: {exc}") from exc


def slug_game(away, home):
    return f"2026_05_{away}_{home}"


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_price(value):
    text = str(value or "").strip()
    if not text:
        return None, None
    match = re.search(r"([+-]\d+)\s+(.+)$", text)
    return (int(match.group(1)), match.group(2).strip()) if match else (None, None)


def player_team(label):
    chunks = str(label).split()
    return (" ".join(chunks[:-3]) or str(label), chunks[-1] if len(chunks) >= 3 else "UNK")


def market_name(label):
    return {"Receiving yards": "receiving_yards", "Rushing yards": "rushing_yards",
            "Passing yards": "passing_yards", "Receptions": "receptions",
            "Anytime TD": "anytime_touchdown", "Longest reception": "longest_reception",
            "Longest rush": "longest_rush"}.get(label, str(label).lower().replace(" ", "_"))


def source(title, url):
    return {"title": title, "url": url}


def source_card_pick(row, event_id):
    quote = row.get("quote") or {}
    price = quote.get("price_american")
    try:
        odds = int(str(price)) if price is not None else None
    except ValueError:
        odds = None
    invalidation = row.get("invalidation") or []
    return {"player": row.get("player") or row.get("name") or "Unknown player", "player_id": row.get("player_id"),
            "team": row.get("team") or "UNK", "market": row.get("market") or "unknown", "side": row.get("side") or "unknown",
            "line": row.get("line"), "book": quote.get("book"), "odds": odds,
            "quote_updated_at": quote.get("captured_at"), "retrieved_at": row.get("run_as_of"),
            "projection": row.get("mean"), "model_probability": row.get("model_p_side"), "calibrated_probability": None,
            "model_run_as_of": row.get("run_as_of"), "status": "research", "rationale": row.get("rationale") or "Saved source card.",
            "counterargument": row.get("countercase") or "Saved source card; role and line can change.",
            "invalidation": "; ".join(invalidation) if isinstance(invalidation, list) else str(invalidation),
            "rank_basis": "Saved native source card; not a current inference or wager approval.",
            "sources": []}


def integrate(scoreboard_path, early_hub_path, late_path, extra_paths=()):
    scoreboard = read_json(scoreboard_path)
    early = read_json(early_hub_path)
    late = read_json(late_path)
    if not isinstance(scoreboard.get("events"), list) or len(scoreboard["events"]) != 15:
        raise ValueError("scoreboard must contain exactly 15 Week 5 events")
    saved = early.get("cards") if isinstance(early, dict) else early
    if not isinstance(saved, list):
        raise ValueError("saved source cards must be a list or hub.json with cards")
    if not isinstance(late, list):
        raise ValueError("late market file must be a top-level list")
    extras = []
    for path in extra_paths:
        doc = read_json(path)
        if not isinstance(doc, list):
            raise ValueError(f"extra payload must be a top-level list: {path}")
        extras.extend(doc)
    by_game = {}
    for row in saved:
        if isinstance(row, dict) and row.get("game_id"):
            by_game.setdefault(row["game_id"], []).append(row)
    late_by_game = {}
    for raw in late:
        if not isinstance(raw, dict) or not raw.get("game"):
            raise ValueError("late market rows must be objects with game")
        late_by_game.setdefault(raw["game"], []).append(raw)
    cards, rows = [], []
    raw_count = 0
    for event in scoreboard["events"]:
        comp = event["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        away, home = teams["away"]["team"]["abbreviation"], teams["home"]["team"]["abbreviation"]
        event_id, gid = str(event["id"]), slug_game(away, home)
        finished = bool(event.get("status", {}).get("type", {}).get("completed"))
        venue = comp.get("venue", {})
        event_sources = [source("ESPN Week 5 scoreboard", "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20261011&seasontype=2&week=5")]
        picks = [source_card_pick(x, event_id) for x in by_game.get(gid, [])]
        late_rows = late_by_game.get(f"{away}_{home}", [])
        for raw in late_rows:
            raw_count += 1
            player, team = player_team(raw.get("player_label"))
            selected = str(raw.get("selected_side") or "").lower()
            line = num(raw.get("Line"))
            prop = market_name(raw.get("Prop"))
            raw_record = dict(raw)
            for side, field in (("over", "Best over"), ("under", "Best under")):
                if raw.get("Prop") == "Anytime TD":
                    side, field = "yes", "Best over"
                    if rows and rows[-1].get("source_row_id") == raw_count:
                        continue
                odds, book = parse_price(raw.get(field))
                disposition = str(raw.get("disposition") or "")
                selected_here = selected == side
                if disposition == "EXCLUDE_REPORTED_OUT":
                    status, outcome = "unavailable", "unavailable"
                elif disposition == "QUARANTINE_QUOTE_ANOMALY":
                    status, outcome = "research", "rejected"
                elif selected_here:
                    status, outcome = "analyst_lean", "reviewed"
                elif odds is None:
                    status, outcome = "unavailable", "unsupported"
                else:
                    status, outcome = "pass", "pass"
                rows.append({"event_id": event_id, "game_id": gid, "player": player, "team": team, "market": prop,
                             "side": side, "period": "full_game", "status": status, "disposition": outcome,
                             "line": None if side == "yes" else line, "odds": odds, "book": book,
                             "captured_at": raw.get("retrieved_at"), "provider_updated_at": None,
                             "reason": raw.get("rationale") or raw.get("history_note"), "sources": [source("Secondary published listing", raw.get("source"))],
                             "source_row_id": raw_count, "raw_source_row": raw_record,
                             "quote_verification": raw.get("quote_verification")})
                if selected_here:
                    picks.append({"player": player, "player_id": None, "team": team, "market": prop, "side": side,
                                  "line": None if side == "yes" else line, "book": book, "odds": odds,
                                  "quote_updated_at": None, "retrieved_at": raw.get("retrieved_at"), "projection": None,
                                  "model_probability": None, "calibrated_probability": None, "model_run_as_of": None,
                                  "status": "analyst_lean", "rationale": raw.get("rationale") or "Conditional analyst lean.",
                                  "counterargument": "Secondary listing and current role/price have not been independently reverified.",
                                  "invalidation": "Any inactive/role change, line movement, or failure to confirm a current executable offer.",
                                  "rank_basis": "Conditional manual lean from descriptive observed history; no calibrated probability.",
                                  "sources": [source("Secondary published listing", raw.get("source"))]})
        outcome = None
        if finished:
            outcome = {"score": f"{away} {teams['away'].get('score', '?')}, {home} {teams['home'].get('score', '?')}",
                       "label": "Final — archived result, not a prospective recommendation."}
        if finished:
            preview = f"Completed game at {venue.get('fullName', 'venue unavailable')}; preserved as an archive result."
            coverage = {"state": "archived_result", "detail": "No prospective props or picks are displayed for a completed game."}
        elif late_rows:
            preview = "Captured secondary player-market screen is available below. Eight conditional analyst leans are separated from pass rows; no calibrated probabilities are claimed."
            coverage = {"state": "captured_secondary_markets", "detail": f"{len(late_rows)} raw source rows retained; outcome rows preserve each offered side."}
        elif picks:
            preview = "Saved source cards are displayed as research only. They are not a current model run, not refreshed prices, and not wager approval."
            coverage = {"state": "saved_source_cards", "detail": f"{len(picks)} saved source cards available; no separate complete-market capture was supplied."}
        else:
            preview = "No researched player-market payload was supplied for this game at build time. This is a coverage gap, not a PASS or a fabricated analysis."
            coverage = {"state": "coverage_gap", "detail": "Awaiting a real game-card/market payload; no market rows were authored to fill the gap."}
        cards.append({"event_id": event_id, "game_id": gid, "away": away, "home": home, "kickoff": event["date"],
                      "source_as_of": "2026-10-10T17:23:00-04:00", "status": "completed" if finished else "upcoming",
                      "preview": preview, "winner_lean": "No current winner lean issued in this read-only publication.",
                      "context": {"venue": venue.get("fullName"), "indoor": venue.get("indoor"), "broadcast": event.get("broadcast")},
                      "coverage": coverage, "sources": event_sources, "picks": picks, "outcome": outcome})
    cards.sort(key=lambda c: c["kickoff"])
    manifest = {"schema": "fablesfable.week5.integration.v1", "season": 2026, "week": 5, "games": len(cards),
                "upcoming_games": sum(c["status"] == "upcoming" for c in cards), "completed_games": sum(c["status"] == "completed" for c in cards),
                "raw_market_rows": raw_count, "outcome_rows": len(rows), "sources": {"scoreboard": str(scoreboard_path), "saved_source_cards": str(early_hub_path), "late_markets": str(late_path), "extra": list(extra_paths)}}
    return cards, rows, manifest


def write_payload(cards, rows, manifest, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "game_cards.json").write_text(json.dumps(cards, indent=2) + "\n")
    (out / "market_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "integration-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def site_page(title, body):
    bridge = (ROOT / "published-site/assets/hub-scroll.js").read_text()
    css = "body{font:16px system-ui;max-width:1100px;margin:auto;padding:1rem;line-height:1.5;color:#182233}nav{display:flex;gap:12px;flex-wrap:wrap}a{color:#17529c}details{border:1px solid #d5dce6;border-radius:8px;margin:8px 0}summary{padding:10px;cursor:pointer;font-weight:600}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #d5dce6;text-align:left;vertical-align:top}input,select{min-height:38px;margin:4px}section{scroll-margin-top:8px}"
    nav = '<nav><a href="index.html">Week 5 board</a><a href="all-props.html">All player props</a><a href="model-cards.html">Saved model cards</a><a href="history.html">Archive</a></nav>'
    return f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head><body>{nav}{body}<script>{bridge}</script></body></html>'


def build_site(cards_path, rows_path, integration_manifest_path, archive, out, published_at):
    cards = read_json(cards_path); rows = read_json(rows_path); integration = read_json(integration_manifest_path)
    expected = {str(c["event_id"]) for c in cards}
    payload = all_props.load(str(cards_path), str(rows_path), expected_event_ids=expected)
    if payload["state"] != "ready":
        raise ValueError("refusing static build: " + "; ".join(payload["errors"]))
    payload["counts"].update({"raw_market_rows": integration["raw_market_rows"], "outcome_rows": integration["outcome_rows"]})
    archive, out = Path(archive), Path(out)
    if not archive.is_dir(): raise ValueError(f"archive not found: {archive}")
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(archive, out, ignore=shutil.ignore_patterns("publication.json"))
    board = all_props.render_page(payload)
    root = site_page("2026 Week 5 full slate", board)
    (out / "index.html").write_text(root)
    (out / "all-props.html").write_text(root)
    (out / "api/all-props.json").write_text(json.dumps(payload, indent=2) + "\n")
    hub = {"season": 2026, "week": 5, "generated_at": published_at, "cards": cards, "all_props": payload["counts"], "label": "research_snapshot", "approved_bets": 0}
    (out / "api/hub.json").write_text(json.dumps(hub, indent=2) + "\n")
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob("*") if p.is_file()}
    manifest = {"schema_version": 2, "kind": "saved-model-analysis", "generator": "scripts/build_week5_full_slate.py", "label": "research_snapshot", "season": 2026, "week": 5, "published_at": published_at, "approved_bets": 0, "model_candidates": 0, "all_props": payload["counts"], "integration": integration, "files": dict(sorted(files.items()))}
    (out / "publication.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    join = sub.add_parser("integrate")
    join.add_argument("--scoreboard", required=True); join.add_argument("--saved-source-cards", required=True)
    join.add_argument("--late-markets", required=True); join.add_argument("--out", required=True); join.add_argument("--extra", action="append", default=[])
    build = sub.add_parser("build")
    build.add_argument("--game-cards", required=True); build.add_argument("--market-rows", required=True); build.add_argument("--integration-manifest", required=True)
    build.add_argument("--archive", required=True); build.add_argument("--out", required=True); build.add_argument("--published-at", required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == "integrate":
            cards, rows, manifest = integrate(args.scoreboard, args.saved_source_cards, args.late_markets, args.extra)
            write_payload(cards, rows, manifest, args.out)
            print(f"integrated {manifest['games']} games; {manifest['raw_market_rows']} raw rows -> {manifest['outcome_rows']} outcome rows")
        else:
            manifest = build_site(args.game_cards, args.market_rows, args.integration_manifest, args.archive, args.out, args.published_at)
            print(f"built {args.out}: {len(manifest['files'])} files")
    except ValueError as exc:
        print(f"[week5-board] not built: {exc}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
