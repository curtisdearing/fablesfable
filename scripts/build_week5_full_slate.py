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
    # Captured labels end in position and team (for example ``RJ Harvey RB DEN``).
    return (" ".join(chunks[:-2]) or str(label), chunks[-1] if len(chunks) >= 2 else "UNK")


def market_name(label):
    return {"Receiving yards": "receiving_yards", "Rushing yards": "rushing_yards",
            "Passing yards": "passing_yards", "Receptions": "receptions",
            "Anytime TD": "anytime_touchdown", "Longest reception": "longest_reception",
            "Longest rush": "longest_rush"}.get(label, str(label).lower().replace(" ", "_"))


def source(title, url):
    return {"title": title, "url": url}


LATE_GAME_ANALYSIS = {
    "DEN_LAC": {
        "preview": "The captured DEN–LAC screen covers 79 public secondary-listing rows. It is a player-prop review, not a game forecast: the retained analysis distinguishes role evidence from conversion and does not treat a posted line as value.",
        "winner_lean": "No game-winner lean was issued: the retained late research assessed player markets, not a side or moneyline.",
        "context": {"countercase": "Payton retaking play-calling, reported receiver absences, a three-back rotation, and offensive-line absences make static workload assumptions fragile."},
    },
    "DET_ARI": {
        "preview": "The captured DET–ARI screen covers 76 public secondary-listing rows. The retained assessment prefers selected catch markets to some yardage/TD comparisons, while keeping target redistribution and game-script uncertainty explicit.",
        "winner_lean": "No game-winner lean was issued: the retained late research assessed player markets, not a side or moneyline.",
        "context": {"countercase": "Detroit receiver outcomes are correlated; a leading script can reduce pass volume, while Harrison's reported absence may concentrate targets without quantifying the allocation."},
    },
    "SF_SEA": {
        "preview": "The captured SF–SEA screen covers 95 public secondary-listing rows. The retained assessment uses observed matching stat appearances only; missing appearances are not imputed as zero and no calibrated probability is claimed.",
        "winner_lean": "No game-winner lean was issued: the retained late research assessed player markets, not a side or moneyline.",
        "context": {"countercase": "Darnold has only two recent full starts, JSN's first two games were largely with Lock, and a trailing/checkdown script can reverse a descriptive receptions case."},
    },
}

LATE_PICK_COUNTERCASES = {
    ("RJ Harvey", "Receiving yards"): "Missing Week 2 is not assumed zero. Pat Bryant/Coleman absences and a new play-caller may alter allocation; a Denver lead can reduce receiving volume.",
    ("Bo Nix", "Rushing yards"): "Payton retaking play-calling creates usage uncertainty; low rushing production in all four observed games is not a stable role forecast.",
    ("Amon-Ra St. Brown", "Receptions"): "A leading script or redistribution can reduce catch volume; this is not a reason to stack correlated Detroit receiver overs.",
    ("Jahmyr Gibbs", "Receptions"): "The receiving case is correlated with other Lions target outcomes; expensive TD/rushing markets and game script remain countercases.",
    ("Trey McBride", "Receptions"): "Harrison's reported absence can concentrate targets but is not quantified; the high catch threshold already reflects opportunity.",
    ("Jaxon Smith-Njigba", "Receptions"): "The first two games were largely with Drew Lock; the last two full Darnold games were 10 and 5 catches on 14 and 6 targets, not a four-game chemistry trend.",
    ("George Kittle", "Receiving yards"): "Explosive-play dependence and coverage risk can overturn the descriptive yardage case.",
    ("Christian McCaffrey", "Receptions"): "A trailing/checkdown script can reverse the descriptive under; shared touches and the Seattle matchup remain material uncertainty.",
}


def is_game_card(item):
    return isinstance(item, dict) and {"event_id", "game_id", "away", "home", "kickoff", "picks"} <= set(item)


def is_market_row(item):
    return isinstance(item, dict) and {"event_id", "game_id", "player", "market", "period", "status", "disposition"} <= set(item)


def source_clock(card, fallback):
    if card and card.get("source_as_of"):
        return card["source_as_of"]
    candidates = [r.get("captured_at") or r.get("retrieved_at") for r in fallback if isinstance(r, dict)]
    return max((str(v) for v in candidates if v), default="Unavailable (no source capture clock supplied)")


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
    extra_cards, extra_rows = [], []
    for path in extra_paths:
        doc = read_json(path)
        if not isinstance(doc, list):
            raise ValueError(f"extra payload must be a top-level list: {path}")
        for item in doc:
            if is_game_card(item):
                extra_cards.append(item)
            elif is_market_row(item):
                extra_rows.append(item)
            else:
                raise ValueError(f"extra payload item is neither a typed game card nor market row: {path}")
    # Saved native cards are deliberately audit-only.  They are old model snapshots,
    # not current issued picks, and must not leak into refreshed manual cards.
    if any(not isinstance(row, dict) for row in saved):
        raise ValueError("saved source cards must contain objects")
    cards_by_event = {str(c["event_id"]): c for c in extra_cards}
    cards_by_game = {c["game_id"]: c for c in extra_cards}
    late_by_game = {}
    for raw in late:
        if not isinstance(raw, dict) or not raw.get("game"):
            raise ValueError("late market rows must be objects with game")
        late_by_game.setdefault(raw["game"], []).append(raw)
    cards, rows = [], list(extra_rows)
    raw_count = 0
    for event in scoreboard["events"]:
        comp = event["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        away, home = teams["away"]["team"]["abbreviation"], teams["home"]["team"]["abbreviation"]
        event_id, gid = str(event["id"]), slug_game(away, home)
        finished = bool(event.get("status", {}).get("type", {}).get("completed"))
        venue = comp.get("venue", {})
        event_sources = [source("ESPN Week 5 scoreboard", "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20261011&seasontype=2&week=5")]
        supplied = cards_by_event.get(event_id) or cards_by_game.get(gid)
        picks = list((supplied or {}).get("picks") or [])
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
                    # A categorical TD listing has one ``yes`` outcome, not an
                    # invented O/U pair.  The second loop pass has no distinct offer.
                    if field != "Best over":
                        continue
                    side = "yes"
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
                source_row_id = f"late:{raw.get('game')}:{raw.get('player_label')}:{raw.get('Prop')}:{raw.get('Line')}"
                rows.append({"event_id": event_id, "game_id": gid, "player": player, "team": team, "market": prop,
                             "side": side, "period": "full_game", "status": status, "disposition": outcome,
                             "line": None if side == "yes" else line, "odds": odds, "book": book,
                             "captured_at": raw.get("retrieved_at"), "provider_updated_at": None,
                             "reason": raw.get("rationale") or raw.get("history_note"), "sources": [source("Secondary published listing", raw.get("source"))],
                             "source_row_id": source_row_id, "raw_source_row": raw_record,
                             "quote_verification": raw.get("quote_verification")})
                if selected_here:
                    picks.append({"player": player, "player_id": None, "team": team, "market": prop, "side": side,
                                  "line": None if side == "yes" else line, "book": book, "odds": odds,
                                  "quote_updated_at": None, "retrieved_at": raw.get("retrieved_at"), "projection": None,
                                  "model_probability": None, "calibrated_probability": None, "model_run_as_of": None,
                                  "status": "analyst_lean", "rationale": raw.get("rationale") or "Conditional analyst lean.",
                                  "counterargument": LATE_PICK_COUNTERCASES.get((player, raw.get("Prop")), "Secondary listing and current role/price have not been independently reverified."),
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
        late_analysis = LATE_GAME_ANALYSIS.get(f"{away}_{home}", {})
        supplied = supplied or {}
        base_context = {"venue": venue.get("fullName"), "indoor": venue.get("indoor"), "broadcast": event.get("broadcast")}
        base_context.update(late_analysis.get("context", {}))
        base_context.update(supplied.get("context") or {})
        card = {"event_id": event_id, "game_id": gid, "away": away, "home": home, "kickoff": event["date"],
                "source_as_of": source_clock(supplied, late_rows), "status": "completed" if finished else "upcoming",
                "preview": supplied.get("preview") or late_analysis.get("preview") or preview,
                "winner_lean": supplied.get("winner_lean") or late_analysis.get("winner_lean") or "No current winner lean issued in this read-only publication.",
                "context": base_context, "coverage": supplied.get("coverage") or coverage,
                "sources": supplied.get("sources") or event_sources, "picks": [] if finished else picks, "outcome": outcome}
        cards.append(card)
    cards.sort(key=lambda c: c["kickoff"])
    priced_extra = [r for r in extra_rows if r.get("raw_market_row_type") not in {"family_availability", "coverage_gap"}]
    extra_families = sum(r.get("raw_market_row_type") == "family_availability" for r in extra_rows)
    extra_gaps = sum(r.get("raw_market_row_type") == "coverage_gap" for r in extra_rows)
    early_raw_ids = {r.get("raw_market_row_id") or r.get("source_row_id") for r in priced_extra if r.get("raw_market_row_id") or r.get("source_row_id")}
    raw_total = raw_count + len(early_raw_ids)
    outcome_total = sum(r.get("raw_market_row_type") not in {"family_availability", "coverage_gap"} for r in rows)
    manifest = {"schema": "fablesfable.week5.integration.v1", "season": 2026, "week": 5, "games": len(cards),
                "upcoming_games": sum(c["status"] == "upcoming" for c in cards), "completed_games": sum(c["status"] == "completed" for c in cards),
                "raw_market_rows": raw_total, "outcome_rows": outcome_total, "coverage_family_rows": extra_families,
                "coverage_gap_rows": extra_gaps, "all_row_records": len(rows),
                "sources": {"scoreboard": str(scoreboard_path), "saved_source_cards": str(early_hub_path), "late_markets": str(late_path), "extra": list(extra_paths)}}
    return cards, rows, manifest


def write_payload(cards, rows, manifest, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "game_cards.json").write_text(json.dumps(cards, indent=2) + "\n")
    (out / "market_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "integration-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def site_page(title, body):
    bridge = (ROOT / "published-site/assets/hub-scroll.js").read_text()
    css = "body{font:16px/1.5 system-ui;max-width:1100px;margin:auto;padding:1rem;color:#182233;overflow-wrap:anywhere}nav{display:flex;gap:8px 12px;flex-wrap:wrap}nav a{min-height:44px;display:inline-flex;align-items:center}a{color:#17529c}details{border:1px solid #d5dce6;border-radius:8px;margin:12px 0}summary{padding:10px;cursor:pointer;font-weight:600}.board-caveat{max-width:72ch}.picks-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.pick-card,.ap-game{border:1px solid #d5dce6;border-radius:10px;padding:16px;background:#fff}.pick-card h3{margin:.2rem 0}.pick-game,.game-time,.pick-price,.no-pick{margin:.25rem 0;color:#4a596d}.pick-price,.model-comparison{font-weight:700;color:#182233}.model-disclaimer{margin:.3em 0;color:#4a596d;font-size:13px}.pick-risk{border-left:3px solid #9b4d00;padding-left:10px}.factor-warning{margin:10px;padding:10px;border-left:3px solid #9b4d00;background:#fff8ee}.game-jumps{margin:10px 0 16px}.game-jumps a{border:1px solid #d5dce6;border-radius:999px;padding:5px 10px;text-decoration:none}.ap-game{margin:10px 0;scroll-margin-top:8px}.ap-game h3{margin:0}.player-projections{margin-top:14px}.projection-scroll{overflow-x:auto}.projection-scroll table{border-collapse:collapse;min-width:760px;width:100%}.projection-scroll th,.projection-scroll td{border-bottom:1px solid #d5dce6;padding:8px;text-align:left;white-space:nowrap}.projection-scroll th{background:#f6f8fb}.research-download{margin:24px 0}@media(max-width:540px){body{padding:.75rem}.pick-card,.ap-game{padding:14px}.picks-grid{grid-template-columns:minmax(0,1fr)}}"
    nav = '<nav><a href="index.html">Week 5 board</a><a href="all-props.html">All player props</a><a href="model-cards.html">Saved model cards</a><a href="history.html">Archive</a></nav>'
    return f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head><body>{nav}{body}<script>{bridge}</script></body></html>'


def build_site(cards_path, rows_path, integration_manifest_path, archive, out, published_at):
    cards = read_json(cards_path); rows = read_json(rows_path); integration = read_json(integration_manifest_path)
    expected = {str(c["event_id"]) for c in cards}
    payload = all_props.load(str(cards_path), str(rows_path), expected_event_ids=expected)
    if payload["state"] != "ready":
        raise ValueError("refusing static build: " + "; ".join(payload["errors"]))
    payload["counts"].update({"raw_market_rows": integration["raw_market_rows"], "outcome_rows": integration["outcome_rows"],
                              "coverage_family_rows": integration.get("coverage_family_rows", 0),
                              "coverage_gap_rows": integration.get("coverage_gap_rows", 0)})
    archive, out = Path(archive), Path(out)
    if not archive.is_dir(): raise ValueError(f"archive not found: {archive}")
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(archive, out, ignore=shutil.ignore_patterns("publication.json"))
    board = all_props.render_page(payload)
    root = site_page("2026 Week 5 full slate", board)
    (out / "index.html").write_text(root)
    (out / "all-props.html").write_text(root)
    (out / "api/all-props.json").write_text(json.dumps(payload, indent=2) + "\n")
    # api/hub.json is the native dashboard contract.  Copy it unchanged from the
    # archived public site; this research board lives at api/all-props.json.
    if not (out / "api/hub.json").is_file():
        raise ValueError("archive is missing native api/hub.json")
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
