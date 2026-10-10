#!/usr/bin/env python3
"""Reprice immutable native candidate distributions at exact captured lines.

This adapter does not forecast, calibrate, request odds, or publish. It preserves all
native candidate distributions and records every exact supported comparison, including
quarantined offers. Current-roster status gates display only; they never alter a mean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "execution" / "fablesfable"))
from nflvalue.projection import p_over  # native probability helper; no surrogate math

TEAM_ALIASES = {"LA": "LAR", "LAR": "LAR", "WAS": "WSH", "WSH": "WSH"}
MARKET_ALIASES = {
    "carries": "rush_attempts", "rushing_attempts": "rush_attempts",
    "rushing_attempt": "rush_attempts", "attempts": "pass_attempts",
    "passing_attempts": "pass_attempts", "pass_attempt": "pass_attempts",
    "receiving_receptions": "receptions", "receptions": "receptions",
    "anytime_touchdown": "anytime_td", "touchdown_scorer": "anytime_td", "tdyes": "anytime_td",
}
SUPPORTED_MARKETS = {"receiving_yards", "receptions", "rushing_yards", "rush_attempts",
                     "passing_yards", "pass_attempts", "anytime_td"}
DISCRETE = {"negbinom", "poisson"}
ACTIVE_STATUSES = {"ACT", "ACTIVE"}
QUARANTINED_DISPOSITIONS = {"rejected", "unavailable", "unsupported"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_team(team: Any) -> str:
    text = str(team or "").strip().upper()
    return TEAM_ALIASES.get(text, text)


def norm_game(game_id: Any) -> str:
    parts = str(game_id or "").strip().upper().split("_")
    if len(parts) == 4:
        parts[2], parts[3] = norm_team(parts[2]), norm_team(parts[3])
    return "_".join(parts)


def norm_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("'", "").replace(".", " ").replace("-", " ")
    return " ".join(t for t in re.sub(r"[^a-z0-9 ]+", " ", text).split()
                    if t not in {"jr", "sr", "ii", "iii", "iv"})


def canonical_market(value: Any) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_")
    return MARKET_ALIASES.get(raw, raw)


def american_decimal(odds: Any) -> float | None:
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(odds) or odds == 0:
        return None
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def probabilities(candidate: dict[str, Any], line: float) -> dict[str, float | None]:
    try:
        mean, sd, dist = float(candidate["mean"]), float(candidate["sd"]), str(candidate["dist"])
        over = float(p_over(mean, sd, line, dist))
    except (KeyError, TypeError, ValueError):
        return {"over": None, "under": None, "push": None}
    push = 0.0
    if dist in DISCRETE and line.is_integer():
        push = max(0.0, min(1.0, float(p_over(mean, sd, line - 1.0, dist)) - over))
    return {"over": over, "under": max(0.0, 1.0 - over - push), "push": push}


def role_gate(candidate: dict[str, Any], roster_row: dict[str, Any] | None) -> tuple[bool, str, str]:
    """Use the current same-season/week roster as authorization, never an old name row."""
    if roster_row is None:
        return False, "current_roster_missing", "unsupported"
    status = str(roster_row.get("status") or "").upper()
    if status not in ACTIVE_STATUSES:
        # RES covers IR/PUP/NFI; INA and explicit OUT likewise fail closed.
        return False, f"current_roster_status_{status or 'unknown'}", "unavailable"
    if candidate.get("absence_qb_identity_state") == "blocked":
        return False, "blocked_native_qb_identity", "unsupported"
    if candidate.get("pos") == "QB":
        gate = candidate.get("qb_role_gate") or {}
        if gate.get("blocks_execution"):
            return False, str(gate.get("reason") or "blocked_native_qb_role"), "unsupported"
        try:
            depth = int(float(candidate.get("player_depth_rank")))
        except (TypeError, ValueError):
            return False, "missing_native_depth_rank", "unsupported"
        if depth != 1:
            return False, "qb_not_native_primary_depth_rank", "unsupported"
        return True, "active_current_roster_native_primary_qb_not_starter_assertion", "active_current_roster"
    if not candidate.get("eligible_for_shortlist"):
        return False, "native_shortlist_ineligible", "unsupported"
    if candidate.get("low_confidence"):
        return False, "native_low_confidence_market", "unsupported"
    try:
        depth = int(float(candidate.get("player_depth_rank")))
    except (TypeError, ValueError):
        return False, "missing_native_depth_rank", "unsupported"
    if depth not in {1, 2}:
        return False, f"native_depth_rank_{depth}_not_supported", "unsupported"
    return True, "active_current_roster", "active_current_roster"


def offer_admissibility(offer: dict[str, Any]) -> str:
    disposition = str(offer.get("disposition") or "").lower()
    status = str(offer.get("status") or "").lower()
    return "quarantined" if disposition in QUARANTINED_DISPOSITIONS or status in QUARANTINED_DISPOSITIONS else "admissible"


def football_explanation(candidate: dict[str, Any], offer: dict[str, Any], side: str, pside: float) -> str:
    market, mean, line = candidate["market"], float(candidate["mean"]), float(offer["line"])
    units = {"passing_yards": "passing yards", "rushing_yards": "rushing yards",
             "receiving_yards": "receiving yards", "receptions": "catches",
             "rush_attempts": "carries", "pass_attempts": "pass attempts"}
    if market == "anytime_td":
        return (f"Model leans {side}: the native touchdown distribution gives {pside:.0%} for at least one score "
                f"at the captured {offer['odds']:+.0f} price. The forecast reflects scoring opportunity and role, "
                "but this is not a calibrated touchdown probability.")
    direction = "above" if mean > line else "below"
    unit = units.get(market, market.replace("_", " "))
    return (f"Model leans {side}: the native forecast centers at {mean:.1f} {unit}, {direction} the {line:g} line "
            f"({pside:.0%} for this side). It combines expected opportunity with per-opportunity production and the opponent adjustment.")


def record_match(candidate: dict[str, Any], offer: dict[str, Any], full_name: str,
                 roster_row: dict[str, Any], model_run_as_of: str | None) -> dict[str, Any]:
    market = canonical_market(offer.get("market"))
    line = 0.5 if market == "anytime_td" else float(offer["line"])
    probs = probabilities(candidate, line)
    side = "over" if str(offer.get("side")).lower() in {"over", "yes"} else "under"
    pside, opposite, push = probs[side], probs["under" if side == "over" else "over"], probs["push"] or 0.0
    dec = american_decimal(offer.get("odds"))
    ev = (pside * dec + push - 1.0) if pside is not None and dec is not None else None
    breakeven = (1.0 - push) / dec if dec else None
    edge = pside - breakeven if pside is not None and breakeven is not None else None
    role_supported, role_reason, role_status = role_gate(candidate, roster_row)
    admissibility = offer_admissibility(offer)
    direction_defensible = pside is not None and opposite is not None and pside > opposite
    shortlist_eligible = bool(role_supported and admissibility == "admissible" and ev is not None and ev > 0 and direction_defensible)
    return {
        "candidate_player_id": candidate["player_id"], "player": full_name, "team": norm_team(candidate["team"]),
        "game_id": norm_game(candidate["game_id"]), "market": market, "side": side, "line": line,
        "book": offer.get("book"), "odds": offer.get("odds"), "quote_updated_at": offer.get("provider_updated_at"),
        "retrieved_at": offer.get("captured_at"), "model_run_as_of": model_run_as_of,
        "source_row_id": offer.get("source_row_id") or offer.get("raw_market_row_id"),
        "offer_admissibility": admissibility, "offer_disposition": offer.get("disposition"),
        "native_mean": candidate.get("mean"), "native_sd": candidate.get("sd"), "native_dist": candidate.get("dist"),
        "native_probability": pside, "p_over": probs["over"], "p_under": probs["under"], "p_push": push,
        "decimal_odds": dec, "breakeven_probability": breakeven, "raw_uncalibrated_edge": edge,
        "raw_uncalibrated_ev_per_unit": ev, "distribution_direction_defensible": direction_defensible,
        "role_supported": role_supported, "role_status": role_status, "role_reason": role_reason,
        "shortlist_eligible": shortlist_eligible,
        "display_why": football_explanation(candidate, {**offer, "line": line}, side, float(pside or 0.0)),
        "sources": offer.get("sources") if isinstance(offer.get("sources"), list) else [], "offer_reason": offer.get("reason"),
    }


def pick_from_match(m: dict[str, Any]) -> dict[str, Any]:
    role_risk = f"Current roster gate: {m['role_reason']}."
    return {
        "player": m["player"], "player_id": m["candidate_player_id"], "team": m["team"], "market": m["market"],
        "side": m["side"], "line": m["line"], "book": m["book"], "odds": m["odds"],
        "quote_updated_at": m["quote_updated_at"], "retrieved_at": m["retrieved_at"], "source_row_id": m["source_row_id"],
        "projection": m["native_mean"], "model_projection": m["native_mean"], "model_probability": m["native_probability"],
        "calibrated_probability": None, "model_run_as_of": m["model_run_as_of"], "status": "analyst_lean",
        "rationale": m["display_why"], "counterargument": f"{role_risk} Current role can change before kickoff; confirm the exact line and price remain available.",
        "invalidation": "Do not use after an inactive, OUT/IR designation, starter/role change, or moved/unverified line or price.",
        "rank_basis": "Exact native distribution at the captured line; positive raw EV only, not calibrated value.",
        "display_why": m["display_why"], "display_risk": f"{role_risk} Confirm the current role and price.",
        "role_status": m["role_status"], "role_supported": m["role_supported"], "role_reason": m["role_reason"],
        "native_model": True, "native_uncalibrated": True, "raw_uncalibrated_edge": m["raw_uncalibrated_edge"],
        "raw_uncalibrated_ev_per_unit": m["raw_uncalibrated_ev_per_unit"], "breakeven_probability": m["breakeven_probability"],
        "p_push": m["p_push"], "sources": m["sources"],
    }


def scoreboard_kickoffs(payload: Any) -> dict[str, str]:
    found: dict[str, str] = {}
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            event_id, date = value.get("id"), value.get("date")
            if event_id is not None and isinstance(date, str) and "T" in date:
                found[str(event_id)] = date
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(payload)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--offers", required=True, type=Path)
    ap.add_argument("--rosters", required=True, type=Path)
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--scoreboard", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    candidates = json.loads(args.candidates.read_text())
    offers = json.loads(args.offers.read_text())
    receipt = json.loads(args.receipt.read_text()) if args.receipt else {}
    model_run_as_of = (receipt.get("native_result") or {}).get("as_of") or receipt.get("completed_at")
    kickoffs = scoreboard_kickoffs(json.loads(args.scoreboard.read_text()))

    roster = pd.read_parquet(args.rosters)
    # nflverse's authoritative current roster asset calls the GSIS identifier gsis_id;
    # retain player_id support only for normalized local test/receipt exports.
    if "player_id" not in roster.columns and "gsis_id" in roster.columns:
        roster = roster.rename(columns={"gsis_id": "player_id"})
    required = {"season", "week", "team", "position", "player_id", "full_name", "status"}
    missing = required - set(roster.columns)
    if missing:
        raise ValueError(f"roster source lacks required current-status columns: {sorted(missing)}")
    roster_by_key = {}
    for row in roster.dropna(subset=["player_id", "full_name"]).itertuples(index=False):
        roster_by_key[(int(row.season), int(row.week), str(row.player_id))] = {
            "full_name": str(row.full_name), "team": norm_team(row.team), "position": str(row.position),
            "status": str(row.status).upper() if row.status is not None else "", "season": int(row.season), "week": int(row.week),
        }

    named_offers = [o for o in offers if o.get("player") != "NO_CAPTURED_NAMED_OFFER"
                    and canonical_market(o.get("market")) in SUPPORTED_MARKETS
                    and str(o.get("side")).lower() in {"over", "under", "yes"}
                    and (o.get("line") is not None or canonical_market(o.get("market")) == "anytime_td")]
    source_quarantine = [o for o in offers if offer_admissibility(o) == "quarantined"]
    offer_idx: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for offer in named_offers:
        offer_idx[(norm_game(offer.get("game_id")), norm_team(offer.get("team")), norm_name(offer.get("player")),
                   canonical_market(offer.get("market")))].append(offer)

    audit, matches = [], []
    forecasts: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for candidate in candidates:
        game_id, team = norm_game(candidate["game_id"]), norm_team(candidate["team"])
        target = (int(candidate["season"]), int(candidate["week"]), str(candidate["player_id"]))
        roster_row = roster_by_key.get(target)
        identity_reason = None
        if roster_row is None:
            identity_reason = "no_current_season_week_roster_id_match"
        elif roster_row["team"] != team:
            identity_reason = f"current_roster_team_mismatch:{roster_row['team']}!=native:{team}"
        full_name = roster_row["full_name"] if roster_row and not identity_reason else None
        role_supported, role_reason, role_status = role_gate(candidate, roster_row if not identity_reason else None)
        forecast_key = (str(candidate["player_id"]), team)
        if forecast_key not in forecasts[game_id]:
            forecasts[game_id][forecast_key] = {
                "name": full_name or candidate["name"], "player": full_name or candidate["name"], "player_id": candidate["player_id"],
                "team": team, "position": candidate["pos"], "pos": candidate["pos"], "means": {},
                "model_run_as_of": model_run_as_of, "role_status": role_status, "role_supported": role_supported,
                "role_reason": role_reason, "native_model": True,
            }
        forecast = forecasts[game_id][forecast_key]
        # Role is a player-level display field. A low-confidence TD market must not
        # overwrite an active player's supported rushing/receiving role.
        if role_supported or not forecast["role_supported"]:
            forecast.update(role_status=role_status, role_supported=role_supported, role_reason=role_reason)
        forecast["means"][candidate["market"]] = candidate.get("mean")
        if candidate["market"] == "anytime_td":
            forecast["native_td_probability"] = probabilities(candidate, .5)["over"]
            forecast["anytime_td_p_ge_1"] = forecast["native_td_probability"]

        base = {"raw_candidate": candidate, "game_id": game_id, "team": team, "player": full_name or candidate["name"],
                "identity_status": "validated" if identity_reason is None else "held", "identity_reason": identity_reason,
                "role_supported": role_supported, "role_status": role_status, "role_reason": role_reason,
                "matched_offers": [], "coverage": None, "suppressed_reason": None}
        if identity_reason:
            base.update(coverage="noquote", suppressed_reason="identity_unresolved:" + identity_reason)
            audit.append(base)
            continue
        matched_offers = offer_idx.get((game_id, team, norm_name(full_name), candidate["market"]), [])
        if not matched_offers:
            base.update(coverage="noquote", suppressed_reason="no_matching_supported_captured_offer")
            audit.append(base)
            continue
        priced = [record_match(candidate, offer, full_name, roster_row, model_run_as_of) for offer in matched_offers]
        base.update(coverage="priced", matched_offers=priced, suppressed_reason=None if role_supported else role_reason)
        audit.append(base)
        matches.extend(priced)

    market_best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for match in matches:
        if not match["shortlist_eligible"]:
            continue
        key = (match["game_id"], match["candidate_player_id"], match["market"])
        old = market_best.get(key)
        if old is None or match["raw_uncalibrated_ev_per_unit"] > old["raw_uncalibrated_ev_per_unit"]:
            market_best[key] = match

    game_ids = sorted({norm_game(c["game_id"]) for c in candidates} | {norm_game(o.get("game_id")) for o in offers})
    cards = []
    for game_id in game_ids:
        parts = game_id.split("_")
        away, home = (parts[2], parts[3]) if len(parts) == 4 else (None, None)
        event_id = next((str(o.get("event_id")) for o in offers if norm_game(o.get("game_id")) == game_id and o.get("event_id") is not None), None)
        picks = [pick_from_match(m) for key, m in market_best.items() if key[0] == game_id]
        picks.sort(key=lambda p: -(p["raw_uncalibrated_ev_per_unit"] or -999))
        game_audit, game_matches = [a for a in audit if a["game_id"] == game_id], [m for m in matches if m["game_id"] == game_id]
        cards.append({
            "event_id": event_id, "game_id": game_id, "away": away, "home": home, "kickoff": kickoffs.get(event_id),
            "model_run_as_of": model_run_as_of, "native_model": True, "status": "upcoming",
            "source_as_of": max((str(o.get("captured_at")) for o in offers if norm_game(o.get("game_id")) == game_id and o.get("captured_at")), default=None),
            "preview": "Native forecasts are shown with current roster-status gates. Model leans are research only.",
            "winner_lean": None, "context": {"native_role_gate": "Current same-week roster status gates availability; primary-QB status is not a starter confirmation.",
                                          "quote_limit": "Captured secondary listings; confirm every current line and price."},
            "coverage": {"native_candidate_rows": len(game_audit), "native_players": len(forecasts.get(game_id, {})),
                         "priced_offer_rows": len(game_matches), "noquote_native_rows": sum(a["coverage"] == "noquote" for a in game_audit),
                         "quarantined_offer_rows": sum(m["offer_admissibility"] == "quarantined" for m in game_matches),
                         "displayed_model_leans": len(picks), "validated_bets": 0},
            "sources": [{"provider": "captured sportsbook-labelled offers", "path": str(args.offers), "sha256": sha256(args.offers)},
                        {"provider": "immutable native candidates", "path": str(args.candidates), "sha256": sha256(args.candidates)},
                        {"provider": "recorded ESPN scoreboard", "path": str(args.scoreboard), "sha256": sha256(args.scoreboard)}],
            "picks": picks, "player_forecasts": sorted(forecasts.get(game_id, {}).values(), key=lambda x: (x["team"], x["name"])),
            "no_pick_reason": "No model lean passed the current-role, captured-offer, positive-raw-EV, and distribution-direction gates." if not picks else "No validated/model-approved wagers.",
        })

    # Exact self-checks: every native row is preserved and no unsafe record is promoted.
    assert len(audit) == len(candidates), "every native candidate must be audited"
    assert Counter(a["raw_candidate"]["market"] for a in audit) == Counter(c["market"] for c in candidates)
    assert all(m["raw_uncalibrated_ev_per_unit"] is None or isinstance(m["native_probability"], float) for m in matches)
    picks = [p for card in cards for p in card["picks"]]
    assert all(p["raw_uncalibrated_ev_per_unit"] > 0 for p in picks)
    assert all(m["offer_admissibility"] == "admissible" for m in market_best.values())
    assert all(m["role_supported"] for m in market_best.values())
    assert all(not (m["market"] in {"passing_yards", "pass_attempts"} and "qb_not_native_primary" in m["role_reason"])
               for m in market_best.values())

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "game_cards.json").write_text(json.dumps(cards, indent=2, allow_nan=False) + "\n")
    (args.out / "native_repricing_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    (args.out / "matched_rows.json").write_text(json.dumps(matches, indent=2, allow_nan=False) + "\n")
    (args.out / "offer_quarantine_audit.json").write_text(json.dumps(source_quarantine, indent=2, allow_nan=False) + "\n")
    summary = {"schema": "native_exact_line_repricing.v2", "generated_at": datetime.now(timezone.utc).isoformat(),
               "inputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in
                          {"candidates": args.candidates, "offers": args.offers, "rosters": args.rosters, "scoreboard": args.scoreboard}.items()},
               "model_run_as_of": model_run_as_of, "native_candidate_count": len(candidates),
               "native_candidate_market_counts": dict(Counter(c["market"] for c in candidates)),
               "outcomes": {"priced_offer_rows": len(matches), "native_noquote_rows": sum(a["coverage"] == "noquote" for a in audit),
                            "quarantined_source_offer_rows": len(source_quarantine),
                            "quarantined_matched_offer_rows": sum(m["offer_admissibility"] == "quarantined" for m in matches),
                            "displayed_model_leans": len(picks), "validated_bets": 0},
               "self_checks": {"all_native_candidate_coverage": len(audit) == len(candidates),
                               "all_market_families_preserved": True, "positive_ev_only": all(p["raw_uncalibrated_ev_per_unit"] > 0 for p in picks),
                               "quarantined_offers_not_promoted": (all(m["offer_admissibility"] == "admissible" for m in market_best.values())
                                                                   and not any(p.get("source_row_id") in {o.get("source_row_id") for o in source_quarantine} for p in picks)),
                               "no_backup_qb_promoted": all("qb_not_native_primary" not in m["role_reason"] for m in market_best.values()),
                               "native_probability_recomputed_at_exact_line": True},
               "notes": ["No publication, odds request, or model rerun occurred.", "Candidate means and distributions are immutable input.",
                         "Current same-season/week roster status is required; reserve/inactive/unknown statuses fail closed."]}
    (args.out / "repricing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["outcomes"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
