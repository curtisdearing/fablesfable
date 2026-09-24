"""Evidence cards for board rows: what is offered, what the model says, and why it is not a bet.

Every card states exact player/market/side/line, the ONE executable quote the
lean was priced at (book, decimal price, capture clock), the run that produced
it (run id, code commit, forecast version, ranker hash), the model's P(side)
with its validation status, breakeven, rationale, countercase, invalidation
conditions and a status:

* ``actionable`` -- only for a market in ``VALIDATED_MARKETS``.  Empty: no
  market has passed an offered-line calibration gate (2026 wk1-2 settled
  exact lines: model Brier 0.263 vs coin 0.250 vs market 0.248), so no card
  can be actionable.  ``publish=True`` elsewhere is software permission, not
  validated edge.
* ``watch``    -- one verified quote, fresh, with a coherent forecast.
* ``research`` -- no offered price (synthetic line), or the player's availability was not
  established by the issuing run (unknown is neither healthy nor ruled out, so the
  forecast stays visible but the quote is not treated as executable).
* ``pass``     -- voided, stale/future/unrecorded quote, quote not found in
  the captured lines, or an invalid forecast/price/side/line.

Provenance and quote identity are what the pipeline persisted with the lean
(``report.persist_leans``, db migration 4). Rows written before that carry
neither; they stay "unknown" and are never matched after the fact by name or
timestamp proximity. The gates are fixed here, not tuned to results.
"""

from __future__ import annotations

import datetime as dt
import html
import math
from typing import Dict, Iterable, List, Optional

VALIDATED_MARKETS: frozenset = frozenset()
STALE_QUOTE_HOURS = 6.0
FUTURE_TOLERANCE_MIN = 5.0
FOOTBALL_ONLY_PREFIX = "ff-football-only-"
VALIDATION_NOTE = ("Model probability is NOT validated as calibrated at offered lines: on 2026 "
                   "weeks 1-2 settled exact lines (307 events, 20 games) it scored Brier 0.263 "
                   "vs coin 0.250 and market consensus 0.248.")


def _f(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _american(dec: Optional[float]) -> Optional[str]:
    if dec is None or dec <= 1:
        return None
    return f"+{round((dec - 1) * 100)}" if dec >= 2 else f"{round(-100 / (dec - 1))}"


def _ts(s) -> Optional[dt.datetime]:
    if s is None or (isinstance(s, float) and math.isnan(s)) or not str(s).strip():
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _s(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    x = str(x).strip()
    return x or None


def _run_hold_reason(row: Dict) -> str:
    """Why the pick's own issuing run did not permit publication (e.g. a T-90 refresh held
    because the inactives feed could not be fetched)."""
    if row.get("_run_publish") is None:
        return "issuing run's publication decision not recorded (not treated as permitted)"
    why = "; ".join(str(r) for r in (row.get("_run_publish_reasons") or []))[:300]
    return f"issuing run held publication ({why or 'no reason recorded'})"


def _availability_hold(row: Dict) -> Optional[str]:
    """Why this pick's own availability blocks execution, from what its run persisted."""
    import json
    try:
        stamps = json.loads(row.get("stage_json") or "null") or {}
    except (TypeError, ValueError):
        stamps = {}
    a = stamps.get("availability")
    if not a:
        return "player availability not recorded by the issuing run (missing, not healthy)"
    if a.get("eligibility") == "degraded":
        return (f"player availability not established ({a.get('availability_state')}): "
                "neither confirmed healthy nor ruled out")
    return None


def build_card(row: Dict, now: dt.datetime) -> Dict:
    side = (_s(row.get("side")) or "").lower()
    line, price = _f(row.get("line")), _f(row.get("price"))
    mean, sd, p = _f(row.get("mean")), _f(row.get("sd")), _f(row.get("p_side"))
    book, quote_ts = _s(row.get("quote_book")), _ts(row.get("quote_ts"))
    age_h = (now - quote_ts).total_seconds() / 3600 if quote_ts else None
    offered = row.get("line_source") == "odds_api"
    version = _s(row.get("forecast_version"))
    prov = {"run_id": _s(row.get("run_id")) or "unknown (not recorded)",
            "code_sha": _s(row.get("code_sha")) or "unknown (not recorded)",
            "forecast_version": version or "unknown (not recorded)",
            "ranker_sha256": _s(row.get("ranker_sha256")),
            "selection_source": _s(row.get("selection_source")) or "unknown (not recorded)"}

    reasons: List[str] = []
    invalid = []
    if side not in ("over", "under"):
        invalid.append(f"invalid side {side!r}")
    if line is None:
        invalid.append("line missing or non-finite")
    if mean is None or mean < 0:
        invalid.append("mean missing or negative")
    if sd is None or sd <= 0:
        invalid.append("sd missing or non-positive")
    if p is None or not 0.0 <= p <= 1.0:
        invalid.append("model probability missing or outside [0, 1]")
    if _s(row.get("status")) == "voided":
        status = "pass"
        reasons.append(f"voided: {_s(row.get('void_reason')) or 'unspecified'}")
    elif invalid:
        status = "pass"
        reasons.extend(invalid)
    elif not offered:
        status = "research"
        reasons.append("no offered price: synthetic line, not a wager")
    elif price is None or price <= 1.0:
        status = "pass"
        reasons.append("offered price missing or not a valid decimal price")
    elif not book or "/" in book:
        status = "pass"
        reasons.append("no single executable quote recorded for this side")
    elif quote_ts is None:
        status = "pass"
        reasons.append("quote clock not recorded")
    elif age_h < -FUTURE_TOLERANCE_MIN / 60:
        status = "pass"
        reasons.append("quote clock is in the future")
    elif not row.get("_quote_verified"):
        status = "pass"
        reasons.append("quote (book, side, line, price, clock) not found in captured lines")
    elif age_h > STALE_QUOTE_HOURS:
        status = "pass"
        reasons.append(f"quote is {age_h:.1f} h old (> {STALE_QUOTE_HOURS:.0f} h)")
    elif "_run_publish" in row and row["_run_publish"] is not True:
        status = "research"
        reasons.append(_run_hold_reason(row))
    elif _availability_hold(row):
        status = "research"
        reasons.append(_availability_hold(row))
    elif row.get("market") in VALIDATED_MARKETS:
        status = "actionable"
    else:
        status = "watch"
        reasons.append("market has not passed an offered-line calibration gate")

    executable = status in ("watch", "actionable")
    football_only = bool(version and version.startswith(FOOTBALL_ONLY_PREFIX))
    if mean is not None and line is not None and sd is not None:
        label = "Football-only projection" if football_only else "Projection (version not recorded)"
        rationale = (f"{label} {mean:.1f} (sd {sd:.1f}) sits {abs(mean - line):.1f} "
                     f"{'above' if mean > line else 'below'} the {line:g} line.")
    else:
        rationale = "No valid projection."
    if _s(row.get("reason")):
        rationale += f" Model drivers: {row['reason']}"
    breakeven = 1 / price if executable and price else None
    invalidation = [
        "any injury designation or inactive listing after the quote clock "
        "(no matched injury row is NOT verified health)",
        f"line moves to the other side of the projection ({mean:.1f})" if mean is not None else "line moves",
        (f"price shortens so breakeven exceeds the model's {p:.0%}" if p is not None and 0 <= p <= 1
         else "price moves"),
        f"quote older than {STALE_QUOTE_HOURS:.0f} h at decision time",
    ]
    return {
        "player": _s(row.get("name")), "player_id": _s(row.get("player_id")),
        "game_id": _s(row.get("game_id")), "market": _s(row.get("market")), "side": side, "line": line,
        "quote": ({"book": book, "price_decimal": price, "price_american": _american(price),
                   "captured_at": row.get("quote_ts")} if executable else None),
        "quote_age_hours": round(age_h, 2) if age_h is not None else None,
        "run_as_of": _s(row.get("as_of")), "clock": _s(row.get("clock")),
        "provenance": prov, "mean": mean, "sd": sd,
        "model_p_side": p, "model_p_status": "unvalidated_at_offered_lines",
        "breakeven": round(breakeven, 4) if breakeven else None,
        "ev_per_unit_unvalidated": round(p * price - 1, 4) if executable and p is not None else None,
        "value_composite_display_only": _f(row.get("composite")),
        "value_composite_note": "includes the price edge; computed after selection; never selects",
        "rationale": rationale,
        "countercase": VALIDATION_NOTE,
        "invalidation": invalidation,
        "status": status, "status_reasons": reasons,
        # built from what this lean's run persisted (stage stamps, receipt, shadow, context);
        # absent when the caller did not load them -- never a synthetic "no change"
        "factor_panel": row.get("_factor_panel"),
    }


_ORDER = {"actionable": 0, "watch": 1, "research": 2, "pass": 3}


def build_cards(rows: Iterable[Dict], now: Optional[dt.datetime] = None) -> List[Dict]:
    now = now or dt.datetime.now(dt.timezone.utc)
    cards = [build_card(r, now) for r in rows]
    return sorted(cards, key=lambda c: (_ORDER[c["status"]], c["game_id"] or "", c["player"] or "",
                                        c["market"] or ""))


def verify_quotes(conn, leans: List[Dict]) -> None:
    """Mark ``_quote_verified`` when the lean's exact quote row exists in ``lines``.

    Exact means same game, book, market, side, point, decimal price and capture
    clock. Player identity is the one the authoritative quote matcher assigned
    when the board was priced (``oddsapi_props.match_player_ids``); nothing here
    re-matches by name.
    """
    for r in leans:
        r["_quote_verified"] = False
        book, ts = _s(r.get("quote_book")), _s(r.get("quote_ts"))
        if not book or not ts or "/" in book:
            continue
        hit = conn.execute(
            "SELECT COUNT(*) FROM lines WHERE game_id=? AND book=? AND market=? AND lower(side)=? "
            "AND point=? AND price=? AND ts=?",
            (r.get("game_id"), book, r.get("market"), str(r.get("side") or "").lower(),
             _f(r.get("line")), _f(r.get("price")), ts)).fetchone()[0]
        r["_quote_verified"] = hit >= 1


def week_cards(conn, season: int, week: int, now: Optional[dt.datetime] = None) -> Dict:
    """Cards for the newest clock of each (game, player, market) lean this week."""
    from . import db as dbmod
    now = now or dt.datetime.now(dt.timezone.utc)
    leans = dbmod.query_df(conn, "SELECT * FROM leans WHERE season=? AND week=?", (season, week))
    rows: List[Dict] = []
    if not leans.empty:
        leans = (leans.assign(_t=leans["clock"].map({"t90": 1}).fillna(0))
                 .sort_values(["_t", "created_at"])
                 .drop_duplicates(["game_id", "player_id", "market"], keep="last"))
        rows = leans.to_dict("records")
        verify_quotes(conn, rows)
        from . import factor_integration as fimod
        receipts = fimod.load_receipts(conn, season, week)
        context = fimod.load_context_records(conn, season, week)
        for r in rows:
            r["_factor_panel"] = fimod.card_panel(r, receipts, context)
            rc = receipts.get(r.get("run_id")) or {}
            r["_run_publish"] = rc.get("publish")          # None: not recorded -> not executable
            r["_run_publish_reasons"] = rc.get("publish_reasons")
    cards = build_cards(rows, now=now)
    return {"season": season, "week": week, "generated_at": now.isoformat(timespec="seconds"),
            "validated_markets": sorted(VALIDATED_MARKETS),
            "counts": {s: sum(c["status"] == s for c in cards) for s in _ORDER},
            "cards": cards}


def _panel_html(panel: Optional[Dict]) -> str:
    if not panel:
        return "<div class=muted>Factor evidence: not loaded for this card.</div>"
    if panel.get("withheld"):
        return f"<div class=muted>Factor evidence: {html.escape(panel['withheld'])}</div>"
    from . import factor_evidence as fe
    return fe.render_panel_html(panel)


def render_cards_html(cards: List[Dict]) -> str:
    e = lambda x: html.escape("" if x is None else str(x))
    parts = []
    for c in cards:
        q = c["quote"]
        quote = (f"{e(q['book'])} {e(q['price_american'])} ({e(q['price_decimal'])}), captured {e(q['captured_at'])}"
                 if q else "no executable quote")
        p = f"{c['model_p_side']:.1%}" if c["model_p_side"] is not None and 0 <= c["model_p_side"] <= 1 else "n/a"
        pv = c["provenance"]
        parts.append(
            f"<div class=card><div><span class=s>{e(c['status'])}</span> &middot; "
            f"<b>{e(c['player'])}</b> {e(c['market'])} <b>{e(c['side'])} {e(c['line'])}</b> &middot; {quote}</div>"
            f"<div>Forecast {e(pv['forecast_version'])}: mean {e(c['mean'])}, sd {e(c['sd'])}; "
            f"model P({e(c['side'])}) {p} (unvalidated at offered lines), breakeven {e(c['breakeven'])}</div>"
            f"<div>Why: {e(c['rationale'])}</div><div>Against: {e(c['countercase'])}</div>"
            f"<div class=muted>Invalid if: {e('; '.join(c['invalidation']))}</div>"
            f"<div class=muted>Status reasons: {e('; '.join(c['status_reasons']) or 'none')}. "
            f"Selected by {e(pv['selection_source'])}; run {e(pv['run_id'])}; code {e(pv['code_sha'])}</div>"
            f"{_panel_html(c.get('factor_panel'))}</div>")
    return "\n".join(parts)


def write_week_cards(conn, season: int, week: int, out_dir: str = "reports") -> Dict:
    import json
    import os
    payload = week_cards(conn, season, week)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pick_cards_latest.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(os.path.join(out_dir, "pick_cards_latest.html"), "w") as f:
        f.write("<!doctype html><meta charset='utf-8'>" + render_cards_html(payload["cards"]))
    # every displayed card goes into the append-only ledger (bookkeeping, not the user's chore)
    from . import issued_ledger
    payload["ledger_written"] = len(issued_ledger.record_cards(conn, season, week, payload["cards"]))
    with open(os.path.join(out_dir, f"issued_picks_{season}_wk{week}.json"), "w") as f:
        f.write(issued_ledger.export(issued_ledger.load(conn, season, week)))
    return payload
