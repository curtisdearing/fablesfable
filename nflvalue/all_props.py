"""Read-only adapter and renderer for externally researched NFL player-prop payloads.

The production forecast database deliberately remains the source for model cards.  This
module consumes the separate all-props research contract without inventing prices,
probabilities, or wagering approval.  It is safe to invoke with absent inputs: callers
receive a visible pending state rather than a demo board.
"""
from __future__ import annotations

import html
import json
import os
from collections import Counter

SCHEMA = "fablesfable.all_props.v1"
VALID_STATUSES = {"analyst_lean", "research", "pass", "unavailable", "model_approved"}
VALID_DISPOSITIONS = {"reviewed", "rejected", "unsupported", "unavailable", "pass", "pending"}
REQUIRED_CARD = {"event_id", "game_id", "away", "home", "kickoff", "source_as_of", "picks"}
REQUIRED_ROW = {"event_id", "game_id", "player", "team", "market", "side", "period", "status", "disposition"}


def _read_list(path, label):
    if not path:
        return None, None
    if not os.path.isfile(path):
        return None, f"{label} file not found: {path}"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        return None, f"{label} is unreadable JSON: {exc}"
    if not isinstance(data, list):
        return None, f"{label} must be a top-level JSON list"
    return data, None


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number_or_none(value):
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))


def _valid_clock(value):
    return value is None or _text(value)


def _row_error(row, n):
    if not isinstance(row, dict):
        return f"market_rows[{n}] is not an object"
    missing = sorted(k for k in REQUIRED_ROW if k not in row)
    if missing:
        return f"market_rows[{n}] missing {', '.join(missing)}"
    invalid_text = [k for k in ("event_id", "game_id", "player", "team", "market", "side", "period")
                    if not _text(row.get(k))]
    if invalid_text:
        return f"market_rows[{n}] has blank {', '.join(invalid_text)}"
    if row.get("status") not in VALID_STATUSES:
        return f"market_rows[{n}] has unknown status {row.get('status')!r}"
    if row.get("disposition") not in VALID_DISPOSITIONS:
        return f"market_rows[{n}] has unknown disposition {row.get('disposition')!r}"
    for key in ("line", "odds", "projection", "model_probability", "calibrated_probability"):
        if key in row and not _number_or_none(row[key]):
            return f"market_rows[{n}].{key} must be numeric or null"
    for key in ("captured_at", "provider_updated_at", "retrieved_at", "quote_updated_at"):
        if key in row and not _valid_clock(row[key]):
            return f"market_rows[{n}].{key} must be an ISO-like string or null"
    return None


def _card_error(card, n):
    if not isinstance(card, dict):
        return f"game_cards[{n}] is not an object"
    missing = sorted(k for k in REQUIRED_CARD if k not in card)
    if missing:
        return f"game_cards[{n}] missing {', '.join(missing)}"
    if not isinstance(card["picks"], list):
        return f"game_cards[{n}].picks must be a list"
    for key in REQUIRED_CARD - {"picks"}:
        if not _text(card.get(key)):
            return f"game_cards[{n}] has blank {key}"
    for p, pick in enumerate(card["picks"]):
        if not isinstance(pick, dict):
            return f"game_cards[{n}].picks[{p}] is not an object"
        required = {"player", "team", "market", "side", "line", "book", "odds", "quote_updated_at",
                    "retrieved_at", "projection", "model_probability", "calibrated_probability",
                    "model_run_as_of", "status", "rationale", "counterargument", "invalidation",
                    "rank_basis", "sources"}
        absent = sorted(k for k in required if k not in pick)
        if absent:
            return f"game_cards[{n}].picks[{p}] missing {', '.join(absent)}"
        if pick.get("status") not in VALID_STATUSES:
            return f"game_cards[{n}].picks[{p}] has unknown status {pick.get('status')!r}"
    return None


def _has_offer(row):
    """Only an exact player-market quote may be labelled as a captured offer."""
    return (row.get("line") is not None and row.get("odds") is not None and _text(row.get("book"))
            and _text(row.get("captured_at")))


def _normalise_row(row):
    out = dict(row)
    out["has_offer"] = _has_offer(out)
    out["offer_label"] = "Captured priced market" if out["has_offer"] else "No verified price captured"
    # Normalize the UI fields but leave all unknown/provenance values as the source supplied them.
    out.setdefault("reason", None)
    out.setdefault("book", None)
    out.setdefault("odds", None)
    out.setdefault("line", None)
    out.setdefault("captured_at", out.get("retrieved_at"))
    out.setdefault("provider_updated_at", out.get("quote_updated_at"))
    out.setdefault("sources", [])
    return out


def _counts(cards, rows):
    athletes = {str(r["player"]).strip() for r in rows if _text(r.get("player"))}
    qualitative = sum(1 for r in rows if r.get("disposition") == "reviewed")
    return {
        "games": len(cards),
        "quote_rows": sum(1 for r in rows if r["has_offer"]),
        "unique_athletes": len(athletes),
        "model_priced": sum(1 for r in rows if r.get("model_probability") is not None or r.get("calibrated_probability") is not None),
        "qualitatively_reviewed": qualitative,
        "unavailable_or_unsupported": sum(1 for r in rows if r.get("disposition") in {"unavailable", "unsupported"}),
    }


def load(game_cards_path, market_rows_path, expected_event_ids=None):
    """Return a contract-preserving browse payload; never synthesize missing research."""
    cards, card_error = _read_list(game_cards_path, "game_cards")
    rows, row_error = _read_list(market_rows_path, "market_rows")
    if cards is None and rows is None and not card_error and not row_error:
        return {"schema": SCHEMA, "state": "pending", "message": "No research payload has been supplied yet.",
                "cards": [], "rows": [], "counts": _counts([], []), "errors": []}
    errors = [e for e in (card_error, row_error) if e]
    cards = cards or []
    rows = rows or []
    errors.extend(e for n, card in enumerate(cards) if (e := _card_error(card, n)))
    errors.extend(e for n, row in enumerate(rows) if (e := _row_error(row, n)))
    if expected_event_ids:
        expected = {str(e) for e in expected_event_ids}
        got = {str(c.get("event_id")) for c in cards if isinstance(c, dict)}
        unexpected = sorted(got - expected)
        missing = sorted(expected - got)
        if unexpected:
            errors.append(f"game_cards contains event(s) outside this board: {', '.join(unexpected)}")
        if missing:
            errors.append(f"game_cards is missing event(s) required for this board: {', '.join(missing)}")
    if errors:
        return {"schema": SCHEMA, "state": "invalid", "message": "Research payload could not be published.",
                "cards": [], "rows": [], "counts": _counts([], []), "errors": errors}
    normal_rows = [_normalise_row(row) for row in rows]
    return {"schema": SCHEMA, "state": "ready", "message": "Research payload loaded; verify clocks before acting.",
            "cards": cards, "rows": normal_rows, "counts": _counts(cards, normal_rows), "errors": []}


def _e(value):
    return html.escape("" if value is None else str(value))


def _status(value):
    return _e(str(value).replace("_", " ").upper())


def render_page(payload):
    """Portable static all-player browser with local filters and disclosure details."""
    state = payload.get("state")
    if state != "ready":
        errors = "".join(f"<li>{_e(error)}</li>" for error in payload.get("errors") or [])
        detail = f"<ul>{errors}</ul>" if errors else ""
        return ("<h1>All player props</h1><p><b>Pending research payload.</b> "
                f"{_e(payload.get('message') or 'No research payload has been supplied yet.')} "
                "No player props or picks are shown until separate game_cards.json and market_rows.json "
                f"pass validation.</p>{detail}")
    c = payload["counts"]
    filters = ("<label>Search <input id='ap-search' type='search' placeholder='player, team, market'></label> "
               "<label>Status <select id='ap-status'><option value=''>All</option></select></label> "
               "<label>Disposition <select id='ap-disposition'><option value=''>All</option></select></label> "
               "<label>Period <select id='ap-period'><option value=''>All</option></select></label>")
    summary = (f"<p>{c['games']} games · {c['quote_rows']} captured priced rows · {c['unique_athletes']} athletes · "
               f"{c['model_priced']} model-priced · {c['qualitatively_reviewed']} qualitatively reviewed · "
               f"{c['unavailable_or_unsupported']} unavailable/unsupported.</p>")
    cards = []
    for card in payload["cards"]:
        picks = card.get("picks") or []
        picks_html = "".join(
            "<li><b>{player}</b> — {market} {side} {line}; <span class='status'>{status}</span>. "
            "{rationale}<details><summary>Why / risk / clock</summary><p><b>Counterargument:</b> {counter}. "
            "<b>Invalidation:</b> {invalid}. <b>Rank basis:</b> {basis}. <b>Quote:</b> {book} {odds}, "
            "updated {updated}; retrieved {retrieved}. <b>Model probability:</b> {prob}.</p></details></li>".format(
                player=_e(p.get("player")), market=_e(p.get("market")), side=_e(p.get("side")),
                line=_e(p.get("line")), status=_status(p.get("status")), rationale=_e(p.get("rationale")),
                counter=_e(p.get("counterargument")), invalid=_e(p.get("invalidation")),
                basis=_e(p.get("rank_basis")), book=_e(p.get("book")), odds=_e(p.get("odds")),
                updated=_e(p.get("quote_updated_at")), retrieved=_e(p.get("retrieved_at")),
                prob=_e(p.get("calibrated_probability") if p.get("calibrated_probability") is not None else p.get("model_probability")))
            for p in picks)
        cards.append(f"<section class='ap-game' id='game-{_e(card.get('event_id'))}'><h2>{_e(card.get('away'))} @ {_e(card.get('home'))}</h2>"
                     f"<p>Kickoff { _e(card.get('kickoff'))}; source as of {_e(card.get('source_as_of'))}.</p>"
                     f"<p>{_e(card.get('preview'))}</p><ol>{picks_html or '<li>No ranked selections supplied.</li>'}</ol></section>")
    rows = []
    for row in payload["rows"]:
        search = " ".join(str(row.get(k) or "") for k in ("game_id", "player", "team", "market", "book", "side"))
        rows.append("<tr data-search='{search}' data-status='{status}' data-disposition='{disposition}' data-period='{period}'>"
                    "<td>{game}</td><td>{player} ({team})</td><td>{market}</td><td>{side} {line}</td><td>{period}</td>"
                    "<td>{book} {odds}<br><small>{offer}</small></td><td>{status}</td><td>{disposition}</td>"
                    "<td><details><summary>Reason & clocks</summary>{reason}<br><small>Captured {captured}; provider {provider}</small></details></td></tr>".format(
                        search=_e(search.lower()), status=_e(row.get("status")), disposition=_e(row.get("disposition")),
                        period=_e(row.get("period")), game=_e(row.get("game_id")), player=_e(row.get("player")),
                        team=_e(row.get("team")), market=_e(row.get("market")), side=_e(row.get("side")),
                        line=_e(row.get("line")), book=_e(row.get("book")), odds=_e(row.get("odds")),
                        offer=_e(row.get("offer_label")), reason=_e(row.get("reason")),
                        captured=_e(row.get("captured_at")), provider=_e(row.get("provider_updated_at"))))
    script = """<script>(function(){const rows=[...document.querySelectorAll('#ap-table tbody tr')];const fields=['status','disposition','period'];for(const f of fields){const s=document.getElementById('ap-'+f);[...new Set(rows.map(r=>r.dataset[f]).filter(Boolean))].sort().forEach(v=>s.add(new Option(v,v)));}function run(){const q=document.getElementById('ap-search').value.toLowerCase();rows.forEach(r=>r.hidden=!!(q&&!r.dataset.search.includes(q))||fields.some(f=>{const v=document.getElementById('ap-'+f).value;return v&&r.dataset[f]!==v;});}document.querySelectorAll('#ap-search,#ap-status,#ap-disposition,#ap-period').forEach(n=>n.addEventListener('input',run));})();</script>"""
    return ("<h1>All player props</h1><p><b>Research view, not wager approval.</b> A row is a captured priced market only when exact line, book, odds and capture clock are present.</p>"
            + summary + filters + "<h2>Game browser & ranked rationales</h2>" + "".join(cards)
            + "<h2>Complete captured/rejected market rows</h2><div style='overflow:auto'><table id='ap-table'><thead><tr><th>Game</th><th>Player</th><th>Market</th><th>Side / line</th><th>Period</th><th>Price</th><th>Status</th><th>Disposition</th><th>Research</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>" + script)
