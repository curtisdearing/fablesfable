"""Read-only adapter and renderer for externally researched NFL player-prop payloads.

The production forecast database deliberately remains the source for model cards.  This
module consumes the separate all-props research contract without inventing prices,
probabilities, or wagering approval.  It is safe to invoke with absent inputs: callers
receive a visible pending state rather than a demo board.
"""
from __future__ import annotations

import html
import json
import math
import os
from collections import Counter

SCHEMA = "fablesfable.all_props.v1"
VALID_STATUSES = {"analyst_lean", "research", "pass", "unavailable", "pending", "model_approved"}
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
    is_coverage = row.get("raw_market_row_type") in {"family_availability", "coverage_gap"}
    invalid_text = [k for k in ("event_id", "game_id", "player", "team", "market", "period")
                    if not _text(row.get(k))]
    if not is_coverage and not _text(row.get("side")):
        invalid_text.append("side")
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
        for key in ("projection", "model_projection"):
            if key in pick and not _number_or_none(pick[key]):
                return f"game_cards[{n}].picks[{p}].{key} must be numeric or null"
    forecasts = card.get("player_forecasts")
    if forecasts is not None and not isinstance(forecasts, list):
        return f"game_cards[{n}].player_forecasts must be a list when supplied"
    for f, forecast in enumerate(forecasts or []):
        if not isinstance(forecast, dict):
            return f"game_cards[{n}].player_forecasts[{f}] is not an object"
        means = forecast.get("means")
        if means is not None and not isinstance(means, dict):
            return f"game_cards[{n}].player_forecasts[{f}].means must be an object when supplied"
    return None


def _has_offer(row):
    """Only an exact player-market quote may be labelled as a captured offer."""
    return ((row.get("line") is not None or row.get("side") == "yes")
            and row.get("odds") is not None and _text(row.get("book"))
            and _text(row.get("captured_at"))
            and row.get("raw_market_row_type") not in {"family_availability", "coverage_gap"})


def _normalise_row(row):
    out = dict(row)
    out["has_offer"] = _has_offer(out)
    out["offer_label"] = ("Published secondary listing — not sportsbook-verified" if out["has_offer"]
                          else "No listed price captured")
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
    athletes = {str(r["player"]).strip() for r in rows if _text(r.get("player"))
                and r.get("raw_market_row_type") not in {"family_availability", "coverage_gap"}}
    qualitative = sum(1 for r in rows if r.get("disposition") == "reviewed")
    return {
        "games": len(cards),
        "raw_market_rows": len({r.get("source_row_id") for r in rows if r.get("source_row_id") is not None}),
        "outcome_rows": sum(1 for r in rows if r.get("raw_market_row_type") not in {"family_availability", "coverage_gap"}),
        "quote_rows": sum(1 for r in rows if r["has_offer"]),
        "unique_athletes": len(athletes),
        "model_priced": len({(r.get('game_id') or card.get('game_id'), r.get('player_id') or r.get('player'), r.get('market'), r.get('side'), r.get('line'), r.get('book'))
                             for card, r in [(c, p) for c in cards for p in c.get('picks', [])] + [({}, r) for r in rows]
                             if r.get('model_probability') is not None or r.get('calibrated_probability') is not None}),
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


def _kickoff(value):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(ZoneInfo('America/New_York')).strftime('%a %b %d, %-I:%M %p ET')
    except (ValueError, TypeError, AttributeError):
        return str(value)


def _source_url(value):
    from urllib.parse import urlsplit
    return isinstance(value, str) and urlsplit(value).scheme in {'https', 'http'}


MARKET_LABELS = {
    "passing_yards": "Passing yards",
    "rushing_yards": "Rushing yards",
    "receiving_yards": "Receiving yards",
    "receptions": "Receptions",
    "anytime_td": "Anytime touchdown",
    "anytime_touchdown": "Anytime touchdown",
    "longest_reception": "Longest reception",
    "longest_rush": "Longest rush",
}


def _display_text(value):
    """Accept only authored reader copy; never stringify arbitrary payload objects."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _market_label(value):
    text = _display_text(value)
    return MARKET_LABELS.get(text, text.replace("_", " ").title())


def _signed_odds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    return f"{value:+g}"


def _numeric(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _number(value):
    """Format a supplied number without fabricating precision or a missing zero."""
    value = _numeric(value)
    return f"{value:g}" if value is not None else ""


def _model_projection(pick):
    """``model_projection`` is the explicit native alias; legacy ``projection`` remains valid."""
    return _numeric(pick.get("model_projection")) if "model_projection" in pick else _numeric(pick.get("projection"))


def _projection_unit(market):
    return {"passing_yards": "yards", "rushing_yards": "yards", "receiving_yards": "yards",
            "receptions": "catches", "rush_attempts": "carries", "pass_attempts": "attempts"}.get(market, "")


def _finite_number(value):
    value = _numeric(value)
    return value if value is not None and math.isfinite(value) else None


def _model_line_gap(pick):
    """Return a unit-safe comparison or None when the supplied fields are incomparable."""
    if pick.get("market") in {"anytime_td", "anytime_touchdown"}:
        # P(TD >= 1) is categorical; it is not a comparable mean against a 0.5 threshold.
        return None
    projection = _finite_number(_model_projection(pick))
    line = _finite_number(pick.get("line"))
    if projection is None or line is None or line == 0:
        return None
    gap = projection - line
    return {"gap": gap, "absolute_percentage": abs(gap) / abs(line) * 100}


def _model_difference_sort_key(pick):
    """Sort comparable discrepancies first, with deterministic player/market ties."""
    comparison = _model_line_gap(pick)
    player = _display_text(pick.get("player")).casefold()
    market = _display_text(pick.get("market")).casefold()
    if comparison is None:
        return (1, 0, player, market)
    return (0, -comparison["absolute_percentage"], player, market)


def _pick_comparison(pick, show_uncomparable=False):
    comparison = _model_line_gap(pick)
    if comparison is None:
        return ("<p class='model-comparison'><b>Model-vs-line gap:</b> Not comparable</p>"
                if show_uncomparable else "")
    projection, line = _model_projection(pick), _numeric(pick.get("line"))
    unit = _projection_unit(pick.get("market"))
    model = f"{_number(projection)} {unit}".rstrip()
    gap = f"{comparison['gap']:+g} {unit}".rstrip()
    percentage = f"{comparison['absolute_percentage']:g}%"
    return (f"<p class='model-comparison'><b>Model:</b> {_e(model)} · <b>Line:</b> {_e(_number(line))}"
            f" · <b>Model-vs-line gap:</b> {_e(gap)} ({_e(percentage)})</p>")


def _forecast_name(forecast):
    return _display_text(forecast.get("name")) or _display_text(forecast.get("player"))


def _forecast_mean(forecast, key):
    means = forecast.get("means")
    return _numeric(means.get(key)) if isinstance(means, dict) else None


def _forecast_td_probability(forecast):
    """Show only an explicit P(TD >= 1), never an expected-TD mean."""
    candidates = (forecast.get("anytime_td_p_ge_1"), forecast.get("td_probability"),
                  forecast.get("anytime_td_probability"))
    probabilities = forecast.get("probabilities")
    if isinstance(probabilities, dict):
        candidates += (probabilities.get("anytime_td_p_ge_1"), probabilities.get("anytime_td"))
    for candidate in candidates:
        value = _numeric(candidate)
        if value is not None and 0 <= value <= 1:
            return value
    return None


def _forecast_role(forecast):
    role = _display_text(forecast.get("role")) or _display_text(forecast.get("model_role"))
    unsupported = forecast.get("supported") is False or _display_text(forecast.get("role_status")).lower() == "unsupported"
    if unsupported:
        return "Not model-supported" + (f" — {role}" if role else "")
    return role


def _forecast_clock(forecast, card):
    return (_display_text(forecast.get("model_run_as_of")) or _display_text(forecast.get("model_clock"))
            or _display_text(forecast.get("forecast_as_of")) or _display_text(card.get("model_run_as_of")))


def _forecast_table(card):
    """Render the native per-player means as supplied; blanks remain blanks, not zeros."""
    if str(card.get("status") or "").lower() == "completed":
        return ""
    forecasts = [forecast for forecast in card.get("player_forecasts") or []
                 if isinstance(forecast, dict) and _forecast_name(forecast)]
    if not forecasts:
        return ""
    td_available = any(_forecast_td_probability(forecast) is not None for forecast in forecasts)
    headers = ["Player", "Pass attempts", "Pass yds", "Carries", "Rush yds", "Catches", "Rec yds"]
    if td_available:
        headers.append("TD probability")
    headers.extend(("Model clock", "Role"))
    rows = []
    for forecast in forecasts:
        player = _forecast_name(forecast)
        position = _display_text(forecast.get("position")) or _display_text(forecast.get("pos"))
        display_player = f"{player} ({position})" if position else player
        values = [display_player, _number(_forecast_mean(forecast, "pass_attempts")), _number(_forecast_mean(forecast, "passing_yards")),
                  _number(_forecast_mean(forecast, "rush_attempts")),
                  _number(_forecast_mean(forecast, "rushing_yards")),
                  _number(_forecast_mean(forecast, "receptions")),
                  _number(_forecast_mean(forecast, "receiving_yards"))]
        if td_available:
            probability = _forecast_td_probability(forecast)
            values.append(f"{probability:.0%}" if probability is not None else "")
        clock = _forecast_clock(forecast, card)
        values.extend((clock, _forecast_role(forecast)))
        rows.append("<tr>" + "".join(f"<td>{_e(value) if value else '—'}</td>" for value in values) + "</tr>")
    warning = (_display_text(card.get("incomplete_factor_warning")) or
               _display_text(card.get("factor_warning")))
    warning_html = f"<p class='factor-warning'>{_e(warning)}</p>" if warning else ""
    return ("<details class='player-projections'><summary>Player projections</summary>" + warning_html
            + "<div class='projection-scroll'><table><thead><tr>"
            + "".join(f"<th>{_e(header)}</th>" for header in headers)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></details>")


def _pick_is_displayable(pick):
    """Keep unsupported/pass placeholders and incomplete offers off the picks board."""
    if not isinstance(pick, dict):
        return False
    if str(pick.get("status") or "").lower() not in {"analyst_lean", "model_approved"}:
        return False
    if str(pick.get("display_status") or pick.get("status") or "").lower() == "pass":
        return False
    player = _display_text(pick.get("player"))
    if not player or player.lower().startswith("no named player"):
        return False
    side = str(pick.get("side") or "").lower()
    categorical_td = pick.get("market") in {"anytime_td", "anytime_touchdown"} and side == "yes"
    has_numeric_line = isinstance(pick.get("line"), (int, float)) and not isinstance(pick.get("line"), bool)
    return (((side in {"over", "under"} and has_numeric_line) or categorical_td)
            and _display_text(pick.get("book")) and bool(_signed_odds(pick.get("odds"))))


def _tier(pick):
    value = _display_text(pick.get("display_tier") or pick.get("tier")).lower()
    if value in {"preferred", "preferred_pick", "preferred pick"}:
        return "preferred"
    if value in {"other", "other_lean", "other lean", "secondary"}:
        return "other"
    return ""


def _pick_card(pick, game, number):
    """Render one reader-facing selection without exposing implementation payload fields."""
    side = str(pick["side"]).upper()
    why = _display_text(pick.get("display_why")) or _display_text(pick.get("rationale"))
    risk = _display_text(pick.get("display_risk")) or _display_text(pick.get("counterargument"))
    quote_clock = _display_text(pick.get("retrieved_at"))
    source_clock = _display_text(game.get("source_as_of"))
    clock = quote_clock or source_clock
    source_links = []
    for source in pick.get("sources") or []:
        if isinstance(source, dict) and _source_url(source.get("url")):
            source_links.append('<a href="{}" rel="noopener noreferrer">{}</a>'.format(
                _e(source["url"]), _e(_display_text(source.get("title")) or _display_text(source.get("provider")) or "Source")))
    detail = []
    if clock:
        detail.append(f"<p><b>Captured:</b> {_e(_kickoff(clock))}</p>")
    model_clock = _display_text(pick.get("model_run_as_of")) or _display_text(game.get("model_run_as_of"))
    if model_clock:
        detail.append(f"<p><b>Model run:</b> {_e(model_clock)}</p>")
    if source_links:
        detail.append(f"<p><b>Source:</b> {' · '.join(source_links)}</p>")
    details = ("<details><summary>Source and timing</summary>" + "".join(detail) + "</details>") if detail else ""
    risk_html = f"<p class='pick-risk'><b>Risk:</b> {_e(risk)}</p>" if risk else ""
    why_html = f"<p>{_e(why)}</p>" if why else ""
    comparison = _pick_comparison(pick, bool(game.get("native_model")))
    line_label = f" {_e(pick.get('line'))}" if pick.get("line") is not None else ""
    return (f"<article class='pick-card' id='pick-{_e(game.get('event_id'))}-{number}'>"
            f"<p class='pick-game'>{_e(game.get('away'))} at {_e(game.get('home'))}</p>"
            f"<h3>{_e(pick['player'])} {side}{line_label} {_e(_market_label(pick.get('market')))}</h3>"
            f"<p class='pick-price'>{_e(_display_text(pick.get('book')))} {_e(_signed_odds(pick.get('odds')))}</p>"
            f"{comparison}{why_html}{risk_html}{details}</article>")


GAME_MARKET_LABELS = {"moneyline": "Moneyline", "spread": "Spread", "total": "Total"}
INJURY_STATUS_CLASSES = {"out": "out", "doubtful": "doubtful", "questionable": "questionable",
                         "injured reserve": "ir", "reserve/injured": "ir"}


def _game_header(card):
    """Render sourced game decisions and both teams' availability before player props.

    Game lines and injury context are an explicit display-only overlay.  Neither is
    converted into a forecast or an implied player workload by this reader.
    """
    markets = {str(item.get("market")).lower(): item for item in card.get("game_markets") or []
               if isinstance(item, dict)}
    market_parts = []
    for key in ("moneyline", "spread", "total"):
        item = markets.get(key, {})
        decision = _display_text(item.get("decision")) or "No supported pick"
        line = _display_text(item.get("line"))
        price = _display_text(item.get("price"))
        provider = _display_text(item.get("provider"))
        source_state = _display_text(item.get("source_state"))
        offer = " · ".join(part for part in (line, price) if part)
        note = f" <span class='game-market-source'>({ _e(source_state) })</span>" if source_state else ""
        if provider:
            note += f" <span class='game-market-provider'>[{_e(provider)}]</span>"
        market_parts.append(f"<article class='game-market game-market-{_e(key)}'><h4>{GAME_MARKET_LABELS[key]}</h4>"
                            f"<p><b>{_e(decision)}</b>{note}</p>"
                            f"<p class='game-market-line'>{_e(offer) if offer else 'No listed game line or price captured.'}</p></article>")
    injuries = card.get("injuries") if isinstance(card.get("injuries"), dict) else {}
    injury_teams = {str(team.get("team")): team for team in injuries.get("teams") or [] if isinstance(team, dict)}
    injury_parts = []
    for team_code in (str(card.get("away")), str(card.get("home"))):
        team = injury_teams.get(team_code, {})
        items = [item for item in team.get("items") or [] if isinstance(item, dict)]
        if items:
            rows = []
            for item in items:
                status = _display_text(item.get("status")) or "No designation"
                css = INJURY_STATUS_CLASSES.get(status.lower(), "unknown")
                position = _display_text(item.get("position"))
                body_part = _display_text(item.get("body_part"))
                description = _display_text(item.get("description"))
                report_date = _display_text(item.get("report_date"))
                impact = _display_text(item.get("impact"))
                details = " — ".join(part for part in (body_part, description) if part)
                reported = f" <span class='injury-date'>(reported {_e(report_date)})</span>" if report_date else ""
                context = f" <span class='injury-impact'> — {_e(impact)}</span>" if impact else ""
                rows.append(f"<li><span class='injury-status injury-{_e(css)}'>{_e(status)}</span> "
                            f"<b>{_e(item.get('name'))}</b>{_e(' (' + position + ')') if position else ''}"
                            f"{_e(' — ' + details) if details else ''}{reported}{context}</li>")
            contents = "<ul>" + "".join(rows) + "</ul>"
        else:
            contents = "<p>No current injury report captured; this is unknown, not healthy.</p>"
        injury_parts.append(f"<section class='injury-team'><h4>{_e(team_code)} injuries</h4>{contents}</section>")
    injury_clock = _display_text(injuries.get("retrieved_at")) or _display_text(card.get("source_as_of"))
    injury_source = _display_text(injuries.get("source")) or "No injury source supplied"
    raw_injury_url = injuries.get("url")
    injury_url = _display_text(raw_injury_url) if _source_url(raw_injury_url) else None
    source_link = (f" <a href='{_e(injury_url)}' rel='noopener noreferrer'>ESPN source</a>" if injury_url else "")
    return ("<div class='game-frontmatter'><section class='game-decisions'><h4>Game decisions</h4>"
            "<div class='game-market-grid'>" + "".join(market_parts) + "</div></section>"
            "<section class='injury-panel'><h4>Both-team injury report</h4>"
            f"<p class='injury-source'>Source: {_e(injury_source)} · retrieved {_e(injury_clock) if injury_clock else 'unknown time'}.{source_link}</p>"
            "<div class='injury-grid'>" + "".join(injury_parts) + "</div></section></div>")


def render_page(payload):
    """Render a picks-first public board; full research remains available as JSON."""
    state = payload.get("state")
    if state != "ready":
        errors = "".join(f"<li>{_e(error)}</li>" for error in payload.get("errors") or [])
        detail = f"<ul>{errors}</ul>" if errors else ""
        return ("<h1>All player props</h1><p><b>Pending research payload.</b> "
                f"{_e(payload.get('message') or 'No research payload has been supplied yet.')} "
                "No player props or picks are shown until separate game_cards.json and market_rows.json "
                f"pass validation.</p>{detail}")
    visible = [(card, pick) for card in payload["cards"] if str(card.get("status") or "").lower() != "completed"
               for pick in card.get("picks") or [] if _pick_is_displayable(pick)]
    tiered = bool(visible) and all(_tier(pick) for _card, pick in visible)
    pick_numbers = {id(pick): n for n, (_card, pick) in enumerate(visible, 1)}
    rendered = []
    native_board = any(card.get('native_model') for card in payload['cards'])
    if native_board:
        rendered.append("<p>Choose a game below. Within each game, comparable model-vs-line discrepancies are ordered by absolute percentage gap; the signed gap shows the actual-unit difference. This is a discrepancy, not confidence. Every supported comparison is retained; none are cut to a fixed number per game.</p>")
    elif tiered:
        for label, key in (("Preferred picks", "preferred"), ("Other leans", "other")):
            choices = [(card, pick) for card, pick in visible if _tier(pick) == key]
            if choices:
                rendered.append(f"<h2>{label}</h2><div class='picks-grid'>" + "".join(
                    _pick_card(pick, card, pick_numbers[id(pick)]) for card, pick in choices) + "</div>")
    elif visible:
        rendered.append("<div class='picks-grid'>" + "".join(
            _pick_card(pick, card, pick_numbers[id(pick)]) for card, pick in visible) + "</div>")
    else:
        rendered.append("<p class='no-pick'>No picks are available.</p>")
    jumps = '<nav class="game-jumps" aria-label="Jump to a game">' + "".join(
        f'<a href="#game-{_e(card.get("event_id"))}">{_e(card.get("away"))} at {_e(card.get("home"))}</a>'
        for card in payload["cards"]) + "</nav>"
    games = []
    for card in payload["cards"]:
        archived = str(card.get("status") or "").lower() == "completed"
        picks = ([] if archived else
                 [pick for pick in card.get("picks") or [] if _pick_is_displayable(pick)])
        if native_board:
            picks.sort(key=_model_difference_sort_key)
        forecasts = _forecast_table(card)
        if picks and native_board:
            content = f"<details class='game-model-picks'><summary>Model differences ({len(picks)})</summary><div class='picks-grid'>" + ''.join(_pick_card(p, card, pick_numbers[id(p)]) for p in picks) + '</div></details>'
        elif picks:
            links = " · ".join(f'<a href="#pick-{_e(card.get("event_id"))}-{n}">{_e(_pick.get("player"))}</a>'
                              for n, (game, _pick) in enumerate(visible, 1) if game is card)
            content = f"<p>{links}</p>"
        elif archived:
            content = "<p class='no-pick'><b>Archived — not a prospective pick.</b></p>"
        else:
            reason = _display_text(card.get("no_pick_reason"))
            if reason and not reason.lower().startswith(('no pick', 'completed')):
                reason = 'No pick. ' + reason
            content = f"<p class='no-pick'>{_e(reason) if reason else 'No pick.'}</p>"
        archived_label = " <span class='archived-label'>Archived</span>" if archived else ""
        games.append(f"<section class='ap-game' id='game-{_e(card.get('event_id'))}'>"
                     f"<h3>{_e(card.get('away'))} at {_e(card.get('home'))}{archived_label}</h3>"
                     f"<p class='game-time'>{_e(_kickoff(card.get('kickoff')))}</p>{_game_header(card)}{content}{forecasts}</section>")
    return ("<h1>Week 5 player props</h1>"
            "<p class='board-caveat'>Uncalibrated model leans: prices can move; confirm every listed price before acting. ESPN pickcenter closes are display-only listed context, not sportsbook-verified executable quotes. No game-market picks are issued because this board has no joint game model.</p>"
            "<h2>Picks</h2>" + "".join(rendered)
            + "<h2>All games</h2>" + jumps + "".join(games)
            + "<p class='research-download'><a href='api/all-props.json' download>Download research data (JSON)</a></p>")
