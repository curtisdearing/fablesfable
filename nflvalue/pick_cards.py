"""Evidence cards for board rows: what is offered, what the model says, and why it is not a bet.

Every card states exact player/market/side/line, bookmaker, price and quote
clock, the forecast version, the model's P(side) and its validation status,
support, rationale, countercase, invalidation conditions and a status:

* ``actionable`` -- only for a market in ``VALIDATED_MARKETS``.  Empty: no
  market has passed an offered-line calibration gate (2026 wk1-2 settled
  exact lines: model Brier 0.263 vs coin 0.250 vs market 0.248), so no card
  can be actionable.  ``publish=True`` elsewhere is software permission, not
  validated edge.
* ``watch``    -- a real offered price with fresh, complete inputs.
* ``research`` -- no offered price (synthetic line) or thin role support.
* ``pass``     -- voided, stale quote, or missing forecast.

The gates are fixed here, not tuned to results.
"""

from __future__ import annotations

import datetime as dt
import html
import math
from typing import Dict, Iterable, List, Optional

VALIDATED_MARKETS: frozenset = frozenset()
STALE_QUOTE_HOURS = 6.0
MIN_ROLE_GAMES = 3
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
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _name_key(name) -> Optional[str]:
    """'D.Moore' / 'DJ Moore' / 'Brian Robinson Jr.' -> 'd moore' / 'd moore' / 'b robinson'."""
    if not isinstance(name, str) or not name.strip():
        return None
    parts = [t for t in name.replace(".", " ").lower().split()
             if t not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else None


def build_card(row: Dict, now: dt.datetime, forecast_version: str) -> Dict:
    side = (row.get("side") or "").lower()
    line, price = _f(row.get("line")), _f(row.get("price"))
    mean, sd, p = _f(row.get("mean")), _f(row.get("sd")), _f(row.get("p_side"))
    quote_ts = _ts(row.get("quote_ts"))
    age_h = (now - quote_ts).total_seconds() / 3600 if quote_ts else None
    offered = row.get("line_source") == "odds_api" and price is not None and bool(row.get("book"))
    roll_games = _f(row.get("roll_games"))
    breakeven = 1 / price if price else None

    reasons: List[str] = []
    if row.get("status") == "voided":
        status = "pass"
        reasons.append(f"voided: {row.get('void_reason') or 'unspecified'}")
    elif mean is None or sd is None or p is None or line is None:
        status = "pass"
        reasons.append("forecast or line missing")
    elif not offered:
        status = "research"
        reasons.append("no offered price: synthetic line, not a wager")
    elif age_h is None or age_h > STALE_QUOTE_HOURS:
        status = "pass"
        reasons.append("quote clock missing" if age_h is None
                       else f"quote is {age_h:.1f} h old (> {STALE_QUOTE_HOURS:.0f} h)")
    elif roll_games is not None and roll_games < MIN_ROLE_GAMES:
        status = "research"
        reasons.append(f"only {roll_games:.0f} games of role history")
    elif row.get("market") in VALIDATED_MARKETS:
        status = "actionable"
    else:
        status = "watch"
        reasons.append("market has not passed an offered-line calibration gate")

    if mean is not None and line is not None:
        direction = "above" if mean > line else "below"
        rationale = (f"Football-only projection {mean:.1f} (sd {sd:.1f}) sits {abs(mean - line):.1f} "
                     f"{direction} the {line:g} line.")
    else:
        rationale = "No projection."
    if row.get("reason"):
        rationale += f" Model drivers: {row['reason']}"
    invalidation = [
        "any injury designation or inactive listing after the quote clock "
        "(no matched injury row is NOT verified health)",
        f"line moves to the other side of the projection ({mean:.1f})" if mean is not None else "line moves",
        f"price shortens so breakeven exceeds the model's {p:.0%}" if p is not None else "price moves",
        f"quote older than {STALE_QUOTE_HOURS:.0f} h at decision time",
    ]
    return {
        "player": row.get("name"), "player_id": row.get("player_id"), "game_id": row.get("game_id"),
        "market": row.get("market"), "side": side, "line": line,
        "book": row.get("book") if offered else None,
        "price_decimal": price if offered else None, "price_american": _american(price) if offered else None,
        "quote_clock": row.get("quote_ts") if quote_ts else None, "run_as_of": row.get("as_of"), "quote_age_hours": round(age_h, 2) if age_h is not None else None,
        "forecast_version": forecast_version, "mean": mean, "sd": sd,
        "model_p_side": p, "model_p_status": "unvalidated_at_offered_lines",
        "breakeven": round(breakeven, 4) if breakeven else None,
        "ordering_score": _f(row.get("composite")),
        "ordering_score_note": "ranker/composite ordering score; market-informed; not a probability",
        "support": {"roll_games": roll_games, "n_books": row.get("n_books")},
        "rationale": rationale,
        "countercase": VALIDATION_NOTE,
        "invalidation": invalidation,
        "status": status, "status_reasons": reasons,
    }


def build_cards(rows: Iterable[Dict], now: Optional[dt.datetime] = None,
                forecast_version: str = "unknown") -> List[Dict]:
    now = now or dt.datetime.now(dt.timezone.utc)
    order = {"actionable": 0, "watch": 1, "research": 2, "pass": 3}
    cards = [build_card(r, now, forecast_version) for r in rows]
    return sorted(cards, key=lambda c: (order[c["status"]], -(c["ordering_score"] or 0)))


def render_html(cards: List[Dict], title: str, generated_at: str) -> str:
    e = lambda x: html.escape("" if x is None else str(x))
    parts = [f"<!doctype html><meta charset='utf-8'><title>{e(title)}</title>",
             "<style>body{font:15px system-ui;max-width:860px;margin:auto;padding:1em}"
             ".card{border:1px solid #ccc;border-radius:8px;padding:.8em;margin:.8em 0}"
             ".s{font-weight:700;text-transform:uppercase}.muted{color:#555;font-size:13px}</style>",
             f"<h1>{e(title)}</h1><p class=muted>Generated {e(generated_at)}. No card is a recommended "
             f"wager. {e(VALIDATION_NOTE)} No guaranteed outcomes.</p>"]
    for c in cards:
        price = (f"{e(c['book'])} {e(c['price_american'])} ({e(c['price_decimal'])}), quote {e(c['quote_clock'])}"
                 if c["book"] else "no offered price")
        p = f"{c['model_p_side']:.1%}" if c["model_p_side"] is not None else "n/a"
        parts.append(
            f"<div class=card><div><span class=s>{e(c['status'])}</span> &middot; "
            f"<b>{e(c['player'])}</b> {e(c['market'])} <b>{e(c['side'])} {e(c['line'])}</b> &middot; {price}</div>"
            f"<div>Forecast {e(c['forecast_version'])}: mean {e(c['mean'])}, sd {e(c['sd'])}; "
            f"model P({e(c['side'])}) {p} (unvalidated), breakeven {e(c['breakeven'])}; "
            f"ordering score {e(c['ordering_score'])} (not a probability)</div>"
            f"<div>Why: {e(c['rationale'])}</div><div>Against: {e(c['countercase'])}</div>"
            f"<div class=muted>Invalid if: {e('; '.join(c['invalidation']))}</div>"
            f"<div class=muted>Status reasons: {e('; '.join(c['status_reasons']) or 'none')}</div></div>")
    return "\n".join(parts)


def write_week_cards(conn, season: int, week: int, out_dir: str = "reports") -> Dict:
    """Cards for the newest clock of each (game, player, market) lean this week."""
    import json
    import os
    from . import db as dbmod
    from .football_forecast import FORECAST_VERSION
    leans = dbmod.query_df(conn, "SELECT * FROM leans WHERE season=? AND week=?", (season, week))
    if not leans.empty:
        leans = (leans.assign(_t=leans["clock"].map({"t90": 1}).fillna(0))
                 .sort_values("_t").drop_duplicates(["game_id", "player_id", "market"], keep="last"))
        # the quote clock is the newest captured quote for this exact book/side/point,
        # not the run's as_of; none found -> quote_ts stays empty and the card passes
        lines = dbmod.query_df(
            conn, "SELECT ts, game_id, book, market, player_id, player_name, side, point FROM lines "
                  "WHERE game_id IN (%s)" % ",".join("?" * leans["game_id"].nunique()),
            tuple(leans["game_id"].unique()))
        if not lines.empty:
            lines["side"] = lines["side"].str.lower()
            newest = (lines.sort_values("ts").drop_duplicates(
                ["game_id", "book", "market", "player_id", "player_name", "side", "point"], keep="last")
                .rename(columns={"ts": "quote_ts", "point": "line"}))
            # a lean's book may list several ("draftkings/fanduel"): newest quote across them
            lk = leans.assign(side=leans["side"].str.lower(), _book=leans["book"].fillna("").str.split("/"))
            ex = lk.explode("_book")
            ex["_key"] = ex["name"].map(_name_key)
            nl = newest.rename(columns={"book": "_book", "player_id": "_qpid"})
            nl["_key"] = nl["player_name"].map(_name_key)
            j = ex.merge(nl, on=["game_id", "_book", "market", "side", "line"], how="inner")
            # identity: same player_id when the quote has one, else same initial+surname
            # and exactly one quoted player at that key (ambiguous -> no clock)
            qpid = j["_qpid"].fillna("")
            same_id = qpid.ne("") & qpid.eq(j["player_id"])
            by_name = qpid.eq("") & j["_key_x"].notna() & j["_key_x"].eq(j["_key_y"])
            j = j[same_id | by_name]
            uniq = j.groupby(["game_id", "player_id", "market"])["player_name"].transform("nunique") == 1
            qt = j[uniq].groupby(["game_id", "player_id", "market"])["quote_ts"].max().reset_index()
            leans = lk.drop(columns="_book").merge(qt, on=["game_id", "player_id", "market"], how="left")
    now = dt.datetime.now(dt.timezone.utc)
    # only rows this run produced are stamped with the running forecast version;
    # the leans table does not record the version of earlier runs
    rows = leans.to_dict("records")
    for r in rows:
        made = _ts(r.get("created_at"))
        r["_fv"] = (FORECAST_VERSION if made and (now - made).total_seconds() < 3 * 3600
                    else "earlier run (version not recorded)")
    cards = [build_card(r, now, r["_fv"]) for r in rows]
    order = {"actionable": 0, "watch": 1, "research": 2, "pass": 3}
    cards.sort(key=lambda c: (order[c["status"]], -(c["ordering_score"] or 0)))
    os.makedirs(out_dir, exist_ok=True)
    payload = {"season": season, "week": week, "generated_at": now.isoformat(timespec="seconds"),
               "forecast_version": FORECAST_VERSION, "validated_markets": sorted(VALIDATED_MARKETS),
               "counts": {s: sum(c["status"] == s for c in cards)
                          for s in ("actionable", "watch", "research", "pass")},
               "cards": cards}
    with open(os.path.join(out_dir, "pick_cards_latest.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(os.path.join(out_dir, "pick_cards_latest.html"), "w") as f:
        f.write(render_html(cards, f"{season} week {week} evidence cards", payload["generated_at"]))
    return payload
