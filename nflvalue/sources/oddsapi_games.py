"""All-event sportsbook GAME lines -- moneyline, spread, total -- for the whole slate.

One metered call prices every listed game. The bulk ``/sports/{sport}/odds``
endpoint is billed per market (x bookmaker group of 10) RETURNED for the whole
response, not per event: three markets at the configured books is an upper
bound of 3 credits for a full week, against 7 per EVENT for the prop markets.
Game lines and player props are different products -- neither stands in for
the other, and each game's coverage row says which one it has.

Same guarantees as the prop client (:mod:`oddsapi_props`):

* the free quota preflight is REQUIRED on the network path; an unknown, NaN,
  negative or infinite quota refuses with zero metered calls;
* credits still HELD for required prop closes are never spent here;
* every answered call writes one receipt row per due slate game
  (``game_line_pulls``), a game or market the answer omitted included;
* each quote keeps the book's own ``last_update`` clock beside our capture
  clock, and an event already under way is never stored as a pregame line;
* an answer younger than ``GAME_LINES_FRESH_HOURS`` is reused, not re-bought.

The provider's billing cycle is UNKNOWN here (it need not be the calendar
month): the preflight's remaining allowance is authoritative. The football
forecast never reads these rows.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Callable, Dict, List, Optional

from .. import db as dbmod
from ..freshness import stamp_now
from . import oddsapi_props as oap
from ._http import get_json_and_headers

GAME_MARKETS = ("h2h", "spreads", "totals")
#: An answer this young is the current board: a re-run reuses it, no new call.
GAME_LINES_FRESH_HOURS = 1.0
GAME_LINES_MAX_AGE_HOURS = oap.MAX_LINE_AGE_HOURS
BILLING_CYCLE = "unknown: the provider quota preflight is the authoritative allowance"

GAME_LINES_DDL = """CREATE TABLE IF NOT EXISTS game_lines (
    ts TEXT, game_id TEXT, event_id TEXT, book TEXT, market TEXT, side TEXT,
    point REAL, price REAL, book_clock TEXT,
    PRIMARY KEY (ts, game_id, book, market, side))"""
GAME_PULLS_DDL = """CREATE TABLE IF NOT EXISTS game_line_pulls (
    ts TEXT, game_id TEXT, event_id TEXT, state TEXT, n_rows INTEGER,
    books TEXT, markets TEXT, commence_time TEXT,
    call_credits_planned REAL, call_credits_billed REAL,
    PRIMARY KEY (ts, game_id))"""


def _ensure(conn) -> None:
    conn.execute(GAME_LINES_DDL)
    conn.execute(GAME_PULLS_DDL)


def request_cost(cfg: Dict, markets=GAME_MARKETS) -> float:
    """Upper bound of ONE bulk call: markets x region groups (10 books = 1)."""
    books = cfg.get("books") or []
    groups = math.ceil(len(books) / 10) if books else len(str(cfg.get("regions", "us")).split(","))
    return float(len(markets) * groups)


def parse_event_game_lines(ev: Dict, ts: str) -> List[Dict]:
    """One v4 event -> moneyline/spread/total rows; a line without its point is not a line."""
    home, away = ev.get("home_team"), ev.get("away_team")
    rows: List[Dict] = []
    for bk in ev.get("bookmakers") or []:
        for mkt in bk.get("markets") or []:
            key = mkt.get("key")
            if key not in GAME_MARKETS:
                continue
            clock = mkt.get("last_update") or bk.get("last_update")
            for o in mkt.get("outcomes") or []:
                name = o.get("name")
                side = ({"over": "over", "under": "under"}.get(str(name).lower()) if key == "totals"
                        else "home" if name == home else "away" if name == away else None)
                try:
                    price = float(o.get("price"))
                except (TypeError, ValueError):
                    continue
                if side is None or not math.isfinite(price) or (key != "h2h" and o.get("point") is None):
                    continue
                rows.append({"ts": ts, "book": bk.get("key"), "market": key, "side": side,
                             "point": float(o["point"]) if o.get("point") is not None else None,
                             "price": price, "book_clock": clock})
    return rows


def _latest(conn, game_ids: List[str], now: dt.datetime) -> Dict[str, Dict]:
    _ensure(conn)
    sql, params = "SELECT game_id, ts, state, n_rows FROM game_line_pulls WHERE ts <= ?", [oap._iso(now)]
    if game_ids:
        sql += f" AND game_id IN ({','.join('?' * len(game_ids))})"
        params.extend(game_ids)
    df = dbmod.query_df(conn, sql, tuple(params))
    if df.empty:
        return {}
    latest = df.sort_values(["game_id", "ts"]).groupby("game_id").tail(1)
    return {r.game_id: {"ts": r.ts, "state": r.state, "n_rows": int(r.n_rows)}
            for r in latest.itertuples(index=False)}


def answered_since(conn, game_ids: List[str], now: Optional[dt.datetime] = None,
                   max_age_hours: float = GAME_LINES_FRESH_HOURS) -> Dict[str, str]:
    """{game_id: ts} of games the bulk call ANSWERED (any state) within the window."""
    now = oap._aware(now)
    floor = oap._iso(now - dt.timedelta(hours=float(max_age_hours)))
    return {g: s["ts"] for g, s in _latest(conn, game_ids, now).items() if s["ts"] >= floor}


def load_game_lines(conn, game_ids: List[str], now: Optional[dt.datetime] = None,
                    max_age_hours: float = GAME_LINES_MAX_AGE_HOURS) -> List[Dict]:
    """Each game's board: every row of its LATEST answer, if that answer is fresh."""
    now = oap._aware(now)
    floor = oap._iso(now - dt.timedelta(hours=float(max_age_hours)))
    keep = {(g, s["ts"]) for g, s in _latest(conn, game_ids, now).items()
            if s["ts"] >= floor and s["n_rows"]}
    if not keep:
        return []
    games, clocks = sorted({g for g, _ in keep}), sorted({t for _, t in keep})
    df = dbmod.query_df(conn, f"SELECT * FROM game_lines WHERE game_id IN ({','.join('?' * len(games))}) "
                              f"AND ts IN ({','.join('?' * len(clocks))})", tuple(games) + tuple(clocks))
    return [r for r in df.to_dict("records") if (r["game_id"], r["ts"]) in keep]


def pull_game_lines(cfg: Dict, identity: Dict, conn=None, fetch: Optional[Callable] = None,
                    quota_fetch: Optional[Callable] = None, budget: Optional[oap.CreditBudget] = None,
                    ts: Optional[str] = None, now: Optional[dt.datetime] = None,
                    kickoffs: Optional[Dict[str, dt.datetime]] = None,
                    markets=GAME_MARKETS, fresh_hours: float = GAME_LINES_FRESH_HOURS) -> Dict:
    """ONE bulk call for every pregame slate game the listing identified.

    ``identity``: ``pipeline_weekly.build_event_map(..., details=identity)``.
    ``fetch(url, params) -> (events, headers)``; omitted = the network, which
    REQUIRES the free quota preflight (an injected fetch is an offline replay
    and preflights only when handed ``quota_fetch``). Returns the call receipt."""
    network = fetch is None
    fetch = fetch or get_json_and_headers
    conn = conn or dbmod.connect()
    _ensure(conn)
    now, ts = oap._aware(now), ts or stamp_now()
    event_map = dict((identity or {}).get("event_map") or {})
    games = list(((identity or {}).get("games") or {}).keys()) or list(event_map)
    started = set(oap.started_games(games, kickoffs, now=now))
    due = [g for g in games if g in event_map and g not in started]
    out = {"called": False, "refused": None, "reused": False, "ts": None, "credits_planned": 0.0,
           "credits_billed": None, "priced": [], "omitted": [], "skipped_started": sorted(started),
           "skipped_unlisted": sorted(g for g in games if g not in event_map),
           "skipped_budget": [], "unmatched_events": 0, "quota_preflight": None,
           "credits_held_for_closes": 0.0, "billing_cycle": BILLING_CYCLE}
    if not due:
        out["refused"] = "no listed pregame game to price"
        return out
    fresh = answered_since(conn, due, now=now, max_age_hours=fresh_hours)
    if all(g in fresh for g in due):
        out.update(reused=True, ts=max(fresh.values()))
        return out
    ob = cfg.get("odds_budget") or {}
    budget = budget or oap.CreditBudget(conn, int(ob.get("monthly_credits", 500)), int(ob.get("reserve", 50)))
    if network or quota_fetch is not None:
        out["quota_preflight"] = pre = oap.quota_preflight(cfg, budget, quota_fetch=quota_fetch)
        if not pre["ok"]:
            out.update(refused=pre["reason"], skipped_budget=due)
            return out
    cost = request_cost(cfg, markets)
    held = sum(oap.outstanding_holds(conn, budget.month, now=now).values())
    out["credits_held_for_closes"] = held
    if not budget.can_spend(cost + held):
        out.update(refused=(f"{cost:.0f} credit(s) not spendable beside {held:.0f} held for "
                            f"prop closes ({budget.remaining:.0f} left)"), skipped_budget=due)
        return out
    params = {"apiKey": cfg.get("odds_api_key", ""), "markets": ",".join(markets),
              "oddsFormat": "decimal", "dateFormat": "iso", "commenceTimeFrom": oap._iso(now)}
    if cfg.get("books"):
        params["bookmakers"] = ",".join(cfg["books"])
    else:
        params["regions"] = str(cfg.get("regions", "us"))
    try:
        payload, headers = fetch(f"{oap.BASE}/sports/{oap.SPORT}/odds", params)
    except Exception as exc:  # noqa: BLE001 -- degrade: the games run without game lines
        out["refused"] = f"bulk game-odds call failed ({type(exc).__name__}: {exc})"
        return out
    budget.spend(cost, headers=headers)
    billed = oap._header_float(headers, "x-requests-last")
    out.update(called=True, ts=ts, credits_planned=cost, credits_billed=billed)
    game_of = {ev: g for g, ev in event_map.items()}
    rows, receipts, seen = [], [], set()
    for ev in payload if isinstance(payload, list) else []:
        gid = game_of.get(ev.get("id")) if isinstance(ev, dict) else None
        if gid is None:
            out["unmatched_events"] += 1
            continue
        if gid not in due or gid in seen:
            continue
        seen.add(gid)
        ct = oap._parse_clock(ev.get("commence_time"))
        ev_rows = [] if ct is not None and ct <= now else parse_event_game_lines(ev, ts)
        state = ("under_way_at_provider" if ct is not None and ct <= now
                 else "answered" if ev_rows else "answered_no_markets")
        for r in ev_rows:
            r.update(game_id=gid, event_id=ev["id"])
        rows.extend(ev_rows)
        receipts.append({"ts": ts, "game_id": gid, "event_id": ev["id"], "state": state,
                         "n_rows": len(ev_rows), "books": json.dumps(sorted({r["book"] for r in ev_rows})),
                         "markets": json.dumps(sorted({r["market"] for r in ev_rows})),
                         "commence_time": oap._iso(ct), "call_credits_planned": cost,
                         "call_credits_billed": billed})
    for gid in due:
        if gid not in seen:
            receipts.append({"ts": ts, "game_id": gid, "event_id": event_map[gid],
                             "state": "omitted_by_provider", "n_rows": 0, "books": "[]", "markets": "[]",
                             "commence_time": None, "call_credits_planned": cost,
                             "call_credits_billed": billed})
    if rows:
        dbmod.upsert(conn, "game_lines", rows, ["ts", "game_id", "book", "market", "side"])
    dbmod.upsert(conn, "game_line_pulls", receipts, ["ts", "game_id"])
    out["priced"] = sorted({r["game_id"] for r in rows})
    out["omitted"] = sorted(g for g in due if g not in out["priced"])
    print(f"[oddsapi-games] one bulk call: {len(out['priced'])}/{len(due)} due game(s) with game lines; "
          f"{cost:.0f} credit(s) planned, billed {billed if billed is not None else 'unknown'}; "
          f"{held:.0f} held for prop closes untouched")
    return out


def game_line_coverage(game_ids: List[str], cfg: Dict, pull: Optional[Dict] = None, conn=None,
                       now: Optional[dt.datetime] = None, requested: bool = True,
                       reason: Optional[str] = None, under_way=None, markets=GAME_MARKETS,
                       max_age_hours: float = GAME_LINES_MAX_AGE_HOURS) -> Dict:
    """One row per scheduled game for the GAME-line product (not props).

    ``state``: ``priced``, ``started`` (a pregame line, if any, is research
    only), ``answered_no_markets``, ``omitted_by_provider``,
    ``under_way_at_provider``, ``not_in_events_listing``, ``skipped_budget``,
    ``stale_quotes_only``, ``not_acquired`` or ``no_current_quotes``; when not
    requested, ``reason``. Each row keeps the capture clock, the books' own
    clocks, books per market and the markets the answer did not carry."""
    now = oap._aware(now)
    floor = oap._iso(now - dt.timedelta(hours=float(max_age_hours)))
    pull = pull or {}
    books_req = [str(b) for b in (cfg.get("books") or [])]
    latest = _latest(conn, list(game_ids), now) if conn is not None and requested else {}
    board = load_game_lines(conn, list(game_ids), now, max_age_hours) if conn is not None and requested else []
    by: Dict[str, List[Dict]] = {}
    for r in board:
        by.setdefault(r["game_id"], []).append(r)
    started = set(pull.get("skipped_started") or []) | set(under_way or [])
    games: Dict[str, Dict] = {}
    for gid in game_ids:
        rows, snap = by.get(gid, []), latest.get(gid)
        per_market = {m: sorted({r["book"] for r in rows if r["market"] == m}) for m in markets}
        clocks = sorted(str(r["book_clock"]) for r in rows if r.get("book_clock"))
        if not requested:
            state = reason or "odds_not_requested"
        elif gid in started:
            state = "started"
        elif rows:
            state = "priced"
        elif snap and snap["ts"] >= floor:
            state = snap["state"]
        elif gid in (pull.get("skipped_unlisted") or []):
            state = "not_in_events_listing"
        elif gid in (pull.get("skipped_budget") or []):
            state = "skipped_budget"
        elif snap:
            state = "stale_quotes_only"
        elif pull.get("refused"):
            state = "not_acquired"
        else:
            state = "no_current_quotes"
        cap = (snap or {}).get("ts")
        cap_dt = oap._parse_clock(cap)
        games[gid] = {
            "state": state, "capture_clock": cap,
            "quote_age_hours": round((now - cap_dt).total_seconds() / 3600, 2) if cap_dt else None,
            "book_clock_min": clocks[0] if clocks else None, "book_clock_max": clocks[-1] if clocks else None,
            "books_by_market": per_market, "markets_offered": [m for m in markets if per_market[m]],
            "markets_missing": [m for m in markets if not per_market[m]] if requested else [],
            "books_missing": ([b for b in books_req if not any(b in v for v in per_market.values())]
                              if requested else []),
            "started": gid in started, "reused_cached_answer": bool(pull.get("reused"))}
    by_state: Dict[str, int] = {}
    for r in games.values():
        by_state[r["state"]] = by_state.get(r["state"], 0) + 1
    return {"games": games, "summary": {
        "product": "game lines (h2h/spreads/totals) -- distinct from player props",
        "n_games": len(games), "priced": by_state.get("priced", 0), "by_state": by_state,
        "call": {k: pull.get(k) for k in ("called", "reused", "refused", "ts", "credits_planned",
                                          "credits_billed", "credits_held_for_closes")},
        "billing_cycle": BILLING_CYCLE, "as_of": oap._iso(now)}}


def coverage_text(cov: Dict) -> str:
    s = cov["summary"]
    missing = [f"{g} ({r['state']})" for g, r in cov["games"].items() if r["state"] != "priced"]
    return (f"Game lines: {s['priced']}/{s['n_games']} game(s) with offered moneyline/spread/total"
            + (f"; not priced: {', '.join(missing)}" if missing else "") + ".")
