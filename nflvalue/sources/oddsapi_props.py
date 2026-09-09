"""Credit-budgeted player-prop line puller (The Odds API v4, free tier).

THE HARD RULE (self-enforced, not advisory): the free tier is 500 credits a
month, and player props are per-event calls costing ``len(markets) x
len(regions)`` credits each. :class:`CreditBudget` keeps a persistent ledger
(``api_credits`` table, one row per calendar month) and REFUSES any call that
would push the month past ``monthly_credits - reserve``. Refusal is not an
error -- the pipeline continues and the untouched games are tagged
``no_market`` (PROP_SHORTLISTER_SPEC.md §3 graceful degradation).

Because you can't afford props for every game, each run pulls a ROTATING
subset: games are ordered least-recently-pulled first (from the ``lines``
table), capped at ``max_prop_games_per_run``. Over a few weeks every game
cycles through.

Snapshots are idempotent: rows key on (ts, game_id, book, market,
player_name, side), so re-running an identical pull cannot duplicate.
Player names from the books are matched to gsis ids against the week's
candidate pool by normalized name (+ conservatively by team when known);
unmatched rows are stored with ``player_id=NULL`` -- visible, never guessed.

Everything is injectable/mockable: pass ``fetch=`` a callable in tests, and
``tests/fixtures/oddsapi_event_props_synthetic.json`` carries a SYNTHETIC
(clearly labeled) v4-shaped payload -- a real one can't be recorded without a
personal API key and an in-season slate.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Callable, Dict, List, Optional

import pandas as pd

from .. import db as dbmod
from ..freshness import stamp_now
from ._http import get_json
from .availability import normalize_name

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# Odds API market key <-> our market names
ODDS_TO_MARKET = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
    "player_rush_attempts": "rush_attempts",
    "player_pass_attempts": "pass_attempts",
    "player_anytime_td": "anytime_td",
}
MARKET_TO_ODDS = {v: k for k, v in ODDS_TO_MARKET.items()}


class BudgetExceeded(RuntimeError):
    """Raised only if a caller tries to FORCE a pull past the hard stop."""


class CreditBudget:
    """Persistent monthly credit ledger with a hard stop.

    ``monthly_credits`` and ``reserve`` come from config ("odds_budget");
    the spendable ceiling is ``monthly_credits - reserve`` (default 500-50 =
    450) so estimation drift can never brush the real limit.
    """

    def __init__(self, conn, monthly_credits: int = 500, reserve: int = 50,
                 month: Optional[str] = None):
        self.conn = conn
        self.ceiling = float(monthly_credits) - float(reserve)
        self.month = month or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m")
        row = dbmod.query_df(conn, "SELECT used FROM api_credits WHERE month=?", (self.month,))
        self.used = float(row.iloc[0]["used"]) if not row.empty else 0.0

    def can_spend(self, credits: float) -> bool:
        return (self.used + credits) <= self.ceiling

    def spend(self, credits: float, headers: Optional[Dict] = None) -> None:
        if not self.can_spend(credits):
            raise BudgetExceeded(
                f"refusing to spend {credits} credits: {self.used}/{self.ceiling} used in {self.month}")
        # trust the API's own accounting when it reports it
        reported = None
        if headers:
            for k in ("x-requests-used", "X-Requests-Used"):
                if headers.get(k) is not None:
                    try:
                        reported = float(headers[k])
                    except (TypeError, ValueError):
                        reported = None
        self.used = reported if reported is not None else self.used + credits
        dbmod.upsert(self.conn, "api_credits", [{
            "month": self.month, "used": self.used,
            "last_headers": json.dumps(dict(headers or {}))[:500],
            "updated_at": stamp_now(),
        }], ["month"])

    @property
    def remaining(self) -> float:
        return max(self.ceiling - self.used, 0.0)


# --------------------------------------------------------------------------- #
# Parsing (pure)
# --------------------------------------------------------------------------- #
def parse_event_props(payload: Dict, ts: str) -> List[Dict]:
    """One v4 event-odds payload -> flat line rows (no ids matched yet)."""
    rows: List[Dict] = []
    for bk in payload.get("bookmakers", []) or []:
        book = bk.get("key")
        for mkt in bk.get("markets", []) or []:
            market = ODDS_TO_MARKET.get(mkt.get("key"))
            if market is None:
                continue
            for o in mkt.get("outcomes", []) or []:
                player_name = o.get("description") or ""
                side_raw = str(o.get("name", "")).lower()
                side = {"over": "over", "yes": "over", "under": "under", "no": "under"}.get(side_raw)
                if not player_name or side is None:
                    continue
                # FAIL CLOSED (Phase 7.2). A malformed/truncated payload used
                # to mint a phantom line here: an outcome carrying no price and
                # no point produced a row with price=None and point defaulted
                # to 0.5, i.e. a "receiving yards over 0.5" quote that never
                # existed. That row was then written to the `lines` table and
                # was indistinguishable from a real quote. A line we did not
                # receive is not a line.
                if o.get("price") is None:
                    continue
                if o.get("point") is None and market != "anytime_td":
                    # 0.5 is the fixed anytime-TD convention ONLY; for every
                    # other market a missing point is missing data.
                    continue
                rows.append({
                    "ts": ts, "game_id": None,  # filled by caller (odds event id != nflverse game_id)
                    "book": book, "market": market,
                    "player_id": None, "player_name": player_name, "side": side,
                    "point": float(o["point"]) if o.get("point") is not None else 0.5,
                    "price": float(o["price"]),
                })
    return rows


def books_in_payload(payload) -> List[str]:
    """Bookmaker keys present in one v4 event-odds payload (sorted, unique)."""
    if not isinstance(payload, dict):
        return []
    keys = {str(bk.get("key")) for bk in (payload.get("bookmakers") or []) if bk.get("key")}
    return sorted(keys)


def book_coverage(cfg: Dict, books_by_game: Dict[str, List[str]]) -> Dict:
    """Requested-vs-returned bookmaker diagnostic for a pull.

    The client stores every bookmaker the provider returns (no per-book
    filter), so a book that is requested via ``bookmakers=`` yet absent here
    was absent from the PROVIDER'S response at that snapshot -- not filtered,
    not rejected.  A stored ``lines`` table with one book therefore means the
    provider returned one book.  This puts that fact in the run log.
    """
    requested = [str(b) for b in (cfg.get("books") or [])]
    returned = sorted({b for books in books_by_game.values() for b in books})
    return {
        "requested": requested,
        "selector": "bookmakers" if requested else f"regions={cfg.get('regions', 'us')}",
        "returned": returned,
        "absent_from_provider_response": [b for b in requested if b not in returned],
        "unrequested_returned": [b for b in returned if requested and b not in requested],
        "by_game": {g: list(b) for g, b in books_by_game.items()},
    }


def match_player_ids(rows: List[Dict], candidates: pd.DataFrame) -> List[Dict]:
    """Attach gsis player_ids by normalized name against the candidate pool.

    Ambiguous or unknown names stay ``player_id=None`` (kept + queryable);
    they simply can't join a projection, so they never mint an edge.
    """
    lookup: Dict[str, set] = {}
    for r in candidates[["player_id", "name"]].drop_duplicates().itertuples(index=False):
        lookup.setdefault(normalize_name(r.name), set()).add(r.player_id)
    # candidate names are abbreviated ("A.St. Brown"); book names are full
    # ("Amon-Ra St. Brown") -- also index by "first-initial lastname"
    fi_lookup: Dict[str, set] = {}
    for key, pids in lookup.items():
        parts = key.split()
        if len(parts) >= 2:
            fi_lookup.setdefault(f"{parts[0][0]} {' '.join(parts[1:])}", set()).update(pids)

    for row in rows:
        key = normalize_name(row["player_name"])
        pids = lookup.get(key, set())
        if not pids:
            parts = key.split()
            if len(parts) >= 2:
                pids = fi_lookup.get(f"{parts[0][0]} {' '.join(parts[1:])}", set())
        row["player_id"] = pids.copy().pop() if len(pids) == 1 else None
    return rows


PROP_LINE_COLS = ["game_id", "market", "player_id", "point", "over_price",
                  "under_price", "book", "consensus_p_over", "n_books"]

#: How old a stored quote may be and still price a board. Long enough to span
#: the gap between two scheduled runs (a game pulled Tuesday still prices
#: Wednesday's board), short enough that a genuinely abandoned line falls out.
MAX_LINE_AGE_HOURS = 60.0


def load_recent_lines(conn, game_ids: Optional[List[str]] = None,
                      max_age_hours: float = MAX_LINE_AGE_HOURS,
                      now: Optional[dt.datetime] = None) -> List[Dict]:
    """The most recent stored quote per (game, book, market, player, side).

    The pipeline used to price a board from ``SELECT * FROM lines WHERE ts=?``
    -- only the rows the CURRENT run had just pulled. Because the rotation
    prices four games a run, a game pulled on Tuesday was invisible on
    Wednesday's board and rendered ``NO_MARKET`` with real quotes sitting in
    the table. (2026-09-09: NE@SEA had DraftKings/BetMGM/HardRock rows from
    2026-09-08T17:56Z and still published with no market.)

    Rows older than ``max_age_hours`` are dropped rather than shown as
    current: a stale quote priced as live is worse than no quote. Freshness
    is the caller's to report -- each returned row keeps its own ``ts``.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    floor = (now - dt.timedelta(hours=float(max_age_hours))
             ).strftime("%Y-%m-%dT%H:%M:%SZ")
    params: List = [floor]
    where = "WHERE ts >= ?"
    if game_ids:
        where += f" AND game_id IN ({','.join('?' * len(game_ids))})"
        params.extend(game_ids)
    sql = f"""
        SELECT l.* FROM lines l
        JOIN (SELECT game_id, book, market, player_name, side, MAX(ts) AS ts
                FROM lines {where}
            GROUP BY game_id, book, market, player_name, side) m
          ON l.game_id = m.game_id AND l.book = m.book AND l.market = m.market
         AND l.player_name = m.player_name AND l.side = m.side AND l.ts = m.ts
    """
    df = dbmod.query_df(conn, sql, tuple(params))
    return [] if df.empty else df.to_dict("records")


def to_prop_lines_frame(rows: List[Dict], sharp_books=("pinnacle",),
                        sharp_weight: float = 2.0) -> pd.DataFrame:
    """Snapshot rows -> prop-line frame with CROSS-BOOK comparison.

    Per (game, market, player): pick the consensus point (the point quoted
    two-sided by the most books; deterministic tie-break), de-vig EVERY book
    at that point into a sharp-weighted CONSENSUS fair probability
    (``oddsmath.consensus_two_way`` -- the same engine the game-line app
    trusts), and carry the BEST available price per side with its book
    (line shopping: edge is judged vs consensus, captured at the best price).
    One-book markets still work (n_books=1 = consensus is that book)."""
    matched = [r for r in rows if r["player_id"] is not None]
    if not matched:
        return pd.DataFrame(columns=PROP_LINE_COLS)
    from .. import oddsmath

    df = pd.DataFrame(matched)
    out = []
    for (gid, market, pid), grp in df.groupby(["game_id", "market", "player_id"]):
        # books quoting BOTH sides, keyed by point
        two_sided: Dict[float, Dict[str, tuple]] = {}
        yes_only: Dict[str, float] = {}
        for book, b in grp.groupby("book"):
            overs = b[b["side"] == "over"]
            unders = b[b["side"] == "under"]
            if overs.empty:
                continue
            over = overs.iloc[0]
            if unders.empty:
                if market == "anytime_td" and over["price"]:
                    yes_only[book] = float(over["price"])
                continue
            pt = float(over["point"])
            two_sided.setdefault(pt, {})[book] = (float(over["price"]),
                                                  float(unders.iloc[0]["price"]))
        if two_sided:
            # consensus point: most two-sided books; ties -> alphabetically
            # first book's point (deterministic)
            point = sorted(two_sided,
                           key=lambda p: (-len(two_sided[p]), min(two_sided[p])))[0]
            cons = oddsmath.consensus_two_way(two_sided[point],
                                              sharp_books=sharp_books,
                                              sharp_weight=sharp_weight)
            if not cons:
                continue
            out.append({
                "game_id": gid, "market": market, "player_id": pid, "point": point,
                "over_price": cons["best_a"], "under_price": cons["best_b"],
                "book": f"{cons['best_a_book']}/{cons['best_b_book']}",
                "consensus_p_over": round(cons["p_a"], 4),
                "n_books": len(two_sided[point]),
            })
        elif yes_only:
            best_book = max(yes_only, key=lambda b: (yes_only[b], b))
            out.append({
                "game_id": gid, "market": market, "player_id": pid, "point": 0.5,
                "over_price": yes_only[best_book], "under_price": None,
                "book": best_book,
                "consensus_p_over": round(float(sum(
                    oddsmath.implied_prob(v) for v in yes_only.values()) / len(yes_only)), 4),
                "n_books": len(yes_only),
            })
    return pd.DataFrame(out, columns=PROP_LINE_COLS)


# --------------------------------------------------------------------------- #
# Fetch + persist
# --------------------------------------------------------------------------- #
def list_events(cfg: Dict, fetch: Optional[Callable] = None) -> List[Dict]:
    """The (credit-free) events listing: [{id, commence_time, home_team, away_team}]."""
    fetch = fetch or get_json
    return fetch(f"{BASE}/sports/{SPORT}/events", {"apiKey": cfg.get("odds_api_key", "")})


#: A game kicking off within this many hours is IMMINENT: it outranks the
#: rotation clock. Sized to comfortably exceed the longest gap between two
#: scheduled runs, so no game can pass through its last chance unpriced.
IMMINENT_HORIZON_HOURS = 24.0


def rotation_order(conn, game_ids: List[str],
                   kickoffs: Optional[Dict[str, dt.datetime]] = None,
                   now: Optional[dt.datetime] = None,
                   horizon_hours: float = IMMINENT_HORIZON_HOURS) -> List[str]:
    """Pull order: imminent kickoffs first, then the rotation clock.

    The original order was purely least-recently-pulled. That is fair but
    blind: on 2026-09-09 it spent all four of the Wednesday run's event-calls
    on Sunday games and skipped the game kicking off that night, so the
    opener reached kickoff with no market at all. Fairness over a month is
    worth nothing to a game that starts in five hours.

    Three tiers:

    0. **IMMINENT** -- kickoff in ``(now, now + horizon_hours]``, soonest
       first. A game we will not get another scheduled chance to price
       before it starts outranks a game five days out, whatever the
       rotation clock says.
    1. **FUTURE** -- kickoff beyond the horizon, or unknown. Least-recently-
       pulled first, then ``game_id``: the original round-robin, unchanged.
    2. **STARTED** -- kickoff already passed. Last, and normally unreachable
       because :func:`pull_week_props` skips them outright; a credit spent on
       a game in progress buys nothing.

    ``kickoffs=None`` reproduces the pre-2026-09-09 behaviour exactly, so
    every existing caller and test is unaffected.
    """
    if not game_ids:
        return []
    df = dbmod.query_df(conn, "SELECT game_id, MAX(ts) AS last_ts FROM lines GROUP BY game_id")
    last = dict(zip(df["game_id"], df["last_ts"])) if not df.empty else {}
    if not kickoffs:
        return sorted(game_ids, key=lambda g: (last.get(g) or "", g))

    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    horizon = dt.timedelta(hours=float(horizon_hours))

    def key(game_id: str):
        ko = kickoffs.get(game_id)
        if ko is None:                       # unknown kickoff -> rotation clock
            return (1, "", last.get(game_id) or "", game_id)
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=dt.timezone.utc)
        if ko <= now:
            return (2, "", "", game_id)
        if ko - now <= horizon:
            return (0, ko.astimezone(dt.timezone.utc).isoformat(), "", game_id)
        return (1, "", last.get(game_id) or "", game_id)

    return sorted(game_ids, key=key)


def started_games(game_ids: List[str], kickoffs: Optional[Dict[str, dt.datetime]],
                  now: Optional[dt.datetime] = None) -> List[str]:
    """Game ids whose kickoff has already passed (never worth a credit)."""
    if not kickoffs:
        return []
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    out = []
    for game_id in game_ids:
        ko = kickoffs.get(game_id)
        if ko is None:
            continue
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=dt.timezone.utc)
        if ko <= now:
            out.append(game_id)
    return out


def pull_week_props(cfg: Dict, event_map: Dict[str, str], conn=None,
                    fetch: Optional[Callable] = None,
                    budget: Optional[CreditBudget] = None,
                    ts: Optional[str] = None,
                    kickoffs: Optional[Dict[str, dt.datetime]] = None,
                    now: Optional[dt.datetime] = None) -> Dict:
    """Pull props for a rotating, budget-capped subset of the week's games.

    ``event_map``: {nflverse game_id -> odds-api event id} (built by the
    pipeline from team names + kickoff dates).

    ``kickoffs`` ({game_id -> aware datetime}) makes the order kickoff-aware:
    games starting within :data:`IMMINENT_HORIZON_HOURS` are pulled first and
    games that have already started are not pulled at all. Omit it and the
    order is the original least-recently-pulled rotation. Returns::

        {"pulled": [game_ids], "skipped_budget": [...], "skipped_cap": [...],
         "skipped_started": [...], "rows_written": int, "credits_spent": float,
         "budget_remaining": float}
    """
    fetch = fetch or get_json
    conn = conn or dbmod.connect()
    ob = cfg.get("odds_budget") or {}
    budget = budget or CreditBudget(conn, int(ob.get("monthly_credits", 500)),
                                    int(ob.get("reserve", 50)))
    from ..config import prop_markets_external
    markets = prop_markets_external(cfg)
    regions = str(cfg.get("regions", "us"))
    cost_per_event = float(len(markets) * len(regions.split(",")))
    cap = int(cfg.get("max_prop_games_per_run", 4))
    ts = ts or stamp_now()

    ordered = rotation_order(conn, list(event_map), kickoffs=kickoffs, now=now)
    started = set(started_games(list(event_map), kickoffs, now=now))
    pulled, skipped_budget, skipped_cap = [], [], []
    skipped_started: List[str] = []
    skipped_error: List[Dict] = []
    all_rows: List[Dict] = []
    spent = 0.0
    books_by_game: Dict[str, List[str]] = {}

    for game_id in ordered:
        # A game already under way cannot be bet from this board; spending a
        # metered credit on its live line buys nothing. Checked before the
        # cap so an in-progress game never consumes a slot either.
        if game_id in started:
            skipped_started.append(game_id)
            continue
        if len(pulled) >= cap:
            skipped_cap.append(game_id)
            continue
        if not budget.can_spend(cost_per_event):
            skipped_budget.append(game_id)
            continue
        params = {"apiKey": cfg.get("odds_api_key", ""),
                  "markets": ",".join(markets), "oddsFormat": "decimal"}
        # user's books (e.g. draftkings, betmgm, hardrockbet) beat a whole-
        # region pull: comparable prices AND a cheaper cost basis
        if cfg.get("books"):
            params["bookmakers"] = ",".join(cfg["books"])
        else:
            params["regions"] = regions
        # DEGRADE, DON'T ABORT (Phase 7.2). A single flaky HTTP call used to
        # propagate out of here and kill the entire weekly run -- after the
        # candidate pool had already been built. One dead event must cost that
        # event only; the game falls through to `no_market`, which is the
        # documented behaviour for a game whose lines we could not pull.
        # BudgetExceeded is deliberately NOT caught: overspending a metered
        # free tier is a hard stop, not a degradation.
        try:
            payload = fetch(f"{BASE}/sports/{SPORT}/events/{event_map[game_id]}/odds", params)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 -- surfaced in skipped_error
            skipped_error.append({"game_id": game_id,
                                  "error": f"{type(exc).__name__}: {exc}"})
            print(f"[oddsapi] pull failed for {game_id}: {type(exc).__name__}: {exc}")
            continue
        headers = payload.pop("_headers", None) if isinstance(payload, dict) else None
        budget.spend(cost_per_event, headers=headers)
        spent += cost_per_event
        rows = parse_event_props(payload, ts)
        for r in rows:
            r["game_id"] = game_id
        all_rows.extend(rows)
        pulled.append(game_id)
        books_by_game[game_id] = books_in_payload(payload)

    written = 0
    if all_rows:
        written = dbmod.upsert(conn, "lines", all_rows,
                               ["ts", "game_id", "book", "market", "player_name", "side"])
    coverage = book_coverage(cfg, books_by_game)
    if coverage["absent_from_provider_response"]:
        print(f"[oddsapi] books requested but absent from the provider response: "
              f"{coverage['absent_from_provider_response']} (returned: {coverage['returned']})")
    return {"pulled": pulled, "skipped_budget": skipped_budget, "skipped_cap": skipped_cap,
            "skipped_started": skipped_started, "skipped_error": skipped_error,
            "rows_written": written, "credits_spent": spent,
            "budget_remaining": budget.remaining, "ts": ts,
            "book_coverage": coverage}


def resnap_lines(cfg: Dict, event_map: Dict[str, str], conn=None,
                 fetch: Optional[Callable] = None, ts: Optional[str] = None) -> Dict:
    """Second snapshot for SPECIFIC games (no rotation, no per-run cap — the
    caller passes exactly the games that already have entry lines and kick
    soon). This is what makes CLV resolvable: entry = Wednesday snapshot,
    close = this pre-kickoff snapshot. Budget hard-stop still applies."""
    fetch = fetch or get_json
    conn = conn or dbmod.connect()
    ob = cfg.get("odds_budget") or {}
    budget = CreditBudget(conn, int(ob.get("monthly_credits", 500)),
                          int(ob.get("reserve", 50)))
    from ..config import prop_markets_external
    markets = prop_markets_external(cfg)
    regions = str(cfg.get("regions", "us"))
    cost = float(len(markets) * len(regions.split(",")))
    ts = ts or stamp_now()
    pulled, skipped, rows = [], [], []
    books_by_game: Dict[str, List[str]] = {}
    for game_id, event_id in sorted(event_map.items()):
        if not budget.can_spend(cost):
            skipped.append(game_id)
            continue
        params = {"apiKey": cfg.get("odds_api_key", ""),
                  "markets": ",".join(markets), "oddsFormat": "decimal"}
        if cfg.get("books"):
            params["bookmakers"] = ",".join(cfg["books"])
        else:
            params["regions"] = regions
        payload = fetch(f"{BASE}/sports/{SPORT}/events/{event_id}/odds", params)
        headers = payload.pop("_headers", None) if isinstance(payload, dict) else None
        budget.spend(cost, headers=headers)
        for r in parse_event_props(payload, ts):
            r["game_id"] = game_id
            rows.append(r)
        pulled.append(game_id)
        books_by_game[game_id] = books_in_payload(payload)
    written = dbmod.upsert(conn, "lines", rows,
                           ["ts", "game_id", "book", "market", "player_name", "side"]) if rows else 0
    coverage = book_coverage(cfg, books_by_game)
    if coverage["absent_from_provider_response"]:
        print(f"[oddsapi] resnap: books requested but absent from the provider response: "
              f"{coverage['absent_from_provider_response']} (returned: {coverage['returned']})")
    return {"pulled": pulled, "skipped_budget": skipped, "rows_written": written,
            "ts": ts, "budget_remaining": budget.remaining, "book_coverage": coverage}
