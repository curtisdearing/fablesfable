"""Credit-budgeted player-prop line puller (The Odds API v4, free tier).

THE HARD RULE (self-enforced, not advisory): the free tier is 500 credits a
month, and player props are per-event calls costing ``len(markets) x
len(regions)`` credits each. :class:`CreditBudget` keeps a persistent ledger
(``api_credits`` table, one row per calendar month) and REFUSES any call that
would push the month past ``monthly_credits - reserve``. Refusal is not an
error -- the pipeline continues and the untouched games are tagged
``no_market`` (PROP_SHORTLISTER_SPEC.md §3 graceful degradation).

Every scheduled game of the week is requested, ordered by kickoff (soonest
first; see :func:`rotation_order`), capped at ``max_prop_games_per_run``
(config.json: 16, the largest NFL slate). The cap is a safety valve, not a
rotation: at 5 credits an event a 16-game Wednesday costs 80 credits, and the
same again for the pre-kick closes. :func:`credit_plan` does that arithmetic
up front, prints it, and :func:`pull_week_props` enforces it -- when the
month cannot afford every game, the games kicking off soonest get their
props and the rest are reported under ``skipped_budget`` (never fetched).
The Wednesday run also HOLDS one close per game it pulls (``reserve_close``)
so a game is never priced on Wednesday and then left without the credit for
its T-90 close.

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
import math
from typing import Callable, Dict, List, Optional

import pandas as pd

from .. import db as dbmod
from ..freshness import stamp_now
from ._http import get_json, get_json_and_headers, get_json_with_headers
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


def _header_float(headers: Optional[Dict], key: str) -> Optional[float]:
    """A finite float from a provider header (case-insensitive), else None."""
    for k, v in (headers or {}).items():
        if str(k).lower() == key and v is not None:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f if math.isfinite(f) and f >= 0 else None
    return None


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
        self.reserve = float(reserve)
        self.month = month or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m")
        row = dbmod.query_df(conn, "SELECT used FROM api_credits WHERE month=?", (self.month,))
        self.used = float(row.iloc[0]["used"]) if not row.empty else 0.0

    def can_spend(self, credits: float) -> bool:
        return (self.used + credits) <= self.ceiling

    def spend(self, credits: float, headers: Optional[Dict] = None) -> None:
        if not self.can_spend(credits):
            raise BudgetExceeded(
                f"refusing to spend {credits} credits: {self.used}/{self.ceiling} used in {self.month}")
        # trust the API's own accounting when it reports it -- but never let a
        # LOWER figure (stale header, provider-side reset) loosen the gate
        # mid-month: then this call is counted on top at its own charge
        # (x-requests-last) or, unknown, at the per-event bound.
        reported = _header_float(headers, "x-requests-used")
        if reported is not None and reported >= self.used:
            self.used = reported
        else:
            last = _header_float(headers, "x-requests-last")
            self.used += last if last is not None and reported is not None else credits
        dbmod.upsert(self.conn, "api_credits", [{
            "month": self.month, "used": self.used,
            "last_headers": json.dumps(dict(headers or {}))[:500],
            "updated_at": stamp_now(),
        }], ["month"])

    @property
    def remaining(self) -> float:
        return max(self.ceiling - self.used, 0.0)

    def reconcile(self, used: float, remaining: float, headers: Optional[Dict] = None) -> None:
        """Adopt the provider's own count BEFORE spending, and never plan past
        what the provider says is left (less the reserve), whatever the
        configured monthly figure claims."""
        self.used = float(used)
        self.ceiling = min(self.ceiling, float(used) + float(remaining) - self.reserve)
        dbmod.upsert(self.conn, "api_credits", [{
            "month": self.month, "used": self.used,
            "last_headers": json.dumps(dict(headers or {}))[:500],
            "updated_at": stamp_now(),
        }], ["month"])


def quota_preflight(cfg: Dict, budget: CreditBudget,
                    quota_fetch: Optional[Callable] = None) -> Dict:
    """Read the provider's quota from the FREE events listing and reconcile
    the ledger before the first metered call.

    The local ledger only learns the provider's count from a metered
    response, so a stale ledger (2026-09-22: 165 local vs 336 at the
    provider) would otherwise authorize the first paid request on its own
    arithmetic. Missing, unparseable or inconsistent quota headers return
    ``ok=False`` and the caller spends nothing."""
    fetch = quota_fetch or get_json_and_headers
    try:
        _, headers = fetch(f"{BASE}/sports/{SPORT}/events", {"apiKey": cfg.get("odds_api_key", "")})
        headers = {str(k).lower(): v for k, v in (headers or {}).items()}
        used = float(headers["x-requests-used"])
        remaining = float(headers["x-requests-remaining"])
        last = float(headers.get("x-requests-last") or 0)
        # float() accepts 'nan'/'inf', and NaN passes every comparison
        if not all(math.isfinite(v) for v in (used, remaining, last)) \
                or used < 0 or remaining < 0 or last != 0:
            raise ValueError(f"inconsistent quota headers {headers}")
    except Exception as exc:  # noqa: BLE001 -- unknown quota is a refusal, never a pass
        return {"ok": False, "reason": f"provider quota unverified ({type(exc).__name__}: {exc})"}
    budget.reconcile(used, remaining, headers=headers)
    return {"ok": True, "used": used, "remaining": remaining, "ceiling": budget.ceiling}


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


def match_player_ids(rows: List[Dict], candidates: pd.DataFrame,
                     roster_rows: Optional[List[Dict]] = None,
                     game_teams: Optional[Dict[str, set]] = None) -> List[Dict]:
    """Attach gsis player_ids by normalized name against the candidate pool.

    Ambiguous or unknown names stay ``player_id=None`` (kept + queryable);
    they simply can't join a projection, so they never mint an edge.

    Order, first unique answer wins:
      1. the candidate's own name, exactly;
      2. the AUTHORITATIVE full name for a candidate id from ``roster_rows``
         (the active-roster snapshot: ``{player_id, name, team}``);
      3. the candidate's abbreviation: equal last name(s) and the book's
         first name starting with the candidate's first token, so 'Bi.Robinson'
         can take 'Bijan Robinson' but never 'Brian Robinson Jr.'.
    ``game_teams`` ({game_id: {teams}}) confines each row to the two teams of
    its own game when the candidates carry ``team``. Ambiguity at any step is
    final: it is never resolved by falling through to a weaker rule.
    """
    has_team = "team" in candidates.columns
    cols = ["player_id", "name"] + (["team"] if has_team else [])
    pool = []
    for r in candidates[cols].drop_duplicates().itertuples(index=False):
        key = normalize_name(r.name)
        parts = key.split()
        pool.append((r.player_id, key, parts[0] if parts else "", parts[1:],
                     getattr(r, "team", None) if has_team else None))
    full: Dict[str, str] = {}
    for rr in roster_rows or []:
        if rr.get("player_id") and rr.get("name"):
            full[str(rr["player_id"])] = normalize_name(rr["name"])

    def unique(pids):
        pids = set(pids)
        return (next(iter(pids)) if len(pids) == 1 else None), len(pids)

    for row in rows:
        key = normalize_name(row["player_name"])
        parts = key.split()
        teams = (game_teams or {}).get(row.get("game_id")) if has_team else None
        cands = [c for c in pool if teams is None or c[4] in teams]
        pid, n = unique(c[0] for c in cands if c[1] == key)
        if n == 0 and full:
            pid, n = unique(c[0] for c in cands if full.get(str(c[0])) == key)
        if n == 0 and len(parts) >= 2:
            pid, n = unique(c[0] for c in cands
                            if c[2] and c[3] == parts[1:] and parts[0].startswith(c[2]))
        row["player_id"] = pid
    return rows


PROP_LINE_COLS = ["game_id", "market", "player_id", "point", "over_price",
                  "under_price", "book", "consensus_p_over", "n_books",
                  "over_book", "under_book", "over_ts", "under_ts"]

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
        # (book, side, point) -> capture clock of THAT quote row, so a lean can
        # carry one exact executable quote identity (book, side, point, price, ts)
        clock: Dict[tuple, Optional[str]] = {}
        for r in grp.itertuples(index=False):
            clock[(r.book, r.side, float(r.point))] = getattr(r, "ts", None)
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
                "over_book": cons["best_a_book"], "under_book": cons["best_b_book"],
                "over_ts": clock.get((cons["best_a_book"], "over", point)),
                "under_ts": clock.get((cons["best_b_book"], "under", point)),
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
                "over_book": best_book, "under_book": None,
                "over_ts": clock.get((best_book, "over", 0.5)), "under_ts": None,
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
    1. **FUTURE** -- kickoff beyond the horizon: soonest kickoff first, then
       least-recently-pulled, then ``game_id``. When the budget rations a
       Wednesday, the games that kick off first are the ones priced (2026
       Week 3, #26). A game with an UNKNOWN kickoff sorts at the head of
       this tier on the rotation clock alone -- it cannot be ranked by time.
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
        iso = ko.astimezone(dt.timezone.utc).isoformat()
        if ko - now <= horizon:
            return (0, iso, "", game_id)
        return (1, iso, last.get(game_id) or "", game_id)

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


def close_reserve_for(game_id: str, cost_per_event: float,
                      kickoffs: Optional[Dict[str, dt.datetime]], month: str) -> float:
    """Credits to HOLD for this game's pre-kick close when pulling it now.

    The ledger is a calendar-month ledger (:class:`CreditBudget`), so a close
    that falls in a later month draws on a fresh quota and needs no hold
    here. A game with no known kickoff is assumed to close this month.
    """
    ko = (kickoffs or {}).get(game_id)
    if ko is None:
        return float(cost_per_event)
    if ko.tzinfo is None:
        ko = ko.replace(tzinfo=dt.timezone.utc)
    return float(cost_per_event) if ko.astimezone(dt.timezone.utc).strftime("%Y-%m") == month else 0.0


def credit_plan(budget: CreditBudget, cost_per_event: float, game_ids: List[str],
                kickoffs: Optional[Dict[str, dt.datetime]] = None,
                reserve_close: bool = False, cap: Optional[int] = None) -> Dict:
    """The credit arithmetic for one pull, BEFORE any credit is spent.

    ``game_ids`` is the pull order (started games already removed). Walks it
    the way :func:`pull_week_props` will -- pull cost plus, with
    ``reserve_close``, the hold for that game's close -- and counts how many
    games the month can afford. Pure: it reads the ledger and touches nothing.

    Returned (all numbers are credits)::

        {month, cost_per_event, ceiling, used, spendable, n_games,
         pull_cost, close_reserve, needed, affordable_games, rationed_games,
         affordable, rationed}
    """
    ids = list(game_ids)
    if cap is not None:
        ids = ids[:max(int(cap), 0)]
    holds = {g: (close_reserve_for(g, cost_per_event, kickoffs, budget.month)
                 if reserve_close else 0.0) for g in ids}
    spendable = budget.remaining
    committed = 0.0
    affordable: List[str] = []
    rationed: List[str] = []
    for g in ids:
        need = float(cost_per_event) + holds[g]
        if committed + need <= spendable and not rationed:
            affordable.append(g)
            committed += need
        else:
            rationed.append(g)
    return {
        "month": budget.month, "cost_per_event": float(cost_per_event),
        "ceiling": budget.ceiling, "used": budget.used, "spendable": spendable,
        "n_games": len(ids), "pull_cost": float(cost_per_event) * len(ids),
        "close_reserve": sum(holds.values()),
        "needed": float(cost_per_event) * len(ids) + sum(holds.values()),
        "affordable_games": len(affordable), "rationed_games": len(rationed),
        "affordable": affordable, "rationed": rationed,
    }


def plan_text(plan: Dict) -> str:
    """One line of the arithmetic, for the run log and the board's line note."""
    return (f"credit plan {plan['month']}: {plan['n_games']} game(s) x "
            f"{plan['cost_per_event']:.0f} = {plan['pull_cost']:.0f} to pull"
            + (f" + {plan['close_reserve']:.0f} held for pre-kick closes"
               if plan['close_reserve'] else "")
            + f" = {plan['needed']:.0f} needed; {plan['spendable']:.0f} spendable "
            f"({plan['used']:.0f} used of {plan['ceiling']:.0f}) -> "
            f"{plan['affordable_games']} affordable, {plan['rationed_games']} rationed "
            f"at full per-event billing (the provider bills per market returned, so a "
            f"thin slate can afford more; every call still re-checks the hard stop)")


class BillingTally:
    """Per-pull credit accounting, kept apart by how each figure is KNOWN.

    ``measured``  -- sum of the provider's ``x-requests-last`` (what each of
                     OUR requests was charged; per market returned).
    ``estimated`` -- calls answered without that header, counted at the
                     per-event bound (never labelled provider billing).
    ``account_delta`` -- provider ``x-requests-used`` at the last answered
                     call minus the figure before the first one. Account-wide:
                     it includes any other use of the same key and can be
                     negative after a provider reset or a stale header.
    ``planned``   -- the per-event upper bound the budget gate charged.
    """

    def __init__(self, start_used: float):
        self.start_used = float(start_used)
        self.measured = self.estimated = self.planned = 0.0
        self.last_used: Optional[float] = None

    def add(self, cost: float, headers: Optional[Dict]) -> None:
        self.planned += cost
        last = _header_float(headers, "x-requests-last")
        if last is None:
            self.estimated += cost
        else:
            self.measured += last
        used = _header_float(headers, "x-requests-used")
        if used is not None:
            self.last_used = used

    @property
    def spent(self) -> float:
        """Request-attributed credits: measured where known, else the bound."""
        return self.measured + self.estimated

    @property
    def account_delta(self) -> Optional[float]:
        return None if self.last_used is None else self.last_used - self.start_used

    def fields(self) -> Dict:
        return {"credits_spent": self.spent, "credits_billed_measured": self.measured,
                "credits_estimated": self.estimated, "credits_planned": self.planned,
                "account_usage_delta": self.account_delta}


def billing_text(res: Dict) -> str:
    """How the credits of one pull are known, for the run log and line note."""
    parts = []
    if res.get("credits_billed_measured") or not res.get("credits_estimated"):
        parts.append(f"provider billed {res.get('credits_billed_measured') or 0:.0f} credit(s) "
                     f"per x-requests-last")
    if res.get("credits_estimated"):
        parts.append(f"{res['credits_estimated']:.0f} credit(s) ESTIMATED at the per-event "
                     f"bound (no provider billing header)")
    delta = res.get("account_usage_delta")
    if delta is not None:
        parts.append(f"account usage {delta:+.0f} (whole key, any consumer)")
    return "; ".join(parts) + f" (upper bound planned {res.get('credits_planned') or 0:.0f})"


def pull_week_props(cfg: Dict, event_map: Dict[str, str], conn=None,
                    fetch: Optional[Callable] = None,
                    budget: Optional[CreditBudget] = None,
                    ts: Optional[str] = None,
                    kickoffs: Optional[Dict[str, dt.datetime]] = None,
                    now: Optional[dt.datetime] = None,
                    reserve_close: bool = False,
                    quota_fetch: Optional[Callable] = None) -> Dict:
    """Pull props for the week's games, kickoff-ordered and budget-capped.

    ``event_map``: {nflverse game_id -> odds-api event id} (built by the
    pipeline from team names + kickoff dates).

    ``kickoffs`` ({game_id -> aware datetime}) makes the order kickoff-aware:
    games starting within :data:`IMMINENT_HORIZON_HOURS` are pulled first,
    the rest soonest-kickoff first, and games that have already started are
    not pulled at all. Omit it and the order is the original
    least-recently-pulled rotation.

    ``reserve_close=True`` (the Wednesday run) holds one more event's worth
    of credits per game pulled, for that game's pre-kick close in the same
    ledger month: a game is either priced on Wednesday AND affordable to
    close, or reported under ``skipped_budget``. The arithmetic is printed
    before the first call and returned as ``plan`` (:func:`credit_plan`).
    Returns::

        {"pulled": [game_ids], "priced": [...], "empty": [...],
         "skipped_budget": [...], "skipped_cap": [...], "skipped_started": [...],
         "rows_written": int, "credits_spent": float, "credits_planned": float,
         "budget_remaining": float, "plan": {...}}

    ``pulled`` is every game whose call was answered; ``priced`` the ones
    that returned at least one quote and ``empty`` the ones that returned
    none (books not posted yet). ``credits_spent`` is what the ledger moved
    by -- the provider's own x-requests-used when it reports it, which bills
    per market RETURNED (an empty event costs nothing) -- and
    ``credits_planned`` the per-event upper bound the budget gate charged
    (2026-09-23: 34 billed, 70 planned, 4 of 14 answered games empty).
    """
    fetch = fetch or get_json_with_headers
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
    # Provider quota BEFORE the plan. Required on the real network path; an
    # injected ``fetch`` (offline tests, captured-payload replay) touches no
    # meter and preflights only when handed a ``quota_fetch``.
    preflight = None
    if fetch is get_json_with_headers or quota_fetch is not None:
        preflight = quota_preflight(cfg, budget, quota_fetch=quota_fetch)
        print(f"[oddsapi] quota preflight: {preflight}")
        if not preflight["ok"]:
            live = [g for g in ordered if g not in started]
            return {"pulled": [], "skipped_budget": live, "skipped_cap": [],
                    "skipped_started": [g for g in ordered if g in started],
                    "skipped_error": [], "rows_written": 0, "credits_spent": 0.0,
                    "credits_planned": 0.0, "priced": [], "empty": [],
                    "close_reserved": 0.0, "budget_remaining": 0.0, "ts": ts,
                    "plan": None, "book_coverage": book_coverage(cfg, {}),
                    "quota_preflight": preflight}
    # The arithmetic, before the first metered call, in the log and the result.
    plan = credit_plan(budget, cost_per_event, [g for g in ordered if g not in started],
                       kickoffs=kickoffs, reserve_close=reserve_close, cap=cap)
    print(f"[oddsapi] {plan_text(plan)}")
    pulled, skipped_budget, skipped_cap = [], [], []
    skipped_started: List[str] = []
    skipped_error: List[Dict] = []
    all_rows: List[Dict] = []
    tally = BillingTally(budget.used)   # how each credit is known (BillingTally)
    empty: List[str] = []   # answered with no quote (books not posted yet)
    reserved = 0.0          # closes held for games pulled in THIS call
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
        hold = (close_reserve_for(game_id, cost_per_event, kickoffs, budget.month)
                if reserve_close else 0.0)
        if not budget.can_spend(cost_per_event + reserved + hold):
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
        tally.add(cost_per_event, headers)
        reserved += hold
        rows = parse_event_props(payload, ts)
        for r in rows:
            r["game_id"] = game_id
        all_rows.extend(rows)
        pulled.append(game_id)
        if not rows:
            empty.append(game_id)
        books_by_game[game_id] = books_in_payload(payload)

    written = 0
    if all_rows:
        written = dbmod.upsert(conn, "lines", all_rows,
                               ["ts", "game_id", "book", "market", "player_name", "side"])
    coverage = book_coverage(cfg, books_by_game)
    if coverage["absent_from_provider_response"]:
        print(f"[oddsapi] books requested but absent from the provider response: "
              f"{coverage['absent_from_provider_response']} (returned: {coverage['returned']})")
    priced = [g for g in pulled if g not in empty]
    print(f"[oddsapi] requested {len(pulled)} game(s): {len(priced)} priced, "
          f"{len(empty)} no quotes: {', '.join(empty) or 'none'}; "
          f"{billing_text(tally.fields())}; "
          f"{reserved:.0f} held for closes, {budget.remaining:.0f} left in {budget.month}; "
          f"skipped: budget={len(skipped_budget)} cap={len(skipped_cap)} "
          f"started={len(skipped_started)} error={len(skipped_error)}")
    return {"pulled": pulled, "priced": priced, "empty": empty,
            "skipped_budget": skipped_budget, "skipped_cap": skipped_cap,
            "skipped_started": skipped_started, "skipped_error": skipped_error,
            "rows_written": written, **tally.fields(),
            "close_reserved": reserved,
            "budget_remaining": budget.remaining, "ts": ts,
            "book_coverage": coverage, "plan": plan, "quota_preflight": preflight}


def resnap_lines(cfg: Dict, event_map: Dict[str, str], conn=None,
                 fetch: Optional[Callable] = None, ts: Optional[str] = None,
                 quota_fetch: Optional[Callable] = None) -> Dict:
    """Second snapshot for SPECIFIC games (no rotation, no per-run cap — the
    caller passes exactly the games that already have entry lines and kick
    soon). This is what makes CLV resolvable: entry = Wednesday snapshot,
    close = this pre-kickoff snapshot. Budget hard-stop still applies."""
    fetch = fetch or get_json_with_headers
    conn = conn or dbmod.connect()
    ob = cfg.get("odds_budget") or {}
    budget = CreditBudget(conn, int(ob.get("monthly_credits", 500)),
                          int(ob.get("reserve", 50)))
    from ..config import prop_markets_external
    markets = prop_markets_external(cfg)
    regions = str(cfg.get("regions", "us"))
    cost = float(len(markets) * len(regions.split(",")))
    ts = ts or stamp_now()
    if fetch is get_json_with_headers or quota_fetch is not None:
        preflight = quota_preflight(cfg, budget, quota_fetch=quota_fetch)
        print(f"[oddsapi] resnap quota preflight: {preflight}")
        if not preflight["ok"]:
            return {"pulled": [], "priced": [], "empty": [],
                    "skipped_budget": sorted(event_map), "rows_written": 0,
                    "credits_spent": 0.0, "credits_planned": 0.0,
                    "ts": ts, "budget_remaining": 0.0,
                    "book_coverage": book_coverage(cfg, {}), "quota_preflight": preflight}
    pulled, skipped, rows, empty = [], [], [], []
    tally = BillingTally(budget.used)
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
        tally.add(cost, headers)
        game_rows = parse_event_props(payload, ts)
        for r in game_rows:
            r["game_id"] = game_id
            rows.append(r)
        pulled.append(game_id)
        if not game_rows:
            empty.append(game_id)
        books_by_game[game_id] = books_in_payload(payload)
    written = dbmod.upsert(conn, "lines", rows,
                           ["ts", "game_id", "book", "market", "player_name", "side"]) if rows else 0
    coverage = book_coverage(cfg, books_by_game)
    if coverage["absent_from_provider_response"]:
        print(f"[oddsapi] resnap: books requested but absent from the provider response: "
              f"{coverage['absent_from_provider_response']} (returned: {coverage['returned']})")
    print(f"[oddsapi] resnap: {len(pulled)} game(s) answered, {len(empty)} no quotes: "
          f"{', '.join(empty) or 'none'}; {billing_text(tally.fields())}")
    return {"pulled": pulled, "priced": [g for g in pulled if g not in empty], "empty": empty,
            "skipped_budget": skipped, "rows_written": written,
            **tally.fields(),
            "ts": ts, "budget_remaining": budget.remaining, "book_coverage": coverage}
