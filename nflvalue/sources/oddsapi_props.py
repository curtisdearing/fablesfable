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
                rows.append({
                    "ts": ts, "game_id": None,  # filled by caller (odds event id != nflverse game_id)
                    "book": book, "market": market,
                    "player_id": None, "player_name": player_name, "side": side,
                    "point": float(o["point"]) if o.get("point") is not None else 0.5,
                    "price": float(o["price"]) if o.get("price") is not None else None,
                })
    return rows


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
                "consensus_p_over": round(float(np_mean := sum(
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


def rotation_order(conn, game_ids: List[str]) -> List[str]:
    """Least-recently-pulled first, never-pulled at the very front (stable)."""
    if not game_ids:
        return []
    df = dbmod.query_df(conn, "SELECT game_id, MAX(ts) AS last_ts FROM lines GROUP BY game_id")
    last = dict(zip(df["game_id"], df["last_ts"])) if not df.empty else {}
    return sorted(game_ids, key=lambda g: (last.get(g) or "", g))


def pull_week_props(cfg: Dict, event_map: Dict[str, str], conn=None,
                    fetch: Optional[Callable] = None,
                    budget: Optional[CreditBudget] = None,
                    ts: Optional[str] = None) -> Dict:
    """Pull props for a rotating, budget-capped subset of the week's games.

    ``event_map``: {nflverse game_id -> odds-api event id} (built by the
    pipeline from team names + kickoff dates). Returns::

        {"pulled": [game_ids], "skipped_budget": [...], "skipped_cap": [...],
         "rows_written": int, "credits_spent": float, "budget_remaining": float}
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

    ordered = rotation_order(conn, list(event_map))
    pulled, skipped_budget, skipped_cap = [], [], []
    all_rows: List[Dict] = []
    spent = 0.0

    for game_id in ordered:
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
        payload = fetch(f"{BASE}/sports/{SPORT}/events/{event_map[game_id]}/odds", params)
        headers = payload.pop("_headers", None) if isinstance(payload, dict) else None
        budget.spend(cost_per_event, headers=headers)
        spent += cost_per_event
        rows = parse_event_props(payload, ts)
        for r in rows:
            r["game_id"] = game_id
        all_rows.extend(rows)
        pulled.append(game_id)

    written = 0
    if all_rows:
        written = dbmod.upsert(conn, "lines", all_rows,
                               ["ts", "game_id", "book", "market", "player_name", "side"])
    return {"pulled": pulled, "skipped_budget": skipped_budget, "skipped_cap": skipped_cap,
            "rows_written": written, "credits_spent": spent,
            "budget_remaining": budget.remaining, "ts": ts}


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
    written = dbmod.upsert(conn, "lines", rows,
                           ["ts", "game_id", "book", "market", "player_name", "side"]) if rows else 0
    return {"pulled": pulled, "skipped_budget": skipped, "rows_written": written,
            "ts": ts, "budget_remaining": budget.remaining}
