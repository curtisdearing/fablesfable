"""The Odds API v4 client (https://the-odds-api.com).

Free tier = 500 credits / month, all sports, all markets. Game lines are cheap
(1 credit each); player props use the per-event endpoint and cost more, so we
cap how many games we pull props for (see config.max_prop_games_per_run).
"""

from __future__ import annotations

from typing import Dict, List

from ._http import get_json

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"


def _normalize_game_odds(ev: Dict) -> Dict:
    home, away = ev.get("home_team"), ev.get("away_team")
    h2h, spreads, totals = {}, {}, {}
    for bk in ev.get("bookmakers", []):
        key = bk.get("key")
        for mkt in bk.get("markets", []):
            outs = mkt.get("outcomes", [])
            if mkt.get("key") == "h2h":
                d = {o["name"]: o["price"] for o in outs}
                if home in d and away in d:
                    h2h[key] = {"home": d[home], "away": d[away]}
            elif mkt.get("key") == "spreads":
                d = {o["name"]: o for o in outs}
                if home in d and away in d:
                    spreads[key] = {
                        "home": {"point": d[home].get("point"), "price": d[home]["price"]},
                        "away": {"point": d[away].get("point"), "price": d[away]["price"]},
                    }
            elif mkt.get("key") == "totals":
                d = {o["name"].lower(): o for o in outs}
                if "over" in d and "under" in d:
                    totals[key] = {
                        "over": {"point": d["over"].get("point"), "price": d["over"]["price"]},
                        "under": {"point": d["under"].get("point"), "price": d["under"]["price"]},
                    }
    return {
        "id": ev.get("id"),
        "commence_time": ev.get("commence_time"),
        "home_team": home, "away_team": away,
        "books": {"h2h": h2h, "spreads": spreads, "totals": totals},
        "context": {}, "props": [],
    }


def fetch_game_odds(cfg: Dict) -> List[Dict]:
    data = get_json(f"{BASE}/sports/{SPORT}/odds", {
        "apiKey": cfg["odds_api_key"],
        "regions": cfg.get("regions", "us"),
        "markets": ",".join(cfg.get("game_markets", ["h2h", "spreads", "totals"])),
        "oddsFormat": "decimal",
    })
    return [_normalize_game_odds(ev) for ev in data]


def fetch_event_props(cfg: Dict, event_id: str) -> List[Dict]:
    """Player props for one event. Returns a list of normalized prop dicts."""
    try:
        from ..config import prop_markets_external
        data = get_json(f"{BASE}/sports/{SPORT}/events/{event_id}/odds", {
            "apiKey": cfg["odds_api_key"],
            "regions": cfg.get("regions", "us"),
            "markets": ",".join(prop_markets_external(cfg)),
            "oddsFormat": "decimal",
        })
    except Exception as exc:  # noqa: BLE001
        print(f"[oddsapi] props fetch failed for {event_id}: {exc}")
        return []

    # group by (market, player, point) -> {book: {over/under: price}}
    grouped: Dict = {}
    label_map = {"player_pass_yds": "Pass Yds", "player_rush_yds": "Rush Yds",
                 "player_reception_yds": "Rec Yds", "player_receptions": "Receptions",
                 "player_pass_tds": "Pass TDs", "player_anytime_td": "Anytime TD"}
    for bk in data.get("bookmakers", []):
        bkey = bk.get("key")
        for mkt in bk.get("markets", []):
            mkey = mkt.get("key")
            ptype = mkey.replace("player_", "")
            for o in mkt.get("outcomes", []):
                player = o.get("description") or o.get("name")
                point = o.get("point", 0.5)
                side = o.get("name", "").lower()
                side = "over" if side in ("over", "yes") else ("under" if side in ("under", "no") else side)
                if side not in ("over", "under"):
                    continue
                gk = (mkey, player, point)
                g = grouped.setdefault(gk, {
                    "player": player, "team": "", "prop_type": ptype,
                    "label": label_map.get(mkey, ptype), "line": point,
                    "books": {}, "ctx": {"weather_sev": 0.0, "opp_def_rating": 0.0,
                                         "usage_trend": 0.0}})
                g["books"].setdefault(bkey, {})[side] = o.get("price")
    # keep only props quoted on both sides by at least 2 books
    out = []
    for g in grouped.values():
        good = {b: pr for b, pr in g["books"].items() if "over" in pr and "under" in pr}
        if len(good) >= 2:
            g["books"] = good
            out.append(g)
    return out


def fetch_scores(cfg: Dict, days_from: int = 3) -> Dict[str, Dict]:
    """Finished-game results keyed by event id."""
    data = get_json(f"{BASE}/sports/{SPORT}/scores", {
        "apiKey": cfg["odds_api_key"], "daysFrom": days_from,
    })
    results = {}
    for ev in data:
        if not ev.get("completed"):
            continue
        scores = {s["name"]: float(s["score"]) for s in (ev.get("scores") or [])
                  if s.get("score") is not None}
        home, away = ev.get("home_team"), ev.get("away_team")
        if home in scores and away in scores:
            results[ev["id"]] = {
                "home_score": scores[home], "away_score": scores[away],
                "prop_actuals": {},  # box-score grading of props is left to extend
                "settled_ts": ev.get("last_update") or ev.get("commence_time"),
                "completed": True,
            }
    return results
