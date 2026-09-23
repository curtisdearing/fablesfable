"""Grade the issued-pick ledger against official final box scores.

Universe: ONLY records from ``issued_picks`` (or its deterministic export). The
full candidate universe is graded elsewhere (``analysis/real_line_backtest.py``)
and never pooled with this. Groups are kept apart by tier (primary /
experimental / analyst_override), pick class, surface and slate tag
(Thursday is its own tag; nothing here refits or writes model adjustments).

Per pick key the decision of record is its LAST revision recorded before
kickoff; revisions recorded at/after kickoff are excluded as post-kick. A
record whose quote clock or decision clock is not before kickoff is excluded
the same way.

Actuals come from ESPN final box JSON (``gamepackageJSON``) with an explicit
capture clock: passing attempts are the official ``completions/passingAttempts``
figure (sacks excluded). Identity: an explicit ledger-id -> ESPN-id map when
given, else an exact normalized full-name (or initial + surname) match that is
UNIQUE among the game's box athletes; anything else is UNRESOLVED with the
reason. A player on no box category is UNRESOLVED (inactive vs. zero is not
knowable from the box).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from . import settlement as st

ALIAS = {"WSH": "WAS", "LAR": "LA"}
BOX_STAT = {"passing_yards": ("passing", "passingYards"),
            "pass_attempts": ("passing", "completions/passingAttempts"),
            "rushing_yards": ("rushing", "rushingYards"), "rush_attempts": ("rushing", "rushingAttempts"),
            "receiving_yards": ("receiving", "receivingYards"), "receptions": ("receiving", "receptions")}
TD_KEYS = (("rushing", "rushingTouchdowns"), ("receiving", "receivingTouchdowns"))
INTERVAL = 0.80


def _norm(s: str) -> str:
    s = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", "", (s or "").strip(), flags=re.I)
    return re.sub("[^a-z]", "", unicodedata.normalize("NFKD", s).lower())


def _ts(s) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


# ------------------------------------------------------------------ actuals --
def box_game(gp: Dict, captured_at: str, source: str) -> Dict:
    """One ESPN final box -> game identity, kickoff, completion and per-athlete stats."""
    c = gp["header"]["competitions"][0]
    teams = {t["homeAway"]: ALIAS.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
             for t in c["competitors"]}
    athletes: Dict[str, Dict] = {}
    for t in gp["boxscore"]["players"]:
        team = ALIAS.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
        for cat in t["statistics"]:
            for a in cat["athletes"]:
                ath = a["athlete"]
                row = athletes.setdefault(ath["id"], {"espn_id": ath["id"], "team": team,
                                                      "name": ath.get("displayName", ""),
                                                      "first": ath.get("firstName", ""),
                                                      "last": ath.get("lastName", ""), "cats": {}})
                row["cats"][cat["name"]] = dict(zip(cat["keys"], a["stats"]))
    return {"espn_event": c.get("id") or gp["header"].get("id"), "home": teams.get("home"),
            "away": teams.get("away"), "kickoff": c.get("date"),
            "completed": bool(c["status"]["type"].get("completed")), "athletes": athletes,
            "captured_at": captured_at, "source": source}


def load_boxes(paths: Iterable[str], captured_at: str) -> Dict[str, Dict]:
    """{(away, home) game key: game} from ESPN box files; sha256 of each file kept."""
    games = {}
    for p in sorted(paths):
        raw = open(p, "rb").read()
        gp = json.loads(raw)
        gp = gp.get("gamepackageJSON", gp)
        g = box_game(gp, captured_at, source=p)
        g["sha256"] = hashlib.sha256(raw).hexdigest()
        games[f"{g['away']}@{g['home']}"] = g
    return games


def _game_for(record: Dict, games: Dict[str, Dict]) -> Optional[Dict]:
    parts = str(record.get("game_id") or "").split("_")
    return games.get(f"{parts[2]}@{parts[3]}") if len(parts) == 4 else None


def identify(record: Dict, game: Dict, id_map: Optional[Dict[str, str]] = None):
    """(athlete, method) or (None, reason). Never guesses between candidates."""
    if id_map and record.get("player_id") in id_map:
        a = game["athletes"].get(str(id_map[record["player_id"]]))
        return (a, "id_map") if a else (None, "mapped ESPN id has no box row")
    name = record.get("player_name") or ""
    full = [a for a in game["athletes"].values() if _norm(a["name"]) == _norm(name)]
    if len(full) == 1:
        return full[0], "full_name_unique_in_game"
    if len(full) > 1:
        return None, "full name ambiguous in game"
    m = re.fullmatch(r"\s*([A-Za-z]+)\.\s*(.+)", name)
    if m:
        cand = [a for a in game["athletes"].values()
                if _norm(a["last"]) == _norm(m.group(2)) and _norm(a["first"]).startswith(_norm(m.group(1)))]
        if len(cand) == 1:
            return cand[0], "initial_surname_unique_in_game"
        if len(cand) > 1:
            return None, "initial + surname ambiguous in game"
    return None, "no box row for this player (inactive vs. played-with-zero not knowable)"


def box_actual(athlete: Dict, market: str) -> Optional[float]:
    cats = athlete["cats"]
    if market == "anytime_td":
        return float(sum(int(cats[c][k]) for c, k in TD_KEYS if c in cats and k in cats[c]))
    cat, key = BOX_STAT[market]
    if cat not in cats:
        return 0.0          # on the box in another category: participated, zero here
    v = cats[cat][key]
    return float(v.split("/")[1]) if market == "pass_attempts" else float(v)


# ------------------------------------------------------------------ records --
def slate_tag(kickoff: Optional[str]) -> str:
    t = _ts(kickoff)
    if t is None:
        return "unknown"
    et = t - dt.timedelta(hours=4)       # regular season Sep-early Nov is EDT; tag is the weekday only
    return et.strftime("%A").lower()


def decisions_of_record(records: List[Dict], games: Dict[str, Dict]):
    """Last pre-kick revision per pick key; the rest with an exclusion reason."""
    by_key = defaultdict(list)
    for r in records:
        by_key[r["pick_key"]].append(r)
    keep, excluded = [], []
    for key, recs in by_key.items():
        recs = sorted(recs, key=lambda r: (r.get("revision") or 0))
        game = _game_for(recs[0], games)
        kick = _ts(game["kickoff"]) if game else None
        pre = []
        for r in recs:
            clocks = [_ts(r.get("recorded_at")), _ts(r.get("decision_ts"))]
            if r.get("quote_ts"):
                clocks.append(_ts(r["quote_ts"]))
            if kick is None:
                excluded.append({**r, "excluded": "no official game/kickoff for this record"})
            elif any(c is None for c in clocks[:2]) or any(c >= kick for c in clocks if c):
                excluded.append({**r, "excluded": "post-kick or unclocked record"})
            else:
                pre.append(r)
        if pre:
            keep.append({**pre[-1], "n_revisions_pre_kick": len(pre)})
            excluded.extend({**r, "excluded": "superseded by a later pre-kick revision"} for r in pre[:-1])
    return keep, excluded


def _quantile(mean: float, sd: float, dist: str, q: float) -> float:
    from .projection import p_over
    lo, hi = 0.0, max(mean + 12 * sd, 1.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        if 1 - p_over(mean, sd, mid, dist) < q:
            lo = mid
        else:
            hi = mid
    return hi


def grade_record(r: Dict, games: Dict[str, Dict], id_map=None) -> Dict:
    game = _game_for(r, games)
    out = {k: r.get(k) for k in ("record_id", "pick_key", "revision", "n_revisions_pre_kick", "season",
                                 "week", "game_id", "player_id", "player_name", "market", "side", "line",
                                 "tier", "pick_class", "surface", "card_status", "quote_book", "quote_price",
                                 "quote_ts", "decision_ts", "recorded_at", "model_p_side", "mean", "sd", "dist")}
    out.update(kickoff=game and game["kickoff"], slate_tag=slate_tag(game and game["kickoff"]),
               actuals_source=game and game["source"], actuals_sha256=game and game.get("sha256"),
               actuals_captured_at=game and game["captured_at"])
    if not game or not game["completed"]:
        v = st.Verdict(st.UNRESOLVED, None, None, "official final box not available")
        ident = None
    else:
        ath, ident = identify(r, game, id_map)
        actual = box_actual(ath, r["market"]) if ath and r["market"] in (*BOX_STAT, "anytime_td") else None
        v = st.settle(r["market"], r.get("side"), r.get("line"), actual, has_stat_row=ath is not None)
        if ath is None:
            v = st.Verdict(st.UNRESOLVED, None, None, ident)
        out["espn_id"] = ath and ath["espn_id"]
    out.update(identity=ident, settlement=v.settlement, hit=v.hit, actual=v.actual, detail=v.detail)
    mean, sd, p = r.get("mean"), r.get("sd"), r.get("model_p_side")
    if v.actual is not None and mean is not None:
        out["point_error"] = v.actual - mean
    if v.actual is not None and r.get("dist") and mean is not None and sd:
        lo = _quantile(mean, sd, r["dist"], (1 - INTERVAL) / 2)
        hi = _quantile(mean, sd, r["dist"], 1 - (1 - INTERVAL) / 2)
        out.update(interval_lo=lo, interval_hi=hi, covered=int(lo <= v.actual <= hi))
    if v.hit is not None and p is not None:
        pc = min(max(p, 1e-6), 1 - 1e-6)
        out.update(brier=(p - v.hit) ** 2, logloss=-(math.log(pc) if v.hit else math.log(1 - pc)))
    return out


def clv_row(r: Dict, closes: Optional[Dict], kickoff: Optional[str]) -> Dict:
    """Valid CLV needs a close captured before kickoff at the SAME line and the entry quote."""
    if not closes or not r.get("quote_price"):
        return {"clv_status": "unavailable: no entry quote or no close capture supplied"}
    c = closes.get((r["game_id"], r["player_id"], r["market"], r["side"]))
    if not c:
        return {"clv_status": "unavailable: no close for this side"}
    kick = _ts(kickoff)
    if kick is None or _ts(c["close_ts"]) is None or _ts(c["close_ts"]) >= kick:
        return {"clv_status": "invalid: close not captured before kickoff"}
    if c.get("close_point") != r.get("line"):
        return {"clv_status": "invalid: close line differs from the issued line"}
    return {"clv_status": "valid", "close_ts": c["close_ts"], "close_prob": c["close_prob"],
            "clv_prob_vs_entry_breakeven": c["close_prob"] - 1 / r["quote_price"]}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def summarize(rows: List[Dict]) -> Dict:
    games = sorted({r["game_id"] for r in rows})
    settled = [r for r in rows if r["settlement"] in st.SETTLED]
    counts = {s: sum(r["settlement"] == s for r in rows) for s in st.SETTLEMENTS}
    errs = [r.get("point_error") for r in rows if r.get("point_error") is not None]
    cov = [r for r in rows if r.get("covered") is not None]
    clv = [r["clv_prob_vs_entry_breakeven"] for r in rows if r.get("clv_status") == "valid"]
    out = {"n_records": len(rows), "n_games": len(games), "games": games,
           "per_game_n": {g: sum(r["game_id"] == g for r in rows) for g in games},
           "settlement_counts": counts, "n_settled": len(settled),
           "wins": sum(r["hit"] == 1 for r in settled),
           "brier": _mean([r.get("brier") for r in settled]), "logloss": _mean([r.get("logloss") for r in settled]),
           "n_with_p": sum(r.get("brier") is not None for r in settled),
           "mae": _mean([abs(e) for e in errs]), "bias_actual_minus_mean": _mean(errs), "n_point": len(errs),
           "interval": (f"{INTERVAL:.0%} central, family as recorded" if cov else "unavailable: no recorded dist"),
           "coverage": _mean([r["covered"] for r in cov]),
           "mean_width": _mean([r["interval_hi"] - r["interval_lo"] for r in cov]), "n_interval": len(cov),
           "clv_valid_n": len(clv), "clv_mean": _mean(clv)}
    out["claims"] = ("none: fewer than 2 independent games; descriptive only" if len(games) < 2 else
                     "descriptive; game-clustered, not quote-row n; no inference without a pre-registered test")
    return out


def grade(records: List[Dict], games: Dict[str, Dict], id_map=None, closes=None,
          prior_rows: Optional[List[Dict]] = None) -> Dict:
    """Row-level grades + grouped summaries. Writes nothing; refits nothing."""
    keep, excluded = decisions_of_record(records, games)
    rows = []
    for r in keep:
        g = grade_record(r, games, id_map)
        g.update(clv_row(r, closes, g["kickoff"]))
        rows.append(g)
    corrections = []
    if prior_rows:
        prev = {p["record_id"]: p for p in prior_rows}
        for g in rows:
            p = prev.get(g["record_id"])
            if p and p.get("actual") != g["actual"]:
                corrections.append({"record_id": g["record_id"], "prior_actual": p.get("actual"),
                                    "prior_captured_at": p.get("actuals_captured_at"),
                                    "actual": g["actual"], "captured_at": g["actuals_captured_at"]})
    groups = defaultdict(list)
    for g in rows:
        groups[(g["tier"], g["pick_class"], g["surface"], g["slate_tag"])].append(g)
        groups[(g["tier"], g["pick_class"], g["surface"], "all_slates")].append(g)
    rows.sort(key=lambda g: (g["season"], g["week"], g["game_id"] or "", g["pick_key"]))
    return {"universe": "issued_picks ledger only (candidate universe graded separately)",
            "book_rules_unverified": list(st.BOOK_RULES_UNVERIFIED),
            "refit": "none: grades never feed model adjustments automatically",
            "rows": rows, "excluded": [{k: e.get(k) for k in ("record_id", "pick_key", "revision", "excluded")}
                                       for e in excluded],
            "stat_corrections": corrections,
            "groups": {"|".join(k): summarize(v) for k, v in sorted(groups.items())}}
