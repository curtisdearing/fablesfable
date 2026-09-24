"""Grade the issued-pick ledger against official final box scores.

Universe: ONLY ledger records (``issued_picks`` + ``issued_pick_events``, or their
export, or a verified saved page). The full candidate universe is graded elsewhere
(``analysis/real_line_backtest.py``) and never pooled with this.

Sections (never pooled):

* ``recommendations_given`` -- DEFAULT. Every distinct recommendation that was
  published or delivered before kickoff. A later record for the same pick (a stale
  re-render turning it to ``pass``, a line/side/price change) is reported beside it
  and never erases it: the user may have acted on the first one. A changed pick
  that was itself given is its own row naming the row it revises. The same decision
  shown on several surfaces (or re-rendered with identical side/line/quote) is one row.
* ``watch_published`` -- watch-list cards. Not recommendations.
* ``generated_not_shown`` -- recommendation/watch cards a run generated whose
  publication or delivery is not evidenced in the ledger.
* ``retrospective`` -- recommendation/watch records with no event recorded before
  kickoff (archive recordings, pages rebuilt at grading time, post-kick records).
* ``latest_pre_kick_snapshot`` -- a separate, predeclared analysis: per pick, the
  last record the ledger held before kickoff, whatever its class. Not the given-pick record.

A record is prospective only if one of its events was recorded in the ledger
(ledger wall clock) before kickoff and its decision, quote and stage clocks are
also before kickoff.

Actuals: ESPN final box JSON (``gamepackageJSON``). A box binds to a canonical
game id ``{season}_{week:02d}_{away}_{home}`` only from its own header (season
year, season type = 2 regular season, week, event id, kickoff, teams); files
missing that metadata, not final, captured before kickoff, or repeated for one
game are rejected. Records bind by EXACT game id. Passing attempts are the
official ``completions/passingAttempts`` (sacks excluded). A player absent from
the market's box category, a missing key or a non-numeric value is UNRESOLVED
(zero is not verified). Nothing here refits or writes model adjustments.
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
from zoneinfo import ZoneInfo

from . import settlement as st

ALIAS = {"WSH": "WAS", "LAR": "LA"}
BOX_STAT = {"passing_yards": ("passing", "passingYards"),
            "pass_attempts": ("passing", "completions/passingAttempts"),
            "rushing_yards": ("rushing", "rushingYards"), "rush_attempts": ("rushing", "rushingAttempts"),
            "receiving_yards": ("receiving", "receivingYards"), "receptions": ("receiving", "receptions")}
TD_KEYS = (("rushing", "rushingTouchdowns"), ("receiving", "receivingTouchdowns"))
INTERVAL = 0.80
ET = ZoneInfo("America/New_York")
SHOWN = ("published", "delivered")
_STAGE_RANK = {"generated": 1, "published": 2, "delivered": 3}


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
    return t if t.tzinfo else None          # a clock without a zone is not a verifiable clock


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ------------------------------------------------------------------ actuals --
def box_game(gp: Dict, captured_at: str, source: str) -> Dict:
    """One ESPN box -> identity from its own header, completion, per-athlete stats. Raises on
    missing or unusable identity metadata."""
    h = gp.get("header") or {}
    season = h.get("season") or {}
    c = (h.get("competitions") or [{}])[0]
    year, stype, week, event = season.get("year"), season.get("type"), h.get("week"), h.get("id") or c.get("id")
    if not isinstance(year, int) or not isinstance(week, int) or not event:
        raise ValueError("box header lacks season year / week / event id")
    if stype != 2:
        raise ValueError(f"season type {stype!r} is not the regular season (2)")
    teams = {t.get("homeAway"): ALIAS.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
             for t in c.get("competitors") or []}
    if set(teams) != {"home", "away"}:
        raise ValueError("box header lacks home/away teams")
    kick = _ts(c.get("date"))
    if kick is None:
        raise ValueError("box header lacks a zoned kickoff clock")
    status = ((c.get("status") or {}).get("type") or {})
    cap = _ts(captured_at)
    if cap is None:
        raise ValueError("capture clock missing or not zoned")
    athletes: Dict[str, Dict] = {}
    for t in (gp.get("boxscore") or {}).get("players") or []:
        team = ALIAS.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
        for cat in t.get("statistics") or []:
            for a in cat.get("athletes") or []:
                ath = a["athlete"]
                row = athletes.setdefault(str(ath["id"]), {"espn_id": str(ath["id"]), "team": team,
                                                           "name": ath.get("displayName", ""),
                                                           "first": ath.get("firstName", ""),
                                                           "last": ath.get("lastName", ""), "cats": {}})
                row["cats"][cat["name"]] = dict(zip(cat.get("keys") or [], a.get("stats") or []))
    return {"game_id": f"{year}_{week:02d}_{teams['away']}_{teams['home']}", "season": year, "week": week,
            "season_type": stype, "espn_event": str(event), "home": teams["home"], "away": teams["away"],
            "kickoff": c.get("date"),
            "final": status.get("name") == "STATUS_FINAL" and bool(status.get("completed")),
            "captured_after_kickoff": cap > kick,
            "athletes": athletes, "captured_at": captured_at, "source": source}


def load_boxes(paths: Iterable[str], captured_at: str) -> Dict:
    """{"games": {canonical game_id: game}, "rejected": [{source, reason}]}.

    A game id seen in more than one file is rejected entirely (never last-one-wins)."""
    games: Dict[str, Dict] = {}
    rejected, seen = [], defaultdict(list)
    for p in sorted(paths):
        raw = open(p, "rb").read()
        try:
            gp = json.loads(raw)
            g = box_game(gp.get("gamepackageJSON", gp), captured_at, source=p)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"source": p, "reason": str(exc)})
            continue
        g["sha256"] = hashlib.sha256(raw).hexdigest()
        if not g["final"]:
            rejected.append({"source": p, "reason": "not an official final (status not STATUS_FINAL)"})
            continue
        if not g["captured_after_kickoff"]:
            rejected.append({"source": p, "reason": "capture clock not after kickoff: cannot be the final box"})
            continue
        seen[g["game_id"]].append(g)
    for gid, gs in sorted(seen.items()):
        if len(gs) > 1:
            rejected.extend({"source": g["source"], "reason": f"duplicate box files for {gid}"} for g in gs)
        else:
            games[gid] = gs[0]
    return {"games": games, "rejected": rejected}


def _game_for(record: Dict, games: Dict[str, Dict]) -> Optional[Dict]:
    """Exact canonical game id only; the record's own season/week must agree with it."""
    gid = str(record.get("game_id") or "")
    g = games.get(gid)
    if g is None:
        return None
    if record.get("season") is not None and int(record["season"]) != g["season"]:
        return None
    if record.get("week") is not None and int(record["week"]) != g["week"]:
        return None
    return g


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


def box_actual(athlete: Dict, market: str):
    """(value, None) or (None, reason). Absence from a category is not a verified zero."""
    cats = athlete["cats"]
    if market == "anytime_td":
        vals = [_num(cats[c].get(k)) if c in cats else None for c, k in TD_KEYS]
        if any(v is not None and v >= 1 for v in vals):
            return float(sum(v for v in vals if v is not None)), None
        if all(v is not None for v in vals):
            return 0.0, None
        return None, "zero touchdowns not verified: player missing from rushing or receiving"
    if market not in BOX_STAT:
        return None, f"market {market!r} has no box definition"
    cat, key = BOX_STAT[market]
    if cat not in cats:
        return None, f"player not listed under {cat}: zero not verified"
    raw = cats[cat].get(key)
    if raw is None:
        return None, f"box {cat} row lacks {key}"
    if market == "pass_attempts":
        parts = str(raw).split("/")
        v = _num(parts[1]) if len(parts) == 2 else None
    else:
        v = _num(raw)
    return (v, None) if v is not None else (None, f"box value {raw!r} is not numeric")


# ------------------------------------------------------------------ records --
def slate_tag(kickoff: Optional[str]) -> str:
    t = _ts(kickoff)
    return t.astimezone(ET).strftime("%A").lower() if t else "unknown"


def _prospective_events(r: Dict, kick: Optional[dt.datetime]) -> List[Dict]:
    if kick is None:
        return []
    rec_clocks = [_ts(r.get("decision_ts"))] + ([_ts(r["quote_ts"])] if r.get("quote_ts") else [])
    if any(c is None or c >= kick for c in rec_clocks):
        return []
    out = []
    for e in r.get("events") or []:
        led, own = _ts(e.get("recorded_at")), _ts(e.get("event_ts")) if e.get("event_ts") else None
        if led is not None and led < kick and (own is None or own < kick):
            out.append(e)
    return out


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
    out = {k: r.get(k) for k in ("record_id", "pick_key", "revision", "season", "week", "game_id", "player_id",
                                 "player_name", "market", "side", "line", "tier", "pick_class", "card_status",
                                 "quote_book", "quote_price", "quote_ts", "decision_ts", "model_p_side",
                                 "mean", "sd", "dist")}
    out.update(kickoff=game and game["kickoff"], slate_tag=slate_tag(game and game["kickoff"]),
               espn_event=game and game["espn_event"], actuals_source=game and game["source"],
               actuals_sha256=game and game.get("sha256"), actuals_captured_at=game and game["captured_at"])
    ident = None
    if not game:
        v = st.Verdict(st.UNRESOLVED, None, None, "no verified official final box for this exact game id")
    else:
        ath, ident = identify(r, game, id_map)
        if ath is None:
            v = st.Verdict(st.UNRESOLVED, None, None, ident)
        else:
            actual, why = box_actual(ath, r["market"])
            v = (st.settle(r["market"], r.get("side"), r.get("line"), actual, has_stat_row=True)
                 if why is None else st.Verdict(st.UNRESOLVED, None, None, why))
            out["espn_id"] = ath["espn_id"]
    out.update(identity=ident, settlement=v.settlement, hit=v.hit, actual=v.actual, detail=v.detail)
    mean, sd, p = _num(r.get("mean")), _num(r.get("sd")), _num(r.get("model_p_side"))
    if v.actual is not None and mean is not None:
        out["point_error"] = v.actual - mean
    if v.actual is not None and r.get("dist") and mean is not None and sd and sd > 0:
        lo = _quantile(mean, sd, r["dist"], (1 - INTERVAL) / 2)
        hi = _quantile(mean, sd, r["dist"], 1 - (1 - INTERVAL) / 2)
        out.update(interval_lo=lo, interval_hi=hi, covered=int(lo <= v.actual <= hi))
    if v.hit is not None:
        if p is None or not 0.0 <= p <= 1.0:
            out["probability_status"] = "invalid or missing model probability: not scored"
        else:
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
    errs = [r.get("point_error") for r in rows if r.get("point_error") is not None]
    cov = [r for r in rows if r.get("covered") is not None]
    clv = [r["clv_prob_vs_entry_breakeven"] for r in rows if r.get("clv_status") == "valid"]
    out = {"n_rows": len(rows), "n_games": len(games), "games": games,
           "per_game_n": {g: sum(r["game_id"] == g for r in rows) for g in games},
           "settlement_counts": {s: sum(r["settlement"] == s for r in rows) for s in st.SETTLEMENTS},
           "n_settled": len(settled), "wins": sum(r["hit"] == 1 for r in settled),
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


def _decision(r: Dict):
    return (r.get("tier"), r["pick_key"], r.get("side"), r.get("line"), r.get("quote_book"),
            r.get("quote_price"), r.get("quote_ts"), r.get("pick_class"))


def _change(first: Dict, later: Dict) -> str:
    if (first.get("side"), first.get("line")) != (later.get("side"), later.get("line")):
        return "side_or_line_changed"
    if later.get("quote_book") is None and first.get("pick_class") != later.get("pick_class"):
        # a non-executable re-render (stale quote, held run) shows no quote: not a new price
        return "status_changed_only (re-render; not a withdrawal)"
    if (first.get("quote_book"), first.get("quote_price"), first.get("quote_ts")) != \
            (later.get("quote_book"), later.get("quote_price"), later.get("quote_ts")):
        return "quote_changed"
    if first.get("pick_class") != later.get("pick_class"):
        return "status_changed_only (re-render; not a withdrawal)"
    return "display_changed_only"


def _sections(records: List[Dict], games: Dict[str, Dict]):
    """Classify records; dedup identical decisions; attach later records without erasing."""
    info = []
    for r in records:
        g = _game_for(r, games)
        kick = _ts(g["kickoff"]) if g else None
        pro = _prospective_events(r, kick)
        stage = max((e["stage"] for e in pro), key=_STAGE_RANK.get, default=None)
        first = min((_ts(e["recorded_at"]) for e in pro), default=None)
        info.append({"r": r, "stage": stage, "first_seen": first, "has_game": g is not None})
    by_key = defaultdict(list)
    for i in info:
        by_key[i["r"]["pick_key"]].append(i)
    for lst in by_key.values():
        lst.sort(key=lambda i: (i["first_seen"] is None, i["first_seen"] or dt.datetime.max.replace(
            tzinfo=dt.timezone.utc), i["r"].get("revision") or 0))

    def section(pred):
        rows, seen = [], {}
        for i in info:
            if not pred(i):
                continue
            key = _decision(i["r"])
            if key in seen:                       # same decision, another record or surface
                seen[key]["same_decision_records"].append(i["r"]["record_id"])
                continue
            unit = {"record": i["r"], "stage": i["stage"], "first_seen": i["first_seen"],
                    "same_decision_records": []}
            seen[key] = unit
            rows.append(unit)
        for u in rows:
            sib = by_key[u["record"]["pick_key"]]
            pos = [s["r"]["record_id"] for s in sib].index(u["record"]["record_id"])
            u["later_records"] = [{"record_id": s["r"]["record_id"], "card_status": s["r"].get("card_status"),
                                   "pick_class": s["r"].get("pick_class"), "side": s["r"].get("side"),
                                   "line": s["r"].get("line"), "stage": s["stage"],
                                   "change": _change(u["record"], s["r"])}
                                  for s in sib[pos + 1:] if s["r"]["record_id"] not in u["same_decision_records"]
                                  and _decision(s["r"]) != _decision(u["record"])]
            earlier = [s for s in sib[:pos] if pred(s)]
            u["revises"] = earlier[-1]["r"]["record_id"] if earlier else None
        return rows

    issued = lambda i: i["r"].get("pick_class") in ("recommendation", "watch")  # noqa: E731
    secs = {
        "recommendations_given": section(lambda i: i["r"].get("pick_class") == "recommendation"
                                         and i["stage"] in SHOWN),
        "watch_published": section(lambda i: i["r"].get("pick_class") == "watch" and i["stage"] in SHOWN),
        "generated_not_shown": section(lambda i: issued(i) and i["stage"] == "generated"),
        "retrospective": section(lambda i: issued(i) and i["stage"] is None),
    }
    latest = []
    for lst in by_key.values():
        pro = [i for i in lst if i["stage"] is not None]
        if pro:
            latest.append({"record": pro[-1]["r"], "stage": pro[-1]["stage"], "first_seen": pro[-1]["first_seen"],
                           "same_decision_records": [], "later_records": [], "revises": None})
    secs["latest_pre_kick_snapshot"] = latest
    counts = {"records": len(records),
              "not_a_pick_records": sum(i["r"].get("pick_class") == "not_a_pick" for i in info),
              **{k: len(v) for k, v in secs.items()}}
    return secs, counts


def grade(records: List[Dict], boxes: Dict, id_map=None, closes=None,
          prior_rows: Optional[List[Dict]] = None) -> Dict:
    """Row-level grades per section + grouped summaries. Writes nothing; refits nothing."""
    games = boxes["games"]
    secs, counts = _sections(records, games)
    cache: Dict[str, Dict] = {}
    out_secs = {}
    for name, units in secs.items():
        rows = []
        for u in units:
            r = u["record"]
            if r["record_id"] not in cache:
                g = grade_record(r, games, id_map)
                g.update(clv_row(r, closes, g["kickoff"]))
                cache[r["record_id"]] = g
            rows.append({**cache[r["record_id"]], "section": name, "evidence_stage": u["stage"],
                         "first_seen_in_ledger": u["first_seen"].isoformat() if u["first_seen"] else None,
                         "same_decision_records": u["same_decision_records"],
                         "later_records": u["later_records"], "revises": u["revises"]})
        rows.sort(key=lambda g: (g["season"], g["week"], g["game_id"] or "", g["pick_key"],
                                 g["first_seen_in_ledger"] or ""))
        groups = defaultdict(list)
        for g in rows:
            groups[(g["tier"], g["slate_tag"])].append(g)
            groups[(g["tier"], "all_slates")].append(g)
        out_secs[name] = {"rows": rows, "groups": {"|".join(k): summarize(v) for k, v in sorted(groups.items())}}
    corrections = []
    if prior_rows:
        prev = {p["record_id"]: p for p in prior_rows}
        for g in cache.values():
            p = prev.get(g["record_id"])
            if p and p.get("actual") != g["actual"]:
                corrections.append({"record_id": g["record_id"], "prior_actual": p.get("actual"),
                                    "prior_captured_at": p.get("actuals_captured_at"),
                                    "actual": g["actual"], "captured_at": g["actuals_captured_at"]})
    return {"universe": "issued-pick ledger only (candidate universe graded separately)",
            "default_section": "recommendations_given",
            "counts": counts,
            "book_rules_unverified": list(st.BOOK_RULES_UNVERIFIED),
            "refit": "none: grades never feed model adjustments automatically",
            "boxes": {"games": sorted(games), "rejected": boxes["rejected"]},
            "sections": out_secs,
            "stat_corrections": sorted(corrections, key=lambda c: c["record_id"])}
