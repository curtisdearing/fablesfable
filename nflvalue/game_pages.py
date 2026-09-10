"""Per-game pages: one hyperlink per matchup on the dashboard, and behind it
the four things a reader wants before touching a prop --

  * WHERE THE ODDS ARE      book / line / price / how many books, per lean
  * WHO IS NOT PLAYING      every listed player on both teams with the ESPN
                            report date (recency), OUT first, then RISK
  * WHERE THE BODIES ARE    venue time zone vs each team's home zone, the
                            kickoff on each team's body clock, rest days,
                            neutral-site flag, hand-curated arrival notes
  * WHY THE MODEL LEANS     the projection ledger's own drivers for the game's
                            priced leans -- the model's reasons, not prose

Display-only by construction: every number here is read from the same
payload the dashboard already renders (``weekly_leans`` + ``explain``); nothing
in this module can move a score.  Two entry points:

``attach_context(games, ...)``  called in the pipeline where the live feeds
    are in hand; stamps ``g["page_context"]`` (travel + availability + hand
    notes) onto each shortlist game dict, next to ``g["notes"]``.
``build_pages(weekly_leans, explain)``  called when the dashboard payload is
    assembled; joins the context with the leans and the explain cards into
    one JSON-able page dict per game.  ``render_html(page)`` turns one into a
    standalone document; ``scripts/prepare_pages.py`` writes them under
    ``_site/games/``.

Hand notes (arrival dates, quoted designations, anything no feed carries)
live in ``data/game_context/{season}/week-{week}.json`` -- committed to the
repository, keyed by nflverse ``game_id``.  Absent file = no notes; a
malformed file is reported on the page rather than swallowed.
"""

from __future__ import annotations

import datetime as dt
import html as _html
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

from . import config as cfgmod

# --------------------------------------------------------------------------- #
# Time zones
# --------------------------------------------------------------------------- #
#: Home-stadium IANA zone per nflverse abbreviation.  Arizona does not observe
#: DST (America/Phoenix), which is exactly the kind of thing a hand-typed
#: "MT" would get wrong for half the season.
TEAM_TZ: Dict[str, str] = {
    "ARI": "America/Phoenix", "ATL": "America/New_York", "BAL": "America/New_York",
    "BUF": "America/New_York", "CAR": "America/New_York", "CHI": "America/Chicago",
    "CIN": "America/New_York", "CLE": "America/New_York", "DAL": "America/Chicago",
    "DEN": "America/Denver", "DET": "America/New_York", "GB": "America/Chicago",
    "HOU": "America/Chicago", "IND": "America/New_York", "JAX": "America/New_York",
    "KC": "America/Chicago", "LV": "America/Los_Angeles", "LAC": "America/Los_Angeles",
    "LA": "America/Los_Angeles", "LAR": "America/Los_Angeles", "MIA": "America/New_York",
    "MIN": "America/Chicago", "NE": "America/New_York", "NO": "America/Chicago",
    "NYG": "America/New_York", "NYJ": "America/New_York", "PHI": "America/New_York",
    "PIT": "America/New_York", "SEA": "America/Los_Angeles", "SF": "America/Los_Angeles",
    "TB": "America/New_York", "TEN": "America/Chicago", "WAS": "America/New_York",
    # legacy codes that still appear in older schedule rows
    "OAK": "America/Los_Angeles", "SD": "America/Los_Angeles", "STL": "America/Chicago",
}

#: Neutral-site venues, matched by substring of the schedule's ``stadium``
#: (case-insensitive).  Order matters only for overlapping keys.
NEUTRAL_VENUE_TZ: Tuple[Tuple[str, str], ...] = (
    ("melbourne", "Australia/Melbourne"), ("mcg", "Australia/Melbourne"),
    ("sydney", "Australia/Sydney"),
    ("tottenham", "Europe/London"), ("wembley", "Europe/London"),
    ("twickenham", "Europe/London"), ("london", "Europe/London"),
    ("croke", "Europe/Dublin"), ("dublin", "Europe/Dublin"),
    ("allianz", "Europe/Berlin"), ("munich", "Europe/Berlin"),
    ("frankfurt", "Europe/Berlin"), ("deutsche bank", "Europe/Berlin"),
    ("olympiastadion", "Europe/Berlin"), ("berlin", "Europe/Berlin"),
    ("bernab", "Europe/Madrid"), ("madrid", "Europe/Madrid"),
    ("azteca", "America/Mexico_City"), ("mexico", "America/Mexico_City"),
    ("corinthians", "America/Sao_Paulo"), ("neo qu", "America/Sao_Paulo"),
    ("sao paulo", "America/Sao_Paulo"), ("são paulo", "America/Sao_Paulo"),
    ("maracan", "America/Sao_Paulo"), ("rio de janeiro", "America/Sao_Paulo"),
    ("rogers centre", "America/Toronto"), ("toronto", "America/Toronto"),
)

#: nflverse ``gametime`` is Eastern, always.
SCHEDULE_TZ = "America/New_York"

#: Body-clock kickoff flags.  Football is normally played early-to-mid
#: afternoon on the body clock; before noon or at/after 21:00 is what the
#: page calls out.  The shift threshold is inclusive at two zones.
EARLY_BODY_CLOCK_HOUR = 12
LATE_BODY_CLOCK_HOUR = 21
SHIFT_FLAG_HOURS = 2.0
SHORT_REST_DAYS = 6
LONG_REST_DAYS = 10


def _zone(name: str):
    from zoneinfo import ZoneInfo  # py3.9+
    return ZoneInfo(name)


def kickoff_from_schedule(gameday: str, gametime: Optional[str]) -> Optional[dt.datetime]:
    """nflverse ``gameday`` + ``gametime`` (Eastern, 'HH:MM') -> aware datetime.
    ``None`` when the row cannot be parsed (never a guessed kickoff)."""
    if not gameday:
        return None
    try:
        hh, mm = (gametime or "13:00").split(":")[:2]
        naive = dt.datetime.fromisoformat(str(gameday)[:10]).replace(
            hour=int(hh), minute=int(mm))
        return naive.replace(tzinfo=_zone(SCHEDULE_TZ))
    except (ValueError, TypeError, KeyError):
        return None


def venue_zone(home_team: str, stadium: Optional[str], location: Optional[str]) -> Tuple[str, str]:
    """(iana zone, provenance).  Neutral rows are matched against the venue
    table; anything else, and any neutral venue we do not know, falls back to
    the home team's zone and SAYS so."""
    neutral = str(location or "").strip().lower() == "neutral"
    if neutral:
        key = str(stadium or "").lower()
        for needle, zone in NEUTRAL_VENUE_TZ:
            if needle in key:
                return zone, "neutral_venue_table"
    tz = TEAM_TZ.get(str(home_team or "").upper())
    if tz is None:
        return SCHEDULE_TZ, "unknown_team_fallback_eastern"
    return tz, ("home_team_fallback_for_unknown_neutral_venue" if neutral else "home_stadium")


def _offset_hours(kick: dt.datetime, zone: str) -> float:
    off = kick.astimezone(_zone(zone)).utcoffset() or dt.timedelta(0)
    return off.total_seconds() / 3600.0


def _team_clock(kick: dt.datetime, team: str, venue_tz: str, rest_days) -> Dict:
    home_tz = TEAM_TZ.get(str(team or "").upper())
    if home_tz is None:
        return {"team": team, "home_tz": None, "shift_hours": None,
                "body_clock_kickoff": None, "flags": ["home_zone_unknown"],
                "rest_days": rest_days}
    shift = round(_offset_hours(kick, venue_tz) - _offset_hours(kick, home_tz), 1)
    body = kick.astimezone(_zone(home_tz))
    flags: List[str] = []
    if abs(shift) >= SHIFT_FLAG_HOURS:
        flags.append("east_of_home" if shift > 0 else "west_of_home")
        flags.append(f"crosses_{abs(shift):g}h")
    if body.hour < EARLY_BODY_CLOCK_HOUR:
        flags.append("early_body_clock")
    elif body.hour >= LATE_BODY_CLOCK_HOUR:
        flags.append("late_body_clock")
    try:
        rd = int(rest_days) if rest_days is not None and rest_days == rest_days else None
    except (TypeError, ValueError):
        rd = None
    if rd is not None:
        if rd < SHORT_REST_DAYS:
            flags.append("short_rest")
        elif rd >= LONG_REST_DAYS:
            flags.append("long_rest")
    return {"team": team, "home_tz": home_tz, "shift_hours": shift,
            "body_clock_kickoff": body.strftime("%a %H:%M"),
            "rest_days": rd, "flags": flags}


def travel_context(row: Dict) -> Dict:
    """One schedule row (dict with gameday, gametime, home_team, away_team,
    stadium, location, roof, away_rest, home_rest) -> the travel/circadian
    block.  Never raises: an unparseable kickoff yields ``available: False``."""
    kick = kickoff_from_schedule(row.get("gameday"), row.get("gametime"))
    home, away = row.get("home_team"), row.get("away_team")
    neutral = str(row.get("location") or "").strip().lower() == "neutral"
    if kick is None:
        return {"available": False, "reason": "kickoff not parseable from schedule",
                "neutral_site": neutral, "stadium": row.get("stadium")}
    vtz, vsrc = venue_zone(home, row.get("stadium"), row.get("location"))
    local = kick.astimezone(_zone(vtz))
    out = {
        "available": True,
        "kickoff_et": kick.strftime("%a %Y-%m-%d %H:%M ET"),
        "kickoff_utc": kick.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "venue_tz": vtz, "venue_tz_source": vsrc,
        "kickoff_venue_local": local.strftime("%a %Y-%m-%d %H:%M"),
        "stadium": row.get("stadium"), "roof": row.get("roof"),
        "neutral_site": neutral,
        "teams": [_team_clock(kick, away, vtz, row.get("away_rest")),
                  _team_clock(kick, home, vtz, row.get("home_rest"))],
    }
    # the headline: the larger of the two shifts, said once
    shifts = [t["shift_hours"] for t in out["teams"] if t["shift_hours"] is not None]
    out["max_shift_hours"] = max((abs(s) for s in shifts), default=None)
    out["flags"] = sorted({f for t in out["teams"] for f in t["flags"]}
                          | ({"neutral_site"} if neutral else set()))
    return out


# --------------------------------------------------------------------------- #
# Availability (both teams, all positions)
# --------------------------------------------------------------------------- #
def _parse_stamp(value) -> Optional[dt.datetime]:
    if not value or not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)


def _age_hours(reported: Optional[str], as_of: Optional[str]) -> Optional[float]:
    a, b = _parse_stamp(reported), _parse_stamp(as_of)
    if a is None or b is None:
        return None
    return round((b - a).total_seconds() / 3600.0, 1)


_STATUS_ORDER = {"OUT": 0, "RISK": 1, "OK": 2}


def availability_for_teams(teams: Iterable[str], injury_rows: Iterable[Dict],
                           inactive_rows: Optional[Iterable[Dict]], as_of: Optional[str],
                           max_ok: int = 0) -> Dict:
    """Every ESPN-listed player on the given teams, with report recency.

    OUT and RISK rows always appear.  ``status_raw == "Active"`` rows (ESPN
    keeps a "returned from injury" note for weeks) are dropped unless
    ``max_ok`` asks for them.  T-90 inactives (``active == False`` in the
    event roster) are appended as OUT with source ``espn_event_roster`` when
    the injury feed did not already list the player."""
    teams = {str(t).upper() for t in teams if t}
    listed: List[Dict] = []
    seen_names = set()
    for r in injury_rows or []:
        if str(r.get("team") or "").upper() not in teams:
            continue
        status = r.get("status") or "OK"
        if status == "OK" and max_ok <= 0:
            continue
        listed.append({
            "team": str(r.get("team") or "").upper(),
            "name": r.get("name") or "",
            "pos": r.get("position") or "",
            "status": status,
            "status_raw": r.get("status_raw") or "",
            "injury_type": r.get("injury_type") or "",
            "reported": r.get("date") or "",
            "age_hours": _age_hours(r.get("date"), as_of),
            "comment": (r.get("comment") or "")[:240],
            "source": "espn_team_injuries",
        })
        seen_names.add((str(r.get("team") or "").upper(), (r.get("name") or "").lower()))
    for r in inactive_rows or []:
        team = str(r.get("team") or "").upper()
        if team not in teams or r.get("active", True):
            continue
        key = (team, (r.get("name") or "").lower())
        if key in seen_names:
            for row in listed:
                if (row["team"], row["name"].lower()) == key:
                    row["status"] = "OUT"
                    row["status_raw"] = (row["status_raw"] + "|inactive_t90").strip("|")
                    row["source"] = "espn_event_roster"
            continue
        listed.append({"team": team, "name": r.get("name") or "", "pos": "",
                       "status": "OUT", "status_raw": "inactive_t90", "injury_type": "",
                       "reported": "", "age_hours": None, "comment": "",
                       "source": "espn_event_roster"})
    listed.sort(key=lambda r: (_STATUS_ORDER.get(r["status"], 3), r["team"], r["name"]))
    return {
        "as_of": as_of,
        "n_out": sum(1 for r in listed if r["status"] == "OUT"),
        "n_risk": sum(1 for r in listed if r["status"] == "RISK"),
        "players": listed,
    }


# --------------------------------------------------------------------------- #
# Hand-curated notes
# --------------------------------------------------------------------------- #
def notes_path(season: int, week: int, root: Optional[str] = None) -> str:
    base = root or os.path.join(cfgmod.DATA_DIR, "game_context")
    return os.path.join(base, str(season), f"week-{int(week):02d}.json")


def load_hand_notes(season: int, week: int, root: Optional[str] = None) -> Dict:
    """{game_id: notes} for the week, plus ``__error__`` when the file exists
    but is not valid JSON (the page shows the error; it never guesses)."""
    path = notes_path(season, week, root)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        return {"__error__": f"{os.path.basename(path)}: {exc}"}
    if not isinstance(data, dict):
        return {"__error__": f"{os.path.basename(path)}: top level is not an object"}
    return data


# --------------------------------------------------------------------------- #
# Pipeline hook
# --------------------------------------------------------------------------- #
def _slate_rows(schedules, season: int, week: int) -> Dict[str, Dict]:
    sl = schedules[(schedules["season"] == season) & (schedules["week"] == week)
                   & (schedules["game_type"] == "REG")]
    cols = [c for c in ("game_id", "gameday", "gametime", "home_team", "away_team",
                        "stadium", "roof", "location", "away_rest", "home_rest")
            if c in sl.columns]
    out = {}
    for rec in sl[cols].to_dict("records"):
        clean = {k: (None if (isinstance(v, float) and v != v) else v) for k, v in rec.items()}
        out[clean["game_id"]] = clean
    return out


def attach_context(games: List[Dict], schedules, season: int, week: int,
                   injury_rows: Optional[Iterable[Dict]] = None,
                   inactive_rows: Optional[Iterable[Dict]] = None,
                   as_of: Optional[str] = None,
                   notes_root: Optional[str] = None) -> None:
    """Stamp ``g["page_context"]`` onto each shortlist game dict, in place."""
    rows = _slate_rows(schedules, season, week)
    notes = load_hand_notes(season, week, notes_root)
    notes_error = notes.get("__error__")
    injury_rows = list(injury_rows or [])
    for g in games:
        row = rows.get(g.get("game_id")) or {}
        teams = [row.get("away_team"), row.get("home_team")]
        ctx = {
            "home_team": row.get("home_team"), "away_team": row.get("away_team"),
            "travel": travel_context(row) if row else {"available": False,
                                                       "reason": "game not on schedule"},
            "availability": availability_for_teams(teams, injury_rows, inactive_rows, as_of),
            "hand_notes": notes.get(g.get("game_id")) or {},
        }
        if notes_error:
            ctx["hand_notes_error"] = notes_error
        g["page_context"] = ctx


# --------------------------------------------------------------------------- #
# Page assembly (dashboard payload -> one dict per game)
# --------------------------------------------------------------------------- #
def _num(v, nd=1):
    try:
        if v is None or v != v:
            return None
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _market_row(lean: Dict) -> Dict:
    prices = lean.get("prices") or {}
    side = lean.get("side")
    comp = lean.get("components") or {}
    return {
        "player_id": lean.get("player_id"), "name": lean.get("name"),
        "pos": lean.get("pos"), "team": lean.get("team"),
        "market": lean.get("market"), "side": side,
        "line": lean.get("line"), "line_source": lean.get("line_source"),
        "market_state": lean.get("market_state"),
        "price": prices.get(side) if side in ("over", "under") else None,
        "price_over": prices.get("over"), "price_under": prices.get("under"),
        "book": prices.get("book"),
        "n_books": comp.get("n_books") or lean.get("n_books"),
        "edge": _num(lean.get("edge"), 4),
        "p_side": _num(lean.get("p_over") if side == "over" else lean.get("p_under"), 4),
        "mean": _num(lean.get("mean"), 1), "sd": _num(lean.get("sd"), 1),
        "ml_p_over": _num(lean.get("ml_p_over"), 4),
        "score": lean.get("ml_score") if lean.get("ml_score") is not None else lean.get("composite"),
    }


def _drivers_for(card: Optional[Dict], side: Optional[str], limit: int = 4) -> List[Dict]:
    """The model's reasons, largest first, baseline excluded.  ``with_side``
    says whether the driver pushes the published side or against it."""
    if not card:
        return []
    out = []
    for d in card.get("drivers") or []:
        # ledger vocabulary: baseline | level | neutral | up | down -- only the
        # last two are directional claims (explain.Contribution.direction)
        if d.get("direction") not in ("up", "down"):
            continue
        push = d.get("direction")
        with_side = (push == "up") if side == "over" else (push == "down") if side == "under" else None
        out.append({
            "label": d.get("label"), "direction": push,
            "multiplier": d.get("multiplier_label"), "delta": d.get("delta_label"),
            "unit": d.get("unit"), "grade": (d.get("evidence") or {}).get("grade"),
            "with_side": with_side,
        })
    def mag(d):
        try:
            return abs(float(str(d.get("delta") or "0").replace("+", "")))
        except ValueError:
            return 0.0
    out.sort(key=lambda d: -mag(d))
    return out[:limit]


def build_pages(weekly_leans: Dict, explain: Optional[Dict] = None,
                max_bets: int = 3) -> List[Dict]:
    """``weekly_leans`` is the pipeline payload (games with ``page_context``);
    ``explain`` is the cards payload.  One dict per game; JSON-able."""
    cards = {}
    for c in (explain or {}).get("cards") or []:
        cards[(str(c.get("player_id")), c.get("market"))] = c
    season, week = weekly_leans.get("season"), weekly_leans.get("week")
    pages: List[Dict] = []
    for g in weekly_leans.get("games") or []:
        ctx = g.get("page_context") or {}
        leans = g.get("leans") or []
        market = [_market_row(l) for l in leans]
        priced = [m for m in market if m["market_state"] == "REAL_MARKET"
                  and m["edge"] is not None]
        priced.sort(key=lambda m: (-(m["edge"] or 0), str(m["player_id"]), m["market"]))
        bets = []
        for m in priced[:max_bets]:
            card = cards.get((str(m["player_id"]), m["market"]))
            bets.append({**m, "drivers": _drivers_for(card, m["side"]),
                         "weakest_grade": (card or {}).get("weakest_grade"),
                         "counter_case_count": (card or {}).get("counter_case_count")})
        books = sorted({b for m in market if m.get("book")
                        for b in str(m["book"]).split("/") if b})
        pages.append({
            "game_id": g.get("game_id"), "matchup": g.get("matchup"),
            "season": season, "week": week, "clock": weekly_leans.get("clock"),
            "as_of": weekly_leans.get("as_of"),
            "publish": weekly_leans.get("publish", True),
            "home_team": ctx.get("home_team"), "away_team": ctx.get("away_team"),
            "travel": ctx.get("travel") or {"available": False, "reason": "no context"},
            "availability": ctx.get("availability") or {"players": [], "n_out": 0, "n_risk": 0},
            "hand_notes": ctx.get("hand_notes") or {},
            "hand_notes_error": ctx.get("hand_notes_error"),
            "notes": g.get("notes") or [],
            "market": market, "books": books,
            "n_priced": len(priced), "n_leans": len(market),
            "best_bets": bets,
            "href": f"games/{g.get('game_id')}.html",
        })
    return pages


def merge_pages(existing: Optional[List[Dict]], new: List[Dict],
                season, week) -> List[Dict]:
    """A T-90 run carries one game; keep the other games' pages from the
    previous payload of the SAME season/week, replace the ones it re-ran."""
    keep = {}
    for p in existing or []:
        if str(p.get("season")) == str(season) and str(p.get("week")) == str(week):
            keep[p.get("game_id")] = p
    for p in new:
        keep[p.get("game_id")] = p
    return sorted(keep.values(), key=lambda p: str(p.get("game_id")))


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = """
:root{--ink:#141a24;--sub:#5a6472;--rule:#e3e7ee;--bg:#fff;--box:#f7f8fa;--red:#b3261e;--amber:#b8860b;--green:#1f7a3a}
body{font:15px/1.5 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);max-width:880px;margin:32px auto;padding:0 20px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 8px;border-bottom:1px solid var(--rule);padding-bottom:4px}
.sub{color:var(--sub);font-size:13px}.box{background:var(--box);border:1px solid var(--rule);border-radius:6px;padding:12px 14px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--rule);vertical-align:top}th{font-size:12px;color:var(--sub);text-transform:uppercase}
.out{color:var(--red);font-weight:600}.risk{color:var(--amber);font-weight:600}.ok{color:var(--green)}
.badge{display:inline-block;font-size:11px;padding:1px 6px;border-radius:3px;border:1px solid var(--rule);margin-left:4px}
.flag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:10px;background:#eef2f8;margin:2px 4px 2px 0}
.warn{border-left:4px solid var(--amber);background:#fbf7ee;padding:10px 12px;margin:10px 0}
.drv{margin:2px 0 2px 12px;font-size:13px}.with{color:var(--green)}.against{color:var(--red)}
a{color:#1a56a8}.top a{font-size:13px}
"""


def _e(v) -> str:
    return _html.escape("" if v is None else str(v))


def _fmt_price(p) -> str:
    if p is None:
        return "—"
    try:
        p = float(p)
    except (TypeError, ValueError):
        return _e(p)
    return f"{p:+.0f}" if p == int(p) else f"{p:+.1f}"


def _fmt_age(h) -> str:
    if h is None:
        return "report date unknown"
    if h < 0:
        return "future-dated"
    if h < 48:
        return f"{h:.0f}h ago"
    return f"{h/24:.1f}d ago"


def _travel_html(t: Dict) -> str:
    if not t.get("available"):
        return f"<div class='box sub'>Travel context unavailable: {_e(t.get('reason'))}</div>"
    rows = []
    for tm in t.get("teams") or []:
        flags = " ".join(f"<span class='flag'>{_e(f.replace('_', ' '))}</span>"
                         for f in tm.get("flags") or [])
        shift = tm.get("shift_hours")
        shift_s = "—" if shift is None else (f"{shift:+g}h" if shift else "none")
        rows.append(f"<tr><td><b>{_e(tm.get('team'))}</b><div class='sub'>{_e(tm.get('home_tz'))}</div></td>"
                    f"<td>{shift_s}</td><td>{_e(tm.get('body_clock_kickoff') or '—')}</td>"
                    f"<td>{_e(tm.get('rest_days') if tm.get('rest_days') is not None else '—')}</td>"
                    f"<td>{flags}</td></tr>")
    neutral = "<span class='badge'>NEUTRAL SITE</span>" if t.get("neutral_site") else ""
    src = t.get("venue_tz_source")
    src_note = ("" if src in ("home_stadium", "neutral_venue_table")
                else f"<div class='warn'>Venue zone fell back to the home team's ({_e(src)}); "
                     "check the stadium.</div>")
    return (f"<div class='box'><b>{_e(t.get('stadium') or 'venue n/a')}</b> {neutral} "
            f"<span class='sub'>roof: {_e(t.get('roof') or 'n/a')} · venue zone {_e(t.get('venue_tz'))}</span>"
            f"<div>Kickoff {_e(t.get('kickoff_et'))} · venue-local {_e(t.get('kickoff_venue_local'))}</div>"
            f"{src_note}"
            "<table><thead><tr><th>Team</th><th>Zone shift</th><th>Kickoff on body clock</th>"
            "<th>Rest days</th><th>Flags</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
            "<div class='sub'>Shift = venue offset minus home-stadium offset at kickoff; body clock = "
            "kickoff in the team's home zone. Flags fire at ≥2h shift, before 12:00 or at/after 21:00 "
            "on the body clock, &lt;6 or ≥10 rest days.</div></div>")


def _hand_notes_html(n: Dict, err: Optional[str]) -> str:
    parts = []
    if err:
        parts.append(f"<div class='warn'>Hand notes file could not be read: {_e(err)}</div>")
    if not n:
        return "".join(parts)
    arr = n.get("arrival") or {}
    if arr:
        parts.append("<div class='box'><b>Arrival / travel (hand notes)</b>" + "".join(
            f"<div><b>{_e(k)}</b> — {_e(v)}</div>" for k, v in arr.items()) + "</div>")
    npl = n.get("not_playing") or []
    if npl:
        rows = "".join(
            f"<tr><td><b>{_e(p.get('team'))}</b></td><td>{_e(p.get('name'))} <span class='sub'>{_e(p.get('pos'))}</span></td>"
            f"<td>{_e(p.get('note'))}</td><td class='sub'>{_e(p.get('published') or '')}"
            + (f" · <a href='{_e(p.get('source'))}'>source</a>" if p.get("source") else "") + "</td></tr>"
            for p in npl)
        parts.append("<div class='box'><b>Not playing / limited (hand notes)</b><table><thead><tr><th>Team</th>"
                     "<th>Player</th><th>Note</th><th>When · source</th></tr></thead><tbody>" + rows + "</tbody></table></div>")
    notes = n.get("notes") or []
    if notes:
        parts.append("<div class='box'><b>Notes (hand-curated)</b>" + "".join(
            f"<div>• {_e(x.get('text') if isinstance(x, dict) else x)}"
            + (f" <span class='sub'>({_e(x.get('published') or '')}"
               + (f" · <a href='{_e(x.get('source'))}'>source</a>" if x.get("source") else "") + ")</span>"
               if isinstance(x, dict) else "") + "</div>" for x in notes) + "</div>")
    return "".join(parts)


def _availability_html(a: Dict) -> str:
    players = a.get("players") or []
    if not players:
        return "<div class='box sub'>No OUT / questionable designations listed by the injury feed for either team.</div>"
    rows = []
    for p in players:
        cls = {"OUT": "out", "RISK": "risk"}.get(p.get("status"), "ok")
        rows.append(
            f"<tr><td><b>{_e(p.get('team'))}</b></td><td>{_e(p.get('name'))} <span class='sub'>{_e(p.get('pos'))}</span></td>"
            f"<td class='{cls}'>{_e(p.get('status_raw') or p.get('status'))}</td>"
            f"<td>{_e(p.get('injury_type') or '')}</td>"
            f"<td>{_e(_fmt_age(p.get('age_hours')))}<div class='sub'>{_e((p.get('reported') or '')[:16])}</div></td>"
            f"<td class='sub'>{_e(p.get('comment') or '')}</td></tr>")
    return (f"<div class='box'><b>{a.get('n_out', 0)} out · {a.get('n_risk', 0)} questionable</b> "
            f"<span class='sub'>ESPN injury feed as of {_e(a.get('as_of'))}; 'reported' is ESPN's own stamp</span>"
            "<table><thead><tr><th>Team</th><th>Player</th><th>Status</th><th>Injury</th><th>Reported</th><th>Note</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></div>")


def _market_html(page: Dict) -> str:
    rows = []
    for m in page.get("market") or []:
        state = m.get("market_state") or ("REAL_MARKET" if m.get("line_source") == "odds_api" else "NO_MARKET")
        line = "—" if m.get("line") is None else _e(m.get("line"))
        if m.get("line") is not None and m.get("line_source") != "odds_api":
            line += "†"
        price = _fmt_price(m.get("price")) if state == "REAL_MARKET" else "—"
        edge = f"{m['edge']*100:+.1f}%" if m.get("edge") is not None else f"<span class='sub'>{_e(state.lower())}</span>"
        rows.append(f"<tr><td>{_e(m.get('name'))} <span class='sub'>{_e(m.get('pos'))} · {_e(m.get('team'))}</span></td>"
                    f"<td>{_e(str(m.get('market') or '').replace('_', ' '))}</td><td><b>{_e((m.get('side') or '').upper())}</b> {line}</td>"
                    f"<td>{price}</td><td>{_e(m.get('book') or '—')}</td><td>{_e(m.get('n_books') if m.get('n_books') is not None else '—')}</td>"
                    f"<td>{edge}</td><td>{_e(m.get('mean'))}</td></tr>")
    books = ", ".join(page.get("books") or []) or "none quoted"
    return (f"<div class='box'><b>Books quoted on this game:</b> {_e(books)} "
            f"<span class='sub'>· {page.get('n_priced', 0)} of {page.get('n_leans', 0)} leans priced by ≥2 books</span>"
            "<table><thead><tr><th>Player</th><th>Market</th><th>Side · line</th><th>Price</th><th>Best book (over/under)</th>"
            "<th>Books</th><th>Edge</th><th>Proj</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
            "<div class='sub'>† synthetic reference line (trailing mean), not a market. Price = best available on the "
            "published side at pull time; the book pair is best-over/best-under.</div></div>")


def _bets_html(page: Dict) -> str:
    bets = page.get("best_bets") or []
    if not bets:
        return ("<div class='box'><b>No bettable lean.</b> <span class='sub'>Nothing on this game "
                "carries a two-book market with a computed edge, so nothing is offered. A synthetic "
                "line is never a bet.</span></div>")
    out = []
    for b in bets:
        drv = "".join(
            f"<div class='drv {'with' if d.get('with_side') else 'against' if d.get('with_side') is False else ''}'>"
            f"{'▲' if d.get('direction') == 'up' else '▼'} {_e(d.get('label'))}"
            + (f" ×{_e(d.get('multiplier'))}" if d.get("multiplier") else "")
            + (f" ({_e(d.get('delta'))} {_e(d.get('unit') or '')})" if d.get("delta") else "")
            + (f" <span class='sub'>· {_e(d.get('grade'))}</span>" if d.get("grade") else "") + "</div>"
            for d in b.get("drivers") or []) or "<div class='drv sub'>no explain card for this lean</div>"
        out.append(
            f"<div class='box'><b>{_e(b.get('name'))}</b> <span class='sub'>{_e(b.get('pos'))} · {_e(b.get('team'))}</span> — "
            f"<b>{_e((b.get('side') or '').upper())} {_e(b.get('line'))}</b> {_e(str(b.get('market') or '').replace('_', ' '))} "
            f"@ {_fmt_price(b.get('price'))} <span class='sub'>({_e(b.get('book'))}, {_e(b.get('n_books'))} books)</span>"
            f"<div>edge {b['edge']*100:+.1f}% · model p(side) {_e(b.get('p_side'))} · proj {_e(b.get('mean'))} ± {_e(b.get('sd'))}"
            + (f" · ranker p(over) {_e(b.get('ml_p_over'))}" if b.get("ml_p_over") is not None else "")
            + (f" · weakest driver evidence: {_e(b.get('weakest_grade'))}" if b.get("weakest_grade") else "")
            + f"</div><div class='sub'>What moves the projection (largest first; green pushes the published side):</div>{drv}</div>")
    return "".join(out)


def render_html(page: Dict) -> str:
    title = f"{page.get('matchup') or page.get('game_id')} — {page.get('season')} week {page.get('week')}"
    pub = "" if page.get("publish", True) else (
        "<div class='warn'><b>NOT PUBLISHED</b> — the run behind this page failed its data gate; "
        "read it as context, not as a board.</div>")
    notes = page.get("notes") or []
    notes_html = ("<div class='box'>" + "".join(f"<div>• {_e(n)}</div>" for n in notes) + "</div>") if notes else ""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>"
        f"<div class='top'><a href='../index.html'>← board</a></div>"
        f"<h1>{_e(page.get('matchup') or page.get('game_id'))}</h1>"
        f"<div class='sub'>{_e(page.get('season'))} week {_e(page.get('week'))} · clock {_e(page.get('clock'))} · "
        f"as of {_e(page.get('as_of'))} · game {_e(page.get('game_id'))}</div>{pub}"
        "<h2>Best bets (props, priced markets only)</h2>" + _bets_html(page) +
        "<h2>Where the odds are</h2>" + _market_html(page) +
        "<h2>Not playing / limited</h2>" + _availability_html(page.get("availability") or {}) +
        _hand_notes_html(page.get("hand_notes") or {}, page.get("hand_notes_error")) +
        "<h2>Location, travel, body clock</h2>" + _travel_html(page.get("travel") or {}) +
        ("<h2>Game notes</h2>" + notes_html if notes_html else "") +
        "<div class='sub' style='margin-top:24px'>Leans, not locks. Every number above is the model's; "
        "the drivers are its projection ledger, not commentary. 1-800-GAMBLER.</div>"
        "</body></html>")


def write_site_pages(pages: List[Dict], site_dir: str) -> List[str]:
    """Render every page under ``{site_dir}/games/``; returns written paths."""
    out_dir = os.path.join(site_dir, "games")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for p in pages:
        gid = str(p.get("game_id") or "")
        if not gid or "/" in gid or ".." in gid:
            continue
        path = os.path.join(out_dir, f"{gid}.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_html(p))
        written.append(path)
    index = {"schema_version": 1, "pages": [{"game_id": p.get("game_id"),
                                            "matchup": p.get("matchup"),
                                            "href": p.get("href"),
                                            "n_bets": len(p.get("best_bets") or [])}
                                           for p in pages]}
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    return written
