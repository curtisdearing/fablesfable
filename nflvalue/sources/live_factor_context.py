"""Live, bounded, slate-wide factor context (QB, OL, defensive absences, game status).

Replaces the hand-curated one-game ``data/factor_context/<season>-w<week>.json`` with a
document built from free structured feeds for EVERY game on the week's slate, in the
same ``factor_context/1`` shape ``factor_integration.load_context`` already reads
(``news`` items for ``factor_evidence.assess_news`` and plain ``records`` for
``normalize_record``).  Nothing here changes a number: every item is context, and
``factor_evidence`` re-derives status and cutoff from the clocks carried on each item.

What each source can and cannot establish
-----------------------------------------
* NFL.com league injury report (``league_official``): the official Week-N practice
  participation and game-status designations.  The page carries no publication time
  (capture clock only) and, before the week's first report, the Week-N URL serves the
  PREVIOUS week's page -- so the page title must name the requested week and season or
  the whole route is rejected.  Practice participation (DNP/Limited/Full) is never a game
  status; a blank game-status cell is "no designation on this report", not "healthy".
* ESPN injuries feed (``data_feed``): per-player status with a note date.  An
  Out/Doubtful/Questionable dated before the team's previous kickoff is that earlier
  game's designation: it is emitted with ``expires_at`` = that kickoff so it reads as
  superseded, never as this week's status.  A newer one is an attributed feed report,
  not the official designation.  "Active" rows are not emitted: a feed saying active is
  not a verified health check.  RotoWire note text is a third-party summary and is
  never used as a claim.
* ESPN depth chart (``data_feed``): QB1 per team, capture clock only -> observed but
  unverified as the target-week starter.

Refresh is bounded: each route is tried once (no retry loop) under ``max_requests``.
Injury routes run official -> ESPN feed -> ESPN per-event summary (structured
fallback); if all fail the game/category is marked unknown with the exact reason.
Every game gets a coverage row for every category, with either counted records or a
specific not-obtained reason -- an empty category is never presented as complete.

Identity: the scoreboard payload must say the requested season, week and regular
season; a player links to a gsis id only through a unique ESPN id on the player's own
team in the supplied id map (no name or initial fallback); a feed row whose team is not
in the game is dropped.  No individual coverage/shadow matchup is ever produced.

Public entrypoint (no network at import; ``http`` is injectable)::

    build_live_context(season, week, *, captured_at=None, http=None, id_map=None,
                       curated=None, max_requests=48) -> dict
"""

from __future__ import annotations

import datetime as dt
import html as _html
import json
import re
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .availability import DISPLAY_TO_ABBR, _espn_id_from_links, canonical_abbr

SCHEMA = "factor_context/1"
COVERAGE_SCHEMA = "live-coverage-v1"
GENERIC_USER_AGENT = "Python-urllib/3"
SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
NFLCOM = "https://www.nfl.com/injuries/league/{season}/reg{week}"

CATEGORIES = ("qb_news", "ol_injury", "def_absence", "game_status")
OL_POS = {"T", "OT", "G", "OG", "C", "OL", "LT", "RT", "LG", "RG"}
DEF_POS = {"DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "MLB", "EDGE", "CB", "S", "FS",
           "SS", "DB"}
DESIGNATIONS = {"out", "doubtful", "questionable"}
PRACTICE = {"did not participate in practice": "DNP", "limited participation in practice": "LP",
            "full participation in practice": "FP"}

#: nfl.com section headers are club nicknames.
NICKNAMES = {
    "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF", "Panthers": "CAR",
    "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE", "Cowboys": "DAL", "Broncos": "DEN",
    "Lions": "DET", "Packers": "GB", "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX",
    "Chiefs": "KC", "Raiders": "LV", "Chargers": "LAC", "Rams": "LA", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG", "Jets": "NYJ",
    "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF", "Seahawks": "SEA", "Buccaneers": "TB",
    "Titans": "TEN", "Commanders": "WAS"}


class IdentityMismatch(ValueError):
    """A payload is not for the requested season/week/game/team."""


class RouteFailed(RuntimeError):
    """One acquisition route failed; the caller moves to the next route."""


# --------------------------------------------------------------------------- #
# clocks / http
# --------------------------------------------------------------------------- #

def _ts(x) -> Optional[dt.datetime]:
    if not x:
        return None
    if isinstance(x, dt.datetime):
        t = x
    else:
        s = str(x).strip().replace("Z", "+00:00")
        if re.match(r".*T\d\d:\d\d\+", s):
            s = s.replace("+", ":00+", 1)
        t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _iso(t: Optional[dt.datetime]) -> Optional[str]:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


def default_http(url: str, timeout: float = 20.0) -> Tuple[int, Dict[str, str], bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": GENERIC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, dict(resp.headers), resp.read()


class _Budget:
    """One attempt per route, a hard request cap, and a log of every call."""

    def __init__(self, http: Callable, max_requests: int):
        self.http, self.left, self.log = http, int(max_requests), []

    def get(self, url: str, kind: str = "json"):
        if self.left <= 0:
            raise RouteFailed(f"request budget exhausted before {url}")
        self.left -= 1
        try:
            status, headers, body = self.http(url)
        except Exception as exc:  # noqa: BLE001 -- network failure degrades, never raises
            self.log.append({"url": url, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
            raise RouteFailed(f"{type(exc).__name__}: {exc}"[:200]) from exc
        self.log.append({"url": url, "ok": status == 200, "status": status,
                         "server_date": (headers or {}).get("Date"),
                         "last_modified": (headers or {}).get("Last-Modified")})
        if status != 200:
            raise RouteFailed(f"HTTP {status} from {url}")
        if kind == "json":
            try:
                return json.loads(body)
            except ValueError as exc:
                raise RouteFailed(f"non-JSON body from {url}") from exc
        return body.decode("utf-8", "ignore")


# --------------------------------------------------------------------------- #
# slate
# --------------------------------------------------------------------------- #

def slate_from_scoreboard(raw: Dict, season: int, week: int) -> List[Dict]:
    """ESPN week scoreboard -> [{game_id, event_id, kickoff, home, away, espn_team_ids}].

    Raises ``IdentityMismatch`` unless the payload itself says this season, this week
    and the regular season, and every event carries the same season."""
    sea, wk = (raw.get("season") or {}), (raw.get("week") or {})
    if sea.get("year") != season or sea.get("type") != 2 or wk.get("number") != week:
        raise IdentityMismatch(f"scoreboard is season={sea.get('year')} type={sea.get('type')} "
                               f"week={wk.get('number')}, wanted {season} REG week {week}")
    games = []
    for ev in raw.get("events") or []:
        if (ev.get("season") or {}).get("year") != season:
            raise IdentityMismatch(f"event {ev.get('id')} is from another season")
        comp = (ev.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if set(sides) != {"home", "away"}:
            raise IdentityMismatch(f"event {ev.get('id')} lacks home/away competitors")
        ab = {k: canonical_abbr((v.get("team") or {}).get("abbreviation")) for k, v in sides.items()}
        games.append({
            "game_id": f"{season}_{week:02d}_{ab['away']}_{ab['home']}", "event_id": str(ev["id"]),
            "kickoff": _iso(_ts(ev.get("date"))), "home": ab["home"], "away": ab["away"],
            "espn_team_ids": {ab[k]: str((v.get("team") or {}).get("id")) for k, v in sides.items()}})
    if not games:
        raise IdentityMismatch("scoreboard has no events")
    return games


def _team_game(games: List[Dict]) -> Dict[str, Dict]:
    out = {}
    for g in games:
        for t in (g["home"], g["away"]):
            out[t] = g
    return out


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #

def link_player(espn_id, team: str, id_map) -> Tuple[str, str, Optional[str]]:
    """(entity_id, entity_type, identity_note).  gsis only via a UNIQUE espn id on the
    same team in ``id_map`` rows {espn_id, team, gsis_id}; otherwise an explicit
    unlinked id -- never a name or initial guess."""
    if espn_id is None:
        return f"{team}:unidentified", "team", "no player id in the source row"
    hits = {r["gsis_id"] for r in (id_map or [])
            if str(r.get("espn_id")) == str(espn_id) and r.get("team") == team and r.get("gsis_id")}
    if len(hits) == 1:
        return hits.pop(), "player", None
    why = "ambiguous" if hits else "not found"
    return f"espn:{espn_id}", "player", f"ESPN id {espn_id} {why} on {team} in the roster id map; not linked"


def category_for(pos: str) -> str:
    p = (pos or "").upper()
    if p == "QB":
        return "qb_news"
    if p in OL_POS:
        return "ol_injury"
    if p in DEF_POS:
        return "def_absence"
    return "team_news"


# --------------------------------------------------------------------------- #
# route 1: NFL.com official league injury report
# --------------------------------------------------------------------------- #

_TITLE = re.compile(r"<title>[^<]*Week\s+(\d+)\s+of\s+the\s+(\d{4})\s+Season", re.I)
_SECTION = re.compile(r'd3-o-section-sub-title"><span>([^<]+)</span>(.*?)</table>', re.S)
_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def parse_official_report(page: str, season: int, week: int) -> List[Dict]:
    """NFL.com league injury page -> [{team, name, pos, injury, practice, game_status}].

    Raises ``IdentityMismatch`` when the page is for another week/season (the Week-N
    URL serves the prior week until Week N's first report is posted)."""
    m = _TITLE.search(page or "")
    if not m or int(m.group(1)) != week or int(m.group(2)) != season:
        got = f"week {m.group(1)} {m.group(2)}" if m else "no week in title"
        raise IdentityMismatch(f"official report page is {got}, wanted week {week} {season}")
    rows = []
    for nick, body in _SECTION.findall(page):
        team = NICKNAMES.get(_html.unescape(nick).strip())
        if not team:
            continue
        for tr in _ROW.findall(body):
            cells = [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in _CELL.findall(tr)]
            if len(cells) != 5:
                continue
            name, pos, injury, practice, status = cells
            rows.append({"team": team, "name": name, "pos": pos, "injury": injury,
                         "practice": PRACTICE.get(practice.lower(), practice or None),
                         "game_status": status or None})
    return rows


def official_items(rows: List[Dict], games: List[Dict], url: str, fetched_at: str) -> List[Dict]:
    """Official rows -> news items.  Practice and game status are separate claims."""
    tg, items = _team_game(games), []
    for r in rows:
        g = tg.get(r["team"])
        if not g:
            continue
        cat = category_for(r["pos"])
        base = dict(entity_id=f"{r['team']}:{r['name']}", entity_type="team", team=r["team"],
                    game_id=g["game_id"], attribution="NFL.com official injury report",
                    source_tier="league_official", source_url=url,
                    source_title=f"Official NFL injury report, {g['game_id'][:7]}",
                    published_at=fetched_at, observed_at=fetched_at, fetched_at=fetched_at)
        slug = re.sub(r"[^a-z0-9]+", "_", r["name"].lower())
        if r.get("practice"):
            items.append({**base, "story_id": f"nflcom_practice:{g['game_id']}:{r['team']}:{slug}",
                          "category": cat, "claim_key": "practice", "claim_kind": "confirmed",
                          "claim_value": r["practice"],
                          "claim": f"{r['pos']} {r['name']} ({r['injury'] or 'no injury listed'}): "
                                   f"{r['practice']} on the latest listed practice day.",
                          "uncertainty": "Practice participation is not a game status."})
        if (r.get("game_status") or "").lower() in DESIGNATIONS:
            items.append({**base, "story_id": f"nflcom_status:{g['game_id']}:{r['team']}:{slug}",
                          "category": cat, "claim_key": "game_status", "claim_kind": "confirmed",
                          "claim_value": r["game_status"],
                          "claim": f"{r['pos']} {r['name']}: {r['game_status']} ({r['injury']})."})
    return items


# --------------------------------------------------------------------------- #
# route 2: ESPN injuries feed
# --------------------------------------------------------------------------- #

def parse_espn_injuries(raw: Dict) -> List[Dict]:
    if not isinstance(raw, dict) or "injuries" not in raw:
        raise RouteFailed("injuries payload missing 'injuries'")
    out = []
    for t in raw["injuries"]:
        for it in t.get("injuries") or []:
            ath = it.get("athlete") or {}
            # ``it["id"]`` is the injury NOTE id, not the athlete: identity comes from
            # the athlete's own profile link only.
            out.append({"espn_id": _espn_id_from_links(ath), "note_id": it.get("id"),
                        "name": ath.get("displayName") or "",
                        "pos": ((ath.get("position") or {}).get("abbreviation") or "").upper(),
                        "team": canonical_abbr((ath.get("team") or {}).get("abbreviation")),
                        "status": str(it.get("status") or ""), "date": it.get("date")})
    return out


def espn_items(rows: List[Dict], games: List[Dict], prior_kickoff: Dict[str, str],
               url: str, fetched_at: str, id_map=None, season: int = 0, week: int = 0) -> List[Dict]:
    """ESPN feed rows -> attributed feed reports (never 'confirmed' official status).

    ``prior_kickoff``: {team: ISO kickoff of the team's previous game}.  A designation
    dated at or before it belonged to that game and expires there (superseded)."""
    tg, items = _team_game(games), []
    for r in rows:
        g = tg.get(r["team"])
        st = r["status"].strip().lower()
        if not g or st in ("active", ""):
            continue
        noted = _ts(r["date"])
        if noted and noted > _ts(fetched_at):
            continue  # a note dated after capture is not a valid clock
        eid, etype, idnote = link_player(r["espn_id"], r["team"], id_map)
        cat = category_for(r["pos"])
        prior = _ts(prior_kickoff.get(r["team"]))
        base = dict(entity_id=eid, entity_type=etype, team=r["team"], game_id=g["game_id"],
                    category=cat, attribution="ESPN injuries feed", source_tier="data_feed",
                    claim_kind="report", source_url=url, source_title="ESPN NFL injuries feed",
                    published_at=_iso(noted), fetched_at=fetched_at, source_id=r.get("note_id") or r["espn_id"],
                    story_id=f"espn_inj:{g['game_id']}:{r['espn_id'] or r['name']}")
        extra = f" {idnote}." if idnote else ""
        if st == "injured reserve":
            items.append({**base, "claim_key": "roster_status", "claim_value": "Injured Reserve",
                          "claim": f"{r['pos']} {r['name']}: on injured reserve per the ESPN feed.",
                          "uncertainty": "Feed roster status; not the official report." + extra})
        elif st in DESIGNATIONS and prior and noted and noted <= prior:
            items.append({**base, "claim_key": "prior_game_status", "claim_value": r["status"],
                          "expires_at": _iso(prior),
                          "claim": f"{r['pos']} {r['name']}: {r['status']} for the previous game "
                                   f"(feed note {_iso(noted)}).",
                          "uncertainty": f"Designation for the game before Week {week}; superseded "
                                         f"and not a Week {week} status." + extra})
        else:
            items.append({**base, "claim_key": "game_status_report", "claim_value": r["status"],
                          "claim": f"{r['pos']} {r['name']}: {r['status']} per the ESPN feed.",
                          "uncertainty": "Feed status, not the official Week-" + str(week) +
                                         " designation." + extra})
    return items


# --------------------------------------------------------------------------- #
# structured fallback: ESPN per-event summary injuries block
# --------------------------------------------------------------------------- #

def summary_rows(raw: Dict) -> List[Dict]:
    out = []
    for blk in (raw or {}).get("injuries") or []:
        team = canonical_abbr((blk.get("team") or {}).get("abbreviation"))
        for it in blk.get("injuries") or []:
            ath = it.get("athlete") or {}
            out.append({"espn_id": str(ath.get("id")) if ath.get("id") else _espn_id_from_links(ath),
                        "name": ath.get("displayName") or "",
                        "pos": ((ath.get("position") or {}).get("abbreviation") or "").upper(),
                        "team": team, "status": str(it.get("status") or ""), "date": it.get("date")})
    return out


# --------------------------------------------------------------------------- #
# QB depth chart
# --------------------------------------------------------------------------- #

def qb_from_depthchart(raw: Dict) -> List[str]:
    """ESPN core depth chart -> QB espn ids in rank order (first offensive chart)."""
    for chart in (raw or {}).get("items") or []:
        qb = (chart.get("positions") or {}).get("qb")
        if qb:
            ath = sorted(qb.get("athletes") or [], key=lambda a: a.get("rank", 99))
            return [re.sub(r"\?.*", "", (a.get("athlete") or {}).get("$ref", "")).rsplit("/", 1)[-1]
                    for a in ath]
    return []


def qb_record(team: str, game: Dict, qb_ids: List[str], url: str, fetched_at: str,
              id_map=None, week: int = 0) -> Dict:
    eid, _, idnote = link_player(qb_ids[0], team, id_map) if qb_ids else (None, None, None)
    return {"factor_id": f"qb_depth:{team}:{game['game_id']}", "category": "qb_news",
            "entity_type": "team", "entity_id": team, "team": team, "game_id": game["game_id"],
            "measurement_kind": "observed", "verified": False, "populated": True,
            "value": eid, "observation": f"QB1 on the ESPN depth chart at capture: {eid}"
            + (f" ({idnote})" if idnote else ""),
            "source_url": url, "source_title": "ESPN depth chart (capture clock only)",
            "fetched_at": fetched_at,
            "reason_not_applied": f"depth chart has no publication time; not verified as the "
                                  f"Week {week} starter"}


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #

def coverage_matrix(games: List[Dict], items: List[Dict], records: List[Dict],
                    route_status: Dict[str, str]) -> Dict[str, Dict[str, Dict]]:
    """{game_id: {category: {state, n_items, sources, reason}}}.  States:
    ``official`` (league/team official items), ``feed_report`` (data-feed only),
    ``not_obtained`` (specific reason).  Zero rows is never 'complete'."""
    out = {}
    for g in games:
        row = {}
        gi = [i for i in items if i.get("game_id") == g["game_id"]]
        gr = [r for r in records if r.get("game_id") == g["game_id"]]
        for cat in CATEGORIES:
            if cat == "game_status":
                sel = [i for i in gi if i.get("claim_key") in ("game_status", "game_status_report")]
            else:
                sel = [i for i in gi if i.get("category") == cat] + [
                    r for r in gr if r.get("category") == cat and r.get("measurement_kind") != "unavailable"]
            tiers = sorted({i.get("source_tier", "data_feed") for i in sel})
            if any(t in ("league_official", "team_official") for t in tiers):
                state, reason = "official", None
            elif sel:
                state, reason = "feed_report", "only data-feed rows; official Week report not obtained"
            else:
                state = "not_obtained"
                reason = ("no rows in any obtained source; not evidence of health. Routes: " +
                          "; ".join(f"{k}={v}" for k, v in sorted(route_status.items())))
            row[cat] = {"state": state, "n_items": len(sel), "sources": tiers, "reason": reason}
        out[g["game_id"]] = row
    return out


def not_obtained_records(coverage: Dict, captured_at: str) -> List[Dict]:
    recs = []
    for gid, row in coverage.items():
        for cat, c in row.items():
            if c["state"] == "official":
                continue
            recs.append({"factor_id": f"coverage:{cat}:{gid}",
                         "category": "team_news" if cat == "game_status" else cat,
                         "entity_type": "game", "entity_id": gid, "game_id": gid,
                         "measurement_kind": "unavailable", "verified": False,
                         "observation": f"{cat.replace('_', ' ')}: official Week report not "
                                        f"obtained at capture {captured_at}",
                         "fetched_at": captured_at,
                         "reason_not_applied": f"{c['reason']}; unknown is not 'healthy'"})
    return recs


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #

def build_live_context(season: int, week: int, *, captured_at=None, http: Optional[Callable] = None,
                       id_map: Optional[Iterable[Dict]] = None, curated: Optional[Dict] = None,
                       prior_kickoff: Optional[Dict[str, str]] = None,
                       club_reports: Optional[Dict[str, str]] = None,
                       max_requests: int = 48) -> Dict:
    """Build a ``factor_context/1`` document covering every game on the slate.

    ``curated``: an earlier document for this season/week; its items/records are kept
    verbatim for their games (never rewritten), live items are added alongside.
    ``prior_kickoff``: {team: previous kickoff ISO}; when omitted it is read from the
    previous week's scoreboard (one request).  ``club_reports``: {game_id: team-official
    injury-report URL} fetched once each and parsed by :func:`parse_club_report`.  Raises ``IdentityMismatch`` only when the
    slate itself cannot be established; every later failure degrades to coverage rows.
    """
    cap = _iso(_ts(captured_at) or dt.datetime.now(dt.timezone.utc))
    b = _Budget(http or default_http, max_requests)
    id_map = list(id_map or [])
    board_url = f"{SITE}/scoreboard?seasontype=2&week={week}&dates={season}"
    games = slate_from_scoreboard(b.get(board_url), season, week)
    route: Dict[str, str] = {}

    if prior_kickoff is None and week > 1:
        prior_kickoff = {}
        try:
            prev = slate_from_scoreboard(
                b.get(f"{SITE}/scoreboard?seasontype=2&week={week - 1}&dates={season}"), season, week - 1)
            for g in prev:
                prior_kickoff[g["home"]] = prior_kickoff[g["away"]] = g["kickoff"]
        except (RouteFailed, IdentityMismatch) as exc:
            route["prior_week_clock"] = f"failed: {exc}"
    prior_kickoff = prior_kickoff or {}

    items: List[Dict] = []
    off_url = NFLCOM.format(season=season, week=week)
    try:
        items += official_items(parse_official_report(b.get(off_url, "text"), season, week),
                                games, off_url, cap)
        route["official_report"] = "ok"
    except (RouteFailed, IdentityMismatch) as exc:
        route["official_report"] = f"failed: {exc}"
    by_id = {g["game_id"]: g for g in games}
    for gid, url in sorted((club_reports or {}).items()):
        g = by_id.get(gid)
        try:
            if g is None:
                raise IdentityMismatch(f"{gid} is not on the Week {week} slate")
            items += club_items(parse_club_report(b.get(url, "text"), season, week, g["home"], g["away"]),
                                g, url, cap)
            route[f"club_report:{gid}"] = "ok"
        except (RouteFailed, IdentityMismatch) as exc:
            route[f"club_report:{gid}"] = f"failed: {exc}"
    try:
        items += espn_items(parse_espn_injuries(b.get(f"{SITE}/injuries")), games, prior_kickoff,
                            f"{SITE}/injuries", cap, id_map, season, week)
        route["espn_injuries"] = "ok"
    except (RouteFailed, IdentityMismatch) as exc:
        route["espn_injuries"] = f"failed: {exc}"
        if not route["official_report"] == "ok":
            got = 0
            for g in games:
                u = f"{SITE}/summary?event={g['event_id']}"
                try:
                    items += espn_items(summary_rows(b.get(u)), [g], prior_kickoff, u, cap,
                                        id_map, season, week)
                    got += 1
                except RouteFailed:
                    continue
            route["espn_summary_fallback"] = f"{got}/{len(games)} events"

    records: List[Dict] = []
    qb_ok = 0
    for g in games:
        for team in (g["away"], g["home"]):
            u = f"{CORE}/seasons/{season}/teams/{g['espn_team_ids'][team]}/depthcharts"
            try:
                ids = qb_from_depthchart(b.get(u))
            except RouteFailed:
                continue
            if ids:
                records.append(qb_record(team, g, ids, u, cap, id_map, week))
                qb_ok += 1
    route["espn_depthchart_qb"] = f"{qb_ok}/{2 * len(games)} teams"

    gids = {g["game_id"] for g in games}
    kept_games = set()
    if curated and curated.get("season") == season and curated.get("week") == week:
        cn = [i for i in curated.get("news", []) if i.get("game_id") in gids]
        cr = [r for r in curated.get("records", []) if r.get("game_id") in gids]
        kept_games = {i["game_id"] for i in cn + cr}
        items, records = cn + items, cr + records  # curated kept verbatim; live added alongside
    coverage = coverage_matrix(games, items, records, route)
    records += not_obtained_records(coverage, cap)
    return {"schema": SCHEMA, "season": season, "week": week,
            "note": ("Built by nflvalue.sources.live_factor_context from free structured feeds at "
                     f"{cap}. Context only; per-item clocks decide what a run may see."),
            "captured_at": cap, "coverage_schema": COVERAGE_SCHEMA,
            "games": [{k: g[k] for k in ("game_id", "event_id", "kickoff", "home", "away")}
                      for g in games],
            "curated_games_kept": sorted(kept_games), "routes": route, "coverage": coverage,
            "sources_checked": sorted({c["url"] for c in b.log if c.get("ok")}),
            "request_log": b.log, "news": items, "records": records}


# --------------------------------------------------------------------------- #
# route 1b: team-official (club site) injury report for one game
# --------------------------------------------------------------------------- #
# The NFL.com league page can lag the clubs: at 2026-09-23T22:33Z its Week-3 ATL/GB
# rows had no game-status cells while packers.com had published the final Thursday
# report (datePublished 20:00Z) with game statuses for BOTH teams.  Club article URLs
# are slugs, so the caller supplies the URL; this parser checks the page is about the
# requested game and never guesses a team.
_CLUB_TABLE = re.compile(r"<table.*?</table>", re.S)
_CLUB_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_PUBLISHED = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')
_OG_TITLE = re.compile(r'<title>([^<]*)</title>', re.I)
_HEADING = re.compile(r"<h[2-4][^>]*>(.*?)</h[2-4]>", re.S)
_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday")


def _text(fragment: str) -> str:
    return _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_club_report(page: str, season: int, week: int, home: str, away: str) -> Dict:
    """Club-site injury report -> {published_at, title, rows}.

    Row: {team, name, pos, injury, practice: [{day, status, estimated}], game_status}.
    ``estimated`` is True for a column whose header is starred (the club's footnote:
    walkthrough / no practice, "participation reports are an estimation").  A blank or
    "--" game status is None: no designation on this report, NOT a health claim.

    Raises ``IdentityMismatch`` when the page is not a Week-``week`` report naming both
    teams, a table's team cannot be read from the heading just before it, a table
    belongs to neither team, or the page carries no publication clock."""
    page = page or ""
    tm = _OG_TITLE.search(page)
    title = _text(tm.group(1)) if tm else ""
    if not re.search(rf"\bWeek\s+{week}\b", title, re.I):
        raise IdentityMismatch(f"club report title {title!r} does not name Week {week}")
    pm = _PUBLISHED.search(page)
    pub = _ts(pm.group(1)) if pm else None
    if pub is None or pub.year != season:
        raise IdentityMismatch("club report has no datePublished in the requested season")
    rows, seen, prev_end = [], set(), 0
    for m in _CLUB_TABLE.finditer(page):
        heads = [_text(h) for h in _HEADING.findall(page[prev_end:m.start()])]
        prev_end = m.end()
        if not heads or heads[-1] not in DISPLAY_TO_ABBR:
            raise IdentityMismatch("club report table without a team-name heading before it")
        team = canonical_abbr(DISPLAY_TO_ABBR[heads[-1]])
        if team not in (home, away):
            raise IdentityMismatch(f"club report table for {team}, not {away}@{home}")
        trs = re.findall(r"<tr.*?</tr>", m.group(0), re.S)
        head = [_text(c) for c in _CLUB_CELL.findall(trs[0])] if trs else []
        low = [h.lstrip("*").strip().lower() for h in head]
        if not low or low[0] != "player" or low[-1] != "game status":
            raise IdentityMismatch(f"club report table header {head!r} not recognized")
        day_cols = [(i, low[i], head[i].startswith("*")) for i in range(len(low)) if low[i] in _DAYS]
        seen.add(team)
        for tr in trs[1:]:
            cells = [_text(c) for c in _CLUB_CELL.findall(tr)]
            if len(cells) != len(head):
                continue
            name, _, pos = cells[0].rpartition(",")
            practice = [{"day": d, "status": PRACTICE.get(f"{cells[i].lower()} in practice", None),
                         "estimated": est} for i, d, est in day_cols if cells[i] not in ("", "--")]
            gs = cells[-1] if cells[-1] not in ("", "--") else None
            rows.append({"team": team, "name": name.strip() or cells[0], "pos": pos.strip(),
                         "injury": cells[1], "practice": practice, "game_status": gs})
    if not rows:
        raise IdentityMismatch("club report has no rows for this game")
    return {"published_at": _iso(pub), "title": title, "teams": sorted(seen), "rows": rows}


def club_items(parsed: Dict, game: Dict, url: str, fetched_at: str) -> List[Dict]:
    """Club rows -> news items.  Game status and latest practice day are separate claims."""
    items, pub = [], parsed["published_at"]
    for r in parsed["rows"]:
        cat = category_for(r["pos"])
        slug = re.sub(r"[^a-z0-9]+", "_", r["name"].lower())
        base = dict(entity_id=f"{r['team']}:{r['name']}", entity_type="team", team=r["team"],
                    game_id=game["game_id"], category=cat, attribution="Team-official injury report",
                    source_tier="team_official", source_url=url, source_title=parsed["title"],
                    published_at=pub, observed_at=pub, fetched_at=fetched_at)
        if r["practice"]:
            last = r["practice"][-1]
            kind = "report" if last["estimated"] else "confirmed"  # an estimate is not observed
            items.append({**base, "story_id": f"club_practice:{game['game_id']}:{r['team']}:{slug}",
                          "claim_key": "practice", "claim_kind": kind, "claim_value": last["status"],
                          "claim": (f"{r['pos']} {r['name']} ({r['injury'] or 'no injury listed'}): "
                                    f"{last['status']} {last['day'].title()}"
                                    f"{' (estimated; no full practice held)' if last['estimated'] else ''}."),
                          "uncertainty": "Practice participation is not a game status."})
        if (r["game_status"] or "").lower() in DESIGNATIONS:
            items.append({**base, "story_id": f"club_status:{game['game_id']}:{r['team']}:{slug}",
                          "claim_key": "game_status", "claim_kind": "confirmed",
                          "claim_value": r["game_status"],
                          "claim": f"{r['pos']} {r['name']}: {r['game_status']} ({r['injury']})."})
    return items
