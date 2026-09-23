"""Personnel / matchup evidence: starting QB, offensive line, opposing defense.

Pure builder of *context* evidence records in the shared factor-evidence
schema (see the parallel contract).  It answers "who is documented to be
available, who is documented to be missing, who is expected to replace them,
and which offensive players' roles touch that personnel" -- and it keeps
documented facts apart from inferred matchups.

What it deliberately does NOT do:

* No numerical adjustment.  Every record is ``context_only`` (or
  ``unavailable_unverified``); ``numerical_effect`` is always ``None``.  No
  pregame-as-of historical test in this repo has validated a conditional
  QB/OL/defense effect beyond what the primary path already consumes (see
  ``OVERLAPS``), so a number here would be invented.
* No double counting.  The primary/ranker path already consumes
  ``qb_continuity`` + ``backup_qb_adj`` (QB), ``oline_outs`` (OL), the
  context_features defensive-out counts, ``opp_pressure_rate`` and FTN
  ``opp_blitz_rate``/``opp_box_avg``.  Records name the overlapping consumer
  in ``overlaps_existing`` and set ``consumed=False``; an integrator that wants
  numbers must replace, not add to, those consumers.
* "No matching injury row" is never "healthy".  A player absent from a
  published report is ``not_listed_unverified``; with no report at all he is
  ``unknown_no_report``.  Unknown is carried as ``None`` / explicit counts,
  never folded into zero.
* No CB-shadows-WR statement without a verified, attributed source row, and a
  defensive absence is linked to the offensive roles it plausibly touches
  with ``direction="not_estimated"`` -- it does not improve every prop.
* Rows published (or, lacking a publication stamp, captured) after ``as_of``
  or at/after kickoff are excluded and counted; rows with neither clock cannot
  pass the cutoff check.  Differing statuses are ordered only by publication
  clocks; otherwise they stay ``conflict_unresolved``.

Entrypoint: :func:`build_personnel_evidence`.  Inputs are plain lists of
dicts (see its docstring); nothing is fetched and inputs are not mutated.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "factor_evidence/1"
MODEL_VERSION = "personnel_matchup-context-v1"
COMPONENT = "personnel_matchup"

# Positions as documented by the source; mapping is to *groups*, and a
# position that plausibly belongs to more than one group is flagged ambiguous
# instead of being silently assigned.
OL_POS = {"T", "OT", "LT", "RT", "G", "OG", "LG", "RG", "C", "OL"}
QB_POS = {"QB"}
DEF_GROUPS: Dict[str, Dict[str, Any]] = {
    "DE": {"groups": ("pass_rush_edge", "run_front"), "ambiguous": False},
    "EDGE": {"groups": ("pass_rush_edge", "run_front"), "ambiguous": False},
    "DT": {"groups": ("pass_rush_interior", "run_front"), "ambiguous": False},
    "NT": {"groups": ("pass_rush_interior", "run_front"), "ambiguous": False},
    "DL": {"groups": ("pass_rush_interior", "pass_rush_edge", "run_front"), "ambiguous": True},
    "OLB": {"groups": ("pass_rush_edge", "run_front"), "ambiguous": True},
    "LB": {"groups": ("run_front",), "ambiguous": True},
    "ILB": {"groups": ("run_front",), "ambiguous": False},
    "MLB": {"groups": ("run_front",), "ambiguous": False},
    "CB": {"groups": ("coverage",), "ambiguous": False},
    "NB": {"groups": ("coverage",), "ambiguous": False},
    "DB": {"groups": ("coverage",), "ambiguous": False},
    "S": {"groups": ("coverage",), "ambiguous": False},
    "SS": {"groups": ("coverage",), "ambiguous": False},
    "FS": {"groups": ("coverage",), "ambiguous": False},
}
DEF_GROUP_NAMES = ("pass_rush_edge", "pass_rush_interior", "run_front", "coverage")
ABSENT_STATUSES = {"out", "doubtful"}
UNCERTAIN_STATUSES = {"questionable"}
# Report statuses that are documented but carry no game designation.
NO_DESIGNATION = {"none", "no_designation", "-", "(-)", ""}

# Existing consumers already carrying (part of) each factor.  Named so the
# integrator can see the double-count risk; this module never feeds them.
OVERLAPS: Dict[str, List[str]] = {
    "qb_starter": ["advanced_features.qb_continuity (ml_ranker feature)",
                   "candidates.apply_backup_qb_adjustment (primary mean x0.92)"],
    "ol_availability": ["advanced_features.oline_outs (ml_ranker feature; missing coerced to 0)"],
    "def_pass_rush_edge": ["context_features defense_outs", "chemistry.opp_pressure_rate",
                           "ftn_features.opp_blitz_rate"],
    "def_pass_rush_interior": ["context_features defense_outs", "chemistry.opp_pressure_rate"],
    "def_run_front": ["context_features defense_outs", "ftn_features.opp_box_avg"],
    "def_coverage": ["context_features defense_outs (DB-only count)"],
}

# Which defensive groups a given offensive role plausibly touches.  This is a
# relevance link, not an effect: sign and size are not estimated.
ROLE_DEF_RELEVANCE = {
    "QB": ("pass_rush_edge", "pass_rush_interior", "coverage"),
    "WR": ("coverage",),
    "TE": ("coverage", "run_front"),
    "RB": ("run_front",),
}
ROLE_OWN_RELEVANCE = {
    "QB": ("qb_starter", "ol_availability"),
    "WR": ("qb_starter",),
    "TE": ("qb_starter",),
    "RB": ("qb_starter", "ol_availability"),
}


class PersonnelInputError(ValueError):
    """Malformed top-level input (e.g. naive ``as_of``)."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _ts(value: Any) -> Optional[dt.datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        d = value
    else:
        try:
            d = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return d if d.tzinfo is not None else None


def _iso(d: Optional[dt.datetime]) -> Optional[str]:
    return d.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z") if d else None


def norm_name(name: Any) -> str:
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _status(raw: Any) -> str:
    s = str(raw if raw is not None else "").strip().lower()
    return "no_designation" if s in NO_DESIGNATION else s


def _record(**kw: Any) -> Dict[str, Any]:
    """One shared-schema factor evidence record (all keys always present)."""
    base = {
        "schema_version": SCHEMA_VERSION, "factor_id": None, "category": None,
        "entity_id": None, "entity_type": None, "game_id": None, "as_of": None,
        "observation": None, "value": None, "unit": None,
        "observed_at": None, "published_at": None, "fetched_at": None,
        "source_url": None, "source_id": None, "sources": [],
        "verified": False, "cutoff_ok": False,
        "measurement_kind": "unavailable", "status": "unavailable_unverified",
        "component": COMPONENT, "model_version": MODEL_VERSION,
        "feature_name": None, "consumed": False, "numerical_effect": None,
        "numerical_effect_unit": None, "numerical_effect_method": None,
        "support_games": None, "support_opportunities": None,
        "reason_not_applied": None, "overlaps_existing": [],
        "rationale": None, "uncertainty": None,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# input screening
# --------------------------------------------------------------------------- #
def _screen(rows: Iterable[Dict], as_of: dt.datetime, kickoff: Optional[dt.datetime],
            excluded: List[Dict], kind: str) -> List[Dict]:
    """Keep rows known at/before as_of (and before kickoff).

    The clock is ``published_at`` when the source states one, else
    ``captured_at`` (when this copy was actually retrieved): a document captured
    before a live decision was demonstrably available to it, even without a
    publication stamp.  A capture clock only bounds THAT capture -- in a
    historical replay a later capture is excluded, never backdated.  Rows with
    neither clock are kept but marked ``cutoff_ok=False``."""
    kept = []
    for r in rows:
        pub = _ts(r.get("published_at"))
        cap = _ts(r.get("captured_at"))
        clock = pub or cap
        basis = "published_at" if pub else ("captured_at" if cap else None)
        tag = "published" if pub else "captured"
        if clock is not None and clock > as_of:
            excluded.append({"kind": kind, "reason": f"{tag}_after_as_of", "row": _brief(r)})
            continue
        if clock is not None and kickoff is not None and clock >= kickoff:
            excluded.append({"kind": kind, "reason": f"{tag}_at_or_after_kickoff",
                             "row": _brief(r)})
            continue
        kept.append(dict(r, _pub=pub, _clock_basis=basis, _cutoff_ok=clock is not None))
    return kept


def _brief(r: Dict) -> Dict:
    return {k: r.get(k) for k in ("player_id", "name", "team", "game_id",
                                  "published_at", "source_id") if r.get(k) is not None}


def _resolve_identity(row: Dict, roster_by_team: Dict[str, List[Dict]]
                      ) -> Tuple[Optional[str], Optional[str]]:
    """Return (player_id, error).  Explicit IDs win; otherwise an exact
    normalized-name match on the same team must be unique."""
    pid = row.get("player_id")
    team = row.get("team")
    if pid:
        return str(pid), None
    nm = norm_name(row.get("name"))
    if not nm or not team:
        return None, "missing_name_or_team"
    hits = {str(p["player_id"]) for p in roster_by_team.get(team, [])
            if p.get("player_id") and norm_name(p.get("name")) == nm}
    if len(hits) == 1:
        return hits.pop(), None
    return None, "ambiguous_name" if hits else "no_roster_match"


# --------------------------------------------------------------------------- #
# availability state per player
# --------------------------------------------------------------------------- #
def _availability(pid: str, team: str, game_id: str, current: Dict[str, List[Dict]],
                  report_published: Dict[Tuple[str, str], List[Dict]],
                  prior: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Consolidated availability decision for one player (one record, even
    when several sources list him -- conflicts are surfaced, not resolved)."""
    rows = current.get(pid, [])
    statuses = sorted({_status(r.get("report_status")) for r in rows})
    sources = [_src(r) for r in rows]
    superseded: List[Dict] = []
    status_basis = "single_status" if len(statuses) <= 1 else "conflict"
    if len(statuses) > 1 and all(r.get("_pub") is not None for r in rows):
        # a strictly newer PUBLICATION is an update, not a contradiction;
        # same-time or capture-only-clocked disagreements stay unresolved
        newest = max(r["_pub"] for r in rows)
        top = {_status(r.get("report_status")) for r in rows if r["_pub"] == newest}
        if len(top) == 1:
            superseded = [dict(_src(r), report_status=r.get("report_status"))
                          for r in rows if r["_pub"] < newest]
            statuses = sorted(top)
            status_basis = "latest_publication"
    prior_rows = prior.get(pid, [])
    prior_absent = any(_status(r.get("report_status")) in ABSENT_STATUSES
                       or r.get("did_not_play") for r in prior_rows)
    reports = report_published.get((team, game_id), [])
    if rows:
        if len(statuses) > 1:
            state = "conflict_unresolved"
        elif statuses[0] in ABSENT_STATUSES:
            state = f"documented_{statuses[0]}"
        elif statuses[0] in UNCERTAIN_STATUSES:
            state = "documented_questionable"
        elif statuses[0] == "no_designation":
            state = "listed_no_designation"
        else:
            state = f"documented_{statuses[0]}"
    elif reports:
        state = "not_listed_unverified"
    else:
        state = "unknown_no_report"
    returning = None
    if prior_absent:
        if state == "listed_no_designation":
            returning = "returning_documented"
        elif state in ("not_listed_unverified", "unknown_no_report"):
            returning = "returning_unverified"
        elif state == "documented_questionable":
            returning = "returning_uncertain"
    practice = [p for r in rows for p in (r.get("practice") or [])]
    return {"player_id": pid, "state": state, "statuses": statuses,
            "practice": practice, "returning": returning,
            "status_basis": status_basis, "superseded": superseded,
            "prior_absent": prior_absent, "sources": sources,
            "cutoff_ok": bool(rows) and all(r["_cutoff_ok"] for r in rows),
            "report_sources": [_src(r) for r in reports]}


def _src(r: Dict) -> Dict:
    return {"source_id": r.get("source_id"), "source_url": r.get("source_url"),
            "published_at": _iso(r.get("_pub")) if r.get("_pub") else r.get("published_at"),
            "captured_at": r.get("captured_at"), "fetched_at": r.get("fetched_at"),
            "clock_basis": r.get("_clock_basis"), "cutoff_ok": bool(r.get("_cutoff_ok"))}


def _is_absent(av: Dict) -> Optional[bool]:
    """True documented absent, False documented not absent, None unknown."""
    s = av["state"]
    if s in ("documented_out", "documented_doubtful"):
        return True
    if s == "listed_no_designation":
        return False
    return None


# --------------------------------------------------------------------------- #
# public entrypoint
# --------------------------------------------------------------------------- #
def build_personnel_evidence(*, games: List[Dict], roster: List[Dict], as_of: Any,
                             depth: Optional[List[Dict]] = None,
                             availability: Optional[List[Dict]] = None,
                             report_index: Optional[List[Dict]] = None,
                             prior_availability: Optional[List[Dict]] = None,
                             starter_claims: Optional[List[Dict]] = None,
                             offensive_players: Optional[List[Dict]] = None,
                             matchup_claims: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Build QB/OL/defense personnel evidence for each game.

    Parameters (all plain dicts; unknown keys ignored, inputs not mutated):

    games            ``{game_id, season, week, home_team, away_team, kickoff_utc,
                     source_url?, source_id?, published_at?, fetched_at?}``
    roster           ``{team, player_id, name, position}`` -- documented positions.
    as_of            timezone-aware datetime / ISO string (decision clock).
    depth            ``{team, player_id, slot, rank, source_id?, published_at?}``;
                     slot e.g. ``QB``, ``LT``, ``LG``, ``C``, ``RG``, ``RT``;
                     rank 1 = documented starter.  Missing -> starters unknown.
    availability     injury-report rows for the TARGET game:
                     ``{game_id, team, player_id?|name, position, report_status,
                     practice?, published_at, fetched_at?, source_url, source_id}``.
    report_index     published reports for the target game even if a player is
                     not on them: ``{game_id, team, published_at, source_url,
                     source_id}``.  Without one, unlisted players are
                     ``unknown_no_report`` (not healthy).
    prior_availability  rows from the team's previous game (report_status
                     Out/Doubtful or ``did_not_play``) for return detection.
    starter_claims   attributable QB-start statements: ``{game_id, team,
                     player_id, claim: 'expected_starter'|'will_not_start',
                     attribution, source_url, published_at}``.
    offensive_players  ``{player_id, team, game_id, position, expected_alignment?}``
                     supplied by the opportunity specialist (not computed here).
    matchup_claims   verified alignment/shadow statements: ``{game_id,
                     offense_player_id, defense_player_id, kind, attribution,
                     source_url, published_at, verified}``.

    Returns ``{schema_version, component, model_version, as_of, games: {...},
    records: [...], relevance_links: [...], excluded_rows: [...],
    identity_errors: [...], errors: [...]}``.
    """
    as_of_dt = _ts(as_of)
    if as_of_dt is None:
        raise PersonnelInputError("as_of must be a timezone-aware datetime or ISO string")
    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "component": COMPONENT,
        "model_version": MODEL_VERSION, "as_of": _iso(as_of_dt),
        "games": {}, "records": [], "relevance_links": [], "excluded_rows": [],
        "identity_errors": [], "errors": []}

    roster_by_team: Dict[str, List[Dict]] = {}
    pos_of: Dict[str, str] = {}
    name_of: Dict[str, str] = {}
    team_of: Dict[str, set] = {}
    for p in roster or []:
        if not p.get("player_id") or not p.get("team"):
            out["identity_errors"].append({"kind": "roster", "reason": "missing_id_or_team",
                                           "row": _brief(p)})
            continue
        pid = str(p["player_id"])
        roster_by_team.setdefault(p["team"], []).append(p)
        team_of.setdefault(pid, set()).add(p["team"])
        prev = pos_of.get(pid)
        pos = str(p.get("position") or "").upper()
        if prev and prev != pos:
            pos_of[pid] = "CONFLICT"
        else:
            pos_of[pid] = pos
        name_of[pid] = p.get("name") or pid

    seen_games = set()
    for g in sorted(games or [], key=lambda x: (str(x.get("kickoff_utc")), str(x.get("game_id")))):
        gid = g.get("game_id")
        kickoff = _ts(g.get("kickoff_utc"))
        if not gid or not g.get("home_team") or not g.get("away_team") or kickoff is None:
            out["errors"].append({"kind": "game", "reason": "missing_id_teams_or_tz_kickoff",
                                  "row": {k: g.get(k) for k in ("game_id", "home_team", "away_team",
                                                                 "kickoff_utc")}})
            continue
        if gid in seen_games:
            out["errors"].append({"kind": "game", "reason": "duplicate_game_id", "game_id": gid})
            continue
        seen_games.add(gid)
        if kickoff <= as_of_dt:
            out["errors"].append({"kind": "game", "reason": "as_of_not_before_kickoff",
                                  "game_id": gid})
            continue
        _build_game(g, kickoff, as_of_dt, out, roster_by_team, pos_of, name_of, team_of,
                    depth or [], availability or [], report_index or [],
                    prior_availability or [], starter_claims or [],
                    offensive_players or [], matchup_claims or [])
    return out


def _build_game(g, kickoff, as_of_dt, out, roster_by_team, pos_of, name_of, team_of,
                depth, availability, report_index, prior_availability, starter_claims,
                offensive_players, matchup_claims) -> None:
    gid = g["game_id"]
    teams = (g["home_team"], g["away_team"])
    ex = out["excluded_rows"]
    avail = _screen([r for r in availability if r.get("game_id") == gid],
                    as_of_dt, kickoff, ex, "availability")
    reports = _screen([r for r in report_index if r.get("game_id") == gid],
                      as_of_dt, kickoff, ex, "report_index")
    prior = _screen([r for r in prior_availability if r.get("team") in teams],
                    as_of_dt, kickoff, ex, "prior_availability")
    claims = _screen([r for r in starter_claims if r.get("game_id") == gid],
                     as_of_dt, kickoff, ex, "starter_claim")
    mclaims = _screen([r for r in matchup_claims if r.get("game_id") == gid],
                      as_of_dt, kickoff, ex, "matchup_claim")
    dep = _screen([r for r in depth if r.get("team") in teams],
                  as_of_dt, kickoff, ex, "depth")

    def index(rows: List[Dict], kind: str) -> Dict[str, List[Dict]]:
        idx: Dict[str, List[Dict]] = {}
        for r in rows:
            pid, err = _resolve_identity(r, roster_by_team)
            if err:
                out["identity_errors"].append({"kind": kind, "reason": err, "game_id": gid,
                                               "row": _brief(r)})
                continue
            if r.get("team") and pid in team_of and r["team"] not in team_of[pid]:
                out["identity_errors"].append({"kind": kind, "reason": "team_mismatch",
                                               "game_id": gid, "row": _brief(r)})
                continue
            idx.setdefault(pid, []).append(r)
        return idx

    current = index(avail, "availability")
    prior_idx = index(prior, "prior_availability")
    rep_pub: Dict[Tuple[str, str], List[Dict]] = {}
    for r in reports:
        rep_pub.setdefault((r.get("team"), gid), []).append(r)

    game_summary: Dict[str, Any] = {"kickoff_utc": _iso(kickoff), "teams": {},
                                    "schedule_source": {k: g.get(k) for k in (
                                        "source_id", "source_url", "published_at", "fetched_at")}}
    records = out["records"]
    for team in teams:
        opp = teams[1] if team == teams[0] else teams[0]
        tdepth = [d for d in dep if d.get("team") == team]

        def av(pid: str) -> Dict[str, Any]:
            return _availability(pid, team, gid, current, rep_pub, prior_idx)

        qb = _qb_record(team, gid, as_of_dt, tdepth, claims, av, pos_of, name_of,
                        roster_by_team)
        ol = _ol_record(team, gid, as_of_dt, tdepth, av, pos_of, name_of)
        records.extend([qb, ol])
        dgroups = _def_records(opp, team, gid, as_of_dt, roster_by_team, tdepth_all=dep,
                               av=lambda pid: _availability(pid, opp, gid, current, rep_pub,
                                                            prior_idx),
                               pos_of=pos_of, name_of=name_of)
        records.extend(dgroups)
        game_summary["teams"][team] = {
            "opponent": opp, "qb": qb["value"], "ol": ol["value"],
            "opp_defense": {r["factor_id"].split(":")[0]: r["value"] for r in dgroups},
            "report_published": bool(rep_pub.get((team, gid)))}
    out["games"][gid] = game_summary
    _links(gid, out, offensive_players, mclaims, pos_of, teams)


def _qb_record(team, gid, as_of_dt, tdepth, claims, av, pos_of, name_of, roster_by_team):
    qbs = sorted([d for d in tdepth if str(d.get("slot", "")).upper() == "QB"],
                 key=lambda d: (int(d.get("rank", 99)), str(d.get("player_id"))))
    team_claims = [c for c in claims if c.get("team") == team]
    claim_pids = {}
    for c in team_claims:
        claim_pids.setdefault(c.get("claim"), set()).add(str(c.get("player_id")))
    expected = claim_pids.get("expected_starter", set())
    ruled_out = claim_pids.get("will_not_start", set())
    candidates = []
    for d in qbs:
        pid = str(d["player_id"])
        a = av(pid)
        candidates.append({"player_id": pid, "name": name_of.get(pid), "depth_rank": d.get("rank"),
                           "availability": a["state"], "returning": a["returning"],
                           "depth_source": d.get("source_id"),
                           "depth_cutoff_ok": bool(d.get("_cutoff_ok"))})
    starter, state, reason = None, "unknown", None
    if len(expected) > 1 or (expected & ruled_out):
        state, reason = "conflict_unresolved", "attributable claims name different starters"
    elif not candidates and not expected:
        state, reason = "unknown", "no documented QB depth and no attributable starter claim"
    else:
        qb1 = candidates[0] if candidates else None
        if expected:
            starter = next(iter(expected))
            state = "expected_starter_attributed"
        elif qb1 and qb1["player_id"] in ruled_out:
            state = "backup_expected"
        elif qb1 and qb1["availability"] in ("documented_out",):
            state = "backup_expected"
        elif qb1 and qb1["availability"] in ("documented_doubtful", "documented_questionable",
                                             "conflict_unresolved"):
            state = "uncertain_starter"
            reason = f"depth QB1 is {qb1['availability']}"
        elif qb1 and qb1["returning"] == "returning_unverified":
            state = "uncertain_starter"
            reason = "depth QB1 was absent last game and no current report documents his status"
        elif qb1:
            starter = qb1["player_id"]
            state = ("expected_starter_documented" if qb1["availability"] == "listed_no_designation"
                     else "expected_starter_availability_unverified")
        if state == "backup_expected":
            nxt = next((c for c in candidates[1:]
                        if c["availability"] not in ("documented_out", "documented_doubtful")
                        and c["player_id"] not in ruled_out), None)
            starter = nxt["player_id"] if nxt else None
            reason = ("depth QB1 documented out / ruled out; replacement is next QB on documented "
                      "depth" if nxt else "QB1 out and no available documented replacement")
    verified = state in ("expected_starter_documented", "expected_starter_attributed") and all(
        c["depth_cutoff_ok"] for c in candidates[:1])
    val = {"state": state, "expected_starter": starter,
           "expected_starter_name": name_of.get(starter) if starter else None,
           "candidates": candidates,
           "claims": [{"player_id": c.get("player_id"), "claim": c.get("claim"),
                       "attribution": c.get("attribution"), **_src(c)} for c in team_claims]}
    kind = "observed" if state.startswith("expected_starter_documented") or \
        state == "expected_starter_attributed" else ("projected" if starter else "unavailable")
    return _record(
        factor_id=f"qb_starter:{team}:{gid}", category="personnel_qb", entity_id=team,
        entity_type="team", game_id=gid, as_of=_iso(as_of_dt), observation=state, value=val,
        unit="categorical", verified=verified,
        cutoff_ok=all(c["depth_cutoff_ok"] for c in candidates) and all(
            c.get("_cutoff_ok") for c in team_claims),
        measurement_kind=kind,
        status="context_only" if state != "unknown" else "unavailable_unverified",
        feature_name="qb_starter_state", sources=[_src(c) for c in team_claims],
        reason_not_applied=("no validated pregame-as-of starter effect beyond the existing "
                            "qb_continuity/backup_qb_adj consumers; adding one would double count"),
        overlaps_existing=OVERLAPS["qb_starter"],
        rationale=reason or f"QB state {state} from documented depth/report/claims",
        uncertainty=("depth charts are team-published and can lag practice reports; a "
                     "Questionable/Doubtful QB1 can still start"))


def _ol_record(team, gid, as_of_dt, tdepth, av, pos_of, name_of):
    slots: Dict[str, List[Dict]] = {}
    for d in tdepth:
        s = str(d.get("slot", "")).upper()
        if s in OL_POS:
            slots.setdefault(s, []).append(d)
    rows, counts = [], {"documented_out": 0, "documented_doubtful": 0,
                        "documented_questionable": 0, "listed_no_designation": 0,
                        "unknown": 0, "conflict_unresolved": 0}
    seen = set()
    for s in sorted(slots):
        ds = sorted(slots[s], key=lambda d: (int(d.get("rank", 99)), str(d.get("player_id"))))
        starter = ds[0]
        pid = str(starter["player_id"])
        if pid in seen:          # same player at two slots: count once
            continue
        seen.add(pid)
        a = av(pid)
        st = a["state"]
        key = st if st in counts else ("unknown" if st in ("not_listed_unverified",
                                                           "unknown_no_report") else st)
        counts[key] = counts.get(key, 0) + 1
        repl = None
        if _is_absent(a) or st in ("documented_questionable", "conflict_unresolved"):
            nxt = next((d for d in ds[1:] if str(d["player_id"]) not in seen), None)
            repl = ({"player_id": str(nxt["player_id"]), "name": name_of.get(str(nxt["player_id"])),
                     "basis": "documented_depth_next", "availability": av(str(nxt["player_id"]))["state"]}
                    if nxt else {"player_id": None, "basis": "missing_depth_replacement"})
        rows.append({"slot": s, "player_id": pid, "name": name_of.get(pid),
                     "position_documented": pos_of.get(pid), "availability": st,
                     "returning": a["returning"], "practice": a["practice"],
                     "replacement": repl, "sources": a["sources"]})
    n_start = len(rows)
    absent = counts["documented_out"] + counts["documented_doubtful"]
    val = {"starters_documented": n_start, "counts": counts,
           "documented_absent": absent if n_start else None,
           "unknown_starters": counts["unknown"] if n_start else None, "starters": rows,
           "returns": [r for r in rows if r["returning"]]}
    return _record(
        factor_id=f"ol_availability:{team}:{gid}", category="personnel_ol", entity_id=team,
        entity_type="team", game_id=gid, as_of=_iso(as_of_dt), observation=(
            f"{absent} documented OL starter absences; {counts['unknown']} unknown"
            if n_start else "no documented OL depth"),
        value=val, unit="players",
        verified=bool(n_start) and counts["unknown"] == 0 and counts["conflict_unresolved"] == 0,
        cutoff_ok=bool(n_start) and all(bool(d.get("_cutoff_ok")) for d in tdepth),
        measurement_kind="observed" if n_start else "unavailable",
        status="context_only" if n_start else "unavailable_unverified",
        feature_name="ol_starter_absences",
        reason_not_applied=("existing oline_outs already feeds the ranker; no pregame-clocked "
                            "historical validation of an OL-absence effect exists here"),
        overlaps_existing=OVERLAPS["ol_availability"],
        rationale="documented depth starters crossed with the target-game injury report",
        uncertainty=("replacement is the next documented depth player, not a confirmed lineup; "
                     "unlisted starters are unknown, not healthy"))


def _def_records(opp, offense, gid, as_of_dt, roster_by_team, tdepth_all, av, pos_of, name_of):
    """Documented absences/returns per defensive group of the opponent.

    Only players who appear in an availability/prior row can be classified;
    the rest of the roster is not asserted healthy."""
    groups: Dict[str, Dict[str, Any]] = {g: {"absent": [], "questionable": [], "returning": [],
                                             "conflict": [], "ambiguous_position": []}
                                         for g in DEF_GROUP_NAMES}
    listed_any = False
    report_published = False
    for p in sorted(roster_by_team.get(opp, []), key=lambda x: str(x.get("player_id"))):
        pid = str(p["player_id"])
        pos = pos_of.get(pid, "")
        spec = DEF_GROUPS.get(pos)
        if spec is None:
            continue
        a = av(pid)
        report_published = report_published or bool(a["report_sources"])
        if a["state"] in ("unknown_no_report", "not_listed_unverified") and not a["returning"]:
            continue
        # only a target-game row makes the group "known"; a prior-week row
        # alone (return detection) must not turn unknown into zero
        listed_any = listed_any or bool(a["sources"])
        entry = {"player_id": pid, "name": name_of.get(pid), "position_documented": pos,
                 "availability": a["state"], "returning": a["returning"], "sources": a["sources"]}
        for grp in spec["groups"]:
            if _is_absent(a):
                groups[grp]["absent"].append(entry)
            elif a["state"] == "documented_questionable":
                groups[grp]["questionable"].append(entry)
            elif a["state"] == "conflict_unresolved":
                groups[grp]["conflict"].append(entry)
            if a["returning"]:
                groups[grp]["returning"].append(entry)
            if spec["ambiguous"]:
                groups[grp]["ambiguous_position"].append(pid)
    recs = []
    for grp in DEF_GROUP_NAMES:
        v = groups[grp]
        known = report_published or listed_any
        val = {"documented_absent": len(v["absent"]) if known else None,
               "questionable": len(v["questionable"]) if known else None,
               "absent": v["absent"], "questionable_players": v["questionable"],
               "returning": v["returning"], "conflict": v["conflict"],
               "ambiguous_position_ids": v["ambiguous_position"],
               "report_published": report_published,
               "starter_depth_known": False}
        recs.append(_record(
            factor_id=f"def_{grp}:{opp}:vs:{offense}:{gid}", category="personnel_defense",
            entity_id=opp, entity_type="team_defense", game_id=gid, as_of=_iso(as_of_dt),
            observation=(f"{len(v['absent'])} documented {grp} absences" if known
                         else "no published report for this game"),
            value=val, unit="players", verified=report_published and not v["conflict"],
            cutoff_ok=report_published,
            measurement_kind="observed" if known else "unavailable",
            status="context_only" if known else "unavailable_unverified",
            feature_name=f"opp_{grp}_documented_absences",
            reason_not_applied=("a defensive absence is not a uniform boost to every offensive "
                                "prop; existing consumers already count defensive outs"),
            overlaps_existing=OVERLAPS[f"def_{grp}"],
            rationale=("documented report statuses for the opponent's players at "
                       f"positions mapped to {grp}; starter vs reserve not known without depth"),
            uncertainty=("position groups are mapped from documented positions; LB/OLB/DL are "
                         "ambiguous and flagged; unlisted players are not asserted healthy")))
    return recs


def _links(gid, out, offensive_players, mclaims, pos_of, teams):
    by_factor = {r["factor_id"]: r for r in out["records"] if r["game_id"] == gid}
    for op in sorted(offensive_players or [], key=lambda x: str(x.get("player_id"))):
        if op.get("game_id") != gid:
            continue
        pid = str(op.get("player_id") or "")
        team = op.get("team")
        if not pid or team not in teams:
            out["identity_errors"].append({"kind": "offensive_player", "reason":
                                           "missing_id_or_team_not_in_game", "game_id": gid,
                                           "row": _brief(op)})
            continue
        opp = teams[1] if team == teams[0] else teams[0]
        pos = str(op.get("position") or pos_of.get(pid) or "").upper()
        if pos not in ROLE_DEF_RELEVANCE:
            out["relevance_links"].append({"game_id": gid, "player_id": pid, "position": pos,
                                           "applicability": "not_applicable_role",
                                           "factor_id": None})
            continue
        for f in ROLE_OWN_RELEVANCE[pos]:
            fid = f"{f}:{team}:{gid}"
            out["relevance_links"].append({
                "game_id": gid, "player_id": pid, "position": pos, "factor_id": fid,
                "relation": "own_team_personnel", "direction": "not_estimated",
                "factor_status": by_factor.get(fid, {}).get("status"),
                "basis": "documented position -> role relevance map"})
        for grp in DEF_GROUP_NAMES:
            fid = f"def_{grp}:{opp}:vs:{team}:{gid}"
            applicable = grp in ROLE_DEF_RELEVANCE[pos]
            out["relevance_links"].append({
                "game_id": gid, "player_id": pid, "position": pos, "factor_id": fid,
                "relation": "opposing_defense_personnel",
                "applicability": "applicable" if applicable else "not_applicable_role",
                "direction": "not_estimated",
                "factor_status": by_factor.get(fid, {}).get("status"),
                "basis": "documented position -> role relevance map"})
        # alignment matchup: documented only with a verified attributable claim
        align = op.get("expected_alignment")
        verified = [c for c in mclaims if str(c.get("offense_player_id")) == pid
                    and c.get("verified") is True and c.get("attribution") and c.get("source_url")
                    and c.get("_cutoff_ok")]
        unverified = [c for c in mclaims if str(c.get("offense_player_id")) == pid
                      and c not in verified]
        out["relevance_links"].append({
            "game_id": gid, "player_id": pid, "position": pos,
            "factor_id": f"alignment_matchup:{pid}:{gid}", "relation": "alignment_matchup",
            "expected_alignment": align,
            "expected_alignment_source": "opportunity_specialist" if align else None,
            "matchup_kind": "documented" if verified else (
                "inferred_unverified" if align else "unknown"),
            "documented_claims": [{"defense_player_id": c.get("defense_player_id"),
                                   "kind": c.get("kind"), "attribution": c.get("attribution"),
                                   **_src(c)} for c in verified],
            "unverified_claims_withheld": len(unverified),
            "shadow_assertion": bool([c for c in verified if c.get("kind") == "shadow"]),
            "direction": "not_estimated"})
