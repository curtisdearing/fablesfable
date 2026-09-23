"""Pipeline -> DB -> card wiring for factor evidence (execution receipts, shadow, context).

What this module records, and what each record means on a card:

* **Per-pick stage stamps** (``leans.stage_json``) are written by the run that made the pick.
  Each primary stage that can change the projected mean gets one of these states for this row:
  ``applied`` (multiplier != 1, the value as executed), ``no_change`` (the stage evaluated
  this row and left it alone), ``not_evaluated`` (the stage did not run, or its input was
  missing for this row; missing is never neutral), or ``not_applicable`` (the market is
  outside the stage's scope).
  A run-level flag alone is not enough: e.g. ``apply_backup_qb_adjustment`` skips rows whose
  ``qb_continuity`` is missing, so that row is ``not_evaluated`` even though the stage ran.
* **Run receipt** (``run_receipts``): what executed this run: stages, the ordering model and
  the features it actually read, the shadow component and the context file.
* **Shadow** (``leans.shadow_json``): ``role_opportunity`` expected volume, UNCONSERVED (there
  is no confirmed pregame active list). Never feeds the mean, SD, side, selection or order.
* **Context** (``factor_context``): sourced, game-scoped records loaded from
  ``data/factor_context/<season>-w<week>.json``, only for games in this run and only when
  published/captured at or before the run clock. A game without a file entry gets an explicit
  "not collected" record.

Cards rebuild labels from these persisted rows through ``factor_evidence``; nothing here
reads the feature schema or configuration to decide that something was used.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from typing import Dict, Iterable, List, Optional

from . import candidates as _cand
from . import factor_evidence as fe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTEXT_DIR = os.path.join(ROOT, "data", "factor_context")

#: stage -> (row column, markets it can touch or None for all).  The market scopes are the
#: executing functions' own constants: a disclosure scope that differs from the consumer's
#: (as it did before 2026-09-23: a QB-only set on the backup-QB stage, which actually
#: touches receiving_yards/receptions/passing_yards) mislabels every row in the gap.
STAGES = {
    "realloc_volume": ("realloc_mult", None),
    "realloc_efficiency": ("realloc_eff_mult", None),
    "backup_qb": ("backup_qb_adj", frozenset(_cand.BACKUP_QB_MARKETS)),
    "absence_qb": ("absence_qb_mult", frozenset(_cand.ABSENCE_QB_MARKETS)),
}
#: ordering-model inputs worth showing by value (all populated ones are listed in the receipt)
ORDERING_SHOWN = ("qb_continuity", "oline_outs", "temp", "wind", "player_depth_rank")

SHADOW_QUANTITY = {  # market -> (share quantity, expected-volume key)
    "receiving_yards": [("target_share", "expected_targets")],
    "receptions": [("target_share", "expected_targets")],
    "rushing_yards": [("carry_share", "expected_carries")],
    "rush_attempts": [("carry_share", "expected_carries")],
    "passing_yards": [("pass_share", "expected_pass_attempts")],
    "pass_attempts": [("pass_share", "expected_pass_attempts")],
    "completions": [("pass_share", "expected_pass_attempts")],
    "anytime_td": [("target_share", "expected_targets"), ("carry_share", "expected_carries")],
}
_VOLUME_WORD = {"expected_targets": "targets", "expected_carries": "carries",
                "expected_pass_attempts": "pass attempts"}


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _position(row: Dict) -> Optional[str]:
    """Candidates carry the position as ``pos`` (``projection.project``)."""
    for k in ("pos", "position", "role"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return None


# --------------------------------------------------------------------------- #
# run time: stamps, receipt, shadow
# --------------------------------------------------------------------------- #

#: stages whose "no change" depends on every teammate's availability being established
TEAM_AVAILABILITY_STAGES = ("realloc_volume", "realloc_efficiency", "absence_qb")


def row_stage_stamps(row: Dict, ran: Dict[str, bool], reasons: Dict[str, str],
                     unknown_teammates: int = 0) -> Dict:
    """Per-row state of every primary stage.  ``ran[stage]`` is whether the stage was
    evaluated in this run; ``reasons[stage]`` explains a run-level non-evaluation.
    ``unknown_teammates``: teammates on this board whose availability this run could not
    establish.  A run-level "report received" does not make those players' absence
    evaluated, so a teammate-absence stage that left this row unchanged is
    ``not_evaluated`` for it, not ``no_change``."""
    market = row.get("market")
    out = {}
    for stage, (col, scope) in STAGES.items():
        v = _num(row.get(col))
        if scope is not None and market not in scope:
            out[stage] = {"state": "not_applicable", "value": None, "reason": None}
        elif not ran.get(stage):
            out[stage] = {"state": "not_evaluated", "value": None,
                          "reason": reasons.get(stage) or "stage did not run in this run"}
        elif (stage in TEAM_AVAILABILITY_STAGES and unknown_teammates
              and (v is None or v == 1.0)):
            out[stage] = {"state": "not_evaluated", "value": None,
                          "reason": f"availability not established for {unknown_teammates} "
                                    f"teammate(s) on this board; a teammate absence could not "
                                    f"be ruled in or out"}
        elif stage == "backup_qb" and _num(row.get("qb_continuity")) is None:
            out[stage] = {"state": "not_evaluated", "value": None,
                          "reason": "qb_continuity missing for this row (missing, not zero): "
                                    "the 2024+ schedule carries no QB ids, and the x0.92 rule "
                                    "is not validated on pregame starter data"}
        elif v is not None and v != 1.0:
            out[stage] = {"state": "applied", "value": v, "reason": None}
        else:
            out[stage] = {"state": "no_change", "value": None, "reason": None}
    return out


_AVAIL_KEYS = ("status", "status_raw", "availability_state", "eligibility", "evidence_kind",
               "report_state", "source", "timestamp", "matched_by", "designation_date",
               "designation_predates_previous_game")


def availability_stamp(st: Optional[Dict]) -> Dict:
    """The per-player availability this run resolved, as persisted with the pick.  A player
    the resolver never returned is UNKNOWN/degraded -- never healthy by omission."""
    if not st:
        return {"status": "UNKNOWN", "availability_state": "not_resolved",
                "eligibility": "degraded", "evidence_kind": "unknown:not_resolved"}
    return {k: st.get(k) for k in _AVAIL_KEYS}


def build_stamps(cands, ran: Dict[str, bool], reasons: Dict[str, str],
                 ordering_features: Iterable[str] = (),
                 availability: Optional[Dict[str, Dict]] = None,
                 qb_context: Optional[Dict[str, Dict]] = None) -> Dict[tuple, Dict]:
    """(player_id, market) -> persisted stamp dict for every candidate row.

    ``availability``: the run's ``resolve_statuses`` map (None = not a live run, nothing
    recorded).  ``qb_context``: {team: starting-QB context record} (see
    ``qb_context_records``)."""
    shown = [f for f in ORDERING_SHOWN if f in set(ordering_features)]
    out = {}
    if cands is None or len(cands) == 0:
        return out
    rows = cands.to_dict("records")
    degraded: Dict[str, set] = {}
    if availability is not None:
        for row in rows:
            if availability_stamp(availability.get(row.get("player_id")))["eligibility"] == "degraded":
                degraded.setdefault(row.get("team"), set()).add(row.get("player_id"))
    for row in rows:
        comps = row.get("components") if isinstance(row.get("components"), dict) else {}
        pid, team = row.get("player_id"), row.get("team")
        others = len(degraded.get(team, set()) - {pid})
        stamp = {
            "team": team, "position": _position(row),
            "stages": row_stage_stamps(row, ran, reasons, unknown_teammates=others),
            "margin_source": row.get("margin_source"),
            "dispersion_role": row.get("dispersion_role"),
            "incumbent_volume": _num(comps.get("volume")),
            "ordering": {f: _num(row.get(f)) for f in shown},
        }
        if availability is not None:
            stamp["availability"] = availability_stamp(availability.get(pid))
        if qb_context is not None:
            stamp["qb_context"] = qb_context.get(team)
        out[(pid, row.get("market"))] = stamp
    return out


def qb_context_records(teams: Iterable[str], *, doc: Optional[Dict], pbp, roster_rows,
                       season: int, week: int, as_of, kickoffs: Dict[str, str]) -> Dict[str, Dict]:
    """{team: compact starting-QB context} for this run: sourced starter claims from the
    context document THIS run used (explicitly confirmed, clocked, captured by the run before
    its decision clock, linked to a unique roster QB), compared with a previous-game proxy.
    Context only: the numeric backup-QB rule stays blocked (``qb_readiness.NUMERIC_BLOCK``)."""
    import pandas as pd
    from . import qb_readiness as qr
    roster = pd.DataFrame([{"player_id": r.get("player_id"), "full_name": r.get("name"),
                            "team": r.get("team"), "position": r.get("position")}
                           for r in (roster_rows or [])],
                          columns=["player_id", "full_name", "team", "position"])
    names = dict(zip(roster["player_id"], roster["full_name"]))
    positions = {p: pos for p, pos in zip(roster["player_id"], roster["position"]) if pos}
    prior = {}
    if pbp is not None and len(pbp):
        prior = qr.prior_realized_starters(pbp, season, week, positions=positions or None)
    claims = qr.starter_claims_from_factor_context(doc or {})
    out = {}
    for team in sorted(set(t for t in teams if t)):
        linked = [qr.link_claim_identity(c, roster) for c in claims if c.get("team") == team]
        rec = qr.resolve_team(team, prior.get(team), linked, as_of,
                              kickoffs.get(team))
        p = rec.get("prior") or {}
        src = next((c for c in rec["usable_claims"] if c["qb_id"] == rec["qb_id"]), {}) \
            if rec.get("qb_id") else {}
        out[team] = {
            "state": rec["state"], "qb_id": rec.get("qb_id"),
            "qb_name": names.get(rec.get("qb_id")) or src.get("claim_value"),
            "source": rec.get("source"), "source_tier": src.get("source_tier"),
            "published_at": rec.get("published_at"), "fetched_at": src.get("fetched_at"),
            "prior": ({"qb_id": p.get("qb_id"), "name": names.get(p.get("qb_id")),
                       "game_id": p.get("game_id"), "basis": p.get("basis")} if p else None),
            "rejected": [{"rejected": c.get("rejected"), "source": c.get("source"),
                          "published_before_as_of": c.get("published_before_as_of")}
                         for c in rec["rejected_claims"]],
            "numeric_blocked": qr.NUMERIC_BLOCK["backup_qb_adj"],
            "pbp_available": bool(pbp is not None and len(pbp)),
        }
    return out


def shadow_opportunity(pw, cands, *, season: int, week: int, as_of: dt.datetime,
                       kickoffs: Dict[str, dt.datetime]) -> Dict:
    """role_opportunity in SHADOW for the candidate players.  Returns
    ``{"status", "component", "hyper_fit_seasons", "players": {player_id: {...}}}``.
    A failure is reported, never raised: the primary run must not depend on the shadow."""
    from . import role_opportunity as ro
    import pandas as pd
    res = {"component": ro.MODEL_VERSION, "consumption": "shadow", "conserved": False,
           "players": {}, "hyper_fit_seasons": None}
    try:
        if cands is None or len(cands) == 0:
            res["status"] = "no candidates"
            return res
        c = cands.drop_duplicates("player_id")
        t = pd.DataFrame({"player_id": c["player_id"].values, "team": c["team"].values,
                          "position": [_position(r) for r in c.to_dict("records")],
                          "game_id": c["game_id"].values})
        t = t[t["position"].isin(["QB", "RB", "WR", "TE"])]
        t["game_start"] = t["game_id"].map(lambda g: kickoffs.get(g))
        t = t[t["game_start"].map(lambda k: k is not None and k > as_of)]
        if t.empty:
            res["status"] = "no target rows before kickoff"
            return res
        t["game_start"] = t["game_start"].map(lambda k: k.isoformat())
        games = ro.player_games_from_player_week(pw)
        hyper = ro.fit_hyperparameters(games, before_season=season)
        out = ro.forecast_opportunity(games, t, season=season, week=week, as_of=as_of,
                                      hyper=hyper, consumption="shadow")
        res["hyper_fit_seasons"] = hyper.get("fit_seasons")
        for p in out["players"]:
            share = {}
            for q in ("target_share", "carry_share", "pass_share"):
                r = p.get(q)
                if r:
                    share[q] = {k: r.get(k) for k in ("prior_estimate", "prior_source", "prior_games",
                                                      "current_estimate", "current_games",
                                                      "current_denominator", "k", "w_current",
                                                      "posterior", "regime")}
            res["players"][p["player_id"]] = {
                "regime": p.get("regime"), "prior_season_team": p.get("prior_season_team"),
                "expected": {k: _num(p.get(f"{k}_unconserved")) for k in _VOLUME_WORD},
                "prior_only": {k: _num(p.get(f"{k}_prior_only")) for k in _VOLUME_WORD},
                "share": share}
        res["status"] = "ok"
    except Exception as exc:  # the shadow must never block the primary
        import re
        msg = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s'\"]+", "<path>", str(exc))  # public receipt: no paths
        res["status"] = f"error: {type(exc).__name__}: {msg}"[:300]
    return res


def context_path(season: int, week: int) -> str:
    return os.path.join(CONTEXT_DIR, f"{season}-w{week:02d}.json")


def load_context(season: int, week: int, game_ids: Iterable[str], as_of,
                 path: Optional[str] = None, doc: Optional[Dict] = None,
                 source_label: Optional[str] = None,
                 extra_records: Optional[List[Dict]] = None) -> Dict:
    """Sourced context for this run's games.  Only records dated at or before ``as_of``
    survive (``factor_evidence`` re-derives status and cutoff).  Games with no entry get an
    explicit unavailable record, never silence.  A context file that exists but cannot be read
    is a source failure (``status`` "error: ..."), reported per game, never a healthy empty."""
    as_of_t = fe._as_of(as_of)
    path = path or context_path(season, week)
    games = sorted(set(g for g in game_ids if g))
    given, status = doc, "not collected"
    doc = None
    recs: List[Dict] = []
    covered = set()
    try:
        if given is not None:
            doc = given
        elif os.path.isfile(path):
            with open(path) as f:
                doc = json.load(f)
            if doc.get("season") != season or doc.get("week") != week:
                doc, status = None, "not collected (file is for another season/week)"
        if doc:
            items = [i for i in doc.get("news", []) if i.get("game_id") in games]
            recs.extend(fe.assess_news(items, as_of_t))
            for raw in list(doc.get("records", [])) + list(extra_records or []):
                if raw.get("game_id") not in games:
                    continue
                recs.append(fe.normalize_record({**raw, "as_of": as_of_t}))
            covered = {r.get("game_id") for r in recs}
            status = "ok"
    except Exception as exc:  # the context layer must never block the primary, nor look healthy
        doc, recs, covered = None, [], set()
        status = f"error: {type(exc).__name__}"
    for g in games:
        if g in covered:
            continue
        if status.startswith("error"):
            recs.append(fe.normalize_record(dict(
                factor_id=f"context_source_failed:{g}", category="team_news", entity_type="game",
                entity_id=g, game_id=g, as_of=as_of_t, measurement_kind="unavailable",
                verified=False, observation="The context source for this game failed to load "
                                            "in this run",
                reason_not_applied=f"context source failed ({status}); missing, not 'nothing "
                                   f"to report'")))
            continue
        recs.append(fe.normalize_record(dict(
            factor_id=f"context_not_collected:{g}", category="team_news", entity_type="game",
            entity_id=g, game_id=g, as_of=as_of_t, measurement_kind="unavailable",
            verified=False, observation="No sourced team news, injury report or starter "
                                        "context was collected for this game in this run",
            reason_not_applied="not collected (missing, not 'nothing to report')")))
    label = (source_label or "context document supplied by the run") if given is not None \
        else os.path.relpath(path, ROOT)
    return {"path": label if doc else None, "status": status,
            "games_with_context": sorted(covered), "records": recs}


def issuing_run_id(base_run_id: str, clock: str, game_id: Optional[str] = None) -> str:
    """The id a run's picks, receipt and context are stored under.  A T-90 refresh is its own
    issuing run per game: the scheduled job refreshes several games in one process (one
    ``run_provenance`` id), and a Wednesday run in the same process shares that id too.
    Reusing it would let a T-90 pick borrow the Wednesday receipt (reloaded data shown as a new
    execution) and let one game's T-90 rewrite another's context."""
    return f"{base_run_id}:t90:{game_id}" if clock == "t90" else base_run_id


def lines_provenance(rows: Optional[List[Dict]], pulled: Iterable[str] = ()) -> Dict:
    """What the prices on this run's picks were, and when they were captured.  Reusing stored
    quotes keeps each quote's own capture clock; the run's clock is never substituted."""
    rows = rows or []
    ts = sorted(str(r.get("ts")) for r in rows if r.get("ts"))
    pulled = sorted(set(pulled or ()))
    if not rows:
        source = "none"
    elif pulled:
        source = "pulled this run" if all(r.get("game_id") in pulled for r in rows) \
            else "pulled this run + stored quotes"
    else:
        source = "stored quotes (captured before this run; no quote acquired)"
    return {"rows": len(rows), "source": source, "games_pulled_this_run": pulled,
            "quote_clock_min": ts[0] if ts else None, "quote_clock_max": ts[-1] if ts else None}


def record_issuing_run(conn, prov: Dict, *, season: int, week: int, clock: str, run_id: str,
                       as_of: str, game_ids: List[str], ran: Dict[str, bool],
                       reasons: Dict[str, str], ordering_component: Optional[str],
                       ordering_features: Iterable[str], shadow: Dict,
                       extra: Optional[Dict] = None, context_doc: Optional[Dict] = None,
                       context_label: Optional[str] = None,
                       extra_records: Optional[List[Dict]] = None) -> Dict:
    """Shared by the Wednesday run and the T-90 refresh: load the context this run could know,
    build its receipt and persist both under ``run_id``.  Returns the receipt.
    ``context_doc``: the document this run refreshed (else the committed file is read);
    ``extra_records``: run-built context records (e.g. observed snaps)."""
    ctx = load_context(season, week, game_ids, as_of, doc=context_doc, source_label=context_label,
                       extra_records=extra_records)
    receipt = run_receipt({**prov, "run_id": run_id}, as_of=as_of, ran=ran, reasons=reasons,
                          ordering_component=ordering_component,
                          ordering_features=ordering_features, shadow=shadow, context=ctx)
    receipt.update({"clock": clock, "game_ids": sorted(game_ids),
                    "base_run_id": prov.get("run_id")}, **(extra or {}))
    persist_run(conn, season, week, clock, receipt, ctx["records"])
    return receipt


def run_receipt(prov: Dict, *, as_of: str, ran: Dict[str, bool], reasons: Dict[str, str],
                ordering_component: Optional[str], ordering_features: Iterable[str],
                shadow: Dict, context: Dict) -> Dict:
    from . import football_forecast as ff
    return {
        "run_id": prov.get("run_id"), "code_sha": prov.get("code_sha"), "as_of": as_of,
        "component": ff.FORECAST_VERSION,
        "stages_executed": sorted([s for s, v in ran.items() if v] + ["dispersion", "game_script"]),
        "stages_not_executed": {s: reasons.get(s) for s, v in ran.items() if not v},
        "primary_margin_source": ff.PRIMARY_MARGIN_SOURCE,
        "ordering_component": ordering_component,
        "ordering_features_populated": list(ordering_features) if ordering_component else [],
        "shadow": {k: shadow.get(k) for k in ("component", "status", "consumption", "conserved",
                                              "hyper_fit_seasons")}
                  | {"players": len(shadow.get("players") or {})},
        "context": {"path": context.get("path"), "status": context.get("status"),
                    "games_with_context": context.get("games_with_context")},
    }


def persist_run(conn, season: int, week: int, clock: str, receipt: Dict,
                context_records: List[Dict], game_ids: Optional[List[str]] = None) -> None:
    """Store this run's receipt and the context it knew, keyed by run_id.  Only this run's own
    rows are replaced (same-run rerun is idempotent); other runs' context is never touched."""
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = receipt["run_id"]
    conn.execute("INSERT OR REPLACE INTO run_receipts (run_id, season, week, clock, as_of, "
                 "receipt_json, created_at) VALUES (?,?,?,?,?,?,?)",
                 (run_id, season, week, clock, receipt["as_of"],
                  json.dumps(receipt, default=str), now))
    conn.execute("DELETE FROM factor_context WHERE run_id=?", (run_id,))
    conn.executemany(
        "INSERT OR REPLACE INTO factor_context (run_id, season, week, game_id, factor_id, "
        "record_json, created_at) VALUES (?,?,?,?,?,?,?)",
        [(run_id, season, week, r.get("game_id"), r["factor_id"],
          json.dumps(r, default=str), now) for r in context_records])
    conn.commit()


def attach_to_leans(games: List[Dict], stamps: Dict[tuple, Dict], shadow: Dict) -> None:
    """Copy the row's stamps and shadow onto each published lean (persist_leans stores them)."""
    players = (shadow or {}).get("players") or {}
    for g in games:
        for l in g.get("leans", []):
            key = (l.get("player_id"), l.get("market"))
            if key in stamps:
                l["stage_stamps"] = stamps[key]
            if l.get("player_id") in players:
                l["shadow"] = {"component": shadow.get("component"), "conserved": False,
                               **players[l["player_id"]]}


# --------------------------------------------------------------------------- #
# card time: persisted rows -> factor records -> panel
# --------------------------------------------------------------------------- #

def _loads(x) -> Optional[Dict]:
    if not x or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return json.loads(x)
    except (TypeError, ValueError):
        return None


def stage_records(lean: Dict, stamps: Optional[Dict], receipt: Optional[Dict], as_of) -> List[Dict]:
    base = dict(entity_id=lean.get("player_id"), entity_type="player", game_id=lean.get("game_id"),
                as_of=as_of)
    if not stamps or not receipt:
        return [fe.normalize_record(dict(
            base, factor_id="model_stages", category="availability_adjustment",
            measurement_kind="unavailable", verified=False,
            observation="Per-pick stage stamps were not recorded for this pick",
            reason_not_applied="no execution receipt for this pick (missing, not 'no change')"))]
    st = stamps.get("stages") or {}
    row = {"player_id": lean.get("player_id"), "game_id": lean.get("game_id"),
           "team": stamps.get("team"), "margin_source": stamps.get("margin_source"),
           "dispersion_role": stamps.get("dispersion_role"), **(stamps.get("ordering") or {})}
    ran = [s for s, v in st.items() if v.get("state") in ("applied", "no_change")]
    for s, v in st.items():
        if v.get("state") == "applied":
            row[STAGES[s][0]] = v.get("value")
    ran += [s for s in ("dispersion", "game_script") if s in (receipt.get("stages_executed") or [])]
    per_row = {**receipt, "stages_executed": ran,
               "ordering_features_populated": [f for f in receipt.get("ordering_features_populated") or []
                                               if f in (stamps.get("ordering") or {})]}
    recs = fe.records_from_forecast_row(row, per_row, as_of)
    out = []
    for r in recs:
        s = st.get(r["factor_id"])
        if s and s.get("state") == "not_applicable":
            continue
        if s and s.get("state") == "not_evaluated":
            r = {**r, "reason_not_applied": s.get("reason")}
        out.append(r)
    return out


def shadow_records(lean: Dict, shadow: Optional[Dict], receipt: Optional[Dict],
                   stamps: Optional[Dict], as_of) -> List[Dict]:
    base = dict(entity_id=lean.get("player_id"), entity_type="player", game_id=lean.get("game_id"),
                as_of=as_of)
    out = [fe.normalize_record(dict(
        base, factor_id="participation:routes", category="role_usage",
        measurement_kind="unavailable", verified=False,
        observation="Routes run are not available",
        reason_not_applied="no free per-player route source (nflverse participation 'route' is "
                           "the targeted receiver's route only; no 2026 file); targets and snaps "
                           "are not a route proxy"))]
    comp = ((receipt or {}).get("shadow") or {}).get("component")
    if not shadow:
        why = ((receipt or {}).get("shadow") or {}).get("status") or "shadow not recorded"
        out.append(fe.normalize_record(dict(
            base, factor_id="shadow:role_opportunity", category="role_usage",
            measurement_kind="unavailable", verified=False,
            observation="Shadow role forecast not available for this pick",
            reason_not_applied=f"shadow output missing ({why})")))
        return out
    inc = (stamps or {}).get("incumbent_volume")
    for q, key in SHADOW_QUANTITY.get(lean.get("market"), []):
        s = (shadow.get("share") or {}).get(q)
        exp = (shadow.get("expected") or {}).get(key)
        if not s or exp is None:
            continue
        cur = s.get("current_estimate")
        prior_src = ("previous-season history" if s.get("prior_source") == "historical_prior_prev_season"
                     else "position average (no previous-season history)")
        obs = (f"Shadow expected {_VOLUME_WORD[key]} {exp:.1f}"
               + (f" (published projection volume {inc:.1f})" if inc is not None else "")
               + f"; {q.replace('_', ' ')}: {prior_src} {s['prior_estimate']:.3f}, this season "
               + ("none" if cur is None else f"{cur:.3f}") + f" over {s.get('current_games')} games, "
               f"weight on this season {s['w_current']:.2f} (k {s['k']:.3g}), regime {s.get('regime')}")
        out.append(fe.normalize_record(dict(
            base, factor_id=f"shadow:{q}", category="role_usage", component=comp or shadow.get("component"),
            model_version=shadow.get("component"), feature_name=key, consumed=False,
            consumed_shadow=True, verified=True, measurement_kind="projected", observation=obs,
            support_games=s.get("current_games"), support_scope="current_season",
            rationale=("Experimental learned weighting of this season vs the prior; unconserved "
                       "(no confirmed pregame active list). The prior is previous-season history or a "
                       "position average, not a preseason projection."),
            uncertainty="failed its frozen bias gate (G2); not validated; hindsight player selection "
                        "in the development evaluation",
            reason_not_applied="shadow only: failed the frozen bias gate, so it does not change the "
                               "published projection")))
    return out


_UNKNOWN_WHY = {
    "report_missing": "no injury report was received by this run",
    "team_not_in_report": "this team had no rows in the injury feed this run received (a team "
                          "with nobody listed cannot be told apart from a missing team)",
    "identity_ambiguous": "the injury feed rows could not be matched to this player uniquely",
    "identity_other_team_only": "the only injury feed row with this name is on another team",
    "not_resolved": "the availability resolver returned nothing for this player",
}
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"


def availability_record(lean: Dict, stamps: Optional[Dict], as_of) -> Dict:
    """This player's own availability as the issuing run resolved it (never inferred later)."""
    pid, gid = lean.get("player_id"), lean.get("game_id")
    base = dict(factor_id=f"availability:{pid}", category="availability", entity_type="player",
                entity_id=pid, game_id=gid, team=(stamps or {}).get("team"), as_of=as_of)
    a = (stamps or {}).get("availability")
    if not a:
        return fe.normalize_record(dict(
            base, measurement_kind="unavailable", verified=False,
            observation="Player availability was not recorded by the run that made this pick",
            reason_not_applied="missing, not 'healthy'; the card is not executable"))
    state, kind = a.get("availability_state"), a.get("evidence_kind") or ""
    raw = str(a.get("status_raw") or "").split("|")[0]
    if a.get("eligibility") == "degraded":
        why = _UNKNOWN_WHY.get(state) or (f"unrecognized feed status '{raw}'" if raw
                                          else f"availability state {state}")
        return fe.normalize_record(dict(
            base, measurement_kind="unavailable", verified=False, fetched_at=a.get("timestamp"),
            observation=f"Availability not established: {why}",
            reason_not_applied="unknown is neither confirmed healthy nor ruled out; the card "
                               "is not executable"))
    rec = dict(base, consumed=True, evaluated_neutral=True, verified=True,
               measurement_kind="observed", fetched_at=a.get("timestamp"),
               source_title="ESPN NFL injuries feed", source_url=ESPN_INJURIES_URL,
               rationale="Availability gate: OUT players are removed before ranking and their "
                         "usage is reallocated; this player was not OUT.")
    if kind == "not_listed_on_received_feed":
        obs = ("Not listed on the ESPN injuries feed this run received (a data feed, not the "
               "official NFL report); kept on the board. Not a medical clearance")
    elif kind == "feed_active_status":
        obs = ("Listed 'Active' on the ESPN injuries feed this run received (a feed status, not "
               "the official report); kept on the board. Not a medical clearance")
    elif kind == "confirmed_active_event_roster":
        obs = "Listed active on the ESPN game roster at the T-90 check"
        rec.update(source_title="ESPN game roster (T-90)", source_url=None)
    else:
        obs = (f"{raw or a.get('status')} on the ESPN injuries feed"
               + (f" (entry dated {a['designation_date']})" if a.get("designation_date") else "")
               + "; a feed status, not the official game-status report or practice "
                 "participation")
        if a.get("designation_predates_previous_game"):
            obs += ("; the entry predates this team's previous game, so it may be that game's "
                    "status, not this week's")
    rec["observation"] = obs
    return fe.normalize_record(rec)


def qb_record(lean: Dict, stamps: Optional[Dict], as_of) -> Dict:
    """Starting-QB context for the pick's team, as the issuing run established it."""
    from . import qb_readiness as qr
    team = (stamps or {}).get("team")
    base = dict(factor_id=f"qb_starter_readiness:{team}", category="qb_news", entity_type="team",
                entity_id=team, team=team, game_id=lean.get("game_id"), as_of=as_of)
    q = (stamps or {}).get("qb_context")
    block = ("context only: the numeric backup-QB rule is blocked (the 2024+ schedule carries "
             "no QB ids, and the x0.92 rule is not validated on pregame starter data)")
    if not q:
        return fe.normalize_record(dict(
            base, measurement_kind="unavailable", verified=False,
            observation="Starting-QB context was not recorded by the run that made this pick",
            reason_not_applied="missing (not recorded), not 'no change'"))
    p = q.get("prior") or {}
    proxy = ("first pass attempt by a roster QB" if p.get("basis") == qr.PRIOR_BASIS_ROSTER_QB
             else "first passer, position unverified")
    prior_txt = (f"previous game's QB by proxy ({proxy}, {p.get('game_id')}): "
                 f"{p.get('name') or p.get('qb_id')}") if p.get("qb_id") else         "no previous-game QB proxy available"
    late = [r for r in q.get("rejected") or [] if r.get("rejected") == qr.REJ_CAPTURED_LATE]
    late_txt = (f"; {len(late)} starter claim(s) published before this run's clock were "
                f"captured later and not used" if late else "")
    state = q.get("state")
    if state in (qr.SOURCED_SAME, qr.SOURCED_CHANGED, qr.NO_PRIOR):
        rel = {qr.SOURCED_SAME: "same player as", qr.SOURCED_CHANGED: "a different player from",
               qr.NO_PRIOR: "no comparison with"}[state]
        return fe.normalize_record(dict(
            base, measurement_kind="observed", verified=True, source_url=q.get("source"),
            published_at=q.get("published_at"), fetched_at=q.get("fetched_at"),
            observation=(f"Confirmed starter per team source: {q.get('qb_name') or q.get('qb_id')} "
                         f"({rel} the {prior_txt})" + late_txt),
            reason_not_applied=block))
    if state == qr.UNCONFIRMED:
        return fe.normalize_record(dict(
            base, measurement_kind="proxy", proxy_name=f"previous game's QB ({proxy})",
            verified=True, observation=("No confirmed starter announcement captured by this run; "
                                        + prior_txt + late_txt),
            reason_not_applied=block))
    why = "conflicting confirmed starter claims" if state == qr.CONFLICT else         "no starter claim and no previous-game QB found"
    return fe.normalize_record(dict(base, measurement_kind="unavailable", verified=False,
                                    observation=f"Starting QB not established: {why}" + late_txt,
                                    reason_not_applied=block))


def _opponent(game_id: Optional[str], team: Optional[str]) -> Optional[str]:
    parts = str(game_id or "").split("_")
    if len(parts) == 4 and team in parts[2:]:
        return parts[3] if team == parts[2] else parts[2]
    return None


def matchup_summary_records(lean: Dict, team: Optional[str], context: List[Dict],
                            as_of) -> List[Dict]:
    """Player-level context the card would otherwise never show: the opposing defense's and
    this team's offensive-line entries known to this run.  Context only; no matchup effect
    is modeled."""
    gid, opp = lean.get("game_id"), _opponent(lean.get("game_id"), team)
    out = []
    for cat, side, label in (("def_absence", opp, "Opposing defense"),
                             ("ol_injury", team, "Offensive line")):
        if not side:
            continue
        items = [r for r in context if r.get("category") == cat and r.get("entity_type") == "player"
                 and r.get("team") == side and r.get("game_id") == gid
                 and r.get("status") != "unavailable_unverified"]
        if not items:
            continue
        clocks = sorted(str(r.get("fetched_at")) for r in items if r.get("fetched_at"))
        obs = f"{label} ({side}) entries known to this run: " + "; ".join(
            sorted({str(r.get("observation") or "")[:90] for r in items})[:8])
        out.append(fe.normalize_record(dict(
            factor_id=f"{cat}_summary:{side}:{gid}", category=cat, entity_type="game",
            entity_id=gid, game_id=gid, team=side, as_of=as_of, measurement_kind="observed",
            verified=False, source_status="attributed_report", observation=obs[:400],
            fetched_at=clocks[-1] if clocks else None,
            source_title="ESPN injuries feed / team reports (per-entry sources on the player cards)",
            reason_not_applied="context only: no validated matchup effect; not an input to the "
                               "projection")))
    return out


def card_records(lean: Dict, receipt: Optional[Dict], context: List[Dict], as_of) -> List[Dict]:
    stamps = _loads(lean.get("stage_json"))
    shadow = _loads(lean.get("shadow_json"))
    team = (stamps or {}).get("team")
    recs = stage_records(lean, stamps, receipt, as_of)
    recs.append(availability_record(lean, stamps, as_of))
    recs.append(qb_record(lean, stamps, as_of))
    recs += shadow_records(lean, shadow, receipt, stamps, as_of)
    recs += fe.select_for_card(context, {"player_id": lean.get("player_id"),
                                         "game_id": lean.get("game_id"), "team": team})
    recs += matchup_summary_records(lean, team, context, as_of)
    return recs


def load_receipts(conn, season: int, week: int) -> Dict[str, Dict]:
    try:
        rows = conn.execute("SELECT run_id, receipt_json FROM run_receipts WHERE season=? AND week=?",
                            (season, week)).fetchall()
    except Exception:
        return {}
    return {r[0]: json.loads(r[1]) for r in rows}


def load_context_records(conn, season: int, week: int) -> Dict[tuple, List[Dict]]:
    """(issuing run_id, game_id) -> the context that run recorded."""
    try:
        rows = conn.execute("SELECT run_id, game_id, record_json FROM factor_context "
                            "WHERE season=? AND week=?", (season, week)).fetchall()
    except Exception:
        return {}
    out: Dict[tuple, List[Dict]] = {}
    for run, g, j in rows:
        out.setdefault((run, g), []).append(json.loads(j))
    return out


def _as_known_at(records: List[Dict], as_of) -> List[Dict]:
    """Defense in depth: a record whose published/observed/fetched clock is after the pick's
    decision time cannot be shown as known then, whatever its stored status says."""
    t = fe._as_of(as_of)
    out = []
    for r in records:
        late = [k for k in ("published_at", "observed_at", "fetched_at")
                if fe._ts(r.get(k)) and fe._ts(r.get(k)) > t]
        if late:
            r = {**r, "status": "unavailable_unverified", "verified": False, "cutoff_ok": False,
                 "numerical_effect": None,
                 "reason_not_applied": "captured or published after this pick's decision time"}
        out.append(r)
    return out


def card_panel(lean: Dict, receipts: Dict[str, Dict], context_by_game: Dict[str, List[Dict]]) -> Dict:
    """Factor panel for one persisted lean, built from what that lean's run recorded."""
    receipt = receipts.get(lean.get("run_id"))
    as_of = (receipt or {}).get("as_of") or lean.get("as_of")
    try:
        ctx = context_by_game.get((lean.get("run_id"), lean.get("game_id")))
        if ctx is None:     # never borrow another run's context
            ctx = [fe.normalize_record(dict(
                factor_id=f"context_not_recorded:{lean.get('game_id')}", category="team_news",
                entity_type="game", entity_id=lean.get("game_id"), game_id=lean.get("game_id"),
                as_of=as_of, measurement_kind="unavailable", verified=False,
                observation="No context was recorded by the run that made this pick",
                reason_not_applied="missing for this pick's run (not borrowed from a later run)"))]
        recs = card_records(lean, receipt, _as_known_at(ctx, as_of), as_of)
        panel = fe.build_panel(recs, as_of)
    except fe.UnsafeCopy as exc:
        return {"withheld": f"panel withheld: {exc}"}
    except fe.FactorRecordError as exc:
        return {"withheld": f"panel withheld: invalid factor record ({exc})"}
    panel["team"] = ((_loads(lean.get("stage_json")) or {}).get("team"))
    return panel
