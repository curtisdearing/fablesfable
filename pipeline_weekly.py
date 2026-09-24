#!/usr/bin/env python3
"""Weekly hands-off pipeline: ingest -> features -> availability -> projection
-> synthesis -> composite -> shortlist -> report -> dashboard (-> Discord).

Two clocks (PHASE1_HANDSOFF_DESIGN.md -- a single Wednesday run cannot be
both hands-off and correct):

  WED provisional   python3 pipeline_weekly.py --season 2025 --week 10 --mode live
  T-90 final        python3 pipeline_weekly.py --season 2025 --week 10 --clock t90 --game 2025_10_CLE_BAL
                    (re-pulls availability, VOIDS leans on OUT/inactive players,
                     re-ranks that game, regenerates report + dashboard)
  post-slate CLV    python3 pipeline_weekly.py --season 2025 --week 10 --resolve-clv

Guardrails wired through, not bolted on:
  * freshness gate: in live mode, stale/missing load-bearing feeds set
    publish=false -- the report renders with a NOT PUBLISHED banner, Discord
    gets (at most) a gate notice, and nothing pretends otherwise.
  * odds budget: the Odds API client hard-stops at the monthly ceiling;
    un-pulled games run no_market. No key / --live-odds absent -> all games
    no_market (synthetic reference lines only), tagged in the report.
  * numbers are deterministic; synthesis (RuleBasedMockLLM by default) runs
    AFTER ranking, on the ranked leans, for the context panel only.
  * idempotent: leans/lines/clv upsert on primary keys -- rerunning a clock
    for the same week overwrites itself, never duplicates.
  * historical mode (completed seasons on the parquet): live feeds are not
    applicable; the report and context panel say so explicitly.

Every injectable seam (feeds, fetchers, inputs) exists so tests run offline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from nflvalue import candidates as candmod
from nflvalue import factor_integration as fimod  # noqa: E402
from nflvalue import clv as clvmod
from nflvalue import config as cfgmod
from nflvalue import db as dbmod
from nflvalue import killcheck as kcmod
from nflvalue import report as rptmod
from nflvalue import prop_decision as pdmod
from nflvalue import shortlist as slmod
from nflvalue import synthesis as synmod
from nflvalue.dashboard import write_dashboard
from nflvalue.freshness import Feed, gate, parse_ts, stamp_now
from nflvalue.sources import availability as avmod
from nflvalue.sources import oddsapi_props as oapmod
from nflvalue.sources import sleeper as slpmod

#: A quote younger than this at T-90 is the close: the scheduled T-90 job
#: (scripts/auto_weekly.job_t90) re-snaps every due game that already has
#: entry lines minutes before run_t90 is called, so run_t90 must not spend a
#: second event-call on the same game. It pulls only when nothing this fresh
#: exists -- the case of a game the Wednesday run could not afford.
T90_LINE_FRESH_HOURS = 1.0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def kickoffs_for(slate: pd.DataFrame) -> Dict[str, str]:
    """{game_id: iso kickoff} from schedules' gameday+gametime (ET naive ->
    stored as-is; CLV only needs a consistent ordering vs snapshot ts)."""
    out = {}
    for g in slate.itertuples(index=False):
        t = f"{g.gameday}T{(g.gametime or '13:00')}:00Z"
        out[g.game_id] = t
    return out


def build_event_map(cfg: Dict, slate: pd.DataFrame,
                    list_events_fn: Optional[Callable] = None) -> Dict[str, str]:
    """{nflverse game_id -> odds api event id} by matching home/away display
    names to abbrs on the same slate. Unmatched games simply aren't pulled."""
    try:
        events = (list_events_fn or oapmod.list_events)(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] odds events listing failed ({exc}); continuing no_market")
        return {}
    by_pair = {}
    for ev in events or []:
        home = avmod.DISPLAY_TO_ABBR.get(ev.get("home_team", ""), "")
        away = avmod.DISPLAY_TO_ABBR.get(ev.get("away_team", ""), "")
        if home and away:
            by_pair[(home, away)] = ev.get("id")
    out = {}
    for g in slate.itertuples(index=False):
        eid = by_pair.get((g.home_team, g.away_team))
        if eid:
            out[g.game_id] = eid
    return out


def slate_kickoffs(slate: pd.DataFrame) -> Dict[str, dt.datetime]:
    """{game_id -> aware kickoff}, built the same way ``auto_weekly`` does it:
    ``gameday`` + ``gametime`` are Eastern clock time on the nflverse slate.
    Games with an unparseable time are omitted, which degrades that game to
    the plain rotation clock rather than mis-ordering the whole slate."""
    out: Dict[str, dt.datetime] = {}
    for g in slate.itertuples(index=False):
        try:
            out[g.game_id] = dt.datetime.strptime(
                f"{g.gameday} {getattr(g, 'gametime', None) or '13:00'}",
                "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo("America/New_York"))
        except (ValueError, TypeError):
            continue
    return out


def _game_teams(slate: pd.DataFrame) -> Dict[str, set]:
    """{game_id: {home, away}} -- confines a quote to its own game's teams."""
    return {g.game_id: {g.home_team, g.away_team} for g in slate.itertuples(index=False)}


def _players_frame(cands: pd.DataFrame) -> pd.DataFrame:
    return (cands[["player_id", "name", "team"]].drop_duplicates()
            .rename(columns={"name": "player_name"}))


def _team_kickoffs(slate: pd.DataFrame) -> Dict[str, str]:
    """{team: aware ISO kickoff} for the games on ``slate``."""
    ko = slate_kickoffs(slate)
    out: Dict[str, str] = {}
    for g in slate.itertuples(index=False):
        if g.game_id in ko:
            out[g.home_team] = out[g.away_team] = ko[g.game_id].isoformat()
    return out


def _prior_kickoffs(schedules: pd.DataFrame, season: int, week: int) -> Dict[str, str]:
    """{team: kickoff of its most recent game before ``week``} -- a feed designation dated
    before it belonged to that game, not this week's."""
    prev = schedules[(schedules["season"] == season) & (schedules["week"] < week)]
    if "game_type" in prev.columns:
        prev = prev[prev["game_type"] == "REG"]
    out: Dict[str, str] = {}
    for g in prev.sort_values("week").itertuples(index=False):
        k = slate_kickoffs(pd.DataFrame([g._asdict()])).get(g.game_id)
        if k is not None:
            out[g.home_team] = out[g.away_team] = k.isoformat()
    return out


_CURATED_RECORD_EXCLUDE = ("coverage:", "qb_depth:")
# Club-report items are live captures: a later run re-fetches them, never inherits them.
_LIVE_CLUB_STORIES = ("club_status:", "club_practice:")


def _run_context_doc(cfg: Dict, season: int, week: int, mode: str,
                     inject: Optional[Dict], roster: Optional[Dict]):
    """The factor-context document THIS run uses, fetched before its decision clock.

    Live runs that fetch their own feeds refresh the whole slate from free structured
    sources (``live_factor_context``), keeping the committed file's hand-curated
    team/league items verbatim (earlier live captures in the file are replaced by this
    capture, never merged as current).  Offline/injected runs and a failed refresh use the
    committed file, and the receipt says which.  Returns ``(doc, label, meta)``; ``doc``
    None means "read the committed file"."""
    inject = inject or {}
    if "factor_context_doc" in inject:
        return inject["factor_context_doc"], "context document injected by the caller", \
            {"refresh": "injected"}
    if mode != "live" or inject or not (cfg.get("factor_context") or {}).get("live_refresh", True):
        return None, None, {"refresh": "not attempted (offline, injected or disabled run); "
                                       "committed context file used"}
    path = fimod.context_path(season, week)
    curated, filed = None, None
    try:
        if os.path.isfile(path):
            with open(path) as f:
                filed = json.load(f)
            curated = {"season": filed.get("season"), "week": filed.get("week"),
                       "news": [i for i in filed.get("news", [])
                                if i.get("source_tier") == "team_official"
                                and not str(i.get("story_id", "")).startswith(_LIVE_CLUB_STORIES)],
                       "records": [r for r in filed.get("records", [])
                                   if not str(r.get("factor_id", "")).startswith(_CURATED_RECORD_EXCLUDE)]}
    except Exception as exc:  # noqa: BLE001 -- the refresh still runs without curated items
        print(f"[pipeline] committed context file unreadable ({type(exc).__name__}); "
              f"refreshing without curated items")
    # Club-site report URLs for this week, registered in the committed file (slugs cannot be
    # discovered).  Each is re-fetched by THIS run under its own capture clock.
    club_reports = {g: u for g, u in ((filed or {}).get("club_reports") or {}).items()
                    if isinstance(g, str) and isinstance(u, str)} if os.path.isfile(path) else {}
    id_map = [{"espn_id": r["espn_id"], "team": r.get("team"), "gsis_id": r["player_id"]}
              for r in (roster or {}).get("rows") or [] if r.get("espn_id")]
    try:
        from nflvalue.sources import live_factor_context as lfc
        doc = lfc.build_live_context(season, week, curated=curated, id_map=id_map,
                                     club_reports=club_reports)
    except Exception as exc:  # noqa: BLE001 -- degrade to the committed file, loudly
        print(f"[pipeline] live context refresh failed ({type(exc).__name__}: {exc}); "
              f"committed context file used")
        return None, None, {"refresh": f"failed ({type(exc).__name__}); committed file used"}
    doc.pop("request_log", None)
    meta = {"refresh": "ok", "captured_at": doc.get("captured_at"), "routes": doc.get("routes"),
            "sources_checked": len(doc.get("sources_checked") or []),
            "curated_games_kept": doc.get("curated_games_kept"), "id_map_rows": len(id_map),
            "club_reports": sorted(club_reports),
            "coverage_states": _coverage_counts(doc)}
    return doc, f"live refresh captured {doc.get('captured_at')} (curated team items kept)", meta


def _coverage_counts(doc: Dict) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in (doc.get("coverage") or {}).values():
        for cell in row.values():
            st = cell.get("state") if isinstance(cell, dict) else None
            if st:
                out[st] = out.get(st, 0) + 1
    return out


def _fetch_snaps(season: int, mode: str, inject: Optional[Dict]):
    """(frame, source, status) for nflverse snap counts, fetched BEFORE the decision clock."""
    from nflvalue.sources import participation_evidence as pe
    inject = inject or {}
    if "snap_counts" in inject:
        return inject["snap_counts"], inject.get("snap_counts_source") or {}, "injected"
    if mode != "live" or inject:
        return None, None, "not attempted (offline or injected run)"
    import io
    import urllib.request
    url = pe.SNAP_URL.format(season=season)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python-urllib/3"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body, lm = r.read(), r.headers.get("Last-Modified")
        frame = pd.read_parquet(io.BytesIO(body))
        from email.utils import parsedate_to_datetime
        source = {"url": url, "fetched_at": stamp_now(),
                  "last_modified": (parsedate_to_datetime(lm).astimezone(dt.timezone.utc)
                                    .strftime("%Y-%m-%dT%H:%M:%SZ") if lm else None)}
        return frame, source, "fetched"
    except Exception as exc:  # noqa: BLE001 -- context only; say so, never zero-fill
        return None, None, f"fetch failed ({type(exc).__name__}); snaps unavailable"


def _participation_records(season: int, week: int, cands: pd.DataFrame, schedules: pd.DataFrame,
                           roster: Optional[Dict], as_of: str, snaps):
    """Observed offensive snaps (nflverse/PFR) for this run's candidate players, as CONTEXT
    records.  ``snaps`` is ``_fetch_snaps``'s result.  Returns ``(records, receipt)``.  Never
    a forecast input; a player/week with no row is "no row", never zero."""
    from nflvalue.sources import participation_evidence as pe
    frame, source, fetch_status = snaps
    if frame is None:
        return [], {"status": fetch_status}
    try:
        frame = frame[pd.to_numeric(frame["week"], errors="coerce") < week]
        rows = (roster or {}).get("rows") or []
        players = pd.DataFrame([{"pfr_id": r["pfr_id"], "gsis_id": r["player_id"]}
                                for r in rows if r.get("pfr_id")], columns=["pfr_id", "gsis_id"])
        sched = schedules[(schedules["season"] == season) & (schedules["week"] < week)]
        games = {int(w): list(g["game_id"]) for w, g in sched.groupby("week")}
        loaded = pe.load_snap_counts(frame, season=season, target_week=week, players=players,
                                     source=source, schedule_games=games)
        recs: List[Dict] = []
        for r in cands[["player_id", "team", "game_id"]].drop_duplicates().itertuples(index=False):
            recs += pe.snap_records(loaded, player_id=r.player_id, team=r.team,
                                    game_id=r.game_id, as_of=as_of)
        return recs, {**loaded["receipt"], "status": f"ok ({fetch_status})", "players": len(recs)}
    except Exception as exc:  # noqa: BLE001
        return [], {"status": f"ingest failed ({type(exc).__name__}: {str(exc)[:160]})"}


def _committed_context(season: int, week: int) -> Optional[Dict]:
    try:
        with open(fimod.context_path(season, week)) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 -- load_context reports the file's own failure
        return None


def _apply_starter_gate(stamps: Dict[tuple, Dict], gate: Dict[tuple, Dict]) -> None:
    """Persist each QB-market row's starter eligibility on its stamp.  A row whose team has a
    different confirmed starter gets its persisted availability eligibility set to degraded
    (state ``not_confirmed_starter``) -- the hold the card builder already honours -- so it is
    never executable.  The resolver's own status fields are kept; no number changes."""
    for key, g in gate.items():
        st = stamps.get(key)
        if st is None:
            continue
        st["qb_eligibility"] = g
        if g["blocks_execution"]:
            a = dict(st.get("availability") or {})
            a["eligibility_before_starter_gate"] = a.get("eligibility")
            a.update({"eligibility": "degraded", "availability_state": "not_confirmed_starter",
                      "evidence_kind": "team_sourced_starter_claim", "starter_gate": g["reason"]})
            st["availability"] = a


def _starter_diagnostics(qb_ctx: Optional[Dict], gate: Dict[tuple, Dict], cands,
                         games: List[Dict]) -> Optional[Dict]:
    """Per team: the starter decision this run used, which QB rows it blocked, and -- when the
    confirmed starter has no published QB-market card -- the exact reason (never a forecast)."""
    if qb_ctx is None:
        return None
    shown = {(l.get("player_id"), l.get("market")) for g in games or [] for l in g.get("leans", [])}
    rows = cands.to_dict("records") if cands is not None and len(cands) else []
    out = {}
    for team, q in sorted(qb_ctx.items()):
        starter = candmod.confirmed_starter(q)
        confirmed = starter is not None
        mine = [r for r in rows if r.get("player_id") == starter
                and r.get("market") in candmod.STARTER_GATED_MARKETS] if starter else []
        carded = sorted(m for (p, m) in shown if p == starter and m in candmod.STARTER_GATED_MARKETS)
        if not confirmed:
            why = None
        elif not mine:
            why = "no candidate row for the confirmed starter in this run (no forecast is invented)"
        elif not carded:
            why = ("candidate rows exist but were not shortlisted (ranked below the per-game "
                   "top_n / max_per_player cut)")
        else:
            why = None
        out[team] = {"state": q.get("state"), "starter_qb_id": starter if confirmed else None,
                     "confirmed": confirmed, "starter_candidate_markets": sorted(r["market"] for r in mine),
                     "starter_cards": carded, "starter_no_card_reason": why,
                     "blocked_rows": sorted([p, m] for (p, m), g in gate.items()
                                            if g["team"] == team and g["blocks_execution"])}
    return out


def _availability_receipt(live: Dict, qb_ctx: Optional[Dict], ctx_meta: Dict,
                          snap_receipt: Dict) -> Dict:
    """Run-receipt fields: what this run established about availability, starting QBs,
    context refresh and participation -- per run, with counts, not a single 'evaluated'."""
    ts = (live or {}).get("ts") or {}
    return {
        "availability": {"report_state": (live or {}).get("report_state"),
                         "injuries_fetched_at": ts.get("injuries"),
                         "summary": (live or {}).get("availability_summary")},
        "qb_context": ({t: {"state": q["state"], "qb_id": q.get("qb_id"),
                            "prior_basis": (q.get("prior") or {}).get("basis"),
                            "rejected": [r["rejected"] for r in q.get("rejected") or []]}
                        for t, q in qb_ctx.items()} if qb_ctx is not None else None),
        "context_refresh": ctx_meta,
        "participation": {k: snap_receipt.get(k) for k in
                          ("status", "source", "weeks", "per_week", "n_rows", "identity",
                           "sha256", "players", "definition") if k in snap_receipt},
        "routes": "unavailable: no free per-player route source",
    }


def _qb_pbp(inputs: candmod.WeekInputs):
    key = tuple(sorted(int(s) for s in inputs.pw["season"].unique()))
    return _PBP_EXT.get(key)


def _apply_forecast_weather(adv, slate: pd.DataFrame) -> None:
    """Override the pack's (post-game, NaN-for-future) schedule weather with
    live Open-Meteo forecasts for this slate's outdoor games (evaluation
    catch: without this, the weather feature is dead all season)."""
    try:
        from build_ratings import ABBR
        from nflvalue.sources.weather import forecast_for_game
        # gameday/gametime are Eastern clock time: the forecast hour is the
        # real (UTC) kickoff, not that clock labelled +00:00.
        kickoffs = slate_kickoffs(slate)
        for g in slate.itertuples(index=False):
            wx = adv.weather.get(g.game_id, (None, None))
            if wx[0] is not None and not pd.isna(wx[0]):
                continue  # dome-neutralized or already known
            if g.game_id not in kickoffs:
                continue  # unparseable kickoff: schedule values kept
            commence = kickoffs[g.game_id].astimezone(dt.timezone.utc).isoformat()
            fc = (forecast_for_game(g.home_team, commence)
                  or forecast_for_game(ABBR.get(g.home_team, g.home_team), commence))
            if not fc:
                continue
            if fc.get("dome"):
                adv.weather[g.game_id] = (70.0, 0.0)
            elif fc.get("temp_f") is not None:
                adv.weather[g.game_id] = (float(fc["temp_f"]), float(fc.get("wind_mph") or 0.0))
    except Exception as exc:  # noqa: BLE001 -- forecast is enhancement, not load-bearing
        print(f"[pipeline] forecast weather unavailable ({exc}); schedule values kept")


_PACK_CACHE: Dict = {}
_PBP_EXT: Dict = {}  # play-by-play the advanced pack loaded (QB prior-starter proxy; context only)


def _feature_packs(inputs: candmod.WeekInputs):
    """Context/advanced packs are expensive (~10s) and season-static: build
    once per (seasons) signature per process; degrade to None loudly."""
    key = tuple(sorted(int(s) for s in inputs.pw["season"].unique()))
    if key in _PACK_CACHE:
        return _PACK_CACHE[key]
    try:
        from nflvalue.context_features import ContextPack
        from nflvalue.sources import rosters as rostersmod
        pack = ContextPack(rostersmod.fetch_rosters_weekly(list(key)), list(key),
                           opd=inputs.opd)
    except Exception as exc:  # noqa: BLE001 -- degrade to neutral stamps, loudly
        print(f"[pipeline] context features unavailable ({exc}); using neutral values")
        pack = None
    try:
        from nflvalue.advanced_features import AdvancedPack, load_pbp_ext
        _pbp_ext = load_pbp_ext()
        _PBP_EXT[key] = _pbp_ext
        adv = AdvancedPack(pbp=_pbp_ext, schedules=inputs.schedules)
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] advanced features unavailable ({exc}); using neutral values")
        adv = None
    try:
        from nflvalue.chemistry import ChemistryPack
        chem = ChemistryPack(pbp=_pbp_ext, pw=inputs.pw, schedules=inputs.schedules)
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] chemistry features unavailable ({exc}); using neutral values")
        chem = None
    try:
        from nflvalue.ftn_features import FTNPack
        ftn = FTNPack(pbp=_pbp_ext)
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] FTN features unavailable ({exc}); using neutral values")
        ftn = None
    try:
        from nflvalue.depth_features import DepthPack
        from nflvalue.sources import rosters as rostersmod
        depthp = DepthPack(rostersmod.fetch_rosters_weekly(list(key)), inputs.pw)
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] depth features unavailable ({exc}); using neutral values")
        depthp = None
    _PACK_CACHE[key] = (pack, adv, chem, ftn, depthp)
    return pack, adv, chem, ftn, depthp


def _maybe_stamp_ml(cfg: Dict, cands: pd.DataFrame,
                    inputs: candmod.WeekInputs) -> pd.DataFrame:
    """Flag-gated ML ranking (config "ml_ranker"): the trained classifier's
    P(over) ORDERS candidates (ordinal only); side, probability, edge, EV and
    Kelly stay the deterministic distribution's. Fails
    LOUD on a walk-forward violation (model trained on/after these weeks) and
    falls back to pure composite if no model artifact exists yet."""
    ml_cfg = cfg.get("ml_ranker") or {}
    if not ml_cfg.get("enabled") or cands.empty:
        return cands
    from nflvalue import ml_ranker as mlrmod
    path = ml_cfg.get("path", mlrmod.MODEL_PATH_DEFAULT)
    try:
        model = mlrmod.MLRanker.load(path)
    except FileNotFoundError:
        print(f"[pipeline] ml_ranker enabled but no model at {path} — "
              "run `python3 ml_test.py --stage fit` after grading; using composite ranking")
        return cands
    pack, adv, chem, ftn, depthp = _feature_packs(inputs)
    if depthp is not None and "player_depth_rank" not in cands.columns:
        cands = depthp.attach(cands)
    feats = mlrmod.build_features(cands, inputs.pw, pack=pack, adv=adv)
    try:
        p = model.predict_p_over(feats)
    except mlrmod.WalkForwardViolation as exc:
        # replaying a week the model already trained on: composite ranks it.
        # (Live future weeks always pass; a future-contaminated model would
        # still fail loudly there -- this fallback is only for the past.)
        print(f"[pipeline] ml_ranker skipped (walk-forward guard): {exc}")
        return cands
    # ``p`` is ORDINAL only. It orders candidates (shortlist.rank_game turns
    # ``ml_p_over`` into a score for the side the distribution/market chose)
    # and must never overwrite ``p_over``/``p_under``: those are the
    # distribution probabilities every edge/EV/Kelly number is derived from.
    # Nothing here is a calibration claim.
    cands = cands.copy()
    yes_only = cands["market"].isin({"anytime_td"})
    p_side = [max(x, 1 - x) for x in p]
    cands["ml_p_over"] = [round(float(x), 4) for x in p]
    cands["ml_score"] = [round(100 * (x if yo else ps), 2)
                         for x, ps, yo in zip(p, p_side, yes_only)]
    cands["rank_source"] = f"ml_{model.model_name}"
    # the artifact's own feature list is what inference read (not config, not NUMERIC_FEATURES)
    used = list(model.features or mlrmod.feature_columns())
    cands.attrs["ml_features_populated"] = [f for f in used if f in feats.columns
                                            and feats[f].notna().any()]
    return cands


def _synthesis_for_games(games: List[Dict], statuses: Dict[str, Dict],
                         sleeper_df: Optional[pd.DataFrame], as_of: str,
                         week: int, freshness_ts: Dict[str, str],
                         client=None,
                         news_by_player: Optional[Dict[str, List[Dict]]] = None) -> Dict[str, Dict]:
    """Run the §3 synthesis layer per game over the RANKED leans (context/
    verification only -- ranking is already final)."""
    slp_idx = {}
    if sleeper_df is not None and not sleeper_df.empty and "gsis_id" in sleeper_df:
        for r in sleeper_df.dropna(subset=["gsis_id"]).itertuples(index=False):
            slp_idx[(r.gsis_id, r.market)] = float(r.sleeper_proj)

    out: Dict[str, Dict] = {}
    for g in games:
        players = []
        for l in g["leans"]:
            pid = l["player_id"]
            st = statuses.get(pid, {})
            fantasy = slp_idx.get((pid, l["market"]))
            players.append({
                "player_id": pid, "name": l["name"], "pos": l.get("pos"),
                "team": l.get("team"),
                "model_projection": {"market": l["market"], "mean": l["mean"],
                                     "sd": l["sd"], "line": l.get("line"),
                                     "p_over": l.get("p_over"), "p_under": l.get("p_under")},
                "recent_usage": {"games_sample": l.get("roll_games")},
                "opponent_context": {},
                # a player the resolver returned nothing for is UNKNOWN, never OK
                "availability": {"report_status": st.get("status", "UNKNOWN"),
                                 "practice_status": None,
                                 "active_flag": None,
                                 "source": st.get("source", "none"),
                                 "timestamp": st.get("timestamp", as_of)},
                "fantasy_ref": ({"source": "sleeper", "proj": fantasy,
                                 "timestamp": freshness_ts.get("fantasy", as_of)}
                                if fantasy is not None else {}),
                "news": (news_by_player or {}).get(pid, []),
            })
        inp = synmod.build_input(as_of=as_of, week=week, game_id=g["game_id"],
                                 matchup=g["matchup"] or "",
                                 data_freshness={
                                     "injuries_updated": freshness_ts.get("injuries"),
                                     "roster_updated": freshness_ts.get("rosters"),
                                     "lines_updated": freshness_ts.get("lines"),
                                     "news_updated": freshness_ts.get("news"),
                                 },
                                 players=players)
        out[g["game_id"]] = synmod.synthesize(inp, client=client)
    return out


# --------------------------------------------------------------------------- #
# Live team identity: the active roster, acquired BEFORE candidates
# --------------------------------------------------------------------------- #
_NOT_ACQUIRED = object()


def _acquire_live_identity(season: int, inject_feeds: Optional[Dict],
                           inputs: candmod.WeekInputs):
    """Acquire the active roster ONCE, before candidate enumeration, and stamp
    the identity decision clock AFTER the acquisition.

    Returns ``(roster, identity_inputs, identity_at)``. ``identity_inputs`` is
    a shallow copy of ``inputs`` whose ``rosters`` is ONLY this payload's rows,
    with ``captured_at`` = the payload's own ``fetched_at`` (never invented,
    never back-dated). A payload whose fetch time is missing or later than
    ``identity_at`` verifies nobody (``features.asof_team_identity``).
    The same payload is handed to ``gather_live_feeds`` so the roster is not
    fetched twice and the roster gate judges the snapshot the seats came from.
    """
    import copy
    from nflvalue import features as featuresmod
    if "active_roster" in (inject_feeds or {}):
        roster = inject_feeds["active_roster"]
    else:
        try:
            from nflvalue.sources import active_roster as armod
            roster = armod.fetch_active_roster(season)
        except Exception as exc:  # noqa: BLE001 -- fail LOUD via the gate, not a crash
            print(f"[pipeline] active roster fetch FAILED: {exc}")
            roster = None
    identity_at = stamp_now()
    payload = dict(roster or {})
    payload.setdefault("season", season)
    identity_inputs = copy.copy(inputs)
    identity_inputs.rosters = featuresmod.roster_frame_from_active_roster(payload)
    return roster, identity_inputs, identity_at


def _identity_receipt(cands: pd.DataFrame, roster: Optional[Dict], identity_at: Optional[str]) -> Dict:
    """Run-receipt ``team_identity``: which clock seated the carry-forward
    players, from which roster capture, and who moved / was left unseated."""
    if identity_at is None:
        return {"clock": "not a live run (as_played seats)"}
    info = dict(cands.attrs.get("asof_team_identity") or {"clock": "missing"})
    info.update({"identity_at": identity_at, "roster_source": (roster or {}).get("source"),
                 "roster_fetched_at": (roster or {}).get("fetched_at"),
                 "roster_snapshot_at": (roster or {}).get("snapshot_at"),
                 "roster_week": (roster or {}).get("week"),
                 "roster_rows": len((roster or {}).get("rows") or [])})
    return info


# --------------------------------------------------------------------------- #
# Live feed gathering (fully injectable)
# --------------------------------------------------------------------------- #
def gather_live_feeds(cfg: Dict, season: int, week: int, players: pd.DataFrame,
                      clock: str = "wed", game_event_ids: Optional[List[str]] = None,
                      inject: Optional[Dict] = None,
                      prior_kickoff: Optional[Dict[str, str]] = None,
                      active_roster=_NOT_ACQUIRED) -> Dict:
    """Fetch injuries (+ inactives at t90) and Sleeper projections; stamp
    everything for the freshness gate. ``inject`` overrides any feed for
    tests/offline runs: {injury_rows, injuries_fetched_at, inactive_rows,
    inactives_fetched_at, sleeper_df, sleeper_fetched_at, active_roster}.
    ``active_roster`` is the ``sources.active_roster.fetch_active_roster``
    payload ({rows: [{player_id, team, status, week}], snapshot_at, ...})."""
    inject = inject or {}
    feeds: List[Feed] = []

    # -- injuries (load-bearing) -------------------------------------------- #
    if "injury_rows" in inject:
        injury_rows = inject["injury_rows"]
        inj_ts = inject.get("injuries_fetched_at", stamp_now())
    else:
        try:
            res = avmod.fetch_team_injuries()
            injury_rows, inj_ts = res["rows"], res["fetched_at"]
        except Exception as exc:  # noqa: BLE001 -- fail LOUD via the gate, not a crash
            print(f"[pipeline] injuries fetch FAILED: {exc}")
            injury_rows, inj_ts = [], None
    feeds.append(Feed("injuries", inj_ts, n_records=len(injury_rows), load_bearing=True))

    # -- inactives (t90 only; load-bearing at t90) --------------------------- #
    # Three distinct states, and conflating any two of them is how a board
    # either voids a whole game or publishes a fiction:
    #   never fetched      -- no event id, or every fetch threw. No timestamp.
    #   fetched, unpopulated -- ESPN answered but has not filled the event in
    #                        (period 0, nobody marked active). Timestamped,
    #                        rows EMPTY, and it must never imply anyone is out.
    #   fetched, populated -- the real actives list.
    inactive_rows, ina_ts = None, None
    inactives_state, inactives_reason = "not_fetched", ""
    if clock == "t90":
        if "inactive_rows" in inject:
            inactive_rows = inject["inactive_rows"]
            ina_ts = inject.get("inactives_fetched_at", stamp_now())
            inactives_state = inject.get("inactives_state", "populated")
            inactives_reason = inject.get("inactives_reason", "")
        else:
            inactive_rows = []
            reasons: List[str] = []
            for eid in game_event_ids or []:
                try:
                    res = avmod.fetch_event_rosters(eid)
                    ina_ts = res["fetched_at"]
                    if res.get("populated"):
                        inactive_rows.extend(res["rows"])
                        inactives_state = "populated"
                    else:
                        reasons.append(res.get("reason") or "unpopulated")
                        if inactives_state != "populated":
                            inactives_state = "unpopulated"
                except Exception as exc:  # noqa: BLE001
                    reasons.append(f"{eid}: {type(exc).__name__}: {exc}")
                    print(f"[pipeline] event roster fetch FAILED for {eid}: {exc}")
            inactives_reason = "; ".join(reasons)
            if not (game_event_ids or []):
                inactives_reason = "no ESPN event id resolved for this game"
        if inactives_state != "populated":
            # Never let an unpopulated feed reach the resolver: `active: false`
            # on every entry becomes OUT on every player.
            inactive_rows = []
        feeds.append(Feed("inactives", ina_ts if inactives_state == "populated" else None,
                          n_records=len(inactive_rows or []),
                          load_bearing=True))

    # -- league news (context only -> not load-bearing; text is untrusted) --- #
    if "news_items" in inject:
        news_items = inject["news_items"]
        news_ts = inject.get("news_fetched_at", stamp_now())
    else:
        try:
            from nflvalue.sources import espn_news
            res = espn_news.fetch_news()
            news_items, news_ts = res["items"], res["fetched_at"]
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] news fetch failed (context panel runs without it): {exc}")
            news_items, news_ts = [], None
    feeds.append(Feed("news", news_ts, n_records=len(news_items or []), load_bearing=False))

    # -- sleeper cross-check (context only -> not load-bearing) -------------- #
    if "sleeper_df" in inject:
        sleeper_df = inject["sleeper_df"]
        slp_ts = inject.get("sleeper_fetched_at", stamp_now())
    else:
        try:
            res = slpmod.fetch_projections(season, week)
            sleeper_df = slpmod.attach_gsis(res["df"], slpmod.fetch_player_map())
            slp_ts = res["fetched_at"]
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] sleeper fetch failed (cross-check unavailable): {exc}")
            sleeper_df, slp_ts = None, None
    feeds.append(Feed("fantasy", slp_ts, n_records=0 if sleeper_df is None else len(sleeper_df),
                      load_bearing=False))

    # -- active roster (load-bearing in live mode) --------------------------- #
    # Carry-forward history is not roster membership. The snapshot's OWN
    # timestamp (nflverse asset Last-Modified) is what the gate ages, so a
    # fresh fetch of a stale asset cannot pass as fresh.
    # ``active_roster``: the payload the run already acquired (and seated its
    # candidates from) -- used as-is, never re-fetched.
    if active_roster is not _NOT_ACQUIRED:
        roster = active_roster
    elif "active_roster" in inject:
        roster = inject["active_roster"]
    else:
        try:
            from nflvalue.sources import active_roster as armod
            roster = armod.fetch_active_roster(season)
        except Exception as exc:  # noqa: BLE001 -- fail LOUD via the gate, not a crash
            print(f"[pipeline] active roster fetch FAILED: {exc}")
            roster = None
    roster_ts = (roster or {}).get("snapshot_at") or (roster or {}).get("fetched_at")
    feeds.append(Feed("active_roster", roster_ts,
                      n_records=len((roster or {}).get("rows") or []), load_bearing=True))

    resolved = avmod.resolve_statuses(players, injury_rows, inactive_rows=inactive_rows,
                                      clock=clock, injuries_fetched_at=inj_ts,
                                      inactives_fetched_at=ina_ts, prior_kickoff=prior_kickoff)
    from nflvalue.sources.espn_news import news_by_player as _nbp
    news_map = _nbp(news_items or [], players) if news_items else {}
    # T-90: names the event roster lists as ACTIVE (a practice-squad elevation
    # shows up here, never in the weekly roster asset)
    # Only from a POPULATED roster. An unpopulated one yields the empty set,
    # which is not "nobody was elevated" -- it is "we do not know", and the
    # eligibility check must not read the two the same way.
    t90_active_names = None
    if clock == "t90" and inactives_state == "populated" and inactive_rows is not None:
        t90_active_names = {avmod.normalize_name(r.get("name"))
                            for r in inactive_rows if r.get("active")}
    return {"feeds": feeds, "statuses": resolved["statuses"],
            "active_roster": roster, "t90_active_names": t90_active_names,
            "inactives_state": inactives_state, "inactives_reason": inactives_reason,
            "unmatched": resolved["unmatched_espn_rows"], "sleeper_df": sleeper_df,
            # per-run availability evidence: a received report is not per-player evidence
            "report_state": resolved.get("report_state"),
            "report_evaluated": avmod.report_evaluated(resolved),
            "availability_summary": resolved.get("summary"),
            "news_by_player": news_map,
            "ts": {"injuries": inj_ts, "inactives": ina_ts, "fantasy": slp_ts,
                   "rosters": roster_ts, "news": news_ts, "lines": None}}


# --------------------------------------------------------------------------- #
# Dashboard merge
# --------------------------------------------------------------------------- #
def update_dashboard(report_payload: Dict, conn) -> str:
    data = cfgmod.load_json(cfgmod.LATEST_PATH, {}) or {}
    data["mode"] = "live"
    data["generated_at"] = report_payload.get("as_of") or stamp_now()
    data["weekly_leans"] = {k: v for k, v in report_payload.items() if k != "markdown"}
    data["leans_clv"] = clvmod.rolling_clv(conn)
    data["leans_killcheck"] = kcmod.report(conn)
    # Phase 8.4-8.6: the explainability payload (cards, trends, honest record).
    # Degrades LOUDLY -- if the ledger cannot be built the dashboard shows an
    # error block rather than a page that silently omits the "why", because a
    # missing explanation on a money-adjacent pick is itself the story.
    try:
        from nflvalue import explain_cards as xcards
        frame = None
        frame_path = os.path.join(cfgmod.DATA_DIR, "ml_frame.parquet")
        if os.path.exists(frame_path):
            frame = pd.read_parquet(
                frame_path,
                columns=["season", "week", "player_id", "market",
                         "proj_volume", "proj_efficiency", "opp_factor"])
        payload = xcards.build_payload(
            data["weekly_leans"], frame=frame, conn=conn,
            eval_results=cfgmod.load_json(
                os.path.join(cfgmod.DATA_DIR, "ml_eval_results.json"), None))
        xcards.write_payload(payload)
        data["explain"] = payload
    except Exception as exc:  # noqa: BLE001 -- visible, never silent
        print(f"[pipeline] explainability payload failed: "
              f"{type(exc).__name__}: {exc}")
        data["explain"] = {"cards": [], "unexplainable": [],
                           "error": f"{type(exc).__name__}: {exc}"}
    data.setdefault("refresh_seconds", 90)
    cfgmod.save_json(cfgmod.LATEST_PATH, data)
    return write_dashboard(data)


# --------------------------------------------------------------------------- #
# WED provisional run
# --------------------------------------------------------------------------- #
def run_week(season: int, week: int, mode: str = "historical", clock: str = "wed",
             live_odds: bool = False, discord: bool = False,
             inputs: Optional[candmod.WeekInputs] = None,
             inject_feeds: Optional[Dict] = None,
             odds_fetch: Optional[Callable] = None,
             list_events_fn: Optional[Callable] = None,
             discord_dry_run: bool = True) -> Dict:
    cfg = cfgmod.load_config()
    conn = dbmod.connect()
    inputs = inputs or candmod.build_week_inputs()
    slate = candmod.games_for_week(season, week, inputs.schedules)
    as_of = stamp_now()

    # 0. live team identity: roster acquired before candidates, clock after it
    live_roster, identity_at = None, None
    if mode == "live":
        live_roster, inputs, identity_at = _acquire_live_identity(season, inject_feeds, inputs)

    # 1. candidates (deterministic numbers; leak-free features)
    roster_mode = "as_played" if mode == "historical" else "carry_forward"
    cands = candmod.enumerate_candidates(
        season, week, inputs=inputs,
        min_usage=(cfg.get("candidates") or {}).get("min_usage"),
        roster_mode=roster_mode, decision_at=identity_at)
    identity_receipt = _identity_receipt(cands, live_roster, identity_at)

    # 2. live feeds + freshness gate
    publish, publish_reasons = True, []
    roster_gate: Dict = {"publish": False, "reason": "not a live run"}
    roster_diag: Dict = {}
    statuses: Dict[str, Dict] = {}
    sleeper_df, feeds_ts, news_by_player = None, {}, {}
    # which primary stages were evaluated this run (per-row states are stamped later)
    stage_ran = {s: False for s in fimod.STAGES}
    stage_why = {s: "not a live run" for s in fimod.STAGES}
    live: Dict = {}
    ctx_doc, ctx_label, ctx_meta = None, None, {"refresh": "not a live run"}
    if mode == "live":
        live = gather_live_feeds(cfg, season, week, _players_frame(cands),
                                 clock="wed", inject=inject_feeds,
                                 prior_kickoff=_prior_kickoffs(inputs.schedules, season, week),
                                 active_roster=live_roster)
        statuses, sleeper_df, feeds_ts = live["statuses"], live["sleeper_df"], live["ts"]
        news_by_player = live.get("news_by_player") or {}
        # slate-wide sourced context, fetched BEFORE the decision clock is stamped
        ctx_doc, ctx_label, ctx_meta = _run_context_doc(cfg, season, week, mode, inject_feeds,
                                                        live.get("active_roster"))
        snaps = _fetch_snaps(season, mode, inject_feeds)
        # as_of is the moment the decision is made, and the decision rests on
        # feeds that were fetched just now.  Stamping it BEFORE candidate
        # enumeration and the fetches made every feed whose fetch outlived
        # the wall-clock second "future-dated" -- H10's leakage guard firing
        # on its own inputs (2026-09-02, run 33578444172: the fantasy feed at
        # :14Z against an as_of of :09Z; injuries would have followed the
        # moment its fetch succeeded).  Re-stamp after the feeds are in hand.
        as_of = stamp_now()
        g = gate(live["feeds"], as_of=as_of,
                 staleness_hours=(cfg.get("freshness") or {}).get("staleness_hours"))
        publish, publish_reasons = g["publish"], g["reasons"]
        # ACTIVE ROSTER GATE (fail closed). History is not membership: a
        # candidate who is retired, released, on reserve, on the practice
        # squad, or on another team is excluded WITH its reason; a snapshot
        # that is missing, stale, future-dated, for the wrong season/week, or
        # missing a slate team blocks publication outright.
        roster_gate = pdmod.validate_roster_snapshot(
            live.get("active_roster"), season=season, week=week,
            slate_teams=set(slate["home_team"]) | set(slate["away_team"]),
            now=parse_ts(as_of))      # the decision clock, same as the freshness gate
        if not roster_gate["publish"]:
            publish = False
            publish_reasons = list(publish_reasons) + [roster_gate["reason"]]
            cands = cands.iloc[0:0].copy()
        else:
            cands, roster_diag = pdmod.apply_roster_eligibility(cands, live["active_roster"])
        # a RECEIVED report (not just a non-empty status map) is required before any
        # teammate-absence stage may be read as evaluated; per-player unknowns are then
        # stamped per row (fimod.build_stamps), never neutral by run-level flag
        avail_evaluated = bool(live.get("report_evaluated")) and roster_gate["publish"]
        for s_ in ("realloc_volume", "realloc_efficiency", "absence_qb"):
            stage_ran[s_] = avail_evaluated
            stage_why[s_] = None if avail_evaluated else (
                f"injury report {live.get('report_state') or 'not received'} this run; "
                f"availability not evaluated" if roster_gate["publish"]
                else "availability statuses not evaluated this run")
        # OUT players never reach the ranker (availability gate) -- and their
        # vacated usage is PRICED into teammates' projections (bounded; H8)
        out_ids = {pid for pid, s in statuses.items() if s["status"] == "OUT"}
        if out_ids:
            realloc = [avmod.reallocate_usage(inputs.pw, season, week, pid)
                       for pid in sorted(out_ids)]
            cands = cands[~cands["player_id"].isin(out_ids)].reset_index(drop=True)
            cands = candmod.apply_reallocation(cands, realloc)

    # 3. real prop lines (budgeted, rotating) -- pulled BEFORE any learning/
    # feature/ML stamping so the re-enumerated frame keeps every layer.
    # (Evaluation catch: the old order re-enumerated AFTER stamping, silently
    # dropping ML/learning/context exactly when real lines existed.)
    prop_lines, line_note = None, None
    line_rows, pulled_games = [], []
    if live_odds and cfg.get("odds_api_key"):
        event_map = build_event_map(cfg, slate, list_events_fn=list_events_fn)
        kickoffs = slate_kickoffs(slate)
        # Every scheduled game, soonest kickoff first, and each game pulled
        # holds the credits for its own pre-kick close (#26; credit_plan).
        pull = oapmod.pull_week_props(cfg, event_map, conn=conn, fetch=odds_fetch,
                                      kickoffs=kickoffs, reserve_close=True)
        feeds_ts["lines"] = pull["ts"]
        unmatched = sorted(set(slate["game_id"]) - set(event_map))
        if unmatched:
            print(f"[pipeline] {len(unmatched)} scheduled game(s) absent from the odds "
                  f"events listing (not pulled): {', '.join(unmatched)}")
        # Every quote we still hold for THIS week's games, not just the rows
        # this run happened to pull. The rotation prices a handful of games a
        # run; reading only `ts = pull["ts"]` threw away every earlier pull and
        # published NO_MARKET for games whose real lines were already stored.
        snap_rows = oapmod.load_recent_lines(conn, game_ids=list(slate["game_id"]))
        line_rows, pulled_games = snap_rows, list(pull["pulled"])
        rows = oapmod.match_player_ids(
            snap_rows, _players_frame(cands).rename(columns={"player_name": "name"}),
            roster_rows=(live.get("active_roster") or {}).get("rows") if mode == "live" else None,
            game_teams=_game_teams(slate))
        prop_lines = oapmod.to_prop_lines_frame(rows)
        carried = sorted({r["game_id"] for r in snap_rows} - set(pull["pulled"]))
        line_note = (f"Odds pull: {len(pull['pulled'])} game(s) pulled "
                     f"({', '.join(pull['pulled']) or 'none'}); "
                     + (f"{len(pull.get('empty') or [])} answered with no quotes "
                        f"({', '.join(pull.get('empty') or []) or 'none'}); ")
                     + f"{len(carried)} game(s) priced from stored quotes "
                     f"({', '.join(carried) or 'none'}); "
                     f"{len(pull['skipped_budget'])} skipped by credit budget, "
                     f"{len(pull['skipped_cap'])} by per-run cap, "
                     f"{len(pull.get('skipped_started') or [])} already under way; "
                     f"{len(unmatched)} not in the odds events listing; "
                     f"{pull['budget_remaining']:.0f} credits left this month; "
                     + oapmod.billing_text(pull) + ". "
                     + (oapmod.plan_text(pull["plan"]) + "." if pull.get("plan") else "")
                     + (f" NO odds pulled: {pull['quota_preflight']['reason']}."
                        if (pull.get("quota_preflight") or {}).get("ok") is False else ""))
        if not prop_lines.empty:
            cands = candmod.enumerate_candidates(
                season, week, inputs=inputs,
                min_usage=(cfg.get("candidates") or {}).get("min_usage"),
                prop_lines=prop_lines, roster_mode=roster_mode, decision_at=identity_at)
            if mode == "live":
                # the re-enumeration must pass the same roster gate
                if roster_gate["publish"]:
                    cands, roster_diag = pdmod.apply_roster_eligibility(
                        cands, live["active_roster"])
                else:
                    cands = cands.iloc[0:0].copy()
            out_ids = {pid for pid, s in statuses.items() if s["status"] == "OUT"}
            if out_ids:
                realloc = [avmod.reallocate_usage(inputs.pw, season, week, pid)
                           for pid in sorted(out_ids)]
                cands = cands[~cands["player_id"].isin(out_ids)].reset_index(drop=True)
                cands = candmod.apply_reallocation(cands, realloc)
    elif live_odds:
        line_note = "live-odds requested but no odds_api_key configured — all games no_market."

    # 4a. learning loop: walk-forward per-market corrections + (evidence-gated,
    # human-promoted) context multipliers. All no-ops until weeks are graded.
    # When the ML ranker is on, the bias-mean correction is SKIPPED -- the
    # classifier was trained on raw deterministic beliefs and subsumes
    # calibration; double-correcting would shift its features off-distribution.
    ml_on = bool((cfg.get("ml_ranker") or {}).get("enabled"))
    learn_cfg = {**{"enabled": True}, **(cfg.get("learning") or {})}
    if learn_cfg.get("enabled") and not ml_on:
        from nflvalue import context_study, prop_learning
        adjustments = prop_learning.load_adjustments(conn, season, week)
        cands = prop_learning.apply_to_candidates(cands, adjustments, enabled=True)
        ctx_mults = context_study.enabled_multipliers(cfg, conn)
        if ctx_mults:
            cands = context_study.apply_context_multipliers(cands, conn, season, week, ctx_mults)

    # 4b. stamp deterministic context/advanced features onto the candidates
    # (leans carry them -> panel + game notes render facts even when the ML
    # layer is off or falls back). Live weather comes from the FORECAST
    # (schedule temp/wind are observed post-game and NaN for future games).
    if mode == "live" and not cands.empty:
        pack, adv, chem, ftn, depthp = _feature_packs(inputs)
        from nflvalue.advanced_features import attach_neutral
        from nflvalue.context_features import attach as ctx_attach
        cands = ctx_attach(cands, pack)
        if adv is not None:
            _apply_forecast_weather(adv, slate)
            cands = adv.attach(cands)
        else:
            cands = attach_neutral(cands)
        outs_now = {pid for pid, s in statuses.items() if s.get("status") == "OUT"}
        if chem is not None:
            cands = chem.attach(cands, out_player_ids=outs_now)
        else:
            from nflvalue.chemistry import attach_neutral as chem_neutral
            cands = chem_neutral(cands)
        from nflvalue.ftn_features import attach_neutral as ftn_neutral
        cands = ftn.attach(cands) if ftn is not None else ftn_neutral(cands)
        from nflvalue.depth_features import attach_neutral as depth_neutral
        cands = depthp.attach(cands) if depthp is not None else depth_neutral(cands)
        # measured second-order: backup QB -> pass-family efficiency x0.92;
        # skill-leader absence -> QB passing markets (absence matrix)
        cands = candmod.apply_backup_qb_adjustment(cands)
        stage_ran["backup_qb"], stage_why["backup_qb"] = True, None
        cands = candmod.apply_absence_qb_adjustment(cands, inputs.pw, season, week, outs_now)
    elif mode == "live":
        for s_ in stage_ran:
            stage_ran[s_], stage_why[s_] = False, "no candidates reached the adjustment stages"

    # 4c. flag-gated ML ranking layer (see reports/ml_improvement_test.md)
    cands = _maybe_stamp_ml(cfg, cands, inputs)
    ml_feats = cands.attrs.get("ml_features_populated") or []
    ordering = (str(cands["rank_source"].iloc[0]) if "rank_source" in cands.columns
                and len(cands) else None)
    qb_ctx, snap_recs, snap_receipt = None, [], {"status": "not a live run"}
    if mode == "live":
        qb_ctx = fimod.qb_context_records(
            set(slate["home_team"]) | set(slate["away_team"]),
            doc=ctx_doc if ctx_doc is not None else _committed_context(season, week),
            pbp=_qb_pbp(inputs), roster_rows=(live.get("active_roster") or {}).get("rows"),
            season=season, week=week, as_of=as_of, kickoffs=_team_kickoffs(slate))
        snap_recs, snap_receipt = _participation_records(
            season, week, cands, inputs.schedules, live.get("active_roster"), as_of, snaps)
    stamps = fimod.build_stamps(cands, stage_ran, stage_why, ordering_features=ml_feats,
                                availability=statuses if mode == "live" else None,
                                qb_context=qb_ctx)
    starter_gate = (candmod.confirmed_starter_gate(cands.to_dict("records"), qb_ctx)
                    if qb_ctx is not None and len(cands) else {})
    _apply_starter_gate(stamps, starter_gate)
    # SHADOW role/opportunity forecast: stored beside the pick, never read by mean/SD/side/order
    shadow = (fimod.shadow_opportunity(inputs.pw, cands, season=season, week=week,
                                       as_of=parse_ts(as_of), kickoffs=slate_kickoffs(slate))
              if mode == "live" else {"status": "not a live run", "players": {}})

    # 4. rank + report (context panel via synthesis on the ranked leans)
    result = rptmod.generate(
        season, week, inputs=inputs, prop_lines=prop_lines,
        synthesis_by_game=None, availability=statuses or None,
        clock=clock, mode=mode, publish=publish, publish_reasons=publish_reasons,
        write_files=False, persist=False, line_note=line_note,
        candidates_df=cands)

    result["roster_gate"] = {k: v for k, v in roster_gate.items()}
    result["roster_eligibility"] = roster_diag
    if mode == "live":
        from nflvalue.game_notes import attach_notes
        attach_notes(result["games"], cands, inputs.schedules, season, week)
        syn = _synthesis_for_games(result["games"], statuses, sleeper_df,
                                   as_of, week, feeds_ts, news_by_player=news_by_player)
        notes = rptmod.load_manual_notes(conn, season, week)
        result["contexts"] = {
            g["game_id"]: slmod.build_context_panel(
                g, synthesis_output=syn.get(g["game_id"]), manual_notes=notes,
                availability=statuses, mode="live")
            for g in result["games"]}
        result["markdown"] = rptmod.render_markdown(
            season, week, result["games"], result["contexts"], result["as_of"],
            clock, publish=publish, publish_reasons=publish_reasons, line_note=line_note)
        # context hypothesis ledger: record every tag we DISPLAYED, so the
        # weekly grade can test whether any of them actually predict outcomes
        from nflvalue import context_study
        context_study.record_tags(conn, season, week, result["games"], result["contexts"])

    # 5. write artifacts + forward log (idempotent)
    import os
    os.makedirs(rptmod.REPORTS_DIR, exist_ok=True)
    md_path = os.path.join(rptmod.REPORTS_DIR, f"props_week_{season}_{week}.md")
    with open(md_path, "w") as f:
        f.write(result["markdown"])
    result["md_path"] = md_path
    from nflvalue.document import write_drop
    result["drop_path"] = write_drop(result, result.get("contexts"))
    cfgmod.save_json(rptmod.WEEKLY_PROPS_JSON, {k: v for k, v in result.items() if k != "markdown"})
    fimod.attach_to_leans(result["games"], stamps, shadow)
    rptmod.persist_leans(conn, season, week, clock, result["games"], result["as_of"])
    from nflvalue.provenance import run_provenance
    prov = run_provenance()
    receipt = fimod.record_issuing_run(
        conn, prov, season=season, week=week, clock=clock, run_id=prov["run_id"],
        as_of=result["as_of"], game_ids=list(slate["game_id"]), ran=stage_ran, reasons=stage_why,
        ordering_component=ordering, ordering_features=ml_feats, shadow=shadow,
        extra={"lines": fimod.lines_provenance(line_rows, pulled_games),
               # the run's own publication decision: cards from a held run are never executable
               "publish": bool(publish), "publish_reasons": list(publish_reasons or []),
               "qb_starter_gate": _starter_diagnostics(qb_ctx, starter_gate, cands, result["games"]),
               "team_identity": identity_receipt,
               **_availability_receipt(live, qb_ctx, ctx_meta, snap_receipt)},
        context_doc=ctx_doc, context_label=ctx_label, extra_records=snap_recs)
    result["factor_receipt"] = receipt
    print(f"[pipeline] factor receipt: stages {receipt['stages_executed']}; shadow "
          f"{receipt['shadow']['status']} ({receipt['shadow']['players']} players); context "
          f"{receipt['context']['path'] or 'NOT COLLECTED'} for {receipt['context']['games_with_context']}")

    # 6. dashboard + (flag-gated) discord
    dash = update_dashboard(result, conn)
    notice = None
    if discord:
        from nflvalue import notify
        notice = notify.post_weekly(result, cfg=cfg, dry_run=discord_dry_run)
    conn.close()
    return {**{k: v for k, v in result.items() if k != "markdown"},
            "dashboard": dash, "discord": notice}


# --------------------------------------------------------------------------- #
# T-90 refresh for one game: void inactive players, re-rank, regenerate
# --------------------------------------------------------------------------- #
def run_t90(season: int, week: int, game_id: str, mode: str = "live",
            inputs: Optional[candmod.WeekInputs] = None,
            inject_feeds: Optional[Dict] = None, discord: bool = False,
            discord_dry_run: bool = True,
            odds_fetch: Optional[Callable] = None,
            list_events_fn: Optional[Callable] = None) -> Dict:
    cfg = cfgmod.load_config()
    conn = dbmod.connect()
    inputs = inputs or candmod.build_week_inputs()
    as_of = stamp_now()

    # live team identity: roster acquired before candidates, clock after it (see run_week)
    live_roster, identity_at = None, None
    if mode == "live":
        live_roster, inputs, identity_at = _acquire_live_identity(season, inject_feeds, inputs)

    roster_mode = "as_played" if mode == "historical" else "carry_forward"
    cands = candmod.enumerate_candidates(
        season, week, inputs=inputs,
        min_usage=(cfg.get("candidates") or {}).get("min_usage"),
        roster_mode=roster_mode, decision_at=identity_at)
    identity_receipt = _identity_receipt(cands, live_roster, identity_at)
    if not cands.empty:          # a board with nobody seated has no columns to filter on
        cands = cands[cands["game_id"] == game_id].reset_index(drop=True)
    if cands.empty:
        conn.close()
        raise ValueError(f"no candidates for game {game_id} — check season/week/game_id; "
                         f"team identity {identity_receipt.get('clock')}, "
                         f"{identity_receipt.get('n_unseated', 0)} player(s) unseated "
                         f"(roster fetched_at {identity_receipt.get('roster_fetched_at')}, "
                         f"identity_at {identity_receipt.get('identity_at')})")

    # REAL LINES AT T-90. This is the best moment of the week to spend a
    # credit on this game: the line is closest to its close and the inactives
    # are out. Before 2026-09-09 T-90 touched `lines` neither way, so a game
    # the rotation had skipped stayed NO_MARKET through kickoff and no run
    # ever priced it. One event, one game, budget-checked like any other pull;
    # pulled BEFORE feature/ML stamping so the re-enumerated frame keeps every
    # layer (the same ordering catch run_week documents).
    t90_line_note = None
    line_rows, pulled_games = [], []
    if mode == "live" and cfg.get("odds_api_key"):
        slate_all = candmod.games_for_week(season, week, inputs.schedules)
        one = slate_all[slate_all["game_id"] == game_id]
        # The scheduled T-90 job has usually just re-snapped this game's close
        # (auto_weekly.job_t90 -> resnap_lines). That quote IS the T-90 line;
        # pulling again would spend a second event-call on the same game.
        fresh = oapmod.load_recent_lines(conn, game_ids=[game_id],
                                         max_age_hours=T90_LINE_FRESH_HOURS)
        if fresh:
            t90_line_note = (f"T-90 odds: {len(fresh)} quote row(s) already refreshed within "
                             f"{T90_LINE_FRESH_HOURS:g}h; no credit spent.")
            print(f"[t90] {game_id}: {t90_line_note}")
        else:
            try:
                event_map = {k: v for k, v in
                             build_event_map(cfg, one, list_events_fn=list_events_fn).items()
                             if k == game_id}
                if event_map:
                    pull = oapmod.pull_week_props(cfg, event_map, conn=conn, fetch=odds_fetch,
                                                  kickoffs=slate_kickoffs(one))
                    pulled_games = list(pull["pulled"])
                    t90_line_note = (f"T-90 odds pull: {len(pull['pulled'])} game(s); "
                                     f"{pull['budget_remaining']:.0f} credits left this month.")
            except oapmod.BudgetExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 -- degrade, don't abort
                t90_line_note = f"T-90 odds pull failed ({type(exc).__name__}: {exc})"
                print(f"[t90] odds pull failed for {game_id}: {exc}")
        line_rows = oapmod.load_recent_lines(conn, game_ids=[game_id])
        rows = oapmod.match_player_ids(
            line_rows, _players_frame(cands).rename(columns={"player_name": "name"}),
            game_teams=_game_teams(one))
        prop_lines = oapmod.to_prop_lines_frame(rows)
        if not prop_lines.empty:
            cands = candmod.enumerate_candidates(
                season, week, inputs=inputs,
                min_usage=(cfg.get("candidates") or {}).get("min_usage"),
                prop_lines=prop_lines, roster_mode=roster_mode, decision_at=identity_at)
            cands = cands[cands["game_id"] == game_id].reset_index(drop=True)

    # Stages THIS refresh evaluates. Every stage starts not-evaluated with the
    # T-90 reason; only what actually runs below is marked. The Wednesday
    # run's stages are never assumed, and a stage added later is missing here
    # until the refresh really executes it.
    stage_ran = {s: False for s in fimod.STAGES}
    stage_why = {s: ("not executed by the T-90 refresh" if mode == "live" else "not a live run")
                 for s in fimod.STAGES}

    # stamp context/advanced features + ML so t90 leans carry the same
    # writeup facts and ranking as the Wednesday run
    if mode == "live" and not cands.empty:
        pack, adv, chem, ftn, depthp = _feature_packs(inputs)
        from nflvalue.advanced_features import attach_neutral
        from nflvalue.context_features import attach as ctx_attach
        cands = ctx_attach(cands, pack)
        cands = adv.attach(cands) if adv is not None else attach_neutral(cands)
        if chem is not None:
            cands = chem.attach(cands)
        else:
            from nflvalue.chemistry import attach_neutral as chem_neutral
            cands = chem_neutral(cands)
        from nflvalue.ftn_features import attach_neutral as ftn_neutral
        cands = ftn.attach(cands) if ftn is not None else ftn_neutral(cands)
        from nflvalue.depth_features import attach_neutral as depth_neutral
        cands = depthp.attach(cands) if depthp is not None else depth_neutral(cands)
        cands = candmod.apply_backup_qb_adjustment(cands)
        stage_ran["backup_qb"], stage_why["backup_qb"] = True, None
    cands = _maybe_stamp_ml(cfg, cands, inputs)
    # The ESPN event id for THIS game. Without it `gather_live_feeds` iterates
    # an empty list and the inactives feed -- the entire reason T-90 exists --
    # arrives empty and unstamped. `game_event_ids` had no caller until now.
    event_ids: List[str] = []
    if "inactive_rows" not in (inject_feeds or {}):
        slate_e = candmod.games_for_week(season, week, inputs.schedules)
        slate_e = slate_e[slate_e["game_id"] == game_id]
        try:
            found = avmod.find_event_ids(slate_e.to_dict("records"))
            event_ids = [found[game_id]] if game_id in found else []
            if not event_ids:
                print(f"[t90] no ESPN event id resolved for {game_id}")
        except Exception as exc:  # noqa: BLE001 -- missing feed, not a dead run
            print(f"[t90] event id lookup failed for {game_id}: {exc}")
    live = gather_live_feeds(cfg, season, week, _players_frame(cands), clock="t90",
                             game_event_ids=event_ids, inject=inject_feeds,
                             prior_kickoff=_prior_kickoffs(inputs.schedules, season, week),
                             **({"active_roster": live_roster} if mode == "live" else {}))
    statuses = live["statuses"]
    # this refresh's own context and participation, fetched before its decision clock
    ctx_doc, ctx_label, ctx_meta = _run_context_doc(cfg, season, week, mode, inject_feeds,
                                                    live.get("active_roster"))
    snaps = _fetch_snaps(season, mode, inject_feeds)
    as_of = stamp_now()  # the decision follows the fetch; see run_week
    g = gate(live["feeds"], as_of=as_of,
             staleness_hours=(cfg.get("freshness") or {}).get("staleness_hours"))
    # Say WHY, precisely. "no/unparseable timestamp" is what the gate sees, but
    # it reads as a parsing bug when the truth is usually that ESPN has not
    # published the event roster yet. A wrong reason sends the next reader
    # hunting for a defect that is not there.
    #
    # OWNER DECISION 2026-09-22 (#27, option 2): ESPN's event roster is a
    # participation record that is unpopulated until kickoff, so a load-
    # bearing inactives feed meant no T-90 board could ever publish. When the
    # source answered but has not published (state "unpopulated"), the feed
    # is downgraded: the board PUBLISHES with a visible banner saying the
    # game-day inactives check did not happen, and the Wednesday availability
    # read (injuries) stands. A source that could not be fetched at all
    # (state "not_fetched": no event id, or every fetch threw) still holds
    # the board -- that is a defect on our side, not the source's timing.
    inactives_state = live.get("inactives_state")
    inactives_banner = None
    if inactives_state == "unpopulated":
        g = gate([f for f in live["feeds"] if f.name != "inactives"], as_of=as_of,
                 staleness_hours=(cfg.get("freshness") or {}).get("staleness_hours"))
        inactives_banner = (
            f"inactives: source has not published yet "
            f"({live.get('inactives_reason') or 'ESPN event roster unpopulated'}); "
            f"published WITHOUT a game-day inactives check -- confirm inactives before acting")
        g["reasons"] = list(g["reasons"]) + [inactives_banner]
    elif inactives_state not in (None, "populated"):
        g["reasons"] = [r for r in g["reasons"] if not r.startswith("inactives:")]
        g["reasons"].append(
            f"inactives: source could not be fetched "
            f"({live.get('inactives_reason') or inactives_state})")
        g["publish"] = False
    # ACTIVE ROSTER GATE at T-90 (same contract as the Wednesday run; a
    # practice-squad player the event roster lists as active is an elevation)
    slate_t = candmod.games_for_week(season, week, inputs.schedules)
    slate_t = slate_t[slate_t["game_id"] == game_id]
    roster_gate = pdmod.validate_roster_snapshot(
        live.get("active_roster"), season=season, week=week,
        slate_teams=set(slate_t["home_team"]) | set(slate_t["away_team"]),
        now=parse_ts(as_of))
    roster_diag: Dict = {}
    if roster_gate["publish"]:
        cands, roster_diag = pdmod.apply_roster_eligibility(
            cands, live["active_roster"], t90_active_names=live.get("t90_active_names"))
    else:
        cands = cands.iloc[0:0].copy()
        g["publish"] = False
        g["reasons"] = list(g["reasons"]) + [roster_gate["reason"]]

    # 1. VOID wed leans whose player is now OUT (auto, with provenance)
    wed = dbmod.query_df(conn, """
        SELECT * FROM leans WHERE season=? AND week=? AND clock='wed' AND game_id=?
        """, (season, week, game_id))
    voided = []
    for l in wed.itertuples(index=False):
        st = statuses.get(l.player_id, {})
        if st.get("status") == "OUT" and l.status != "voided":
            conn.execute("""
                UPDATE leans SET status='voided', void_reason=?
                WHERE season=? AND week=? AND clock='wed' AND game_id=? AND player_id=? AND market=?
                """, (f"t90: {st.get('status_raw') or 'inactive'} ({st.get('source')})",
                      season, week, game_id, l.player_id, l.market))
            voided.append({"player_id": l.player_id, "name": l.name, "market": l.market,
                           "reason": st.get("status_raw") or "inactive"})
    conn.commit()

    # 2. re-rank without OUT players; downgrade note for RISK
    out_ids = {pid for pid, s in statuses.items() if s["status"] == "OUT"}
    cands2 = cands[~cands["player_id"].isin(out_ids)].reset_index(drop=True)
    if cands2.empty:
        for s_ in stage_ran:
            stage_ran[s_], stage_why[s_] = False, "no candidates reached the adjustment stages"
    ml_feats = cands2.attrs.get("ml_features_populated") or cands.attrs.get("ml_features_populated") or []
    ordering = (str(cands2["rank_source"].iloc[0]) if "rank_source" in cands2.columns
                and len(cands2) else None)
    qb_ctx = fimod.qb_context_records(
        set(slate_t["home_team"]) | set(slate_t["away_team"]),
        doc=ctx_doc if ctx_doc is not None else _committed_context(season, week),
        pbp=_qb_pbp(inputs), roster_rows=(live.get("active_roster") or {}).get("rows"),
        season=season, week=week, as_of=as_of, kickoffs=_team_kickoffs(slate_t))
    snap_recs, snap_receipt = _participation_records(
        season, week, cands2, inputs.schedules, live.get("active_roster"), as_of, snaps)
    stamps = fimod.build_stamps(cands2, stage_ran, stage_why, ordering_features=ml_feats,
                                availability=statuses, qb_context=qb_ctx)
    starter_gate = (candmod.confirmed_starter_gate(cands2.to_dict("records"), qb_ctx)
                    if len(cands2) else {})
    _apply_starter_gate(stamps, starter_gate)
    # SHADOW at the refresh's own clock (never read by mean/SD/side/order)
    shadow = (fimod.shadow_opportunity(inputs.pw, cands2, season=season, week=week,
                                       as_of=parse_ts(as_of), kickoffs=slate_kickoffs(slate_t))
              if mode == "live" else {"status": "not a live run", "players": {}})
    games = slmod.shortlist_week(cands2,
                                 weights=(cfg.get("composite") or {}).get("weights"),
                                 params=(cfg.get("composite") or {}).get("params"),
                                 top_n=int((cfg.get("shortlist") or {}).get("top_n", 5)),
                                 max_per_player=int((cfg.get("shortlist") or {})
                                                    .get("max_per_player", 2)))
    from nflvalue.game_notes import attach_notes
    attach_notes(games, cands2, inputs.schedules, season, week)
    contexts = {gm["game_id"]: slmod.build_context_panel(
        gm, availability=statuses, mode=mode) for gm in games}

    md = rptmod.render_markdown(
        season, week, games, contexts, as_of, "t90",
        publish=g["publish"], publish_reasons=g["reasons"],
        line_note=(f"T-90 refresh of {game_id}: {len(voided)} Wednesday lean(s) auto-voided "
                   f"({', '.join(v['name'] for v in voided) or 'none'})."))
    import os
    os.makedirs(rptmod.REPORTS_DIR, exist_ok=True)
    md_path = os.path.join(rptmod.REPORTS_DIR, f"props_week_{season}_{week}_t90_{game_id}.md")
    with open(md_path, "w") as f:
        f.write(md)
    from nflvalue.provenance import run_provenance
    prov = run_provenance()
    run_id = fimod.issuing_run_id(prov["run_id"], "t90", game_id)
    fimod.attach_to_leans(games, stamps, shadow)
    rptmod.persist_leans(conn, season, week, "t90", games, as_of, game_ids=[game_id],
                         run_id=run_id)
    receipt = fimod.record_issuing_run(
        conn, prov, season=season, week=week, clock="t90", run_id=run_id, as_of=as_of,
        game_ids=[game_id], ran=stage_ran, reasons=stage_why, ordering_component=ordering,
        ordering_features=ml_feats, shadow=shadow,
        extra={"lines": fimod.lines_provenance(line_rows, pulled_games),
               "inactives_state": inactives_state,
               "inactives_reason": live.get("inactives_reason") or None,
               "publish": bool(g["publish"]), "publish_reasons": list(g["reasons"] or []),
               "qb_starter_gate": _starter_diagnostics(qb_ctx, starter_gate, cands2, games),
               "team_identity": identity_receipt,
               **_availability_receipt(live, qb_ctx, ctx_meta, snap_receipt)},
        context_doc=ctx_doc, context_label=ctx_label, extra_records=snap_recs)
    print(f"[t90] {game_id} factor receipt {run_id}: stages {receipt['stages_executed']}; "
          f"shadow {receipt['shadow']['status']}; context {receipt['context']['status']}")

    payload = {"season": season, "week": week, "clock": "t90", "as_of": as_of,
               "publish": g["publish"], "publish_reasons": g["reasons"],
               "mode": mode, "games": games, "contexts": contexts,
               "voided": voided, "md_path": md_path,
               "roster_gate": dict(roster_gate), "roster_eligibility": roster_diag,
               "line_note": t90_line_note,
               "inactives_state": inactives_state,
               "inactives_banner": inactives_banner, "factor_receipt": receipt}
    from nflvalue.document import write_drop
    payload["drop_path"] = write_drop(payload, contexts)
    dash = update_dashboard(payload, conn)
    notice = None
    if discord:
        from nflvalue import notify
        notice = notify.post_weekly(payload, cfg=cfg, dry_run=discord_dry_run)
    conn.close()
    return {**payload, "dashboard": dash, "discord": notice}


# --------------------------------------------------------------------------- #
# Post-slate CLV resolution
# --------------------------------------------------------------------------- #
def resolve_clv(season: int, week: int,
                inputs: Optional[candmod.WeekInputs] = None) -> Dict:
    conn = dbmod.connect()
    inputs = inputs or candmod.build_week_inputs()
    slate = candmod.games_for_week(season, week, inputs.schedules)
    kickoffs = kickoffs_for(slate)
    resolved = clvmod.log_close_for_week(conn, season, week, kickoffs)
    # P0: also persist the durable opens/closes record for EVERY snapshotted
    # market (not just published leans) so a real-line reliability/CLV backtest
    # can be built retroactively once enough weeks accrue.
    opens_closes = clvmod.log_open_close_for_week(conn, season, week, kickoffs)
    stats = clvmod.rolling_clv(conn)
    verdict = kcmod.report(conn)
    conn.close()
    # Refresh the standing real-line report from whatever has accrued so far
    # (fail-closed sections; safe on thin data — see analysis/real_line_backtest.py).
    try:
        from analysis import real_line_backtest as rlb
        import json as _json
        _report = rlb.build_report()
        os.makedirs(os.path.dirname(rlb.BOOK_PATH), exist_ok=True)
        with open(rlb.BOOK_PATH, "w") as _fh:
            _json.dump(_report, _fh, indent=1, default=str)
    except Exception as exc:  # a report refresh must never break close capture
        print(f"real_line_backtest refresh skipped: {exc}")
    return {"resolved": int(len(resolved)),
            "opens_closes_logged": int(len(opens_closes)),
            "clv": stats, "killcheck": verdict}


def run_grade(season: int, week: int, inputs: Optional[candmod.WeekInputs] = None) -> Dict:
    """The Tuesday learning step: grade last week, attribute, update adjustments."""
    from nflvalue import prop_learning
    cfg = cfgmod.load_config()
    conn = dbmod.connect()
    inputs = inputs or candmod.build_week_inputs()
    res = prop_learning.grade_and_learn(conn, season, week, inputs,
                                        params=cfg.get("learning"))
    conn.close()
    return res


def _emit_top_bets() -> None:
    try:
        from nflvalue import top_bets as _tb
        _tb.main()
    except Exception as exc:  # never break the pipeline on the display layer
        print(f"top_bets generation skipped: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--clock", choices=["wed", "t90"], default="wed")
    ap.add_argument("--game", help="game_id for the t90 refresh (e.g. 2025_10_CLE_BAL)")
    ap.add_argument("--mode", choices=["historical", "live"], default="live")
    ap.add_argument("--live-odds", action="store_true",
                    help="pull real prop lines (needs odds_api_key; budgeted)")
    ap.add_argument("--discord", action="store_true", help="post to Discord (flag-gated in config too)")
    ap.add_argument("--discord-live", action="store_true",
                    help="actually POST to the webhook (default is dry-run)")
    ap.add_argument("--resolve-clv", action="store_true", help="post-slate CLV resolution")
    ap.add_argument("--grade", action="store_true",
                    help="grade a completed week + update the learning loop")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip the automatic current-season data ingest")
    args = ap.parse_args()

    if args.mode == "live" and not args.no_refresh:
        from nflvalue import ingest
        r = ingest.refresh()
        print(f"[ingest] season {r['season']}: pbp_rows={r['pbp_rows']} "
              f"sched_rows={r['sched_rows']} stale={r['stale']}"
              + (f" errors={r['errors']}" if r["errors"] else ""))
        if r["stale"]:
            print("[ingest] WARNING: serving cached data; the freshness gate "
                  "and report banners reflect anything load-bearing that's missing.")

    if args.grade:
        res = run_grade(args.season, args.week)
        import json as _json
        print(f"Graded {res['graded']} leans (hit rate {res['hit_rate']}); "
              f"adjustments effective {res['adjustments_effective']}:")
        print(_json.dumps(res["adjustments"], indent=1, default=str))
        print("Miss reasons:", res["why"].get("recent_miss_reasons"))
        return
    if args.resolve_clv:
        res = resolve_clv(args.season, args.week)
        print(f"CLV resolved: {res['resolved']} · rolling: {res['clv']} · "
              f"kill-check: {res['killcheck']['verdict']}")
        return
    if args.clock == "t90":
        if not args.game:
            ap.error("--clock t90 requires --game GAME_ID")
        res = run_t90(args.season, args.week, args.game, mode=args.mode,
                      discord=args.discord, discord_dry_run=not args.discord_live)
        print(f"T-90 refresh {args.game}: {len(res['voided'])} lean(s) voided → {res['md_path']}")
        return
    res = run_week(args.season, args.week, mode=args.mode, clock="wed",
                   live_odds=args.live_odds, discord=args.discord,
                   discord_dry_run=not args.discord_live)
    print(f"Week {args.season}/{args.week} ({args.mode}): {len(res['games'])} games → "
          f"{res['md_path']} · publish={res['publish']} · dashboard={res['dashboard']}")
    _emit_top_bets()


if __name__ == "__main__":
    main()
