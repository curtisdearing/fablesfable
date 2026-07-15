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
from typing import Callable, Dict, List, Optional

import pandas as pd

from nflvalue import candidates as candmod
from nflvalue import clv as clvmod
from nflvalue import config as cfgmod
from nflvalue import db as dbmod
from nflvalue import killcheck as kcmod
from nflvalue import report as rptmod
from nflvalue import shortlist as slmod
from nflvalue import synthesis as synmod
from nflvalue.dashboard import write_dashboard
from nflvalue.freshness import Feed, gate, stamp_now
from nflvalue.sources import availability as avmod
from nflvalue.sources import oddsapi_props as oapmod
from nflvalue.sources import sleeper as slpmod


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


def _players_frame(cands: pd.DataFrame) -> pd.DataFrame:
    return (cands[["player_id", "name", "team"]].drop_duplicates()
            .rename(columns={"name": "player_name"}))


def _apply_forecast_weather(adv, slate: pd.DataFrame) -> None:
    """Override the pack's (post-game, NaN-for-future) schedule weather with
    live Open-Meteo forecasts for this slate's outdoor games (evaluation
    catch: without this, the weather feature is dead all season)."""
    try:
        from build_ratings import ABBR
        from nflvalue.sources.weather import forecast_for_game
        for g in slate.itertuples(index=False):
            wx = adv.weather.get(g.game_id, (None, None))
            if wx[0] is not None and not pd.isna(wx[0]):
                continue  # dome-neutralized or already known
            commence = f"{g.gameday}T{(g.gametime or '13:00')}:00+00:00"
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
    _PACK_CACHE[key] = (pack, adv, chem, ftn)
    return pack, adv, chem, ftn


def _maybe_stamp_ml(cfg: Dict, cands: pd.DataFrame,
                    inputs: candmod.WeekInputs) -> pd.DataFrame:
    """Flag-gated ML ranking (config "ml_ranker"): the trained classifier's
    P(over) replaces the ranking probability (side + ordering), while every
    published NUMBER (mean/sd/line) stays the deterministic model's. Fails
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
    pack, adv, chem, ftn = _feature_packs(inputs)
    feats = mlrmod.build_features(cands, inputs.pw, pack=pack, adv=adv)
    try:
        p = model.predict_p_over(feats)
    except mlrmod.WalkForwardViolation as exc:
        # replaying a week the model already trained on: composite ranks it.
        # (Live future weeks always pass; a future-contaminated model would
        # still fail loudly there -- this fallback is only for the past.)
        print(f"[pipeline] ml_ranker skipped (walk-forward guard): {exc}")
        return cands
    cands = cands.copy()
    cands["p_over"] = [round(float(x), 4) for x in p]
    cands["p_under"] = [round(1 - float(x), 4) for x in p]
    yes_only = cands["market"].isin({"anytime_td"})
    p_side = [max(x, 1 - x) for x in p]
    cands["ml_score"] = [round(100 * (x if yo else ps), 2)
                         for x, ps, yo in zip(p, p_side, yes_only)]
    cands["prob_source"] = f"ml_{model.model_name}"
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
                "availability": {"report_status": st.get("status", "OK"),
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
# Live feed gathering (fully injectable)
# --------------------------------------------------------------------------- #
def gather_live_feeds(cfg: Dict, season: int, week: int, players: pd.DataFrame,
                      clock: str = "wed", game_event_ids: Optional[List[str]] = None,
                      inject: Optional[Dict] = None) -> Dict:
    """Fetch injuries (+ inactives at t90) and Sleeper projections; stamp
    everything for the freshness gate. ``inject`` overrides any feed for
    tests/offline runs: {injury_rows, injuries_fetched_at, inactive_rows,
    inactives_fetched_at, sleeper_df, sleeper_fetched_at}."""
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
    inactive_rows, ina_ts = None, None
    if clock == "t90":
        if "inactive_rows" in inject:
            inactive_rows = inject["inactive_rows"]
            ina_ts = inject.get("inactives_fetched_at", stamp_now())
        else:
            inactive_rows = []
            for eid in game_event_ids or []:
                try:
                    res = avmod.fetch_event_rosters(eid)
                    inactive_rows.extend(res["rows"])
                    ina_ts = res["fetched_at"]
                except Exception as exc:  # noqa: BLE001
                    print(f"[pipeline] event roster fetch FAILED for {eid}: {exc}")
        feeds.append(Feed("inactives", ina_ts, n_records=len(inactive_rows or []),
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

    resolved = avmod.resolve_statuses(players, injury_rows, inactive_rows=inactive_rows,
                                      clock=clock, injuries_fetched_at=inj_ts,
                                      inactives_fetched_at=ina_ts)
    from nflvalue.sources.espn_news import news_by_player as _nbp
    news_map = _nbp(news_items or [], players) if news_items else {}
    return {"feeds": feeds, "statuses": resolved["statuses"],
            "unmatched": resolved["unmatched_espn_rows"], "sleeper_df": sleeper_df,
            "news_by_player": news_map,
            "ts": {"injuries": inj_ts, "inactives": ina_ts, "fantasy": slp_ts,
                   "rosters": stamp_now(), "news": news_ts, "lines": None}}


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

    # 1. candidates (deterministic numbers; leak-free features)
    roster_mode = "as_played" if mode == "historical" else "carry_forward"
    cands = candmod.enumerate_candidates(
        season, week, inputs=inputs,
        min_usage=(cfg.get("candidates") or {}).get("min_usage"),
        roster_mode=roster_mode)

    # 2. live feeds + freshness gate
    publish, publish_reasons = True, []
    statuses: Dict[str, Dict] = {}
    sleeper_df, feeds_ts, news_by_player = None, {}, {}
    if mode == "live":
        live = gather_live_feeds(cfg, season, week, _players_frame(cands),
                                 clock="wed", inject=inject_feeds)
        statuses, sleeper_df, feeds_ts = live["statuses"], live["sleeper_df"], live["ts"]
        news_by_player = live.get("news_by_player") or {}
        g = gate(live["feeds"], as_of=as_of,
                 staleness_hours=(cfg.get("freshness") or {}).get("staleness_hours"))
        publish, publish_reasons = g["publish"], g["reasons"]
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
    if live_odds and cfg.get("odds_api_key"):
        event_map = build_event_map(cfg, slate, list_events_fn=list_events_fn)
        pull = oapmod.pull_week_props(cfg, event_map, conn=conn, fetch=odds_fetch)
        feeds_ts["lines"] = pull["ts"]
        snap = dbmod.query_df(conn, "SELECT * FROM lines WHERE ts=?", (pull["ts"],))
        rows = oapmod.match_player_ids(snap.to_dict("records"), _players_frame(cands)
                                       .rename(columns={"player_name": "name"}))
        prop_lines = oapmod.to_prop_lines_frame(rows)
        line_note = (f"Odds pull: {len(pull['pulled'])} game(s) pulled "
                     f"({', '.join(pull['pulled']) or 'none'}); "
                     f"{len(pull['skipped_budget'])} skipped by credit budget, "
                     f"{len(pull['skipped_cap'])} by per-run cap; "
                     f"{pull['budget_remaining']:.0f} credits left this month.")
        if not prop_lines.empty:
            cands = candmod.enumerate_candidates(
                season, week, inputs=inputs,
                min_usage=(cfg.get("candidates") or {}).get("min_usage"),
                prop_lines=prop_lines, roster_mode=roster_mode)
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
        pack, adv, chem, ftn = _feature_packs(inputs)
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
        # measured second-order: backup QB -> pass-family efficiency x0.92;
        # skill-leader absence -> QB passing markets (absence matrix)
        cands = candmod.apply_backup_qb_adjustment(cands)
        cands = candmod.apply_absence_qb_adjustment(cands, inputs.pw, season, week, outs_now)

    # 4c. flag-gated ML ranking layer (see reports/ml_improvement_test.md)
    cands = _maybe_stamp_ml(cfg, cands, inputs)

    # 4. rank + report (context panel via synthesis on the ranked leans)
    result = rptmod.generate(
        season, week, inputs=inputs, prop_lines=prop_lines,
        synthesis_by_game=None, availability=statuses or None,
        clock=clock, mode=mode, publish=publish, publish_reasons=publish_reasons,
        write_files=False, persist=False, line_note=line_note,
        candidates_df=cands)

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
    rptmod.persist_leans(conn, season, week, clock, result["games"], result["as_of"])

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
            discord_dry_run: bool = True) -> Dict:
    cfg = cfgmod.load_config()
    conn = dbmod.connect()
    inputs = inputs or candmod.build_week_inputs()
    as_of = stamp_now()

    roster_mode = "as_played" if mode == "historical" else "carry_forward"
    cands = candmod.enumerate_candidates(
        season, week, inputs=inputs,
        min_usage=(cfg.get("candidates") or {}).get("min_usage"),
        roster_mode=roster_mode)
    cands = cands[cands["game_id"] == game_id].reset_index(drop=True)
    if cands.empty:
        conn.close()
        raise ValueError(f"no candidates for game {game_id} — check season/week/game_id")

    # stamp context/advanced features + ML so t90 leans carry the same
    # writeup facts and ranking as the Wednesday run
    if mode == "live" and not cands.empty:
        pack, adv, chem, ftn = _feature_packs(inputs)
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
        cands = candmod.apply_backup_qb_adjustment(cands)
    cands = _maybe_stamp_ml(cfg, cands, inputs)
    live = gather_live_feeds(cfg, season, week, _players_frame(cands), clock="t90",
                             inject=inject_feeds)
    statuses = live["statuses"]
    g = gate(live["feeds"], as_of=as_of,
             staleness_hours=(cfg.get("freshness") or {}).get("staleness_hours"))

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
    rptmod.persist_leans(conn, season, week, "t90", games, as_of)

    payload = {"season": season, "week": week, "clock": "t90", "as_of": as_of,
               "publish": g["publish"], "publish_reasons": g["reasons"],
               "mode": mode, "games": games, "contexts": contexts,
               "voided": voided, "md_path": md_path}
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
    resolved = clvmod.log_close_for_week(conn, season, week, kickoffs_for(slate))
    stats = clvmod.rolling_clv(conn)
    verdict = kcmod.report(conn)
    conn.close()
    return {"resolved": int(len(resolved)), "clv": stats, "killcheck": verdict}


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


if __name__ == "__main__":
    main()
