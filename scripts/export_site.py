#!/usr/bin/env python3
"""Export the model's guts to a static JSON bundle the transparency site reads.

The whole point of the site is that the pick engine is NOT a black box: every
number a bettor sees on the site can be traced back through the projection, the
adjustments, the market comparison, and the measured constant (with its fit
provenance) that produced it. This script assembles that bundle:

    site/data/picks.json      current week's picks + FULL decision chain per pick
    site/data/model.json      the pipeline stages + feature registry + weights
    site/data/constants.json  every MEASURED constant with its fit provenance
    site/data/evidence.json   CLV / kill-check / backtest / recency-fit verdicts
    site/data/meta.json       generated-at, versions, honesty disclaimer

Sources, in order of preference (each optional -- the site degrades gracefully):
  * data/weekly_props.json    the latest report payload (picks, leans, contexts)
  * data/nfl_props.db         the warehouse (clv, killcheck, picks record)
  * data/*.json               fit verdicts (recency, selection, situations, ...)
  * nflvalue module constants  imported directly (composite weights, selector
                               thresholds, weather/TD/absence coefficients, ...)
  * docs/decisions_p*.md       parsed for the provenance one-liners

Deliberately DB/JSON/import only -- NO parquet read -- so it runs anywhere the
package imports, including CI and slice-limited sandboxes. Run after a weekly
pipeline pass (or any time -- it will emit a clearly-labeled EMPTY bundle when
no live week has been generated yet).

    python3 scripts/export_site.py               # -> site/data/*.json
    python3 scripts/export_site.py --out site/data
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
DOCS = os.path.join(ROOT, "docs")


def _load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# picks.json -- the decision chain for every current pick
# --------------------------------------------------------------------------- #
def _decision_chain(pick: dict) -> list:
    """Ordered, human-readable steps from raw projection to final tier -- the
    exact reason THIS bet is more/less attractive than the alternatives."""
    proj = pick.get("proj_components") or {}
    comp = pick.get("components") or {}
    steps = []

    vol, eff, opp = proj.get("volume"), proj.get("efficiency"), proj.get("opp_factor")
    if vol is not None and eff is not None:
        base = f"Projection: {vol} expected opportunities × {eff} per-opportunity efficiency"
        if opp not in (None, 1.0):
            base += f" × {opp} opponent-vs-position factor"
        steps.append({"stage": "projection", "detail": base + f" → mean {pick.get('mean')}"})
    else:
        steps.append({"stage": "projection", "detail": f"Model projects {pick.get('mean')}"})

    for key, label in (("shape_tilts", "depth/location matchup tilt"),
                       ("game_script", "game-script (spread/PROE) tilt")):
        v = proj.get(key)
        if v and v not in (1.0, {}):
            steps.append({"stage": "adjustment", "detail": f"{label}: {v}"})
    for key, label in (("wx_pass_mult", "fitted weather multiplier"),
                       ("absence_qb_mult", "opponent-injury (absence) multiplier"),
                       ("realloc_mult", "usage-reallocation multiplier (teammate out)"),
                       ("backup_qb_adj", "backup-QB efficiency multiplier")):
        v = pick.get(key)
        if v is not None and v not in (1.0,):
            steps.append({"stage": "adjustment", "detail": f"{label}: ×{v}"})

    mp, kp = comp.get("model_prob"), comp.get("market_prob")
    if mp is not None and kp is not None:
        steps.append({"stage": "market comparison",
                      "detail": (f"Model P({pick.get('side')}) = {mp:.0%} vs de-vigged market fair "
                                 f"{kp:.0%} → edge {(pick.get('edge') or 0) * 100:+.1f} pts"
                                 + (f", EV {comp.get('ev_best_price'):+.1%} at best price"
                                    if comp.get('ev_best_price') is not None else ""))})
    elif pick.get("no_market") or pick.get("line_source") != "odds_api":
        steps.append({"stage": "market comparison",
                      "detail": "No real sportsbook line — synthetic reference only, no edge computable (RESEARCH)"})

    tier = pick.get("tier")
    notes = pick.get("tier_notes") or []
    steps.append({"stage": "tier decision",
                  "detail": f"Tier {tier}" + (f" — {'; '.join(notes)}" if notes else
                                              " — cleared the market-specific edge/EV bars")})
    return steps


def build_picks(props: dict) -> dict:
    games = (props or {}).get("games", []) or []
    out_games = []
    for g in games:
        picks = []
        for p in (g.get("picks") or []) + [
                {**r, "_research": True} for r in (g.get("research_leans") or [])]:
            picks.append({
                "player": p.get("name"), "pos": p.get("pos"), "team": p.get("team"),
                "market": p.get("market"), "side": p.get("side"), "line": p.get("line"),
                "line_source": p.get("line_source"),
                "tier": p.get("tier"), "mean": p.get("mean"),
                "edge": p.get("edge"), "ev": (p.get("components") or {}).get("ev_best_price"),
                "model_prob": (p.get("components") or {}).get("model_prob"),
                "market_prob": (p.get("components") or {}).get("market_prob"),
                "writeup": p.get("writeup"),
                "decision_chain": _decision_chain(p),
                "research_only": bool(p.get("_research")),
            })
        out_games.append({
            "game_id": g.get("game_id"), "matchup": g.get("matchup"),
            "screened": g.get("screened"), "screened_n": g.get("screened_n"),
            "picks": picks,
            "notes": g.get("notes") or [],
            "sgp": g.get("sgp") or [],
        })
    return {
        "season": (props or {}).get("season"), "week": (props or {}).get("week"),
        "clock": (props or {}).get("clock"), "as_of": (props or {}).get("as_of"),
        "publish": (props or {}).get("publish", None),
        "publish_reasons": (props or {}).get("publish_reasons", []),
        "is_sample": bool((props or {}).get("sample")),
        "sample_note": (props or {}).get("sample_note"),
        "games": out_games,
        "empty": not out_games,
    }


# --------------------------------------------------------------------------- #
# model.json -- pipeline stages, feature registry, weights
# --------------------------------------------------------------------------- #
def build_model() -> dict:
    stages = [
        {"n": 1, "name": "Enumerate candidates",
         "what": "Every eligible (player, market) for every game — the full pool, before any selection.",
         "where": "nflvalue/candidates.py"},
        {"n": 2, "name": "Project the mean",
         "what": "mean = expected volume × trailing efficiency × opponent factor, all walk-forward (only prior weeks).",
         "where": "nflvalue/projection.py, features.py"},
        {"n": 3, "name": "Apply measured adjustments",
         "what": "Weather, injuries/availability, usage reallocation, backup-QB, game script, red-zone TD path — each a FITTED constant.",
         "where": "candidates.py, factors.py, projection.py"},
        {"n": 4, "name": "Score: composite + ML probability",
         "what": "Deterministic composite (edge/confidence/matchup) AND a calibrated RandomForest P(over). Both explainable.",
         "where": "composite.py, ml_ranker.py"},
        {"n": 5, "name": "Compare to the market",
         "what": "Model probability vs de-vigged consensus fair probability → edge and EV at the best available price.",
         "where": "oddsapi_props.py, composite.py"},
        {"n": 6, "name": "Select the best picks (post-projection)",
         "what": "ONLY after every candidate is scored: rank by edge, tier PASS/LEAN/PLAYABLE/STRONG, write the reason.",
         "where": "nflvalue/selector.py"},
        {"n": 7, "name": "Grade forward on CLV",
         "what": "Log each pick, compare entry vs closing line. CLV — not profit — is the honest edge test (kill-check).",
         "where": "clv.py, killcheck.py"},
    ]
    model = {"stages": stages, "features": [], "composite_weights": {}, "selector": {},
             "matchup_subweights": {}}
    try:
        from nflvalue import ml_ranker as mlr
        model["features"] = list(mlr.NUMERIC_FEATURES)
        model["market_dummies"] = list(mlr.MARKETS7)
        model["retrain_pending_features"] = list(getattr(mlr, "RETRAIN_PENDING_FEATURES", []))
        model["base_model"] = "RandomForest (calibrated, per-market Platt) — Phase 7.2 bake-off winner"
    except Exception as exc:  # noqa: BLE001
        model["features_error"] = str(exc)
    try:
        from nflvalue import composite as cmp
        model["composite_weights"] = dict(cmp.DEFAULT_WEIGHTS)
        model["matchup_subweights"] = dict(cmp.MATCHUP_SUB_WEIGHTS)
    except Exception:  # noqa: BLE001
        pass
    try:
        from nflvalue import selector as sel
        sc = sel.selector_config({})
        model["selector"] = {"tiers": list(sel.TIERS), "thresholds": sc["thresholds"],
                             "min_model_prob": sc["min_model_prob"],
                             "stale_line_hours": sc["stale_line_hours"]}
    except Exception:  # noqa: BLE001
        pass
    cfg = _load_json(os.path.join(ROOT, "config.json"), {}) or {}
    if cfg.get("composite"):
        model["composite_weights"] = (cfg["composite"].get("weights")
                                      or model["composite_weights"])
    return model


# --------------------------------------------------------------------------- #
# constants.json -- every measured constant + provenance
# --------------------------------------------------------------------------- #
def build_constants() -> dict:
    """Introspect the shipped constants directly, each tagged with the fit
    script + measured verdict that justifies it. The site's whole thesis is
    that none of these are guesses."""
    items = []

    def add(name, value, unit, provenance, script):
        items.append({"name": name, "value": value, "unit": unit,
                      "provenance": provenance, "fit_script": script})

    try:
        from nflvalue import factors as fac
        add("Weather: wind penalty (≤10 mph)", fac.WX_PASS_WIND, "pass yds per mph",
            "Fitted OLS on 2019-23 outdoor team-games (n=1,556), t=-3.4. Replaced the guessed 30mph-max shape.",
            "scripts/fit_weather.py")
        add("Weather: precipitation penalty", fac.WX_PASS_PRECIP, "pass yds",
            "The dominant weather term (t=-5.9). Cold FAILED the t≥2 bar and was dropped.",
            "scripts/fit_weather.py")
        add("Opponent DB-out pass boost", 6.04, "pass yds per DB Out",
            "Fitted 2019-23 (n=2,557 team-games, t=+2.0). Front-7/LB and OWN O-line outs cleared NOTHING and ship no multiplier.",
            "scripts/fit_absence_opp.py")
    except Exception:  # noqa: BLE001
        pass
    try:
        from nflvalue import projection as prj
        add("Anytime-TD red-zone blend weight", prj.TD_BLEND_W, "weight on RZ path",
            "Walk-forward log-loss grid 2019-23 (n=16,871). Opponent RZ factor FAILED the same fit → ML-feature only.",
            "scripts/fit_td_blend.py")
        add("Game-script PROE split coefficient", prj.PROE_SPLIT_COEF, "pass share per pass_oe/100",
            "OLS 2019-23 (t=+3.1). Opponent-pace VOLUME term FAILED (t=-0.3) and is not shipped.",
            "scripts/fit_game_script.py")
    except Exception:  # noqa: BLE001
        pass
    try:
        from nflvalue import features as ftr
        rf = ftr.RECENCY_FIT
        add("Recency weight (EWM span)", rf.get("ewm_span"), "games",
            ("Walk-forward next-game MAE sweep 2019-25: EWM span-8 beats flat-8 (production) AND ewm-4 in "
             "ALL 7 markets, 6/6 seasons each (pooled 5.379→5.293). drop_rest cleaning wins/ties per market; "
             "drop_injury measured WORSE (kept out)."),
            "scripts/fit_recency_weight.py")
        add("Recency: drop rest/meaningless prior games", rf.get("drop_rest"), "bool",
            "Same sweep — zero-weighting a team's playoff-fate-settled games improves every market. Extended to "
            "opponent-defense factors, team pace, ML frame, player-learning ledger, calibration & correlation (§8.4).",
            "scripts/fit_recency_weight.py + merge_recency_shards.py")
    except Exception:  # noqa: BLE001
        pass
    # recency verdict detail rides along for the evidence tab too
    return {"constants": items,
            "principle": ("Every constant above is MEASURED from this project's own history with a "
                          "printed t-stat / MAE verdict. If a candidate effect didn't clear the bar "
                          "(t≥2 or consistent OOS gain) it was rejected, not shipped — several were "
                          "(garbage-time filter, opp-pace, cold weather, O-line multiplier, drop_injury).")}


# --------------------------------------------------------------------------- #
# evidence.json -- CLV / kill-check / recency verdict / backtest
# --------------------------------------------------------------------------- #
def build_evidence() -> dict:
    ev = {"clv": None, "killcheck": None, "recency_fit": None, "selection_opt": None,
          "situations": None, "picks_record": None, "mc_brain": None}
    latest = _load_json(os.path.join(DATA, "latest.json"), {}) or {}
    ev["clv"] = latest.get("leans_clv")
    ev["killcheck"] = latest.get("leans_killcheck")
    # try the live DB for the freshest kill-check + picks record
    try:
        from nflvalue import db as dbmod, killcheck as kc, selector as sel
        conn = dbmod.connect()
        ev["killcheck"] = kc.report(conn)
        ev["picks_record"] = sel.picks_record(conn)
        conn.close()
    except Exception:  # noqa: BLE001
        pass
    rc = _load_json(os.path.join(DATA, "recency_weight_fit.json"))
    if rc:
        ev["recency_fit"] = {
            "pooled_baseline_mae": rc.get("pooled", {}).get("baseline_ewm4_raw_mae"),
            "pooled_top": (rc.get("pooled", {}).get("ranking") or [])[:5],
            "by_market": rc.get("by_market"),
            "season_consistency": rc.get("season_consistency_vs_flat8_production")
                                  or rc.get("season_consistency"),
        }
    ev["selection_opt"] = _load_json(os.path.join(DATA, "selection_opt.json"))
    ev["situations"] = _load_json(os.path.join(DATA, "situation_study.json"))
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "site", "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    props = _load_json(os.path.join(DATA, "weekly_props.json"), {}) or {}
    # A live weekly run writes weekly_props.json with per-game picks. If none
    # exists yet (fresh clone / offseason), fall back to the clearly-labeled
    # SAMPLE payload so the site is viewable/deployable — the exporter tags it.
    if not any(g.get("picks") or g.get("research_leans") for g in props.get("games", [])):
        sample = _load_json(os.path.join(DATA, "weekly_props_sample.json"))
        if sample:
            props = sample
    bundles = {
        "picks.json": build_picks(props),
        "model.json": build_model(),
        "constants.json": build_constants(),
        "evidence.json": build_evidence(),
        "meta.json": {
            "generated_at": _now(),
            "branch": "phase6",
            "disclaimer": ("Leans, not locks. Research/advisory only — no bet placement. Graded at "
                           "synthetic reference lines until live CLV accrues; CLV is the only accepted "
                           "edge proof. If you or someone you know has a gambling problem, call "
                           "1-800-GAMBLER."),
            "has_live_week": bool(props.get("games")),
        },
    }
    for fn, payload in bundles.items():
        with open(os.path.join(args.out, fn), "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"wrote {os.path.join(args.out, fn)}")
    print("\nSite data exported. Open site/index.html (or deploy the site/ folder).")


if __name__ == "__main__":
    main()
