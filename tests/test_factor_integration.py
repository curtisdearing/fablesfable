"""Factor evidence through the real consumers: run_week -> DB -> pick_cards -> site HTML/JSON.

Labels must come from what the run executed and persisted (per-pick stage stamps + run
receipt), never from configuration or a feature list. Missing is never "no change"; the
shadow never touches the published pick; context is scoped to its game and its clock.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue import factor_evidence as fe  # noqa: E402
from nflvalue import factor_integration as fimod  # noqa: E402
from nflvalue import pick_cards as pc  # noqa: E402
from tests.test_pipeline_weekly import GAME_ID, _fresh_feeds, env  # noqa: E402,F401
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402

UTC = dt.timezone.utc
RAN = {s: True for s in fimod.STAGES}


def _labels(panel):
    return {it["factor_id"]: it for g in panel["groups"] for it in g["items"]}


# ---------------------------------------------------------------- per-row stamps

def test_row_stamps_distinguish_applied_no_change_missing_and_out_of_scope():
    row = {"market": "passing_yards", "qb_continuity": 0.3, "backup_qb_adj": 0.92,
           "realloc_mult": float("nan"), "absence_qb_mult": None}
    st = fimod.row_stage_stamps(row, RAN, {})
    assert st["backup_qb"] == {"state": "applied", "value": 0.92, "reason": None}
    assert st["realloc_volume"]["state"] == "no_change"
    assert st["absence_qb"]["state"] == "no_change"
    # the stage ran, but this row's input was missing -> not evaluated, not neutral
    st = fimod.row_stage_stamps({"market": "passing_yards", "qb_continuity": float("nan")}, RAN, {})
    assert st["backup_qb"]["state"] == "not_evaluated"
    assert "missing, not zero" in st["backup_qb"]["reason"]
    # out of scope for a rushing market
    st = fimod.row_stage_stamps({"market": "rushing_yards"}, RAN, {})
    assert st["backup_qb"]["state"] == "not_applicable" and st["absence_qb"]["state"] == "not_applicable"
    # a stage that did not run is not evaluated, with the run's reason
    st = fimod.row_stage_stamps({"market": "receptions"}, {**RAN, "realloc_volume": False},
                                {"realloc_volume": "availability statuses not evaluated this run"})
    assert st["realloc_volume"] == {"state": "not_evaluated", "value": None,
                                    "reason": "availability statuses not evaluated this run"}


def _lean(stamps=None, shadow=None, market="passing_yards"):
    return {"player_id": "P1", "game_id": "G1", "market": market, "run_id": "r1",
            "as_of": "2026-09-23T03:00:00Z",
            "stage_json": json.dumps(stamps) if stamps else None,
            "shadow_json": json.dumps(shadow) if shadow else None}


RECEIPT = {"run_id": "r1", "as_of": "2026-09-23T03:00:00Z", "component": "ff-football-only-v1",
           "stages_executed": ["absence_qb", "backup_qb", "dispersion", "game_script",
                               "realloc_efficiency", "realloc_volume"],
           "primary_margin_source": "neutral", "ordering_component": "ml_gbdt",
           "ordering_features_populated": ["qb_continuity", "temp"],
           "shadow": {"component": "role-opportunity-pooling-v1", "status": "ok"}}


def _stamps(**stage_states):
    st = {s: {"state": "no_change", "value": None, "reason": None} for s in fimod.STAGES}
    st.update(stage_states)
    return {"team": "ATL", "position": "QB", "stages": st, "margin_source": "neutral",
            "dispersion_role": "primary", "incumbent_volume": 33.1,
            "ordering": {"qb_continuity": 0.3, "temp": 60.0}}


def test_card_labels_follow_persisted_execution_not_configuration():
    lean = _lean(_stamps(backup_qb={"state": "applied", "value": 0.92, "reason": None},
                         absence_qb={"state": "not_evaluated", "value": None,
                                     "reason": "availability statuses not evaluated this run"}))
    lab = _labels(fimod.card_panel(lean, {"r1": RECEIPT}, {}))
    assert lab["backup_qb"]["status"] == "numeric_applied"
    assert lab["backup_qb"]["status_label"] == "Used in projection: x0.92 on the projected mean (as executed)"
    assert lab["realloc_volume"]["status"] == "considered_no_change"
    assert lab["absence_qb"]["status"] == "unavailable_unverified"
    assert "availability statuses not evaluated" in lab["absence_qb"]["detail"]
    assert lab["dispersion"]["status"] == "numeric_applied"        # used; contribution not isolated
    assert lab["dispersion"]["status_label"] == fe.NOT_ISOLATED
    assert lab["game_script"]["status"] == "shadow_only"           # neutral primary, football margin shadow
    # ranking-only inputs are labelled as ordering, never as changing the projection
    assert lab["ordering:qb_continuity"]["status_label"] == fe.ORDERING_ONLY_LABEL
    assert lab["ordering:temp"]["status"] == "context_only"


def test_missing_stamps_or_receipt_are_never_no_change():
    for lean, receipts in ((_lean(None), {"r1": RECEIPT}), (_lean(_stamps()), {})):
        lab = _labels(fimod.card_panel(lean, receipts, {}))
        assert lab["model_stages"]["status"] == "unavailable_unverified"
        assert not any(i["status"] == "considered_no_change" for i in lab.values())


SHADOW = {"component": "role-opportunity-pooling-v1", "conserved": False, "regime": "same_team",
          "expected": {"expected_targets": None, "expected_carries": None, "expected_pass_attempts": 31.4},
          "share": {"pass_share": {"prior_estimate": 0.97, "prior_source": "historical_prior_prev_season",
                                   "prior_games": 17, "current_estimate": 0.5, "current_games": 2,
                                   "current_denominator": 60, "k": 67.5, "w_current": 0.47,
                                   "posterior": 0.75, "regime": "same_team"}}}


def test_shadow_is_labelled_shadow_with_its_support_and_prior_source():
    lab = _labels(fimod.card_panel(_lean(_stamps(), SHADOW), {"r1": RECEIPT}, {}))
    sh = lab["shadow:pass_share"]
    assert sh["status"] == "shadow_only"
    assert sh["status_label"] == fe.STATUS_LABELS["shadow_only"]
    assert "31.4" in sh["observation"] and "previous-season history" in sh["observation"]
    assert "weight on this season 0.47" in sh["observation"] and sh["support"] == "2 games this season"
    assert lab["participation:snaps_routes"]["status"] == "unavailable_unverified"
    # no shadow persisted -> shown as missing, with the run's reason
    lab = _labels(fimod.card_panel(_lean(_stamps()), {"r1": RECEIPT}, {}))
    assert lab["shadow:role_opportunity"]["status"] == "unavailable_unverified"


# ---------------------------------------------------------------- sourced context

def test_context_respects_game_scope_cutoff_supersession_and_missing_games(tmp_path):
    ctx = json.load(open(fimod.context_path(2026, 3)))
    p = tmp_path / "ctx.json"
    p.write_text(json.dumps(ctx))
    late = fimod.load_context(2026, 3, ["2026_03_ATL_GB", "2026_03_XXX_YYY"],
                              "2026-09-23T03:00:00Z", path=str(p))
    by = {r["factor_id"]: r for r in late["records"]}
    assert by["news:atl_terrell_ir"]["status"] == "context_only" and by["news:atl_terrell_ir"]["verified"]
    assert by["news:gb_ol_practice_0921"]["status"] == "unavailable_unverified"   # superseded by Sept 22
    assert "superseded" in by["news:gb_ol_practice_0921"]["reason_not_applied"]
    assert by["news:atl_penix_named_starter"]["verified"]
    assert by["context_not_collected:2026_03_XXX_YYY"]["status"] == "unavailable_unverified"
    assert by[f"game_status_designations:2026_03_ATL_GB"]["status"] == "unavailable_unverified"
    # nothing from this file may assert GB had no Week 3 report
    assert "no published report" not in json.dumps(late["records"])
    # before the Terrell article was published it cannot be used
    early = fimod.load_context(2026, 3, ["2026_03_ATL_GB"], "2026-09-22T12:00:00Z", path=str(p))
    t = {r["factor_id"]: r for r in early["records"]}["news:atl_terrell_ir"]
    assert t["status"] == "unavailable_unverified" and not t["cutoff_ok"]
    # a different season/week file is never applied to this week
    other = fimod.load_context(2026, 4, ["2026_03_ATL_GB"], "2026-09-23T03:00:00Z", path=str(p))
    assert other["path"] is None and other["games_with_context"] == []


def test_context_selects_player_team_and_game_records_for_a_card():
    ctx = fimod.load_context(2026, 3, ["2026_03_ATL_GB"], "2026-09-23T03:00:00Z")["records"]
    reed = {"player_id": "00-0039146", "game_id": "2026_03_ATL_GB", "team": "GB"}
    ids = {r["factor_id"] for r in fe.select_for_card(ctx, reed)}
    assert "news:gb_reed_practice_0922" in ids and "news:atl_terrell_ir" in ids
    assert "news:atl_penix_named_starter" not in ids                 # ATL team news, GB card
    atl = {r["factor_id"] for r in fe.select_for_card(ctx, {"player_id": "X", "game_id": "2026_03_ATL_GB",
                                                            "team": "ATL"})}
    assert "news:atl_penix_named_starter" in atl and "news:gb_reed_practice_0922" not in atl
    # practice participation is never presented as a game ruling or as a numeric input
    for r in ctx:
        assert r["status"] in ("context_only", "unavailable_unverified")
        assert r.get("numerical_effect") is None


# ---------------------------------------------------------------- real pipeline

def test_run_week_persists_receipt_stamps_and_cards_render_panels(env):
    from nflvalue.freshness import stamp_now
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_fresh_feeds(stamp_now()))
    conn = dbmod.connect()
    rec = conn.execute("SELECT receipt_json FROM run_receipts").fetchall()
    assert len(rec) == 1
    receipt = json.loads(rec[0][0])
    assert {"backup_qb", "realloc_volume", "absence_qb"} <= set(receipt["stages_executed"])
    assert receipt["shadow"]["status"]                                  # ok or an explicit reason
    assert receipt["context"]["games_with_context"] == []               # no file entry for this game
    leans = dbmod.query_df(conn, "SELECT * FROM leans WHERE clock='wed'")
    assert len(leans) and leans["stage_json"].notna().all()
    payload = pc.week_cards(conn, SEASON, WEEK)
    for c in payload["cards"]:
        lab = _labels(c["factor_panel"])
        assert f"context_not_collected:{GAME_ID}" in lab
        assert "model_stages" not in lab                                # stamps were found
        assert "participation:snaps_routes" in lab
    html = pc.render_cards_html(payload["cards"])
    assert html.count('class="fe-panel"') == len(payload["cards"])
    assert res["factor_receipt"]["run_id"] == receipt["run_id"]


def test_shadow_output_cannot_change_the_published_pick(env, monkeypatch):
    from nflvalue.freshness import stamp_now

    def published(shadow_value):
        monkeypatch.setattr(fimod, "shadow_opportunity", lambda pw_, cands, **k: {
            "status": "ok", "component": "x", "players": {
                pid: {"expected": {"expected_targets": shadow_value}, "share": {}}
                for pid in cands["player_id"]}})
        pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                    inject_feeds=_fresh_feeds(stamp_now()))
        conn = dbmod.connect()
        df = dbmod.query_df(conn, "SELECT player_id, market, side, line, mean, sd, p_side, composite "
                                  "FROM leans ORDER BY rowid")
        conn.close()
        return df

    a, b = published(0.0), published(999.0)
    pd.testing.assert_frame_equal(a, b)


def test_stamps_and_shadow_targets_are_invariant_to_price_spread_total_and_consensus(pbp_fast,
                                                                                     schedules_fast):
    from tests import test_football_only_forecast as t
    pbp = pbp_fast[(pbp_fast["season"] < t.SEASON)
                   | ((pbp_fast["season"] == t.SEASON) & (pbp_fast["week"] < t.WEEK))]
    inputs = t.WeekInputs(pw=t.build_player_week(pbp), opd=t.build_opp_pos_def(pbp),
                          tw=t.build_team_week(pbp), schedules=schedules_fast.copy())
    base = t._run(inputs)
    a = t._run(inputs, prop_lines=t._lines(base, 1.91, 1.91, 0.50)).reset_index()
    b = t._run(inputs, schedules=t._perturbed_schedule(inputs.schedules),
               prop_lines=t._lines(base, 1.30, 3.40, 0.15)).reset_index()
    assert not a["total_line"].equals(b["total_line"])                  # the market really moved
    sa, sb = fimod.build_stamps(a, RAN, {}), fimod.build_stamps(b, RAN, {})
    assert sa and sa == sb


def test_shadow_runs_on_real_enumerated_candidates(pbp_fast, schedules_fast):
    """Regression (found in the final replay): candidates carry position as ``pos``; the
    shadow silently produced zero players. It must forecast real candidate players."""
    from tests import test_football_only_forecast as t
    pbp = pbp_fast[(pbp_fast["season"] < t.SEASON)
                   | ((pbp_fast["season"] == t.SEASON) & (pbp_fast["week"] < t.WEEK))]
    pwk = t.build_player_week(pbp)
    inputs = t.WeekInputs(pw=pwk, opd=t.build_opp_pos_def(pbp), tw=t.build_team_week(pbp),
                          schedules=schedules_fast.copy())
    cands = t._run(inputs).reset_index()
    as_of = dt.datetime(t.SEASON, 1, 1, tzinfo=UTC)
    far = {g: dt.datetime(t.SEASON + 1, 1, 1, tzinfo=UTC) for g in cands["game_id"].unique()}
    before = cands.copy()
    res = fimod.shadow_opportunity(pwk, cands, season=t.SEASON, week=t.WEEK, as_of=as_of, kickoffs=far)
    assert res["status"] == "ok" and len(res["players"]) > 0
    pd.testing.assert_frame_equal(cands, before)                      # shadow never mutates the pick frame
    stamps = fimod.build_stamps(cands, RAN, {})
    assert all(s["position"] in ("QB", "RB", "WR", "TE") for s in stamps.values())
