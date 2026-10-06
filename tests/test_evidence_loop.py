"""Evidence-loop contract tests (analysis-only; fixtures here are never empirical evidence)."""
import copy
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from analysis import evidence_loop as el

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _row(eid="r1", season=2026, week=5, decided="2026-10-11T15:00:00+00:00", kick="2026-10-11T17:00:00Z",
         outcome="win", p=0.6, price=1.91, capture="c" * 64, cls="recommendation", game="g1", market="passing_yards"):
    # trusted capture = ledger receipt digest + ledger clock before kickoff (see el.trusted_capture)
    return {"evidence_id": eid, "season": season, "week": week, "game_id": game, "market": market, "side": "over",
            "line": 250.5, "issued_class": cls, "decision_ts": decided, "kickoff": kick, "capture_sha256": capture,
            "capture_recorded_at": decided, "capture_basis": "ledger_pre_kickoff_event",
            "outcome": outcome, "price_decimal": price, "p": {"issued": p}}


# ------------------------------------------------------------- chronology --
def test_week_1_to_4_is_never_prospective_even_with_clean_clock_and_capture():
    for week in (1, 2, 3, 4):
        assert el.window_of(_row(week=week)) == "retrospective_exploratory"
    assert el.window_of(_row(season=2025, week=17)) == "retrospective_exploratory"
    assert el.window_of(_row(week=5)) == "prospective_confirmation"


def test_prospective_needs_capture_and_strictly_pre_kickoff_explicit_offset_clock():
    assert el.window_of(_row(capture=None)) == "retrospective_exploratory"
    assert el.window_of(_row(decided="2026-10-11T17:00:00+00:00")) == "excluded_late"
    assert el.window_of(_row(decided="2026-10-11T18:00:00+00:00")) == "excluded_late"
    assert el.window_of(_row(decided="2026-10-11T15:00:00")) == "unknown_clock"  # naive: never assumed UTC
    assert el.window_of(_row(decided=None)) == "unknown_clock"
    assert el.window_of(_row(kick=None)) == "unknown_clock"


def test_many_already_read_rows_cannot_fill_the_prospective_gate():
    rows = [_row(eid=str(i), week=1 + i % 4) for i in range(5000)]
    assert sum(el.window_of(r) == "prospective_confirmation" for r in rows) == 0


# --------------------------------------------------------------- coverage --
def test_coverage_counts_every_row_once_and_scores_only_settled_recommendations():
    rows = [_row("a"), _row("b", outcome="loss"), _row("c", outcome="push"), _row("d", outcome="void"),
            _row("e", outcome="pending"), _row("f", p=None), _row("g", p=1.3), _row("h", cls="pass")]
    cov = el.coverage(rows, ["issued"])
    assert cov["rows_received"] == 8 and sum(cov["by_status"].values()) == 8
    assert cov["by_status"] == {"scored": 2, "push": 1, "void": 1, "pending": 1, "missing_probability": 2,
                                "not_in_scope": 1}
    assert sum(cov["by_window"].values()) == 8
    assert [r["evidence_id"] for r in el.scored_rows(rows, ["issued"])] == ["a", "b"]


# ---------------------------------------------------------------- scoring --
def test_proper_scores_and_cluster_interval_rules():
    rows = [_row("a", p=0.8), _row("b", outcome="loss", p=0.3)]
    s = el.score_arms(rows, ["issued"])
    arm = s["arms"]["issued"]
    assert arm["brier"]["point"] == pytest.approx(((0.8 - 1) ** 2 + 0.3 ** 2) / 2)
    assert arm["logloss"] == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)
    assert arm["brier"]["low"] is None and "unavailable" in arm["brier"]["method"]  # 1 game < 5 clusters
    assert arm["account_pl"] is None
    many = [_row(str(i), p=0.7, outcome="win" if i % 2 else "loss", game=f"g{i}") for i in range(40)]
    b = el.score_arms(many, ["issued"], n_boot=500)["arms"]["issued"]["brier"]
    assert b["n_clusters"] == 40 and b["low"] <= b["point"] <= b["high"]


def test_paired_delta_is_clustered_by_game_not_row():
    rows = []
    for g in range(10):  # 10 games x 30 rows: rows inside a game are not independent
        for i in range(30):
            r = _row(f"{g}-{i}", game=f"g{g}", outcome="win" if g % 2 else "loss")
            r["p"] = {"base": 0.5, "chal": 0.55 if g % 2 else 0.45}
            rows.append(r)
    d = el.score_arms(rows, ["base", "chal"], baseline="base", n_boot=500)["arms"]["chal"]["vs_base"]
    assert d["d_brier_game_cluster"]["n_clusters"] == 10
    assert d["d_brier_game_cluster"]["point"] < 0 and d["d_logloss"] < 0


def test_roi_requires_a_real_price_receipt_and_is_not_account_pl():
    assert el.flat_roi([_row(price=None)], "issued")["roi"] is None
    won = el.flat_roi([_row(p=0.6, price=1.8)], "issued")
    assert (won["bets"], won["units"]) == (1, pytest.approx(0.8))
    lost = el.flat_roi([_row(p=0.6, price=1.8, outcome="loss")], "issued")
    assert lost["units"] == pytest.approx(-1.0) and "not account P/L" in lost["status"]
    below = {**_row(p=0.5, price=1.8), "issued_class": "candidate_event"}
    assert el.flat_roi([below], "issued")["bets"] == 0  # candidate event below breakeven: no bet
    assert el.american_to_decimal(-113) == pytest.approx(1 + 100 / 113)
    assert el.american_to_decimal(150) == pytest.approx(2.5)
    assert el.american_to_decimal(50) is None and el.american_to_decimal(None) is None


# --------------------------------------------------------------- adapters --
def test_card_grade_pass_is_counted_not_scored_and_chat_clock_is_unknown():
    card = {"rows": [
        {"pick_id": "w", "category": "player prop", "side": "Over", "line": 258.5, "price": -113, "book": "FD",
         "issued_status": "lean", "raw_probability": 0.57, "actual": 286, "count_in_recommendation_record": True,
         "threshold_outcome": "WIN", "result": "WIN"},
        {"pick_id": "p", "category": "PASS", "side": "Over", "line": 46.5, "price": -113, "book": "FD",
         "issued_status": "PASS", "raw_probability": 0.50, "actual": 59, "count_in_recommendation_record": False,
         "threshold_outcome": "WIN", "result": "PASS"}]}
    rows = el.from_card_grade(card, 2026, 4, "2026_04_ATL_NO", "2026-10-06T00:15:00Z")
    assert [r["issued_class"] for r in rows] == ["recommendation", "pass"]
    assert all(el.window_of(r) == "unknown_clock" for r in rows)
    assert el.coverage(rows, ["issued"])["by_status"]["not_in_scope"] == 1
    assert len(el.scored_rows(rows, ["issued"])) == 1


def test_settlement_adapter_maps_issued_grading_rows():
    out = {"sections": {"recommendations_given": {"rows": [
        {"record_id": "rid1", "season": 2026, "week": 6, "game_id": "2026_06_X_Y", "market": "rushing_yards",
         "side": "under", "line": 60.5, "pick_class": "recommendation", "tier": "primary",
         "decision_ts": "2026-10-18T15:00:00+00:00", "kickoff": "2026-10-18T17:00:00Z", "settlement": "loss",
         "actual": 75, "quote_price": 1.87, "quote_book": "dk", "model_p_side": 0.56,
         "capture_receipt_sha256": "d" * 64, "first_seen_in_ledger": "2026-10-18T15:00:01+00:00"},
        {"record_id": "rid2", "season": 2026, "week": 6, "game_id": "2026_06_X_Y", "market": "receptions",
         "side": "over", "line": 4.5, "pick_class": "watch", "settlement": "unresolved", "model_p_side": 0.5}]}}}
    rows = el.from_issued_grading(out)
    assert rows[0]["outcome"] == "loss" and rows[0]["price_decimal"] == 1.87
    assert el.window_of(rows[0]) == "prospective_confirmation"
    assert rows[0]["capture_sha256"] == "d" * 64 != rows[0]["record_id"]
    # the record's content id alone is no capture receipt
    bare = el.from_issued_grading({"sections": {"recommendations_given": {"rows": [
        {k: v for k, v in out["sections"]["recommendations_given"]["rows"][0].items()
         if k not in ("capture_receipt_sha256", "first_seen_in_ledger")}]}}})
    assert el.window_of(bare[0]) == "retrospective_exploratory"
    assert rows[1]["issued_class"] == "watch" and rows[1]["outcome"] == "pending"


# ------------------------------------------------------- append-only store --
def test_ledger_is_idempotent_appends_revisions_and_detects_tampering(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    assert el.append_evidence(path, [_row("a"), _row("b")])["appended"] == 2
    assert el.append_evidence(path, [_row("a"), _row("b")]) == {"appended": 0, "skipped_identical": 2, "revisions": 0}
    rev = _row("a", outcome="loss")  # stat correction: appended, never edited in place
    assert el.append_evidence(path, [rev]) == {"appended": 1, "skipped_identical": 0, "revisions": 1}
    entries = el.read_ledger(path)
    assert len(entries) == 3 and entries[0]["row"]["outcome"] == "win"
    assert {r["evidence_id"]: r["outcome"] for r in el.latest_rows(entries)} == {"a": "loss", "b": "win"}
    lines = open(path).read().splitlines()
    lines[0] = lines[0].replace('"outcome":"win"', '"outcome":"loss"')
    open(path, "w").write("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="chain broken"):
        el.read_ledger(path)


# --------------------------------------------------------------- registry --
def _registry():
    with open(os.path.join(ROOT, "analysis", "research_registry.json")) as fh:
        return json.load(fh)


def test_committed_registry_is_frozen_and_within_active_limit():
    reg = _registry()
    assert el.validate_registry(reg) == []
    assert sum(c["status"] == el.ACTIVE for c in reg["candidates"]) <= el.MAX_ACTIVE
    assert {c["candidate_id"] for c in reg["candidates"]} >= {"C2-allocation-aware-sd", "QBROLE-TOTAL-v1",
                                                                "OPPFORM-TOTAL-v1", "C1m-opportunity-state"}


def test_registry_rejects_post_freeze_criteria_edits_third_active_and_unevidenced_rejection():
    reg = _registry()
    edited = copy.deepcopy(reg)
    edited["candidates"][-1]["criteria"]["screening_gate"]["alpha_familywise"] = 0.2
    assert any("criteria changed" in e for e in el.validate_registry(edited))
    crowded = copy.deepcopy(reg)
    for c in crowded["candidates"][-2:]:
        c["status"] = el.ACTIVE
    assert any("max 2" in e for e in el.validate_registry(crowded))
    bare = copy.deepcopy(reg)
    bare["candidates"].append({"candidate_id": "x", "status": "rejected_as_implemented"})
    assert any("without retained evidence" in e for e in el.validate_registry(bare))
    status_only = copy.deepcopy(reg)
    status_only["candidates"][-1]["status"] = "retired_retrospective_fail"  # outcome log, not a criteria edit
    assert el.validate_registry(status_only) == []


def test_screen_gates_sample_sign_and_holm():
    crit = {"A": {"screening_gate": {"min_eval_graded": 200, "min_eval_exposed": 50, "alpha_familywise": 0.05,
                                     "expected_beta_sign": -1}}}
    crit["B"] = copy.deepcopy(crit["A"])
    good = {"n": 300, "n_exposed": 80, "d_brier_game_cluster": {"p_ge_0": 0.001},
            "d_brier_week_cluster": {"high": -0.001}, "d_logloss": -0.01}
    out = el.screen({"A": {"beta_dev": -0.05, "eval": good}, "B": {"beta_dev": 0.05, "eval": good}}, crit)
    assert out["A"]["verdict"] == "eligible_for_prospective" and out["A"]["holm_adjusted_p"] == pytest.approx(0.002)
    assert out["B"]["verdict"] == "retired_retrospective_fail" and not out["B"]["checks"]["S4_mechanism_sign"]
    thin = el.screen({"A": {"beta_dev": -0.05, "eval": {**good, "n_exposed": 10}}}, {"A": crit["A"]})
    assert thin["A"]["verdict"] == "insufficient_data"
    assert el.holm({"x": 0.01, "y": 0.04, "z": None}) == {"x": 0.02, "y": 0.04, "z": None}


# ------------------------------------------- challenger signals: no leakage --
def _write_hist(d, target_scores=(17, 24), week3_leader="QA2", injured=False):
    games = pd.DataFrame([
        ("2024_01_A_B", 2024, "REG", 1, "2024-09-08", "13:00", "A", 10, "B", 20, 44.5),
        ("2024_02_A_C", 2024, "REG", 2, "2024-09-15", "13:00", "A", 27, "C", 24, 41.5),
        ("2024_03_B_A", 2024, "REG", 3, "2024-09-22", "16:25", "B", target_scores[0], "A", target_scores[1], 43.0)],
        columns=["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "away_score",
                 "home_team", "home_score", "total_line"])
    games["total"] = games.away_score + games.home_score
    games.to_parquet(os.path.join(d, "lines_extra.parquet"))
    plays = ([(2024, 1, "2024_01_A_B", "A", "QA1")] * 30 + [(2024, 2, "2024_02_A_C", "A", "QA1")] * 5 +
             [(2024, 2, "2024_02_A_C", "A", "QA2")] * 25 + [(2024, 1, "2024_01_A_B", "B", "QB1")] * 33 +
             [(2024, 3, "2024_03_B_A", "A", week3_leader)] * 40 + [(2024, 3, "2024_03_B_A", "B", "QB1")] * 30)
    pbp = pd.DataFrame(plays, columns=["season", "week", "game_id", "posteam", "passer_player_id"])
    pbp["pass_attempt"] = 1
    pbp.to_parquet(os.path.join(d, "pbp_2024.parquet"))
    inj = pd.DataFrame([(2024, 3, "B", "QB1", "Out" if injured else "Questionable")],
                       columns=["season", "week", "team", "gsis_id", "report_status"])
    inj.to_parquet(os.path.join(d, "injuries.parquet"))


def test_signals_use_only_strictly_prior_games_and_pregame_injury_report(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _write_hist(str(a))
    _write_hist(str(b), target_scores=(55, 3), week3_leader="QA9")  # the target game's own result changes
    fa = el.build_totals_frame(str(a)).set_index("game_id")
    fb = el.build_totals_frame(str(b)).set_index("game_id")
    t = "2024_03_B_A"
    assert fa.loc[t, "m2_signal"] == pytest.approx(fb.loc[t, "m2_signal"])
    assert (fa.loc[t, "home_qb_change"], fa.loc[t, "away_qb_change"]) == (fb.loc[t, "home_qb_change"],
                                                                         fb.loc[t, "away_qb_change"])
    # A: QA1 is primary (35 att) but QA2 led week 2 -> change; B: QB1 primary and last leader -> no change
    assert fa.loc[t, "home_qb_change"] == 1 and fa.loc[t, "away_qb_change"] == 0
    assert np.isnan(fa.loc["2024_01_A_B", "m2_signal"])  # no prior games: signal missing, never imputed
    c = tmp_path / "c"
    c.mkdir()
    _write_hist(str(c), injured=True)
    assert el.build_totals_frame(str(c)).set_index("game_id").loc[t, "away_qb_change"] == 1


def test_every_issued_recommendation_counts_in_hypothetical_roi_even_below_breakeven():
    # an analyst-issued pick whose model probability is below breakeven was still issued: its loss must count
    issued_loss = _row("ovr", p=0.45, price=1.91, outcome="loss")
    roi = el.flat_roi([issued_loss, _row("w", p=0.6, price=1.8)], "issued")
    assert (roi["bets"], roi["won"], roi["lost"]) == (2, 1, 1)
    assert roi["units"] == pytest.approx(-0.2)
    event = {**_row("e", p=0.45, price=1.91), "issued_class": "candidate_event"}
    assert el.flat_roi([event], "issued")["bets"] == 0  # candidate events still need p > breakeven


def test_issued_confidence_diagnostic_uses_frozen_probabilities_by_market_and_game_cluster_only():
    """The report is descriptive: issued p/settlement only, never prices or a refit target."""
    rows = []
    for i in range(6):
        rows.append({
            **_row(f"p{i}", game=f"g{i}", market="passing_yards", p=0.7,
                    outcome="win" if i % 2 else "loss", price=1.50 + i),
            "section": "recommendations_given",
            "factor_panel": {"status": "recorded", "counts": {
                "numeric_applied": 1, "considered_no_change": 2, "context_only": 3,
                "shadow_only": 1, "unavailable_unverified": 1,
            }},
        })
    # These rows must appear only as exclusions, not silently alter reliability.
    rows += [
        {**_row("watch", game="g7", market="receptions", p=0.9), "section": "watch_published"},
        {**_row("push", game="g8", market="receptions", p=0.9, outcome="push"),
         "section": "recommendations_given"},
        {**_row("missing", game="g9", market="receptions", p=None),
         "section": "recommendations_given", "factor_panel": {"status": "absent", "counts": None}},
        {**_row("untrusted", game="g10", market="receptions", p=0.9, capture=None),
         "section": "recommendations_given"},
    ]

    out = el.issued_confidence_diagnostic(rows, n_boot=100, seed=7)

    assert out["included"] == 6
    assert out["exclusions"] == {"not_recommendation": 1, "untrusted_capture": 1,
                                  "outside_frozen_prospective_window": 0, "missing_game_cluster": 0,
                                  "non_win_loss": 1, "missing_or_invalid_probability": 1}
    market = out["by_market"]["passing_yards"]
    assert market["n"] == 6 and market["game_clusters"] == 6
    assert market["brier"]["n_clusters"] == 6 and market["brier"]["low"] is not None
    assert market["reliability"]["n"] == 6
    assert out["factor_panels"] == {
        "recorded": 6, "absent": 0, "withheld": 0,
        "status_counts": {"numeric_applied": 6, "considered_no_change": 12, "context_only": 18,
                          "shadow_only": 6, "unavailable_unverified": 6},
    }
    assert "descriptive" in out["statement"] and out["promotion"] is False and out["refit"] is False

    # The diagnostic has no price input: changing every price leaves it byte-for-byte unchanged.
    repriced = [{**r, "price_decimal": 99.0} for r in rows]
    assert el.issued_confidence_diagnostic(repriced, n_boot=100, seed=7) == out


def test_research_status_wires_the_issued_confidence_diagnostic_from_the_append_only_ledger(tmp_path):
    row = {**_row("status", game="g1", market="pass_attempts", p=0.6), "source": "issued_ledger",
           "section": "recommendations_given", "factor_panel": {"status": "withheld", "counts": None}}
    ledger = str(tmp_path / "evidence.jsonl")
    el.append_evidence(ledger, [row], recorded_at="2026-10-12T00:00:00Z")
    doc = el.research_status(ledger, checked_at="2026-10-12T00:01:00Z")
    assert doc["issued_confidence"]["included"] == 1
    assert doc["issued_confidence"]["factor_panels"]["withheld"] == 1
    assert doc["issued_confidence"]["promotion"] is False and doc["issued_confidence"]["refit"] is False


def test_issued_confidence_diagnostic_excludes_rows_outside_the_frozen_prospective_window():
    late = _row("late", decided="2026-10-11T17:00:00+00:00")
    historical = _row("historical", week=4)
    no_game = _row("no-game", game=None)
    for row in (late, historical, no_game):
        row["section"] = "recommendations_given"

    out = el.issued_confidence_diagnostic([late, historical, no_game])

    assert out["included"] == 0
    assert out["exclusions"] == {"not_recommendation": 0, "untrusted_capture": 0,
                                  "outside_frozen_prospective_window": 2, "missing_game_cluster": 1,
                                  "non_win_loss": 0, "missing_or_invalid_probability": 0}
    assert out["by_market"] == {}


def test_issued_confidence_diagnostic_labels_malformed_factor_panel_without_counting_it_or_crashing():
    row = {**_row("bad-panel"), "section": "recommendations_given",
           "factor_panel": {"status": "withheld", "counts": ["not", "counts"]}}

    out = el.issued_confidence_diagnostic([row])

    assert out["included"] == 1
    assert out["factor_panels"] == {"recorded": 0, "absent": 0, "withheld": 0,
                                    "malformed": 1, "status_counts": {}}
