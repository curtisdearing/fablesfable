"""Phase 8.4-8.6 acceptance tests -- cards, trends, and the honest record.

The display layer is where an honest model becomes a dishonest product, so
these tests are about what the reader SEES, not what the pipeline computed:

* a thin case must be visually distinguishable from a strong one;
* magnitude must survive greyscale (length and label, never hue alone);
* a synthetic-line card must have no edge field AT ALL;
* the as-of boundary must be present on every trend;
* n=0 must render as "n=0", never as blank.

Offline and deterministic.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import dashboard                    # noqa: E402
from nflvalue import evidence as evmod            # noqa: E402
from nflvalue import explain_cards as xc          # noqa: E402
from nflvalue import explain_render as rn         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEEKLY = os.path.join(ROOT, "data", "weekly_props.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(WEEKLY),
    reason="data/weekly_props.json is a pipeline artifact; run pipeline_weekly first")


def _payload():
    with open(WEEKLY, encoding="utf-8") as fh:
        weekly = json.load(fh)
    return xc.build_payload(weekly)


# --------------------------------------------------------------------------- #
# 8.4 Cards
# --------------------------------------------------------------------------- #
def test_every_published_lean_produces_a_card():
    with open(WEEKLY, encoding="utf-8") as fh:
        weekly = json.load(fh)
    expected = sum(len(g.get("leans") or []) for g in weekly["games"])
    payload = _payload()
    assert len(payload["cards"]) + len(payload["unexplainable"]) == expected


def test_unexplainable_picks_are_surfaced_not_dropped():
    """A pick whose ledger will not reconcile is the one a reader most needs to
    distrust. Silently omitting it from the display would hide exactly that."""
    payload = _payload()
    assert "unexplainable" in payload
    assert isinstance(payload["unexplainable"], list)


def test_synthetic_line_card_has_no_edge_field_at_all():
    """Not "edge: n/a" -- the key must be ABSENT. A rendered field, even an
    empty one, invites the reader to go looking for a number."""
    payload = _payload()
    synthetic = [c for c in payload["cards"] if c["is_synthetic_line"]]
    assert synthetic, "fixture contained no synthetic-line cards"
    for card in synthetic:
        assert "edge_label" not in card, (
            f"{card['name']} {card['market']} rendered an edge against a "
            f"synthetic reference line")


def test_synthetic_cards_carry_a_visible_badge():
    payload = _payload()
    for card in payload["cards"]:
        assert card["line_source"] in ("synthetic", "real", "none")
        if card["is_synthetic_line"]:
            assert card["line_source"] == "synthetic"


def test_every_driver_shows_a_grade_with_n_and_an_interval_status():
    payload = _payload()
    for card in payload["cards"]:
        for d in card["drivers"]:
            e = d["evidence"]
            assert e["grade"] in evmod.GRADES
            assert e["n_label"], "driver rendered without a sample-size label"
            assert e["ci_label"], "driver rendered without an interval label"


def test_weakest_grade_is_surfaced_not_an_average():
    """A case is only as strong as its weakest load-bearing driver. Averaging
    would let one strong driver launder three thin ones."""
    payload = _payload()
    order = {"unproven": 0, "thin": 1, "moderate": 2, "strong": 3}
    for card in payload["cards"]:
        grades = [d["evidence"]["grade"] for d in card["drivers"]
                  if d["direction"] != "baseline"]
        if grades:
            assert card["weakest_grade"] == min(grades, key=lambda g: order[g])


def test_magnitude_is_encoded_by_length_and_label_not_colour_alone():
    """Greyscale-survival: every bar carries a numeric percentage, a direction
    GLYPH and a direction WORD, so no meaning is lost without hue."""
    payload = _payload()
    for card in payload["cards"]:
        for d in card["drivers"]:
            bar = d["bar"]
            assert "pct" in bar and isinstance(bar["pct"], (int, float))
            assert "glyph" in bar and "word" in bar
            if bar["pct"] > 0:
                assert bar["glyph"] in ("▲", "▼", "="), bar
                assert bar["word"] in ("raises", "lowers", "no change")
            assert bar.get("basis"), "bar drawn without stating what it measures"


def test_level_bars_measure_deviation_not_unit_conversion():
    """A 0.568 catch rate converts targets to receptions; its raw log
    contribution is huge for arithmetic reasons. Drawing that as a 75% bar
    would tell the reader efficiency did three-quarters of the work. Level
    bars must therefore measure deviation from the measured reference."""
    with_ref = xc._bar({"kind": "level", "multiplier": 0.5679, "reference": 0.6261,
                        "log_contribution": -0.5657})
    assert with_ref["basis"] == "deviation from the position average"
    assert with_ref["pct"] < 25, (
        f"level bar of {with_ref['pct']}% is showing the unit conversion, "
        f"not the argument")

    no_ref = xc._bar({"kind": "level", "multiplier": 0.5679, "reference": None,
                      "log_contribution": -0.5657})
    assert no_ref["pct"] == 0.0
    assert "not claimed" in no_ref["basis"]


def test_tilt_bars_measure_distance_from_no_effect():
    tilt = xc._bar({"kind": "tilt", "multiplier": 1.35,
                    "log_contribution": 0.3001})
    assert tilt["basis"] == "distance from no-effect (1.0)"
    assert tilt["glyph"] == "▲" and tilt["word"] == "raises"
    assert tilt["pct"] > 0


def test_a_thin_card_is_distinguishable_from_a_strong_one():
    """Screenshot-diff proxy: the rendered HTML for a thin card differs from a
    strong one in class names, so any visual diff picks it up."""
    thin_html = f'<span class="chip thin">thin</span>'
    strong_html = f'<span class="chip strong">strong</span>'
    assert thin_html != strong_html
    template = dashboard.TEMPLATE
    for grade in evmod.GRADES:
        assert f".chip.{grade}" in template, f"no distinct style for '{grade}'"


def test_grade_styles_differ_by_border_not_only_colour():
    """Colourblind safety: the four grade chips must differ in BORDER STYLE
    (solid / dashed / dotted), so they remain separable in greyscale."""
    template = dashboard.TEMPLATE
    styles = {}
    for grade in evmod.GRADES:
        match = re.search(rf"\.chip\.{grade}\{{([^}}]*)\}}", template)
        assert match, f"missing .chip.{grade} rule"
        border = re.search(r"border:[^;]*", match.group(1))
        assert border, f".chip.{grade} has no border rule"
        styles[grade] = border.group(0)
    kinds = {re.search(r"(solid|dashed|dotted)", v).group(1)
             for v in styles.values() if re.search(r"(solid|dashed|dotted)", v)}
    assert len(kinds) >= 2, (
        f"grade chips rely on colour alone; border styles seen: {kinds}")


def test_counter_case_is_present_on_every_card():
    payload = _payload()
    for card in payload["cards"]:
        assert "counter_case" in card["prose"]
        assert card["prose"]["counter_case"], f"{card['name']}: empty counter-case"


def test_card_copy_never_contains_imperative_betting_language():
    payload = _payload()
    for card in payload["cards"]:
        joined = " ".join(s["text"] for block in card["prose"].values()
                          for s in block)
        rn.check_vocabulary(joined)


def test_cards_state_their_reconciliation():
    payload = _payload()
    for card in payload["cards"]:
        rec = card["reconciliation"]
        assert rec["drift"] is not None and rec["rounding_bound"] is not None
        assert rec["drift"] <= rec["rounding_bound"]


# --------------------------------------------------------------------------- #
# 8.5 Trends
# --------------------------------------------------------------------------- #
def test_trend_points_are_strictly_before_the_predicted_week():
    """The whole purpose of the trend view: let a reader confirm with their own
    eyes that the model never saw the week it predicted."""
    pd = pytest.importorskip("pandas")
    frame_path = os.path.join(ROOT, "data", "ml_frame.parquet")
    if not os.path.exists(frame_path):
        pytest.skip("ml_frame.parquet is a build artifact")
    frame = pd.read_parquet(frame_path, columns=[
        "season", "week", "player_id", "market",
        "proj_volume", "proj_efficiency", "opp_factor"])
    trends = xc.build_trends(frame, season=2023, week=10)
    assert trends, "no trends built"
    for key, t in list(trends.items())[:50]:
        assert t["as_of_boundary"] == {"season": 2023, "week": 10}
        for p in t["points"]:
            assert (p["season"], p["week"]) < (2023, 10), (
                f"{key} plotted week {p['season']}w{p['week']}, which is not "
                f"strictly before the predicted week")


def test_every_trend_carries_an_explicit_as_of_boundary():
    pd = pytest.importorskip("pandas")
    frame_path = os.path.join(ROOT, "data", "ml_frame.parquet")
    if not os.path.exists(frame_path):
        pytest.skip("ml_frame.parquet is a build artifact")
    frame = pd.read_parquet(frame_path, columns=[
        "season", "week", "player_id", "market",
        "proj_volume", "proj_efficiency", "opp_factor"])
    trends = xc.build_trends(frame, season=2023, week=10)
    for t in list(trends.values())[:20]:
        assert "as_of_boundary" in t and t["as_of_boundary"]
        assert "strictly before" in t["note"].lower()


def test_sparkline_renderer_draws_the_boundary():
    assert "as-of" in dashboard.TEMPLATE
    assert "stroke-dasharray" in dashboard.TEMPLATE


# --------------------------------------------------------------------------- #
# 8.6 The honest record
# --------------------------------------------------------------------------- #
def test_record_leads_with_the_absence_of_demonstrated_edge():
    record = xc.build_record()
    assert "NOT demonstrated an edge" in record["headline"]


def test_zero_clv_renders_as_n_equals_zero_not_blank():
    """A blank reads as 'nothing to worry about'. A zero reads as 'no evidence
    has been collected', which is the true state."""
    record = xc.build_record()
    assert record["clv"]["n_resolved"] == 0
    assert record["clv"]["label"] == "n=0 resolved"
    assert record["clv"]["label"].strip() != ""


def test_real_line_record_is_explicitly_none_collected():
    record = xc.build_record()
    assert record["real_line_record"]["n"] == 0
    assert "n=0" in record["real_line_record"]["label"]


def test_synthetic_accuracy_is_labelled_as_not_evidence_of_profit():
    eval_path = os.path.join(ROOT, "data", "ml_eval_results.json")
    results = None
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as fh:
            results = json.load(fh)
    record = xc.build_record(eval_results=results)
    syn = record["synthetic_line_accuracy"]
    assert "SYNTHETIC" in syn["what_it_is"]
    assert "NOT evidence of profitability" in syn["what_it_is_not"]


def test_synthetic_accuracy_carries_n_and_an_interval_per_season():
    eval_path = os.path.join(ROOT, "data", "ml_eval_results.json")
    if not os.path.exists(eval_path):
        pytest.skip("ml_eval_results.json is a build artifact")
    with open(eval_path, encoding="utf-8") as fh:
        results = json.load(fh)
    syn = xc.build_record(eval_results=results)["synthetic_line_accuracy"]
    if syn["status"] != "measured":
        pytest.skip("no measured seasons in the artifact")
    for row in syn["seasons"]:
        assert row["n"] > 0
        assert row["ci"] and len(row["ci"]) == 2
        assert "n=" in row["label"]


def test_insufficient_sample_is_not_described_as_passing():
    record = xc.build_record()
    plain = record["kill_check"]["plain_english"]
    assert "not a passing grade" in plain.lower()


def test_kill_check_plain_english_covers_every_verdict():
    for verdict, expect in (("NO_GO", "stop staking"),
                            ("GO", "not proof of one"),
                            ("INSUFFICIENT_SAMPLE", "not a passing grade")):
        text = xc._kill_check_plain({"verdict": verdict, "n": 10,
                                     "min_sample": 150})
        assert expect in text.lower() or expect in text


def test_record_degrades_loudly_when_the_kill_check_cannot_be_read():
    class _Boom:
        def __getattr__(self, _):
            raise RuntimeError("db gone")

    record = xc.build_record(conn=_Boom())
    assert record["kill_check"]["verdict"] in ("UNAVAILABLE", "INSUFFICIENT_SAMPLE")
    assert "unknown" in record["kill_check"]["plain_english"].lower() or \
           "not a passing grade" in record["kill_check"]["plain_english"].lower()


# --------------------------------------------------------------------------- #
# Payload / dashboard wiring
# --------------------------------------------------------------------------- #
def test_payload_carries_the_disclaimer():
    payload = _payload()
    assert "Leans, not locks" in payload["disclaimer"]
    assert "1-800-GAMBLER" in payload["disclaimer"]


def test_dashboard_template_renders_both_new_panels():
    template = dashboard.TEMPLATE
    for probe in ('data-t="why"', 'data-t="record"', 'id="why"', 'id="record"',
                  "renderWhy()", "renderRecord()"):
        assert probe in template, f"dashboard is missing {probe}"


def test_dashboard_escapes_card_text():
    """Player names and notes reach the DOM; they must be escaped."""
    assert "function esc(" in dashboard.TEMPLATE
    assert "&amp;" in dashboard.TEMPLATE


def test_payload_is_deterministic():
    first = json.dumps(_payload(), sort_keys=True, default=str)
    for _ in range(2):
        assert json.dumps(_payload(), sort_keys=True, default=str) == first
