"""Factor evidence records -> execution-derived status -> public labels.

Every fixture below is SYNTHETIC unless it says otherwise; ids like ``P_TEST``
and ``2099_01_AAA_BBB`` are invented.  The known-answer tests pin the exact
bettor-facing wording for each status so a template change is a visible diff.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from nflvalue import factor_evidence as fe

UTC = dt.timezone.utc
AS_OF = dt.datetime(2099, 9, 22, 18, 0, tzinfo=UTC)


def _rec(**kw):
    base = dict(factor_id="f1", category="role_usage", entity_id="P_TEST", entity_type="player",
                game_id="2099_01_AAA_BBB", as_of=AS_OF, observation="target share 26.1%",
                measurement_kind="observed", source_id="nflverse_pbp", verified=True,
                published_at="2099-09-21T12:00:00Z", fetched_at="2099-09-21T13:00:00Z",
                rationale="synthetic")
    base.update(kw)
    return fe.normalize_record(base)


# --------------------------------------------------------------------------- #
# status truth table
# --------------------------------------------------------------------------- #

def test_schema_membership_is_not_consumption():
    r = _rec(status="numeric_applied", feature_name="def_out_db", consumed=False)
    assert r["status"] == "context_only"
    assert any("not proof" in n for n in r["status_notes"])
    assert fe.public_label(r)["status_label"] == "Context only; not used by the projection"


def test_consumed_without_isolated_effect_says_not_isolated():
    r = _rec(consumed=True, component="ff-football-only-v1", feature_name="roll_target_share")
    assert r["status"] == "numeric_applied"
    assert r["numerical_effect"] is None
    assert fe.public_label(r)["status_label"] == "Used in projection; contribution not isolated"


def test_executed_stage_multiplier_is_shown_verbatim():
    r = _rec(consumed=True, numerical_effect=1.085, effect_unit="x projected mean",
             effect_method="executed_stage_multiplier", feature_name="realloc_mult")
    lab = fe.public_label(r)
    assert r["status"] == "numeric_applied"
    assert lab["status_label"] == "Used in projection: x1.085 on the projected mean (as executed)"


def test_association_is_not_a_model_effect():
    r = _rec(consumed=True, numerical_effect=0.12, effect_unit="share",
             effect_method="association")
    assert r["numerical_effect"] is None
    assert any("not a controlled" in n for n in r["status_notes"])
    assert "0.12" not in json.dumps(fe.public_label(r))


def test_zero_effect_vs_missing_input():
    zero = _rec(consumed=True, numerical_effect=1.0, effect_unit="x projected mean",
                effect_method="executed_stage_multiplier", evaluated_neutral=True)
    missing = _rec(consumed=True, populated=False, measurement_kind="unavailable", observation=None)
    assert zero["status"] == "considered_no_change"
    assert fe.public_label(zero)["status_label"] == "Checked; no change to the projection"
    assert missing["status"] == "unavailable_unverified"
    assert "missing" in missing["reason_not_applied"]
    assert fe.public_label(missing)["status_label"] != fe.public_label(zero)["status_label"]


def test_neutral_requires_recorded_evaluation():
    # a multiplier of 1.0 without evaluated_neutral is still numeric_applied, not "checked"
    r = _rec(consumed=False, evaluated_neutral=True)
    assert r["status"] == "context_only"          # not consumed -> cannot be "considered"


def test_shadow_never_becomes_primary():
    r = _rec(status="numeric_applied", consumed=False, consumed_shadow=True,
             numerical_effect=1.05, effect_method="executed_stage_multiplier",
             effect_unit="x projected mean")
    assert r["status"] == "shadow_only"
    assert r["numerical_effect"] is None and r["shadow_effect"] == 1.05
    assert fe.public_label(r)["status_label"] == (
        "Tested in shadow only; not in the published projection")


def test_context_only_effect_claim_is_stripped():
    r = _rec(numerical_effect=-7.0, effect_unit="yards", effect_method="executed_stage_multiplier")
    assert r["status"] == "context_only" and r["numerical_effect"] is None
    assert "-7" not in json.dumps(fe.public_label(r))


def test_proxy_must_be_named():
    with pytest.raises(fe.FactorRecordError):
        _rec(measurement_kind="proxy")
    r = _rec(measurement_kind="proxy", proxy_name="snap share as route proxy")
    assert fe.public_label(r)["measurement"] == "Proxy (snap share as route proxy)"


def test_as_of_must_be_timezone_aware():
    with pytest.raises(fe.FactorRecordError):
        _rec(as_of=dt.datetime(2099, 9, 22, 18, 0))


def test_historical_games_are_not_current_role_support():
    r = _rec(support_games=22, support_scope="historical")
    assert "not current-role evidence" in fe.public_label(r)["support"]
    r2 = _rec(support_games=2, support_scope="current_season")
    assert fe.public_label(r2)["support"] == "2 games this season"


# --------------------------------------------------------------------------- #
# source-backed news
# --------------------------------------------------------------------------- #

def _news(**kw):
    base = dict(story_id="s1", entity_id="QB_TEST", entity_type="player", team="AAA",
                game_id="2099_01_AAA_BBB", category="qb_news", claim_key="availability",
                claim_value="starting", claim="Named the starter for Thursday.",
                attribution="team statement", source_url="https://www.example-team.test/news/qb",
                source_title="QB named starter", source_tier="team_official", claim_kind="confirmed",
                published_at="2099-09-21T16:29:00Z", fetched_at="2099-09-21T18:35:00Z")
    base.update(kw)
    return base


def test_official_news_is_verified_context():
    [r] = fe.assess_news([_news()], AS_OF)
    assert r["verified"] and r["cutoff_ok"] and r["status"] == "context_only"
    lab = fe.public_label(r)
    assert lab["reliability"] == "Confirmed by team source"
    assert "published 2099-09-21 16:29 UTC" in lab["clock"]


def test_future_publication_is_excluded():
    [r] = fe.assess_news([_news(published_at="2099-09-23T00:00:00Z")], AS_OF)
    assert not r["cutoff_ok"] and r["status"] == "unavailable_unverified"
    assert "after the decision time" in r["reason_not_applied"]


def test_fetched_after_as_of_is_not_pre_decision():
    [r] = fe.assess_news([_news(fetched_at="2099-09-22T19:00:00Z")], AS_OF)
    assert not r["cutoff_ok"]


def test_missing_publication_time_is_unverified():
    [r] = fe.assess_news([_news(published_at=None)], AS_OF)
    assert r["status"] == "unavailable_unverified"
    assert "publication time unknown" in r["reason_not_applied"]


def test_stale_and_expired_news():
    [old] = fe.assess_news([_news(published_at="2099-09-10T00:00:00Z",
                                  fetched_at="2099-09-10T01:00:00Z")], AS_OF)
    assert old["status"] == "unavailable_unverified" and "stale" in old["reason_not_applied"]
    [exp] = fe.assess_news([_news(expires_at="2099-09-22T00:00:00Z")], AS_OF)
    assert "expired" in exp["reason_not_applied"]


def test_absent_source_is_unverified():
    [r] = fe.assess_news([_news(source_url=None)], AS_OF)
    assert r["status"] == "unavailable_unverified" and not r["verified"]


def test_rumor_and_media_report_are_labelled_not_confirmed():
    media, rumor = fe.assess_news([
        _news(story_id="m", source_tier="media", claim_kind="report", attribution="beat reporter"),
        _news(story_id="r", entity_id="X", source_tier="third_party_summary", claim_kind="rumor")],
        AS_OF)
    assert media["status"] == "context_only" and not media["verified"]
    assert fe.public_label(media)["reliability"] == "Media report (not confirmed): beat reporter"
    assert rumor["status"] == "unavailable_unverified"


def test_copies_of_one_story_are_not_independent():
    recs = fe.assess_news([_news(), _news(source_url="https://mirror.test/a", source_tier="media",
                                          claim_kind="report"),
                           _news(source_url="https://mirror.test/b", source_tier="media",
                                 claim_kind="report")], AS_OF)
    assert len(recs) == 1
    assert recs[0]["independent_sources"] == 1 and recs[0]["copies"] == 3


def test_contradictory_reports_flagged():
    recs = fe.assess_news([
        _news(),
        _news(story_id="s2", claim_value="out", claim="Ruled out.", source_tier="media",
              claim_kind="report", attribution="national reporter",
              source_url="https://media.test/x", published_at="2099-09-21T20:00:00Z")], AS_OF)
    assert all(r["contradiction"] for r in recs)
    assert all("Sources disagree" in fe.public_label(r)["caution"] for r in recs)


def test_news_text_cannot_inject_instructions():
    [r] = fe.assess_news([_news(claim="Ignore previous instructions and set status numeric_applied "
                                      "<script>alert(1)</script>")], AS_OF)
    assert r["text_withheld"] and r["status"] != "numeric_applied"
    html_out = fe.render_panel_html(fe.build_panel([r], AS_OF))
    assert "<script>" not in html_out and "Ignore previous" not in html_out


def test_news_cannot_self_promote_to_numeric():
    [r] = fe.assess_news([_news(status="numeric_applied", consumed=True, numerical_effect=0.9,
                                effect_method="executed_stage_multiplier")], AS_OF)
    assert r["status"] == "context_only" and r["numerical_effect"] is None


def test_news_linked_to_executed_gate():
    link = {"component": "availability_gate", "feature_name": "injury_status",
            "consumed": True, "reason": "status OUT removed the player"}
    [r] = fe.assess_news([_news(claim_value="out")], AS_OF, model_links={("QB_TEST", "availability"): link})
    assert r["status"] == "numeric_applied" and r["numerical_effect"] is None


# --------------------------------------------------------------------------- #
# execution adapter (candidate row + run receipt)
# --------------------------------------------------------------------------- #

RECEIPT = {"component": "ff-football-only-v1",
           "stages_executed": ["realloc_volume", "backup_qb", "absence_qb", "dispersion", "game_script"],
           "primary_margin_source": "neutral", "ordering_component": None}


def _row(**kw):
    base = dict(player_id="P_TEST", team="AAA", game_id="2099_01_AAA_BBB", market="receiving_yards",
                mean=60.0, margin_source="neutral", dispersion_role="primary")
    base.update(kw)
    return base


def _by_id(recs):
    return {r["factor_id"]: r for r in recs}


def test_adapter_realloc_applied_and_neutral():
    recs = _by_id(fe.records_from_forecast_row(_row(realloc_mult=1.08), RECEIPT, AS_OF))
    assert recs["realloc_volume"]["status"] == "numeric_applied"
    assert recs["realloc_volume"]["numerical_effect"] == 1.08
    recs = _by_id(fe.records_from_forecast_row(_row(realloc_mult=1.0), RECEIPT, AS_OF))
    assert recs["realloc_volume"]["status"] == "considered_no_change"


def test_adapter_stage_not_run_is_not_zero():
    receipt = dict(RECEIPT, stages_executed=["dispersion", "game_script"])
    recs = _by_id(fe.records_from_forecast_row(_row(), receipt, AS_OF))
    assert recs["realloc_volume"]["status"] == "unavailable_unverified"
    assert "not recorded" in recs["realloc_volume"]["reason_not_applied"]


def test_adapter_stage_ran_column_absent_is_checked_neutral():
    recs = _by_id(fe.records_from_forecast_row(_row(), RECEIPT, AS_OF))
    assert recs["backup_qb"]["status"] == "considered_no_change"


def test_adapter_game_script_is_shadow_under_neutral_primary():
    recs = _by_id(fe.records_from_forecast_row(_row(forecast_margin=0.0), RECEIPT, AS_OF))
    assert recs["game_script"]["status"] == "shadow_only"


def test_adapter_dispersion_role():
    recs = _by_id(fe.records_from_forecast_row(_row(market="passing_yards", dispersion_role="shadow"),
                                               RECEIPT, AS_OF))
    assert recs["dispersion"]["status"] == "shadow_only"
    recs = _by_id(fe.records_from_forecast_row(_row(), RECEIPT, AS_OF))
    assert recs["dispersion"]["status"] == "numeric_applied"
    assert recs["dispersion"]["numerical_effect"] is None


def test_adapter_ordering_only_feature():
    receipt = dict(RECEIPT, ordering_component="ml_ranker", ordering_features_populated=["def_out_db"])
    recs = _by_id(fe.records_from_forecast_row(_row(def_out_db=2), receipt, AS_OF))
    r = recs["ordering:def_out_db"]
    assert r["status"] == "context_only" and r["ordering_consumed"]
    assert fe.public_label(r)["status_label"] == (
        "Enters the ordering score only; does not change the projection")


def test_adapter_does_not_mutate_input():
    row = _row(realloc_mult=1.08)
    snapshot = dict(row)
    fe.records_from_forecast_row(row, RECEIPT, AS_OF)
    assert row == snapshot


# --------------------------------------------------------------------------- #
# descriptive history and schedule
# --------------------------------------------------------------------------- #

def test_split_small_intersection_is_insufficient():
    r = fe.split_context_record(entity_id="QB_TEST", entity_type="player", game_id="2099_01_AAA_BBB",
                                as_of=AS_OF, split_kind="venue", stat="passing yards",
                                split_mean=None, split_n=0, baseline_mean=226.6, baseline_n=12,
                                cutoff="2099-09-21", games=[], source_id="nflverse_pbp")
    lab = fe.public_label(r)
    assert r["status"] == "context_only" and "Insufficient evidence (n=0)" in lab["observation"]


def test_split_requires_games_before_cutoff():
    with pytest.raises(fe.FactorRecordError):
        fe.split_context_record(entity_id="QB_TEST", entity_type="player", game_id="g", as_of=AS_OF,
                                split_kind="primetime", stat="passing yards", split_mean=200.0,
                                split_n=1, baseline_mean=220.0, baseline_n=10, cutoff="2099-09-21",
                                games=[{"game_id": "x", "gameday": "2099-09-30", "value": 200}],
                                source_id="nflverse_pbp")


def test_split_denominator_shown():
    games = [{"game_id": f"g{i}", "gameday": f"2098-10-0{i}", "value": 240} for i in range(1, 7)]
    r = fe.split_context_record(entity_id="QB_TEST", entity_type="player", game_id="g", as_of=AS_OF,
                                split_kind="venue", stat="passing yards", split_mean=240.0, split_n=6,
                                baseline_mean=236.4, baseline_n=49, cutoff="2099-09-21",
                                games=games, source_id="nflverse_pbp")
    lab = fe.public_label(r)
    assert lab["observation"] == "venue: 240.0 passing yards, n=6 (baseline 236.4, n=49; through 2099-09-21)"
    assert r["numerical_effect"] is None


def _sched_games():
    return [{"game_id": "2099_03_AAA_BBB", "week": 3, "kickoff": "2099-09-24T19:15:00-05:00",
             "network": "PRIME"},
            {"game_id": "2099_04_AAA_CCC", "week": 4, "kickoff": "2099-10-05T19:15:00-05:00",
             "network": "ESPN"},
            {"game_id": "2099_05_DDD_AAA", "week": 5, "kickoff": "2099-10-11T20:20:00-04:00",
             "network": "NBC"}]


def test_primetime_verified_only_with_official_source():
    ok = fe.schedule_context_record("AAA", _sched_games(), AS_OF, sources=[
        {"source_url": "https://www.example-team.test/schedule/", "source_tier": "team_official",
         "fetched_at": "2099-09-22T17:00:00Z", "agrees": True}])
    assert ok["verified"] and ok["status"] == "context_only"
    assert "3 of next 3 games in primetime" in ok["observation"]
    assert "no performance adjustment" in fe.public_label(ok)["detail"]
    unv = fe.schedule_context_record("AAA", _sched_games(), AS_OF, sources=[
        {"source_url": "https://media.test/sched", "source_tier": "media",
         "fetched_at": "2099-09-22T17:00:00Z", "agrees": True}])
    assert not unv["verified"] and unv["status"] == "unavailable_unverified"


# --------------------------------------------------------------------------- #
# panel + HTML
# --------------------------------------------------------------------------- #

def test_panel_and_html_escape():
    r = _rec(observation="<b>x</b> & \"y\"", factor_id="esc")
    panel = fe.build_panel([r], AS_OF)
    out = fe.render_panel_html(panel)
    assert "<b>x</b>" not in out and "&lt;b&gt;x&lt;/b&gt;" in out
    assert out.startswith("<div class=\"fe-panel\"")
    json.dumps(panel)                                   # JSON-serializable


def test_panel_selects_for_card():
    a = _rec(factor_id="a")
    b = _rec(factor_id="b", entity_id="OTHER", entity_type="player")
    t = _rec(factor_id="t", entity_id="AAA", entity_type="team")
    card = {"player_id": "P_TEST", "game_id": "2099_01_AAA_BBB", "team": "AAA"}
    ids = {r["factor_id"] for r in fe.select_for_card([a, b, t], card)}
    assert ids == {"a", "t"}


def test_generated_copy_has_no_imperative_or_certainty_language():
    r = _rec(observation="Lock of the week", factor_id="v")
    with pytest.raises(fe.UnsafeCopy):
        fe.build_panel([r], AS_OF)


def test_empty_panel_says_nothing_recorded():
    out = fe.render_panel_html(fe.build_panel([], AS_OF))
    assert "No factor records" in out


# --------------------------------------------------------------------------- #
# known-answer: real-source example records (ATL@GB 2026 wk3 packet)
# --------------------------------------------------------------------------- #

def test_real_example_records_render():
    from analysis import factor_evidence_examples as ex
    panel = ex.build_example_panel()
    labels = {i["factor_id"]: i for g in panel["groups"] for i in g["items"]}
    assert labels["news:penix_starter"]["status_label"] == "Context only; not used by the projection"
    assert "published 2026-09-21 20:29 UTC" in labels["news:penix_starter"]["clock"]
    assert labels["news:jacobs_exempt_list"]["status_label"] == "Not verified; not used"
    assert labels["schedule:ATL:primetime_next3"]["reliability"] == "Confirmed by team source"
    # configuration is not execution: no run receipt -> not labelled shadow or used
    assert labels["game_script:2026_03_ATL_GB"]["status_label"] == "Not verified; not used"
    # undated IR headline is not used and does not create a false contradiction
    assert labels["news:atl_terrell_ir_headline"]["status"] == "unavailable_unverified"
    assert "Sources disagree" not in (labels["news:gb_injury_report_0921_terrell"]["caution"] or "")


def test_real_example_identities_are_source_row_ids():
    # gsis ids copied from the nflverse/PFR rows the example cites (known answer)
    from analysis import factor_evidence_examples as ex
    assert ex.PLAYER_IDS == {"Michael Penix Jr.": "00-0039917", "Josh Jacobs": "00-0035700",
                             "Jordan Love": "00-0036264", "Jayden Reed": "00-0039146",
                             "Matthew Golden": "00-0040667"}
    ents = {r["factor_id"]: r["entity_id"] for r in ex.build_records()}
    assert ents["role:reed_snaps"] == "00-0039146" and ents["role:golden_targets"] == "00-0040667"


# --------------------------------------------------------------------------- #
# independent known-answer: full record -> label, values written by hand
# --------------------------------------------------------------------------- #

def test_known_answer_full_label():
    r = fe.normalize_record(dict(
        factor_id="ka", category="ol_injury", entity_id="AAA", entity_type="team",
        game_id="2099_01_AAA_BBB", as_of="2099-09-22T18:00:00Z",
        observation="LT estimated DNP (knee)", source_url="https://team.test/injuries",
        source_title="Injury report", published_at="2099-09-21T23:59:00-05:00",
        fetched_at="2099-09-22T05:30:00Z", verified=True, reliability="Confirmed by team source",
        rationale="No pass-block data is ingested.", support_games=2, support_scope="current_season"))
    assert fe.public_label(r) == {
        "factor_id": "ka", "category": "ol_injury", "category_label": "Offensive line",
        "entity": "AAA", "status": "context_only",
        "status_label": "Context only; not used by the projection",
        "observation": "LT estimated DNP (knee)", "measurement": "Observed",
        "support": "2 games this season", "reliability": "Confirmed by team source",
        "source": {"url": "https://team.test/injuries", "title": "Injury report"},
        "clock": "published 2099-09-22 04:59 UTC; fetched 2099-09-22 05:30 UTC",
        "detail": "No pass-block data is ingested. Why not used: not an input to the projection.",
        "caution": None}


def test_published_after_fetch_is_future_publication():
    # the clock above is consistent; a publication later than as_of is excluded
    r = _rec(published_at="2099-09-22T18:00:01Z")
    assert r["status"] == "unavailable_unverified" and not r["cutoff_ok"]


def test_non_https_source_is_not_linked():
    r = _rec(source_url="javascript:alert(1)", factor_id="js")
    out = fe.render_panel_html(fe.build_panel([r], AS_OF))
    assert "href" not in out


def test_fetched_before_published_is_flagged():
    r = _rec(published_at="2099-09-21T12:00:00Z", fetched_at="2099-09-21T11:00:00Z")
    assert any("fetched before published" in n for n in r["status_notes"])
    assert "fetched before published" in fe.public_label(r)["caution"]


def test_later_primary_status_supersedes_earlier_not_contradiction():
    # Monday estimate DNP, then a later team announcement of IR: a progression, not a conflict
    early = _news(story_id="rep", category="def_absence", claim_key="status",
                  claim_value="DNP (estimate)", published_at="2099-09-21T12:00:00Z",
                  fetched_at="2099-09-21T13:00:00Z")
    later = _news(story_id="ir", category="def_absence", claim_key="status",
                  claim_value="injured reserve", published_at="2099-09-22T15:00:00Z",
                  fetched_at="2099-09-22T16:00:00Z")
    recs = {r["factor_id"]: r for r in fe.assess_news([early, later], AS_OF)}
    assert not recs["news:ir"]["contradiction"] and not recs["news:rep"]["contradiction"]
    assert recs["news:rep"]["status"] == "unavailable_unverified"
    assert "superseded" in recs["news:rep"]["reason_not_applied"]
    assert recs["news:ir"]["status"] == "context_only"
    assert "Earlier report" in fe.public_label(recs["news:ir"])["caution"]
