"""personnel_matchup: pure QB/OL/defense context evidence (no numeric effect)."""

import copy
import datetime as dt
import json

import pytest

from nflvalue import personnel_matchup as pm

AS_OF = "2026-10-03T18:00:00Z"
KICK = "2026-10-04T17:00:00Z"
PUB = "2026-10-02T20:00:00Z"


def _base():
    games = [{"game_id": "2026_05_AAA_BBB", "season": 2026, "week": 5, "home_team": "BBB",
              "away_team": "AAA", "kickoff_utc": KICK, "source_id": "sched"}]
    roster = [
        {"team": "AAA", "player_id": "qa1", "name": "Alpha One", "position": "QB"},
        {"team": "AAA", "player_id": "qa2", "name": "Alpha Two", "position": "QB"},
        {"team": "AAA", "player_id": "lt1", "name": "Left Tackle", "position": "T"},
        {"team": "AAA", "player_id": "lt2", "name": "Swing Tackle", "position": "T"},
        {"team": "AAA", "player_id": "c1", "name": "Center Man", "position": "C"},
        {"team": "AAA", "player_id": "wr1", "name": "Wide Out", "position": "WR"},
        {"team": "AAA", "player_id": "rb1", "name": "Run Back", "position": "RB"},
        {"team": "BBB", "player_id": "qb1", "name": "Bravo One", "position": "QB"},
        {"team": "BBB", "player_id": "cb1", "name": "Corner One", "position": "CB"},
        {"team": "BBB", "player_id": "de1", "name": "Edge One", "position": "DE"},
        {"team": "BBB", "player_id": "lb1", "name": "Backer One", "position": "LB"},
        {"team": "BBB", "player_id": "dt1", "name": "Tackle One", "position": "DT"},
    ]
    depth = [
        {"team": "AAA", "player_id": "qa1", "slot": "QB", "rank": 1, "published_at": PUB},
        {"team": "AAA", "player_id": "qa2", "slot": "QB", "rank": 2, "published_at": PUB},
        {"team": "AAA", "player_id": "lt1", "slot": "LT", "rank": 1, "published_at": PUB},
        {"team": "AAA", "player_id": "lt2", "slot": "LT", "rank": 2, "published_at": PUB},
        {"team": "AAA", "player_id": "c1", "slot": "C", "rank": 1, "published_at": PUB},
        {"team": "BBB", "player_id": "qb1", "slot": "QB", "rank": 1, "published_at": PUB},
    ]
    reports = [{"game_id": "2026_05_AAA_BBB", "team": t, "published_at": PUB,
                "source_id": f"rep_{t}", "source_url": f"https://example.test/{t}"}
               for t in ("AAA", "BBB")]
    return dict(games=games, roster=roster, depth=depth, report_index=reports, as_of=AS_OF)


def _row(pid, team, status, pub=PUB, **kw):
    r = {"game_id": "2026_05_AAA_BBB", "team": team, "player_id": pid,
         "report_status": status, "published_at": pub, "source_id": f"rep_{team}",
         "source_url": f"https://example.test/{team}"}
    r.update(kw)
    return r


def _rec(out, prefix):
    return next(r for r in out["records"] if r["factor_id"].startswith(prefix))


def test_naive_as_of_rejected():
    kw = _base()
    kw["as_of"] = dt.datetime(2026, 10, 3, 18)
    with pytest.raises(pm.PersonnelInputError):
        pm.build_personnel_evidence(**kw)


def test_deterministic_stable_ids_and_no_input_mutation():
    kw = _base()
    kw["availability"] = [_row("lt1", "AAA", "Out")]
    snap = copy.deepcopy(kw)
    a = pm.build_personnel_evidence(**kw)
    b = pm.build_personnel_evidence(**kw)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    assert kw == snap
    ids = [r["factor_id"] for r in a["records"]]
    assert "qb_starter:AAA:2026_05_AAA_BBB" in ids
    assert "def_coverage:BBB:vs:AAA:2026_05_AAA_BBB" in ids
    assert len(ids) == len(set(ids))


def test_every_record_is_shared_schema_and_never_numeric_applied():
    kw = _base()
    kw["availability"] = [_row("lt1", "AAA", "Out"), _row("cb1", "BBB", "Out")]
    out = pm.build_personnel_evidence(**kw)
    required = {"factor_id", "category", "entity_id", "entity_type", "game_id", "as_of",
                "observation", "value", "unit", "observed_at", "published_at", "fetched_at",
                "source_url", "source_id", "verified", "cutoff_ok", "measurement_kind",
                "status", "component", "model_version", "feature_name", "consumed",
                "numerical_effect", "support_games", "support_opportunities",
                "reason_not_applied", "rationale", "uncertainty"}
    for r in out["records"]:
        assert required <= set(r)
        assert r["status"] in ("context_only", "unavailable_unverified")
        assert r["numerical_effect"] is None and r["consumed"] is False
        assert r["measurement_kind"] in ("observed", "projected", "proxy", "unavailable")


def test_future_rows_excluded():
    kw = _base()
    kw["availability"] = [_row("qa1", "AAA", "Out", pub="2026-10-03T20:00:00Z")]
    out = pm.build_personnel_evidence(**kw)
    qb = _rec(out, "qb_starter:AAA")
    assert qb["value"]["expected_starter"] == "qa1"          # future Out not seen
    assert qb["value"]["state"] == "expected_starter_availability_unverified"
    assert any(e["reason"] == "published_after_as_of" for e in out["excluded_rows"])


def test_game_already_kicked_off_is_refused():
    kw = _base()
    kw["as_of"] = "2026-10-04T17:30:00Z"
    out = pm.build_personnel_evidence(**kw)
    assert out["records"] == []
    assert out["errors"][0]["reason"] == "as_of_not_before_kickoff"


def test_backup_qb_expected_when_qb1_documented_out():
    kw = _base()
    kw["availability"] = [_row("qa1", "AAA", "Out"), _row("qa2", "AAA", "(-)")]
    qb = _rec(pm.build_personnel_evidence(**kw), "qb_starter:AAA")
    assert qb["value"]["state"] == "backup_expected"
    assert qb["value"]["expected_starter"] == "qa2"
    assert "advanced_features.qb_continuity (ml_ranker feature)" in qb["overlaps_existing"]


def test_uncertain_qb_when_questionable_and_no_starter_asserted():
    kw = _base()
    kw["availability"] = [_row("qa1", "AAA", "Questionable")]
    qb = _rec(pm.build_personnel_evidence(**kw), "qb_starter:AAA")
    assert qb["value"]["state"] == "uncertain_starter"
    assert qb["value"]["expected_starter"] is None
    assert qb["verified"] is False


def test_returning_qb1_without_current_report_is_uncertain():
    kw = _base()
    kw["report_index"] = []
    kw["prior_availability"] = [{"game_id": "2026_04_AAA_CCC", "team": "AAA",
                                 "player_id": "qa1", "report_status": "Out"}]
    qb = _rec(pm.build_personnel_evidence(**kw), "qb_starter:AAA")
    assert qb["value"]["state"] == "uncertain_starter"
    assert qb["value"]["candidates"][0]["returning"] == "returning_unverified"
    assert qb["value"]["expected_starter"] is None


def test_conflicting_starter_claims_stay_unresolved():
    kw = _base()
    kw["starter_claims"] = [
        {"game_id": "2026_05_AAA_BBB", "team": "AAA", "player_id": "qa1",
         "claim": "expected_starter", "attribution": "coach", "published_at": PUB},
        {"game_id": "2026_05_AAA_BBB", "team": "AAA", "player_id": "qa2",
         "claim": "expected_starter", "attribution": "reporter", "published_at": PUB}]
    qb = _rec(pm.build_personnel_evidence(**kw), "qb_starter:AAA")
    assert qb["value"]["state"] == "conflict_unresolved"
    assert qb["value"]["expected_starter"] is None


def test_no_injury_row_is_not_healthy():
    kw = _base()
    out = pm.build_personnel_evidence(**kw)
    ol = _rec(out, "ol_availability:AAA")
    assert ol["value"]["counts"]["unknown"] == 2 and ol["value"]["documented_absent"] == 0
    assert ol["verified"] is False
    assert all(s["availability"] == "not_listed_unverified" for s in ol["value"]["starters"])
    qb = _rec(out, "qb_starter:AAA")
    assert qb["value"]["state"] == "expected_starter_availability_unverified"
    kw.pop("report_index")
    ol2 = _rec(pm.build_personnel_evidence(**kw), "ol_availability:AAA")
    assert {s["availability"] for s in ol2["value"]["starters"]} == {"unknown_no_report"}


def test_unknown_is_not_zero_without_depth_or_report():
    kw = _base()
    kw["depth"] = []
    kw["report_index"] = []
    out = pm.build_personnel_evidence(**kw)
    ol = _rec(out, "ol_availability:AAA")
    assert ol["value"]["documented_absent"] is None
    assert ol["status"] == "unavailable_unverified"
    cov = _rec(out, "def_coverage:BBB")
    assert cov["value"]["documented_absent"] is None
    assert cov["status"] == "unavailable_unverified"


def test_backup_ol_replacement_from_documented_depth():
    kw = _base()
    kw["availability"] = [_row("lt1", "AAA", "Out"), _row("lt2", "AAA", "Questionable")]
    ol = _rec(pm.build_personnel_evidence(**kw), "ol_availability:AAA")
    lt = next(s for s in ol["value"]["starters"] if s["slot"] == "LT")
    assert lt["availability"] == "documented_out"
    assert lt["replacement"] == {"player_id": "lt2", "name": "Swing Tackle",
                                 "basis": "documented_depth_next",
                                 "availability": "documented_questionable"}
    assert ol["value"]["documented_absent"] == 1


def test_returning_player_documented_vs_unverified():
    kw = _base()
    kw["prior_availability"] = [
        {"game_id": "2026_04_AAA_CCC", "team": "AAA", "player_id": "lt1", "report_status": "Out",
         "published_at": "2026-09-26T20:00:00Z"},
        {"game_id": "2026_04_AAA_CCC", "team": "AAA", "player_id": "c1", "did_not_play": True,
         "published_at": "2026-09-28T20:00:00Z"}]
    kw["availability"] = [_row("lt1", "AAA", "(-)", practice=["FP", "FP", "FP"])]
    ol = _rec(pm.build_personnel_evidence(**kw), "ol_availability:AAA")
    st = {s["slot"]: s for s in ol["value"]["starters"]}
    assert st["LT"]["returning"] == "returning_documented"
    assert st["C"]["returning"] == "returning_unverified"
    assert len(ol["value"]["returns"]) == 2


def test_prior_week_rows_alone_do_not_make_defense_known():
    kw = _base()
    kw["report_index"] = []
    kw["prior_availability"] = [{"game_id": "2026_04_BBB_CCC", "team": "BBB",
                                 "player_id": "cb1", "report_status": "Out"}]
    cov = _rec(pm.build_personnel_evidence(**kw), "def_coverage:BBB")
    assert cov["value"]["documented_absent"] is None
    assert [r["player_id"] for r in cov["value"]["returning"]] == ["cb1"]
    assert cov["status"] == "unavailable_unverified"


def test_defensive_absence_applies_by_role_not_to_every_prop():
    kw = _base()
    kw["availability"] = [_row("cb1", "BBB", "Out")]
    kw["offensive_players"] = [
        {"player_id": "wr1", "team": "AAA", "game_id": "2026_05_AAA_BBB", "position": "WR"},
        {"player_id": "rb1", "team": "AAA", "game_id": "2026_05_AAA_BBB", "position": "RB"}]
    out = pm.build_personnel_evidence(**kw)
    cov = _rec(out, "def_coverage:BBB")
    assert cov["value"]["documented_absent"] == 1
    assert _rec(out, "def_run_front:BBB")["value"]["documented_absent"] == 0

    def link(pid):
        return next(l for l in out["relevance_links"] if l["player_id"] == pid
                    and l["factor_id"] == cov["factor_id"])
    assert link("wr1")["applicability"] == "applicable"
    assert link("rb1")["applicability"] == "not_applicable_role"
    assert all(l["direction"] == "not_estimated" for l in out["relevance_links"])


def test_ambiguous_lb_position_flagged():
    kw = _base()
    kw["availability"] = [_row("lb1", "BBB", "Doubtful")]
    rf = _rec(pm.build_personnel_evidence(**kw), "def_run_front:BBB")
    assert rf["value"]["documented_absent"] == 1
    assert rf["value"]["ambiguous_position_ids"] == ["lb1"]


def test_no_shadow_claim_without_verified_evidence():
    kw = _base()
    kw["offensive_players"] = [{"player_id": "wr1", "team": "AAA", "game_id": "2026_05_AAA_BBB",
                                "position": "WR", "expected_alignment": "wide"}]
    kw["matchup_claims"] = [{"game_id": "2026_05_AAA_BBB", "offense_player_id": "wr1",
                             "defense_player_id": "cb1", "kind": "shadow", "verified": False,
                             "published_at": PUB}]
    out = pm.build_personnel_evidence(**kw)
    al = next(l for l in out["relevance_links"] if l["relation"] == "alignment_matchup")
    assert al["matchup_kind"] == "inferred_unverified"
    assert al["shadow_assertion"] is False and al["unverified_claims_withheld"] == 1
    kw["matchup_claims"][0].update(verified=True, attribution="team beat, named",
                                   source_url="https://example.test/a")
    al2 = next(l for l in pm.build_personnel_evidence(**kw)["relevance_links"]
               if l["relation"] == "alignment_matchup")
    assert al2["matchup_kind"] == "documented" and al2["shadow_assertion"] is True


def test_duplicate_source_rows_do_not_double_count_and_conflicts_surface():
    kw = _base()
    kw["availability"] = [_row("dt1", "BBB", "Out", source_id="team_site"),
                          _row("dt1", "BBB", "Out", source_id="league_site")]
    out = pm.build_personnel_evidence(**kw)
    assert _rec(out, "def_pass_rush_interior:BBB")["value"]["documented_absent"] == 1
    assert _rec(out, "def_run_front:BBB")["value"]["documented_absent"] == 1
    kw["availability"][1]["report_status"] = "Doubtful"
    out2 = pm.build_personnel_evidence(**kw)
    inter = _rec(out2, "def_pass_rush_interior:BBB")
    assert inter["value"]["documented_absent"] == 0
    assert [c["player_id"] for c in inter["value"]["conflict"]] == ["dt1"]
    assert inter["verified"] is False


def test_malformed_and_ambiguous_identities_fail_safe():
    kw = _base()
    kw["roster"].append({"team": "BBB", "player_id": "cb9", "name": "Corner One",
                         "position": "CB"})
    kw["availability"] = [
        {"game_id": "2026_05_AAA_BBB", "team": "BBB", "name": "Corner One",
         "report_status": "Out", "published_at": PUB},
        {"game_id": "2026_05_AAA_BBB", "team": "BBB", "name": "Nobody Here",
         "report_status": "Out", "published_at": PUB},
        {"game_id": "2026_05_AAA_BBB", "team": "BBB", "player_id": "lt1",
         "report_status": "Out", "published_at": PUB}]
    kw["games"].append({"game_id": "bad", "home_team": "X", "away_team": "Y",
                        "kickoff_utc": "2026-10-04T13:00:00"})
    out = pm.build_personnel_evidence(**kw)
    reasons = sorted(e["reason"] for e in out["identity_errors"])
    assert reasons == ["ambiguous_name", "no_roster_match", "team_mismatch"]
    assert _rec(out, "def_coverage:BBB")["value"]["documented_absent"] == 0
    assert _rec(out, "ol_availability:AAA")["value"]["documented_absent"] == 0
    assert out["errors"][0]["reason"] == "missing_id_teams_or_tz_kickoff"


def test_name_match_resolves_unique_identity():
    kw = _base()
    kw["availability"] = [{"game_id": "2026_05_AAA_BBB", "team": "BBB", "name": "corner one.",
                           "report_status": "Out", "published_at": PUB}]
    out = pm.build_personnel_evidence(**kw)
    assert _rec(out, "def_coverage:BBB")["value"]["absent"][0]["player_id"] == "cb1"


def test_row_without_publication_clock_is_not_cutoff_ok():
    kw = _base()
    r = _row("qa1", "AAA", "(-)")
    r.pop("published_at")
    kw["availability"] = [r]
    kw["depth"] = [dict(d) for d in kw["depth"]]
    qb = _rec(pm.build_personnel_evidence(**kw), "qb_starter:AAA")
    assert qb["value"]["state"] == "expected_starter_documented"
    kw2 = _base()
    for d in kw2["depth"]:
        d.pop("published_at")
    qb2 = _rec(pm.build_personnel_evidence(**kw2), "qb_starter:AAA")
    assert qb2["cutoff_ok"] is False and qb2["verified"] is False


def test_multi_game_generality_no_team_hardcoding():
    kw = _base()
    kw["games"].append({"game_id": "2026_05_CCC_DDD", "season": 2026, "week": 5,
                        "home_team": "DDD", "away_team": "CCC",
                        "kickoff_utc": "2026-10-04T20:25:00Z"})
    kw["roster"] += [{"team": "CCC", "player_id": "qc1", "name": "C Q", "position": "QB"},
                     {"team": "DDD", "player_id": "sd1", "name": "D S", "position": "FS"}]
    kw["availability"] = [{"game_id": "2026_05_CCC_DDD", "team": "DDD", "player_id": "sd1",
                           "report_status": "Out", "published_at": PUB}]
    out = pm.build_personnel_evidence(**kw)
    assert set(out["games"]) == {"2026_05_AAA_BBB", "2026_05_CCC_DDD"}
    cov = _rec(out, "def_coverage:DDD:vs:CCC")
    assert cov["value"]["documented_absent"] == 1
    assert _rec(out, "def_coverage:BBB")["value"]["documented_absent"] == 0


def test_newer_publication_supersedes_older_status():
    kw = _base()
    kw["availability"] = [_row("dt1", "BBB", "Questionable", pub="2026-10-01T20:00:00Z"),
                          _row("dt1", "BBB", "Out", pub="2026-10-02T20:00:00Z")]
    inter = _rec(pm.build_personnel_evidence(**kw), "def_pass_rush_interior:BBB")
    assert inter["value"]["documented_absent"] == 1 and inter["value"]["conflict"] == []
    kw["availability"][0].pop("published_at")        # unclocked -> cannot order them
    inter2 = _rec(pm.build_personnel_evidence(**kw), "def_pass_rush_interior:BBB")
    assert [c["player_id"] for c in inter2["value"]["conflict"]] == ["dt1"]


def test_captured_at_bounds_live_use_but_not_later_captures():
    kw = _base()
    for d in kw["depth"]:
        d.pop("published_at")
        d["captured_at"] = "2026-10-03T12:00:00Z"
    qb = _rec(pm.build_personnel_evidence(**kw), "qb_starter:AAA")
    assert qb["cutoff_ok"] is True and qb["value"]["expected_starter"] == "qa1"
    for d in kw["depth"]:
        d["captured_at"] = "2026-10-03T19:00:00Z"     # captured after as_of
    out = pm.build_personnel_evidence(**kw)
    assert _rec(out, "qb_starter:AAA")["value"]["state"] == "unknown"
    assert any(e["reason"] == "captured_after_as_of" for e in out["excluded_rows"])


# Adapter extraction on a verbatim text excerpt of the retained NFL.com page
# (2026 Week 2, Packers section) -- the original parser gave Cisse Hargrave's
# "Doubtful" and dropped Hargrave.
NFLCOM_GB_EXCERPT = (
    "<div>Packers Player Position Injuries Practice Status Game Status Aaron Banks G Knee "
    "Limited Participation in Practice Questionable Warren Brinson DT Calf Did Not Participate "
    "In Practice Out Brandon Cisse CB Full Participation in Practice Javon Hargrave DT Knee, "
    "Concussion Did Not Participate In Practice Doubtful Ty'Ron Hopper LB Limited Participation "
    "in Practice Will McDonald IV DE Ankle Limited Participation in Practice Questionable "
    "Jets Player Position Injuries Practice Status Game Status Omar Cooper Jr. WR Ankle Did Not "
    "Participate In Practice Out</div>")


def _adapter():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "analysis", "personnel_matchup_example.py")
    spec = importlib.util.spec_from_file_location("pm_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_adapter_nflcom_rows_do_not_bleed_into_next_player(tmp_path):
    f = tmp_path / "nflcom.raw"
    f.write_text(NFLCOM_GB_EXCERPT)
    rows, meta = _adapter().parse_nflcom_report(str(f), "Packers")
    got = {r["name"]: r["report_status"] for r in rows}
    assert got == {"Aaron Banks": "Questionable", "Warren Brinson": "Out",
                   "Brandon Cisse": "(-)", "Javon Hargrave": "Doubtful",
                   "Ty'Ron Hopper": "(-)", "Will McDonald IV": "Questionable"}
    assert meta["complete"] is True


def test_adapter_flags_incomplete_team_table(tmp_path):
    f = tmp_path / "team.raw"
    f.write_text("<p>Player Position Injury Wed Thu Fri Game Status Warren Brinson DT Calf DNP "
                 "DNP DNP OUT Will McDonald IV DE Ankle LP LP QUESTIONABLE Other Team Table - "
                 "Injury report</p>")
    rows, meta = _adapter().parse_team_report(str(f))
    assert meta["complete"] is False and meta["unparsed"]
