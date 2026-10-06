"""Pre-send authorization vs truthful recording of what was sent.

Two different acts, tested on both sides:

* a non-approved pick cannot be authorized for delivery without an explicit, complete
  exception, and an exception never relabels the forecast as approved;
* an already-sent pick that violated the policy is still captured by the append-only
  ledger -- exact text, quote, clock, message id -- flagged, graded in the analyst-issued
  denominator, and never counted as approved.

REHEARSAL ONLY: synthetic cards and message ids; the box is the real official ESPN final
for 2026 week 2 WSH@DAL (event 401872944, kickoff 2026-09-20T20:25Z), shared with the
ledger tests.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue import delivery_policy as dp  # noqa: E402
from nflvalue import issued_grading as ig  # noqa: E402
from nflvalue import issued_ledger as il  # noqa: E402
from nflvalue import pick_cards as pc  # noqa: E402
from nflvalue import settlement as st  # noqa: E402
from scripts import pick_delivery as pdl  # noqa: E402
from scripts import record_issued_pick as rip  # noqa: E402

BOX = Path(__file__).resolve().parent / "fixtures" / "espn_box_2026wk2"
BOX_CAPTURED = "2026-09-22T01:20:00Z"
GAME = "2026_02_WAS_DAL"
KICK = "2026-09-20T20:25:00Z"
DAK = "00-0033077"
AVAIL = json.dumps({"availability": {"status": "OK", "eligibility": "eligible",
                                     "availability_state": "not_listed"}})


def Z(hhmm, day=20):
    return f"2026-09-{day}T{hhmm}:00Z"


def _card(status="research", market="pass_attempts", **kw):
    c = {"game_id": GAME, "player_id": DAK, "player": "Dak Prescott", "market": market, "side": "under",
         "line": 32.5, "run_as_of": Z("10:30"), "model_p_side": 0.6, "model_p_status": "unvalidated_at_offered_lines",
         "mean": 31.0, "sd": 5.0, "status": status, "status_reasons": ["market has not passed an offered-line calibration gate"],
         "quote": {"book": "draftkings", "price_decimal": 1.87, "price_american": "-115", "captured_at": Z("10:00")}}
    c.update(kw)
    return c


EXC = {"by": "curtis", "reason": "balanced-offense read; discretionary 0.25u", "clock": Z("10:45")}


# ------------------------------------------------------------ authorize --
def test_non_actionable_card_is_blocked_without_an_explicit_exception():
    r = dp.authorize(_card("watch"))
    assert r["decision"] == "blocked" and "no explicit exception" in r["reasons"]
    assert r["approval_status"].startswith("none")
    r = dp.authorize(_card("research"))
    assert r["decision"] == "blocked"


def test_complete_exception_allows_sending_but_never_approves():
    r = dp.authorize(_card("research"), EXC)
    assert r["decision"] == "exception" and r["exception_complete"]
    assert r["approval_status"].startswith("none") and "unvalidated" in r["approval_status"]
    line = dp.approval_line(r, 0.25)
    assert line.startswith("APPROVAL: none") and "approved" not in line.lower()
    assert "by curtis" in line and "RISK CAP: 0.25u" in line


@pytest.mark.parametrize("exc", [{"by": "curtis", "reason": "x"}, {"by": "", "reason": "x", "clock": Z("10:45")},
                                 {"by": "curtis", "reason": "x", "clock": "2026-09-20 10:45"}, "curtis said so"])
def test_incomplete_exception_is_blocked(exc):
    r = dp.authorize(_card("research"), exc)
    assert r["decision"] == "blocked" and any("exception" in x for x in r["reasons"])
    assert not r["exception_complete"]


def test_game_lines_are_blocked_citing_the_ledger_decision():
    for kw in ({"market": "spread", "line": -9.0, "player": "MIN"}, {"market": "total", "line": 48.0},
               {"kind": "game_line", "market": "custom"}):
        r = dp.authorize(_card("actionable", **kw))
        assert r["decision"] == "blocked" and dp.GAME_LINE_DECISION in r["reasons"] and r["game_line"]
    # even a complete exception leaves the decision as 'exception', never 'approved'
    r = dp.authorize(_card("actionable", market="spread", line=-9.0), EXC)
    assert r["decision"] == "exception" and r["approval_status"].startswith("none")


def test_actionable_card_is_approved_and_watch_is_a_watch():
    r = dp.authorize(_card("actionable"))
    assert r["decision"] == "approved" and r["approval_status"].startswith("approved")
    assert dp.approval_line(r).startswith("APPROVAL: approved")
    w = dp.authorize(_card("watch"), pick_class="watch")
    assert w["decision"] == "watch" and w["approval_status"].startswith("none")
    assert dp.authorize(_card("research"), pick_class="watch")["decision"] == "blocked"


# -------------------------------------------------------------- prepare --
def _row(**kw):
    r = {"game_id": GAME, "player_id": DAK, "name": "Dak Prescott", "market": "pass_attempts",
         "side": "under", "line": 32.5, "line_source": "odds_api", "price": 1.87, "mean": 31.0, "sd": 5.0,
         "p_side": 0.6, "status": "active", "as_of": Z("10:30"), "quote_book": "draftkings",
         "quote_ts": Z("10:00"), "run_id": "gha:1-1", "code_sha": "abc", "forecast_version": "ff-football-only-v1",
         "selection_source": "football_selection_score", "stage_json": AVAIL, "_quote_verified": True,
         "_run_publish": True}
    r.update(kw)
    return r


def _prepare(tmp_path, rows, extra=(), now=Z("11:00"), out="prep"):
    cards = [pc.build_card(r, pdl._ts(now)) for r in rows]
    cp = tmp_path / "cards.json"
    cp.write_text(json.dumps({"season": 2026, "week": 2, "label": "fresh", "cards": cards}, default=str))
    kp = tmp_path / "kick.json"
    kp.write_text(json.dumps({GAME: KICK}))
    rc = pdl.main(["--now", now, "prepare", "--cards", str(cp), "--season", "2026", "--week", "2",
                   "--kickoffs", str(kp), "--out", str(tmp_path / out), *extra])
    man = tmp_path / out / "manifest.json"
    return rc, (json.loads(man.read_text()) if man.exists() else None), man


def test_prepare_withholds_blocked_cards_and_prints_the_approval_line(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "VALIDATED_MARKETS", frozenset({"pass_attempts"}))
    rows = [_row(),                                                           # actionable -> approved
            _row(market="passing_yards", line=250.5, mean=240.0, sd=60.0),    # watch
            _row(player_id="00-1", name="No Price", line_source="synthetic")]  # research: blocked
    rc, m, _ = _prepare(tmp_path, rows)
    assert rc == 0 and m["counts"] == {"recommendation": 1, "approved": 1, "exception": 0, "watch": 1,
                                       "no_recommendation": 1}
    rec = m["items"][0]
    assert rec["policy"]["decision"] == "approved" and rec["tier"] == "primary"
    assert rec["text"].splitlines()[0] == "APPROVAL: approved: card status actionable (validated market) · EXCEPTION: none · RISK CAP: none"
    held = m["withheld"][0]
    assert held["player"] == "No Price" and held["policy_decision"] == "blocked"
    msg = (tmp_path / "prep" / "message.txt").read_text()
    assert msg.startswith("APPROVAL: approved") and "No Price" not in msg


def test_prepare_exception_needs_by_reason_and_a_named_card(tmp_path):
    rows = [_row()]                                        # watch in production (no validated market)
    key = f"{GAME}|{DAK}|pass_attempts|under|32.5"
    rc, _, _ = _prepare(tmp_path, rows, ["--exception-key", key])                       # no by/reason
    assert rc == 2
    rc, _, _ = _prepare(tmp_path, rows, ["--exception-by", "curtis", "--exception-key", key], out="p2")
    assert rc == 2
    rc, _, _ = _prepare(tmp_path, rows, ["--exception-by", "curtis", "--exception-reason", "lean",
                                         "--exception-key", "nope|x|y|z|1"], out="p3")
    assert rc == 2                                                                      # key matches no card
    rc, m, _ = _prepare(tmp_path, rows, ["--exception-by", "curtis", "--exception-reason", "discretionary lean",
                                         "--exception-key", key, "--risk-cap", "0.25"], out="p4")
    assert rc == 0 and m["counts"]["exception"] == 1 and m["counts"]["approved"] == 0
    it = m["items"][0]
    assert it["pick_class"] == "recommendation" and it["tier"] == "analyst_override"
    assert it["exception"] == {"by": "curtis", "reason": "discretionary lean", "clock": Z("11:00")}
    first = it["text"].splitlines()[0]
    assert first.startswith("APPROVAL: none") and "by curtis: discretionary lean" in first and "0.25u" in first
    assert it["text"].splitlines()[1].startswith("ANALYST EXCEPTION PICK (not model-approved)")
    assert "approved" not in first.lower()
    # the card itself is untouched: still unvalidated
    assert it["card"]["model_p_status"] == "unvalidated_at_offered_lines" and it["card"]["status"] == "watch"


# ------------------------------------------------------------- ledger --
def test_sent_violating_pick_is_captured_flagged_and_not_promoted(tmp_path):
    conn = dbmod.connect(str(tmp_path / "l.db"))
    text = "Cousins UNDER 34.5 pass attempts, DK -121 (2:01 pm ET quote)"
    card = _card("research")
    rec = il.record_delivered(conn, 2026, 2, card, text, "tg-77", Z("11:05"), "telegram",
                              kickoff=KICK, recorded_at=Z("11:06"))
    assert rec["tier"] == "analyst_override" and rec["pick_class"] == "recommendation"
    assert rec["policy"] == {"decision": "blocked", "violation": True, "tier_source": "derived_from_policy",
                             "delivery_evidence_kind": "live_message"}
    stored = il.load(conn)[0]
    assert stored["display_html"] == text and stored["quote_book"] == "draftkings" and stored["quote_price"] == 1.87
    assert stored["quote_ts"] == Z("10:00") and stored["decision_ts"] == Z("10:30")
    ev = json.loads(stored["events"][0]["evidence_json"])
    assert ev["policy_violation"] is True and ev["policy_decision"] == "blocked" and ev["message_id"] == "tg-77"
    assert ev["approval_status"].startswith("none") and ev["exception"] is None
    assert json.loads(stored["card_json"])["model_p_status"] == "unvalidated_at_offered_lines"
    # an explicit complete exception: recorded as exception, no violation, still analyst_override
    rec2 = il.record_delivered(conn, 2026, 2, card, text + " (exception)", "tg-78", Z("11:07"), "telegram",
                               exception=EXC, kickoff=KICK, recorded_at=Z("11:08"))
    assert rec2["policy"]["decision"] == "exception" and rec2["policy"]["violation"] is False
    assert rec2["tier"] == "analyst_override"
    # an explicit tier is kept and labelled explicit; it does not change the violation flag
    rec3 = il.record_delivered(conn, 2026, 2, card, text + " (x)", "tg-79", Z("11:09"), "telegram",
                               tier="experimental", kickoff=KICK, recorded_at=Z("11:10"))
    assert rec3["tier"] == "experimental" and rec3["policy"]["tier_source"] == "explicit"
    assert rec3["policy"]["violation"] is True
    conn.close()


def test_evidence_kind_is_live_only_before_a_supplied_kickoff(tmp_path):
    conn = dbmod.connect(str(tmp_path / "k.db"))
    card = _card("research")
    late = il.record_delivered(conn, 2026, 2, card, "t", "m1", Z("21:00"), "chat", kickoff=KICK, recorded_at=Z("21:01"))
    assert late["policy"]["delivery_evidence_kind"] == "retrospective_import"
    flagged = il.record_delivered(conn, 2026, 2, card, "t2", "m2", Z("11:00"), "chat", kickoff=KICK,
                                  retrospective=True, recorded_at=Z("11:01"))
    assert flagged["policy"]["delivery_evidence_kind"] == "retrospective_import"
    unknown = il.record_delivered(conn, 2026, 2, card, "t3", "m3", Z("11:00"), "chat", recorded_at=Z("11:01"))
    assert unknown["policy"]["delivery_evidence_kind"].startswith("kickoff_not_supplied")
    # the ledger still refuses what it always refused
    with pytest.raises(ValueError, match="exact text"):
        il.record_delivered(conn, 2026, 2, card, "  ", "m4", Z("11:00"), "chat")


def test_record_cli_previews_policy_and_still_records_the_violation(tmp_path, capsys):
    card = _card("research")
    (tmp_path / "card.json").write_text(json.dumps(card))
    (tmp_path / "msg.txt").write_text("Prescott under 32.5 pass attempts (DK -115)")
    base = ["--db", str(tmp_path / "a.db"), "delivered", "--season", "2026", "--week", "2", "--card",
            str(tmp_path / "card.json"), "--text-file", str(tmp_path / "msg.txt"), "--channel", "chat",
            "--delivered-at", Z("11:00"), "--message-id", "msg-1", "--kickoff", KICK]
    assert rip.main(base + ["--exception-by", "curtis"]) == 2           # half an exception: nothing recorded
    assert rip.main(base) == 0
    out = capsys.readouterr().out
    assert "policy_decision=blocked policy_violation=True" in out and "tier=analyst_override (derived)" in out
    assert "kept and flagged, never promoted" in out
    conn = sqlite3.connect(tmp_path / "a.db")
    recs = il.load(conn)
    conn.close()
    assert len(recs) == 1 and recs[0]["display_html"].startswith("Prescott under")
    assert json.loads(recs[0]["events"][0]["evidence_json"])["policy_violation"] is True


# ------------------------------------------------------------- grading --
def test_violating_sent_pick_is_graded_in_the_denominator_but_never_as_approved(tmp_path, monkeypatch):
    conn = dbmod.connect(str(tmp_path / "g.db"))
    # 1) a sent pick the policy would have blocked (research card given as a recommendation)
    il.record_delivered(conn, 2026, 2, _card("research"), "Prescott UNDER 32.5 (DK -115)", "m-v", Z("11:00"), "chat",
                        kickoff=KICK, recorded_at=Z("11:01"))
    # 2) an explicit exception on a different line
    il.record_delivered(conn, 2026, 2, _card("research", line=33.5), "Prescott UNDER 33.5 (DK -115)", "m-e",
                        Z("11:02"), "chat", exception=EXC, kickoff=KICK, recorded_at=Z("11:03"))
    # 3) an approved pick (validated market for the test only)
    monkeypatch.setattr(pc, "VALIDATED_MARKETS", frozenset({"pass_attempts"}))
    il.record_delivered(conn, 2026, 2, _card("actionable", line=31.5), "Prescott UNDER 31.5 (DK -115)", "m-a",
                        Z("11:04"), "chat", kickoff=KICK, recorded_at=Z("11:05"))
    recs = il.load(conn)
    conn.close()
    res = ig.grade(recs, ig.load_boxes([str(BOX / "401872944.json")], BOX_CAPTURED))
    assert res["counts"]["recommendations_given"] == 3                 # all three given picks are graded
    assert res["counts"]["policy_violation_records"] == 1
    assert res["counts"]["policy_exception_records"] == 1 and res["counts"]["policy_approved_records"] == 1
    sec = res["sections"]["recommendations_given"]
    by_line = {r["line"]: r for r in sec["rows"]}
    assert by_line[32.5]["policy_class"] == "violation" and by_line[32.5]["policy_violation"] is True
    assert by_line[33.5]["policy_class"] == "exception" and by_line[31.5]["policy_class"] == "approved"
    assert all(r["settlement"] in st.SETTLEMENTS for r in sec["rows"])
    pg = sec["policy_groups"]
    assert pg["policy:approved|all_slates"]["n_rows"] == 1 and pg["policy:violation|all_slates"]["n_rows"] == 1
    assert pg["policy:exception|all_slates"]["n_rows"] == 1
    # the approved group never contains the violating or exception pick
    assert "policy:approved|all_slates" in pg and pg["policy:approved|all_slates"]["n_rows"] == 1
    assert by_line[32.5]["tier"] == "analyst_override" and by_line[31.5]["tier"] == "primary"
