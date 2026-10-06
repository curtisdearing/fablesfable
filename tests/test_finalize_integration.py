"""Caller-path integration of the finalize slices (settlement -> research loop -> public site).

Trusted pre-kickoff capture comes only from the issued ledger's own clock; the results-only job
appends the cumulative evidence ledger after verified grading and persists it through
scripts/state_store.py; a delivered card is imported without its private text; every page keeps
publish, forecast, quote and settlement clocks apart. Frozen fixtures: engineering checks only,
never live-operation or edge evidence.
"""
import datetime as dt
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from analysis import evidence_loop as el
from nflvalue import db as dbmod
from nflvalue import issued_grading as ig
from nflvalue import issued_ledger as il
from nflvalue import issued_results as ir

sys.path.insert(0, str(Path(__file__).parent))
import test_issued_game_total_settlement as mnf  # noqa: E402
import test_issued_results_loop as loop  # noqa: E402
import test_pregame_scheduler as tps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import record_issued_pick as rip  # noqa: E402
import state_store  # noqa: E402

PRIVATE = "PRIVATE ORIGINAL MESSAGE TEXT that must never reach the ledger"


def _grade_one(tmp_path, *, week, delivered_at, recorded_at, retrospective):
    conn = dbmod.connect(str(tmp_path / f"w{week}-{recorded_at[:13]}.db"))
    il.record_delivered(conn, 2026, week, mnf.SHOUGH, "Shough over 258.5 passing yards, FanDuel -113",
                        message_id="m1", delivered_at=delivered_at, channel="chat", kickoff=mnf.KICK,
                        retrospective=retrospective, recorded_at=recorded_at)
    graded = ig.grade(il.load(conn), ig.load_boxes([str(mnf._box_file(tmp_path))], mnf.CAPTURED))
    conn.close()
    return graded


def test_only_a_ledger_clocked_pre_kickoff_receipt_is_prospective_confirmation(tmp_path):
    live = _grade_one(tmp_path, week=4, delivered_at="2026-10-05T23:00:00Z",
                      recorded_at="2026-10-05T23:00:05Z", retrospective=False)
    row = el.from_issued_grading(live)[0]
    assert el.parse_aware(row["kickoff"]) == el.parse_aware(mnf.KICK)        # the real grader supplies kickoff
    assert el.SHA256_RE.match(row["capture_sha256"]) and row["capture_sha256"] != row["record_id"]
    assert row["capture_basis"] == el.TRUSTED_CAPTURE and not row["historical_import"]
    assert el.trusted_capture(row)
    assert el.window_of(row) == "retrospective_exploratory"                  # week 4 was already read
    assert el.window_of({**row, "week": 5}) == "prospective_confirmation"
    # the same pregame claim imported after the game: graded honestly, never confirmation
    late = _grade_one(tmp_path, week=4, delivered_at="2026-10-05T23:00:00Z",
                      recorded_at="2026-10-06T04:00:00Z", retrospective=True)
    assert late["sections"]["recommendations_given"]["rows"] == []
    hist = el.from_issued_grading(late, section="retrospective")[0]
    assert hist["historical_import"] and hist["capture_sha256"] is None
    assert el.window_of({**hist, "week": 5}) == "retrospective_exploratory"  # pregame decision kept, not relabelled late


@pytest.mark.parametrize("forged", [
    {"capture_sha256": "abc"},                                           # any nonempty string
    {"capture_basis": "source_says_pregame"},                            # a claim, not a ledger receipt
    {"historical_import": True},
    {"capture_recorded_at": "2026-10-11T18:00:00+00:00"},                # ledger saw it after kickoff
    {"capture_recorded_at": None},
])
def test_content_ids_backdated_clocks_and_imports_never_qualify(forged):
    row = {"season": 2026, "week": 5, "decision_ts": "2026-10-11T15:00:00+00:00", "kickoff": "2026-10-11T17:00:00Z",
           "capture_sha256": "a" * 64, "capture_recorded_at": "2026-10-11T15:00:01+00:00",
           "capture_basis": el.TRUSTED_CAPTURE, "historical_import": False}
    assert el.window_of(row) == "prospective_confirmation"
    assert el.window_of({**row, **forged}) == "retrospective_exploratory"


def test_confirmation_is_untouched_only_after_the_challengers_freeze():
    row = {"season": 2026, "week": 5, "decision_ts": "2026-10-11T15:00:00+00:00", "kickoff": "2026-10-11T17:00:00Z",
           "capture_sha256": "a" * 64, "capture_recorded_at": "2026-10-11T15:00:01+00:00",
           "capture_basis": el.TRUSTED_CAPTURE}
    assert el.unused_confirmation(row, "2026-10-06T04:07:37Z")
    assert not el.unused_confirmation(row, "2026-10-12T00:00:00Z")


def _results_job(tmp_path, monkeypatch, fetcher):
    aw = loop._aw()
    db_path = str(tmp_path / "state.db")
    if not Path(db_path).exists():
        mnf._ledger(tmp_path).close()
        Path(tmp_path / "ledger.db").rename(db_path)
    real_connect = dbmod.connect
    monkeypatch.setattr(dbmod, "connect", lambda *a, **k: real_connect(db_path))
    monkeypatch.setattr(ir, "fetch", fetcher)
    monkeypatch.setattr(aw, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(aw, "write_pipeline_heartbeat", lambda *a: None)
    (tmp_path / "data").mkdir(exist_ok=True)
    return aw, db_path


def test_results_job_appends_the_evidence_ledger_after_grading_and_is_idempotent(tmp_path, monkeypatch):
    aw, db_path = _results_job(tmp_path, monkeypatch, loop.Fetcher())
    monkeypatch.setattr(aw, "now_et", lambda: loop.NOW)
    assert aw.job_results() == 0
    ledger = tmp_path / "data/evidence_ledger.jsonl"
    entries = el.read_ledger(str(ledger))
    assert len(entries) == 2 and {e["row"]["section"] for e in entries} == {"retrospective"}
    assert all(e["row"]["historical_import"] and el.window_of(e["row"]) != "prospective_confirmation"
               for e in entries)
    status = json.loads((tmp_path / "data/research_status.json").read_text())
    assert status["sections"]["retrospective"]["outcomes"]["loss"] == 1          # the loss is kept
    assert status["promotion"]["passed_predeclared_gate"] == [] and not status["promotion"]["market_blend"]
    assert any(g.startswith("prospective confirmation: 0") for g in status["missing_data_gates"])
    summary = json.loads((tmp_path / "reports/results/summary.json").read_text())
    assert summary["evidence"]["appended"] == 2
    head = ledger.read_bytes()
    # identical official source, after the recheck window: no grade and no evidence row
    monkeypatch.setattr(aw, "now_et", lambda: loop.NOW + dt.timedelta(hours=ir.RECHECK_HOURS + 1))
    assert aw.job_results() == 0
    assert ledger.read_bytes() == head
    assert json.loads((tmp_path / "reports/results/summary.json").read_text())["written"] == 0
    # a stat revision: exactly one reconciled revision, then identical re-reads add nothing
    monkeypatch.setattr(ir, "fetch", loop.Fetcher(summary=loop._summary(shough_yards=250)))
    for hours in (2, 3):
        monkeypatch.setattr(aw, "now_et", lambda h=hours: loop.NOW + dt.timedelta(hours=h * (ir.RECHECK_HOURS + 1)))
        assert aw.job_results() == 0
    entries = el.read_ledger(str(ledger))
    first = next(e for e in entries if e["row"]["evidence_id"] == entries[-1]["row"]["evidence_id"])
    assert len(entries) == 3 and entries[-1]["revision_of"] == first["content_sha256"]
    assert (first["row"]["market"], first["row"]["outcome"], entries[-1]["row"]["outcome"]) == \
        ("passing_yards", "win", "loss")                                    # 286 -> 250 under 258.5
    issued = sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM issued_picks").fetchone()[0]
    assert issued == 3                                                     # frozen records untouched


def test_evidence_ledger_has_a_single_writer(tmp_path):
    import fcntl
    path = str(tmp_path / "data" / "evidence_ledger.jsonl")
    (tmp_path / "data").mkdir()
    with open(path + ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            el.append_evidence(path, [{"evidence_id": "x"}])


def test_state_archive_carries_the_evidence_ledger_and_research_status(tmp_path):
    (tmp_path / "data").mkdir()
    for name in ("evidence_ledger.jsonl", "research_status.json", "nfl_props.db"):
        (tmp_path / "data" / name).write_text("{}")
    names = {p.name for p in state_store.state_files(tmp_path)}
    assert {"evidence_ledger.jsonl", "research_status.json"} <= names
    assert state_store.STATE_ROOTS == frozenset({"data", "drops"})


def _card_input(**over):
    doc = {"schema": rip.CARD_SCHEMA, "season": 2026, "week": 4, "game_id": mnf.GAME, "kickoff": mnf.KICK,
           "channel": "chat", "capture": "historical_postgame",
           "source": {"source_id": "analyst-chat:session-x/message-1", "issued_at": "2026-10-06T00:11:17Z",
                      "content_sha256": hashlib.sha256(PRIVATE.encode()).hexdigest()},
           "selections": [
               {"role": "recommendation", "player_id": "00-0039152", "player": "Tyler Shough",
                "market": "passing_yards", "side": "over", "line": 258.5, "book": "fanduel", "american": -113,
                "model_p_side": 0.5749455726414532, "mean": 276.869,
                "status_text": "preferred research lean; not model-approved",
                "text": "Tyler Shough over 258.5 passing yards, FanDuel -113"},
               {"role": "recommendation", "player_id": None, "player": "Game total", "market": "game_total",
                "side": "under", "line": 47.5, "book": "draftkings", "american": -102, "model_p_side": 0.6476,
                "mean": 42.7, "status_text": "weaker research lean; not model-approved",
                "text": "ATL at NO under 47.5 points, DraftKings -102"},
               {"role": "pass", "player_id": "00-0035656", "player": "Juwan Johnson", "market": "receiving_yards",
                "side": "over", "line": 46.5, "book": "fanduel", "american": -113,
                "model_p_side": 0.503233017477728, "mean": 54.687, "status_text": "explicit PASS, not a recommendation",
                "text": "PASS: Juwan Johnson over 46.5 receiving yards"}]}
    doc.update(over)
    return doc


def test_delivered_card_import_records_both_leans_and_the_pass_without_private_text(tmp_path, capsys):
    db = tmp_path / "led.db"
    inp = tmp_path / "card.json"
    inp.write_text(json.dumps(_card_input()))
    args = ["--db", str(db), "delivered-card", "--input", str(inp), "--now", "2026-10-06T05:00:00Z"]
    assert rip.main(args) == 0
    out = capsys.readouterr().out
    assert out.count("retrospective_import") == 2 and out.count("violation=True") == 2 and "pass:" in out
    assert rip.main(args) == 0                                              # re-import: no duplicates
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM issued_picks").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM issued_pick_events").fetchone()[0] == 3
    dump = "\n".join(conn.iterdump())
    assert PRIVATE not in dump and hashlib.sha256(PRIVATE.encode()).hexdigest() in dump
    conn.close()
    doc = ir.export_readonly(str(db))
    conn = dbmod.connect(str(db))
    ir.settle(conn, now=loop.NOW, http=loop.Fetcher())
    doc = ir.export(conn)
    conn.close()
    retro = {r["market"]: r for r in doc["sections"]["retrospective"]["rows"]}
    assert (retro["passing_yards"]["settlement"], retro["game_total"]["settlement"]) == ("win", "loss")
    assert {r["policy_class"] for r in retro.values()} != {"approved"}
    assert doc["sections"]["recommendations_given"]["rows"] == []           # postgame import is never "given"


def test_live_pregame_capture_after_kickoff_is_refused(tmp_path):
    conn = dbmod.connect(str(tmp_path / "x.db"))
    with pytest.raises(ValueError, match="historical_postgame"):
        rip.record_card_input(conn, _card_input(capture="live_pregame"), now="2026-10-06T05:00:00Z")
    with pytest.raises(ValueError, match="sha256"):
        rip.record_card_input(conn, _card_input(source={"source_id": "s", "issued_at": "2026-10-06T00:11:17Z",
                                                        "content_sha256": "abc"}), now="2026-10-06T05:00:00Z")
    conn.close()


def test_republishing_saved_output_never_claims_a_new_forecast():
    import build_public_site as bps
    payload = {"runs": [{"as_of": "2026-10-04T22:50:31Z", "code_sha": "44de45e" + "0" * 33,
                         "forecast_version": "football-only-v1"}],
               "quote_clocks": {"earliest": "2026-10-04T16:00:00Z", "latest": "2026-10-04T22:50:31Z",
                                "leans_with_quote": 30}}
    stale = bps.clocks(payload, "fresh", "2026-10-06T02:58:23+00:00", None, None)
    assert not stale["new_forecast_in_this_publication"] and stale["forecast_age_hours_at_publish"] > 24
    assert {"settlement check", "research evidence loop"} <= set(stale["missing"])
    assert "NO NEW FORECAST OR PRICE" in bps.clock_line(stale)
    fresh = bps.clocks(payload, "fresh", "2026-10-04T23:05:00+00:00", "2026-10-05T05:40:00Z", "2026-10-05T05:40:00Z")
    assert fresh["new_forecast_in_this_publication"] and fresh["missing"] == []
    assert not bps.clocks(payload, "results", "2026-10-04T23:05:00+00:00")["new_forecast_in_this_publication"]


def test_public_build_exports_research_status_and_clocks(tmp_path):
    import build_public_site as bps
    import test_public_site_generator as tpg
    db = tpg._settled_db(tmp_path)
    (Path(db).parent / "research_status.json").write_text(json.dumps(
        {"schema": "fablesfable.research_status.v1", "checked_at": "2026-10-06T09:40:00Z",
         "missing_data_gates": ["prospective confirmation: 0 settled"], "candidates": [],
         "sections": {}, "prospective": {}}))
    out = tmp_path / "site" / "published-site"
    assert bps.main(["--db", str(db), "--season", "2026", "--week", "3", "--archive", str(tpg._archive(tmp_path)),
                     "--out", str(out), "--label", "fresh", "--now", "2026-10-06T10:00:00Z"]) == 0
    tpg._run_checker(out.parent)
    m = json.loads((out / "publication.json").read_text())
    assert {"research.html", "api/research.json"} <= set(m["files"])
    assert m["clocks"]["research_checked_at"] == "2026-10-06T09:40:00Z" and m["clocks"]["results_updated_at"]
    assert m["clocks"]["new_forecast_in_this_publication"] is False          # saved week-3 cards, not new
    assert "NO NEW FORECAST OR PRICE" in (out / "index.html").read_text()
    assert "prospective confirmation: 0 settled" in (out / "research.html").read_text()
    cov = json.loads((out / "api" / "coverage.json").read_text())
    assert {"coverage.html", "api/coverage.json"} <= set(m["files"]) and cov["runs"]
    assert all(r["recorded"] is False for r in cov["runs"])                 # old runs: missing, not invented


def test_scheduler_retry_uses_the_current_kickoff_and_never_relaunches_a_dispatched_slot(tmp_path):
    moved = "2026-10-09T00:30:00Z"
    h = tps.Harness(tmp_path, {"current": tps._board(5, [tps._ev("1", "TB", "DAL", "2026-10-09T00:15:00Z")])})
    h.tick("2026-10-08T22:46:00Z")
    h.finish("not ready: refused before dispatch", dispatched=False)
    h.fetch = tps._fetch({"current": tps._board(5, [tps._ev("1", "TB", "DAL", moved)])})
    h.tick("2026-10-08T23:01:00Z")
    cmd = h.spawned[-1][0]
    assert cmd[cmd.index("--kickoff") + 1] == moved.replace("Z", "+00:00") or cmd[cmd.index("--kickoff") + 1] == moved
    h.finish("dispatched; NOT processed: 2026_05_TB_DAL", dispatched=True)
    launches = len(h.spawned)
    for t in ("2026-10-08T23:06:00Z", "2026-10-08T23:11:00Z"):
        h.tick(t)
    assert all("--execute" not in c for c, _ in h.spawned[launches:])      # read-back only, no second charge


class _FakeOdds:
    def __init__(self, answered):
        self.answered, self.resnapped = answered, []

    def answered_since(self, conn, game_ids, now=None, max_age_hours=1.0):
        return {g: self.answered[g] for g in game_ids if g in self.answered}

    def resnap_lines(self, cfg, emap, conn=None):
        self.resnapped.append(dict(emap))
        return {"pulled": list(emap), "empty": [], "rows_written": 12, "budget_remaining": 100.0}

    def latest_snapshots(self, conn, game_ids, now=None, max_age_hours=1.0):
        return {g: {"ts": "2026-10-11T15:31:00Z", "n_rows": 12, "fresh": True} for g in game_ids}

    def billing_text(self, res):
        return "7 credits"


class _FakePipeline:
    def __init__(self):
        self.asked = []

    def build_event_map(self, cfg, slate, details=None):
        self.asked.append(sorted(slate["game_id"]))
        details["games"] = {"G4": {"reason": "event_outside_kickoff_window"}}
        return {g: f"ev-{g}" for g in slate["game_id"] if g != "G4"}


def test_closing_resnap_never_rebills_answered_games_and_reports_every_game(capsys):
    import pandas as pd
    aw = loop._aw()
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE lines (game_id TEXT)")
    conn.executemany("INSERT INTO lines VALUES (?)", [("G1",), ("G2",), ("G3",), ("G4",)])
    soon = pd.DataFrame({"game_id": ["G1", "G2", "G3", "G4", "G5"]})
    odds, pipe = _FakeOdds({"G1": "2026-10-11T15:20:00Z"}), _FakePipeline()
    out = aw.closing_resnap(conn, {}, soon, {"G3"}, odds, pipe, now=dt.datetime(2026, 10, 11, 15, 35,
                                                                             tzinfo=dt.timezone.utc))
    assert out["skipped_recent"] == ["G1"] and out["resnapped"] == ["G2", "G4"]
    assert pipe.asked == [["G2", "G4"]] and odds.resnapped == [{"G2": "ev-G2"}]     # G1 never billed twice
    text = capsys.readouterr().out
    assert "[auto] closing resnap G1: skipped, provider answered" in text
    assert "[auto] closing resnap G2: 12 rows at 2026-10-11T15:31:00Z" in text
    assert "[auto] closing resnap G4: no provider event (event_outside_kickoff_window)" in text
    # a re-fired slot after the close was taken: nothing is billed again
    again = aw.closing_resnap(conn, {}, soon, {"G3"}, _FakeOdds({"G1": "t", "G2": "t", "G4": "t"}), pipe)
    assert again["resnapped"] == [] and pipe.asked == [["G2", "G4"]]
