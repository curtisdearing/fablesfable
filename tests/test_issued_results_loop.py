"""The results-only settlement loop: ledger -> official finals -> append-only grades -> public results.

Uses the real 2026 week-4 Monday final (ATL at NO) frozen from its capture receipts; the
scoreboard/summary are served by a fake fetcher, so nothing here touches the network. Fixture
replays are engineering checks, not live-operation evidence.
"""

import datetime as dt
import gzip
import json
import re
import sys
from pathlib import Path

import pytest

from nflvalue import db as dbmod
from nflvalue import issued_ledger as il
from nflvalue import issued_results as ir

sys.path.insert(0, str(Path(__file__).parent))
import test_issued_game_total_settlement as mnf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIX = mnf.FIX
NOW = dt.datetime(2026, 10, 6, 9, 40, tzinfo=dt.timezone.utc)      # the daily 05:40Z slot, run late


def _summary(**edit):
    original = gzip.decompress((FIX / "espn-summary.json.gz").read_bytes())
    if not edit:
        return original                                   # the captured bytes, sha256 as receipted
    raw = json.loads(original)
    if "shough_yards" in edit:
        for t in raw["boxscore"]["players"]:
            for cat in t["statistics"]:
                for a in cat["athletes"]:
                    if a["athlete"]["displayName"] == "Tyler Shough" and cat["name"] == "passing":
                        a["stats"][cat["keys"].index("passingYards")] = str(edit["shough_yards"])
    if "drop_shough" in edit:
        for t in raw["boxscore"]["players"]:
            for cat in t["statistics"]:
                cat["athletes"] = [a for a in cat["athletes"] if a["athlete"]["displayName"] != "Tyler Shough"]
    return json.dumps(raw).encode()


def _board(status="STATUS_FINAL", completed=True):
    raw = json.loads(gzip.decompress((FIX / "scoreboard.json.gz").read_bytes()))
    for ev in raw["events"]:
        for c in ev["competitions"]:
            c["status"]["type"].update(name=status, completed=completed)
    return json.dumps(raw).encode()


class Fetcher:
    def __init__(self, board=None, summary=None):
        self.board, self.summary, self.urls = board or _board(), summary or _summary(), []

    def __call__(self, url):
        self.urls.append(url)
        if "/scoreboard?" in url:
            return self.board
        assert url == ir.summary_url("401872979"), url
        return self.summary


def _ledger(tmp_path):
    return mnf._ledger(tmp_path)


def _given(doc):
    return {r["market"]: r for r in doc["sections"]["retrospective"]["rows"]}


def test_real_monday_final_settles_the_delivered_card(tmp_path):
    conn = _ledger(tmp_path)
    frozen = il.export(il.load(conn))
    f = Fetcher()
    out = ir.settle(conn, now=NOW, http=f, evidence_dir=str(tmp_path / "evidence"))
    assert out["weeks"] == [[2026, 4]] and out["graded_games"] == ["2026_04_ATL_NO"]
    assert f.urls == [ir.scoreboard_url(2026, 4), ir.summary_url("401872979")]
    assert out["results_written"] == 2 and out["captures"] == 1 and out["errors"] == []
    doc = ir.export(conn, checked_at=out["checked_at"])
    rows = _given(doc)
    assert (rows["passing_yards"]["settlement"], rows["passing_yards"]["actual"]) == ("win", 286.0)
    assert (rows["game_total"]["settlement"], rows["game_total"]["actual"]) == ("loss", 69.0)
    # postgame capture of a chat card: retrospective, a policy violation, never "given before kickoff"
    assert doc["sections"]["recommendations_given"]["rows"] == []
    assert {r["delivery_evidence_kind"] for r in rows.values()} == {"retrospective_import"}
    assert {r["policy_class"] for r in rows.values()} == {"violation"}
    # the PASS is never settled into any section and is not pending either
    assert all(r["market"] != "receiving_yards" for s in doc["sections"].values() for r in s["rows"])
    assert doc["pending"] == []
    # provenance: the public URL and the capture's sha256, never a local path
    assert {r["actuals_url"] for r in rows.values()} == {ir.summary_url("401872979")}
    sha = json.loads((FIX / "source-receipt.json").read_text())["sha256"]
    assert {r["actuals_sha256"] for r in rows.values()} == {sha}
    assert "/tmp" not in json.dumps(doc) and "results-boxes-" not in json.dumps(doc)
    # issued records (the frozen predictions) are untouched
    assert il.export(il.load(conn)) == frozen


def test_two_runs_are_idempotent_and_a_recheck_of_identical_stats_writes_no_grade(tmp_path):
    conn = _ledger(tmp_path)
    ir.settle(conn, now=NOW, http=Fetcher())
    before = conn.execute("SELECT COUNT(*) FROM issued_results").fetchone()[0]
    f = Fetcher()
    again = ir.settle(conn, now=NOW + dt.timedelta(hours=1), http=f)
    assert again["written"] == 0 and again["skipped_recent"] == ["2026_04_ATL_NO"]
    assert f.urls == [ir.scoreboard_url(2026, 4)]                     # no second box read inside the window
    later = ir.settle(conn, now=NOW + dt.timedelta(hours=RECHECK + 1), http=Fetcher())
    assert later["results_written"] == 0 and later["captures"] == 1     # re-read, receipt kept, no new grade
    assert conn.execute("SELECT COUNT(*) FROM issued_results").fetchone()[0] == before


RECHECK = ir.RECHECK_HOURS


def test_stat_correction_appends_and_supersedes_without_touching_the_issued_record(tmp_path):
    conn = _ledger(tmp_path)
    ir.settle(conn, now=NOW, http=Fetcher())
    frozen = il.export(il.load(conn))
    ir.settle(conn, now=NOW + dt.timedelta(hours=RECHECK + 1), http=Fetcher(summary=_summary(shough_yards=250)))
    rows = _given(ir.export(conn))
    sh = rows["passing_yards"]
    assert (sh["settlement"], sh["actual"]) == ("loss", 250.0)
    assert [(c["settlement"], c["actual"]) for c in sh["corrections"]] == [("win", 286.0)]
    assert sh["supersedes"] is not None
    assert rows["game_total"]["corrections"] == []                    # unchanged total: no new row
    assert il.export(il.load(conn)) == frozen
    with pytest.raises(Exception, match="append-only"):
        conn.execute("UPDATE issued_results SET settlement='win'")
    with pytest.raises(Exception, match="append-only"):
        conn.execute("DELETE FROM result_captures")


def test_live_game_stays_pending_then_settles_when_final(tmp_path):
    conn = _ledger(tmp_path)
    f = Fetcher(board=_board("STATUS_IN_PROGRESS", completed=False))
    live = ir.settle(conn, now=NOW, http=f)
    assert f.urls == [ir.scoreboard_url(2026, 4)]                     # no box read for a live game
    assert live["written"] == 0 and live["pending"] == [{"game_id": "2026_04_ATL_NO",
                                                          "reason": "not final (STATUS_IN_PROGRESS)"}]
    doc = ir.export(conn)
    assert all(not s["rows"] for s in doc["sections"].values()) and len(doc["pending"]) == 2
    final = ir.settle(conn, now=NOW + dt.timedelta(minutes=30), http=Fetcher())
    assert final["results_written"] == 2 and ir.export(conn)["pending"] == []


@pytest.mark.parametrize("status", ["STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_SCHEDULED"])
def test_postponed_or_unplayed_game_is_never_zeroed_or_voided(tmp_path, status):
    conn = _ledger(tmp_path)
    out = ir.settle(conn, now=NOW, http=Fetcher(board=_board(status, completed=False)))
    assert out["written"] == 0 and conn.execute("SELECT COUNT(*) FROM issued_results").fetchone()[0] == 0
    assert len(ir.export(conn)["pending"]) == 2


def test_player_missing_from_the_final_box_is_unresolved_not_zero(tmp_path):
    conn = _ledger(tmp_path)
    ir.settle(conn, now=NOW, http=Fetcher(summary=_summary(drop_shough=True)))
    sh = _given(ir.export(conn))["passing_yards"]
    assert sh["settlement"] == "unresolved" and sh["actual"] is None and sh["hit"] is None


def test_a_non_final_summary_is_refused_and_recorded_as_a_refused_capture(tmp_path):
    conn = _ledger(tmp_path)
    raw = json.loads(_summary())
    raw["header"]["competitions"][0]["status"]["type"].update(name="STATUS_IN_PROGRESS", completed=False)
    out = ir.settle(conn, now=NOW, http=Fetcher(summary=json.dumps(raw).encode()))
    assert out["results_written"] == 0 and out["captures"] == 1
    assert conn.execute("SELECT accepted, reason FROM result_captures").fetchone()[0] == 0
    assert out["pending"][0]["reason"].startswith("final box refused")


def test_bounded_lookback_and_request_budget(tmp_path, monkeypatch):
    conn = _ledger(tmp_path)
    f = Fetcher()
    old = ir.settle(conn, now=NOW + dt.timedelta(days=ir.LOOKBACK_DAYS + 1), http=f)
    assert old["weeks"] == [] and f.urls == []
    monkeypatch.setattr(ir, "MAX_REQUESTS", 1)
    capped = ir.settle(conn, now=NOW, http=Fetcher())
    assert capped["requests"] == 1 and capped["written"] == 0
    assert any("request budget" in e for e in capped["errors"])


def test_fetch_failures_leave_games_pending_and_exit_cleanly(tmp_path):
    conn = _ledger(tmp_path)

    def down(url):
        raise RuntimeError("503")
    out = ir.settle(conn, now=NOW, http=down)
    assert out["written"] == 0 and out["errors"] and out["pending"][0]["reason"] == "scoreboard unavailable"


# ------------------------------------------------------------------ the scheduled job (caller) --
def _aw():
    sys.path.insert(0, str(ROOT / "scripts"))
    import auto_weekly
    return auto_weekly


def test_results_job_is_results_only_and_writes_state_export_and_summary(tmp_path, monkeypatch):
    aw = _aw()
    db_path = str(tmp_path / "state.db")
    seed = _ledger(tmp_path)
    seed.close()
    import shutil
    shutil.copy(tmp_path / "ledger.db", db_path)
    real_connect = dbmod.connect
    monkeypatch.setattr(dbmod, "connect", lambda *a, **k: real_connect(db_path))
    monkeypatch.setattr(ir, "fetch", Fetcher())
    monkeypatch.setattr(aw, "now_et", lambda: NOW)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(aw, "RESULTS_ROOT", tmp_path)
    beats = []
    monkeypatch.setattr(aw, "write_pipeline_heartbeat", lambda *a: beats.append(a))

    def forbidden(*a, **k):
        raise AssertionError("the results job must not ingest feeds, price, fit or notify")
    from nflvalue import ingest, notify
    monkeypatch.setattr(ingest, "refresh", forbidden)
    monkeypatch.setattr(notify, "resolve_webhook", forbidden)
    monkeypatch.setitem(sys.modules, "pipeline_weekly", None)              # importing it would fail
    monkeypatch.setattr(aw.subprocess if hasattr(aw, "subprocess") else __import__("subprocess"),
                        "run", forbidden)
    assert aw.job_results() == 0
    summary = json.loads((tmp_path / "reports/results/summary.json").read_text())
    # 2 grades + 1 capture + 2 research evidence rows (appended after verified grading)
    assert summary["written"] == 5 and summary["results_written"] == 2 and summary["evidence"]["appended"] == 2
    doc = json.loads((tmp_path / "data/issued_results.json").read_text())
    assert {r["settlement"] for r in doc["sections"]["retrospective"]["rows"]} == {"win", "loss"}
    assert beats and beats[-1][0] == "active" and beats[-1][2] == "results"
    assert list((tmp_path / "reports/results/boxes").glob("2026_04_ATL_NO-*.json"))
    # second run inside the recheck window: nothing new to save or publish
    assert aw.job_results() == 0
    assert json.loads((tmp_path / "reports/results/summary.json").read_text())["written"] == 0


def test_results_job_fails_closed_on_a_ledger_failure(tmp_path, monkeypatch):
    aw = _aw()
    monkeypatch.setattr(aw, "RESULTS_ROOT", tmp_path)

    class Broken:
        def execute(self, *a):
            raise RuntimeError("disk I/O error")

        def rollback(self):
            pass

        def close(self):
            pass
    monkeypatch.setattr(dbmod, "connect", lambda *a, **k: Broken())
    monkeypatch.setattr(aw, "write_pipeline_heartbeat", lambda *a: None)
    assert aw.job_results() == 1
    assert not (tmp_path / "data/issued_results.json").exists()


def test_cli_accepts_the_results_job():
    aw = _aw()
    src = (ROOT / "scripts/auto_weekly.py").read_text()
    assert '"results": job_results' in src and callable(aw.job_results)


# ------------------------------------------------------------------ workflow wiring --
WF = (ROOT / ".github/workflows/live-weekly.yml").read_text()
RESULTS_CRONS = ["40 5 * 9,10,11,12,1 *", "40 21 * 9,10,11,12,1 0", "40 1 * 9,10,11,12,1 1"]


def test_workflow_schedules_and_dispatches_the_results_job():
    sys.path.insert(0, str(Path(__file__).parent))
    import test_t90_schedule_contract as contract
    crons = contract._crons()
    for c in RESULTS_CRONS:
        assert c in crons and contract._job_for(c) == "results", c
    assert "options: [deploy, wed, t90, tuesday, wed-early, results]" in WF


def test_results_crons_cover_thursday_sunday_and_monday_night_finals():
    sys.path.insert(0, str(Path(__file__).parent))
    import test_t90_schedule_contract as contract
    # a final at ~23:40 ET Thu (TNF), Sun (SNF), Mon (MNF); ~16:30 and ~19:45 ET Sunday windows
    ends = [dt.datetime(2026, 10, 9, 3, 40), dt.datetime(2026, 10, 12, 3, 40), dt.datetime(2026, 10, 13, 3, 40),
            dt.datetime(2026, 10, 11, 20, 30), dt.datetime(2026, 10, 11, 23, 45),
            dt.datetime(2026, 11, 13, 4, 40), dt.datetime(2026, 11, 17, 4, 40)]
    for end in ends:
        hits = [end + dt.timedelta(minutes=m) for m in range(0, 6 * 60)
                if any(contract._cron_matches(c, end + dt.timedelta(minutes=m)) for c in RESULTS_CRONS)]
        assert hits, f"no results slot within 6 h after a final at {end}Z"


def test_state_and_site_are_saved_only_when_the_results_job_wrote_something():
    guard = 'if [[ "${JOB:-}" == "results" && "${RESULTS_WRITTEN:-}" == "0" ]]; then'  # nounset-safe
    for step in ("Publish successful production state", "Keep the eight newest state archives"):
        block = WF[WF.index(f"- name: {step}"):]
        block = block[:block.index("\n      - ", 1)]
        assert "\n        if:" not in block, step                 # still success-only (no step condition)
        assert "RESULTS_WRITTEN: ${{ steps.results.outputs.written }}" in block and guard in block, step
        assert block.index(guard) < block.index("gh release"), step
    site = WF[WF.index("- name: Build public site from this run"):]
    assert "steps.select.outputs.job == 'results' && steps.results.outputs.written != '0'" in site.splitlines()[2]
    assert "--label results" in site and "--label fresh" in site
    feeds = WF[WF.index("- name: Save rebuildable seasonal feeds"):].splitlines()[1]
    assert "results" not in feeds                                          # the results job fetches no feeds


def test_website_publisher_accepts_the_results_files():
    site = (ROOT / ".github/workflows/website.yml").read_text()
    assert re.search(r"'results\.html', 'api/results\.json'", site)
