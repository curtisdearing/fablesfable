"""scripts/pregame_dispatch.py: bounded operator-triggered T-90 dispatch.

Every remote interaction goes through an injected fake ``gh`` runner and an
injected official-schedule fetcher; the dry-run and refusal paths are also run
with the REAL defaults while sockets, subprocesses and urlopen are booby-
trapped, proving they make no network call at all. The fixtures below are
synthetic software-test inputs (a toy state DB, fake run logs), never
football or betting evidence.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import shutil
import socket
import sqlite3
import subprocess
import urllib.request
from pathlib import Path

import pytest

from scripts import pregame_dispatch as pd
from scripts import state_store

UTC = dt.timezone.utc
SHA = "a" * 40
GAME = "2026_03_ATL_GB"
KICK = dt.datetime(2026, 9, 25, 0, 15, tzinfo=UTC)
IN_WINDOW = KICK - dt.timedelta(minutes=80)          # 22:55Z


def base_args(*extra, receipt=None):
    args = ["--season", "2026", "--week", "3", "--game", GAME, "--kickoff", "2026-09-25T00:15:00Z"]
    if receipt is not None:
        args += ["--expect-sha", SHA, "--receipt-dir", str(receipt)]
    return args + list(extra)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += dt.timedelta(seconds=seconds)


def make_state(tmp: Path, name: str, t90_rows: int, line_rows: int = 3) -> tuple[Path, str]:
    root = tmp / f"root-{name}"
    (root / "data").mkdir(parents=True)
    conn = sqlite3.connect(root / "data" / "nfl_props.db")
    conn.execute("CREATE TABLE leans (game_id TEXT, clock TEXT)")
    conn.execute("CREATE TABLE lines (game_id TEXT)")
    conn.executemany("INSERT INTO leans VALUES (?, ?)", [(GAME, "wed")] + [(GAME, "t90")] * t90_rows)
    conn.executemany("INSERT INTO lines VALUES (?)", [(GAME,)] * line_rows)
    conn.commit()
    conn.close()
    archive = tmp / "assets" / name
    meta = state_store.pack(archive, root=root)
    return archive, meta["sha256"]


class FakeGitHub:
    """Stands in for the ``gh`` CLI; records every argv it receives."""

    def __init__(self, tmp: Path, clock: Clock):
        self.tmp, self.clock = tmp, clock
        self.calls: list[list] = []
        self.dispatches: list = []
        self.main_sha = SHA
        self.ci_runs = [{"id": 1, "status": "completed", "conclusion": "success"}]
        self.wf_state = "active"
        self.wf_text = ("on:\n  workflow_dispatch:\n    inputs:\n      job:\n"
                        "        options: [deploy, wed, t90, tuesday]\n")
        self.runs = {"live-weekly.yml": [], "publication-ingest.yml": None}
        self.run_infos, self.jobs, self.logs = {}, {}, {}
        self.assets = {}
        self.set_state("state-100-1.tar.gz", t90_rows=0)
        self.on_dispatch = self.successful_run
        self.next_id = 500

    def set_state(self, asset, t90_rows):
        path, sha = make_state(self.tmp, asset, t90_rows)
        self.assets[asset] = path
        self.pointer = {"schema_version": 1, "asset": asset, "sha256": sha}

    def successful_run(self):
        rid = self.next_id
        self.next_id += 1
        self.runs["live-weekly.yml"].append({"id": rid, "status": "completed", "conclusion": "success",
                                             "event": "workflow_dispatch",
                                             "created_at": pd.iso(self.clock())})
        self.run_infos[rid] = {"status": "completed", "conclusion": "success", "event": "workflow_dispatch",
                               "head_sha": SHA, "run_attempt": 1, "html_url": f"u/{rid}"}
        self.jobs[rid] = [{"steps": [{"name": pd.RUN_STEP, "status": "completed", "conclusion": "success"}]}]
        self.logs[rid] = (f"gate\tjob=t90\nrun\t[auto] closing resnap: 1 game(s), 0 with no quotes\n"
                          f"run\t[auto] t90 {GAME}: 0 voided\n")
        self.set_state(f"state-{rid}-1.tar.gz", t90_rows=4)

    def __call__(self, argv):
        self.calls.append(list(argv))
        assert argv[0] == "gh"
        if argv[1] == "workflow" and argv[2] == "run":
            assert argv[3:] == ["live-weekly.yml", "-R", pd.REPO, "--ref", "main", "-f", "job=t90"]
            self.dispatches.append(pd.iso(self.clock()))
            self.on_dispatch()
            return 0, "", ""
        if argv[1] == "release" and argv[2] == "download":
            asset = argv[argv.index("--pattern") + 1]
            dest = Path(argv[argv.index("--dir") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.assets[asset], dest / asset)
            return 0, "", ""
        if argv[1] == "run" and argv[2] == "view":
            return 0, self.logs[int(argv[3])], ""
        assert argv[1] == "api"
        path = argv[2].removeprefix(f"repos/{pd.REPO}/")
        body = self.api(path)
        if body is None:
            return 1, "", "gh: Not Found (HTTP 404)"
        return 0, json.dumps(body), ""

    def api(self, path):
        if path == "branches/main":
            return {"commit": {"sha": self.main_sha}}
        if path.startswith("actions/workflows/ci.yml/runs"):
            return {"workflow_runs": self.ci_runs}
        if path == "actions/workflows/live-weekly.yml":
            return {"state": self.wf_state}
        if path.startswith("contents/.github/workflows/live-weekly.yml?ref="):
            return {"content": base64.b64encode(self.wf_text.encode()).decode()}
        if path.startswith("actions/workflows/") and "/runs?" in path:
            name, query = path.split("/")[2], path.split("?", 1)[1]
            runs = self.runs.get(name)
            if runs is None:
                return None
            if "event=workflow_dispatch" in query:
                runs = [r for r in runs if r["event"] == "workflow_dispatch"]
            return {"workflow_runs": runs}
        if path.startswith("actions/runs/"):
            parts = path.split("/")
            rid = int(parts[2])
            return {"jobs": self.jobs[rid]} if len(parts) == 4 else self.run_infos[rid]
        if path == f"releases/tags/{pd.STATE_TAG}":
            return {"body": json.dumps(self.pointer)}
        raise AssertionError(f"unexpected api path {path}")


def scoreboard(kickoff="2026-09-25T00:15Z", status="STATUS_SCHEDULED", away="ATL", home="GB"):
    calls = []

    def fetch(url):
        calls.append(url)
        assert url == pd.ESPN_SCOREBOARD.format(week=3, season=2026)
        return json.dumps({"events": [{
            "id": "401872948", "date": kickoff, "status": {"type": {"name": status}},
            "competitions": [{"competitors": [
                {"team": {"abbreviation": home}, "homeAway": "home"},
                {"team": {"abbreviation": away}, "homeAway": "away"}]}]}]}).encode()
    fetch.calls = calls
    return fetch


@pytest.fixture
def env(tmp_path):
    clock = Clock(IN_WINDOW)
    gh = FakeGitHub(tmp_path, clock)
    receipt = tmp_path / "receipts"
    return clock, gh, receipt


def go(args, clock, gh, fetch=None):
    return pd.run(args, runner=gh, fetch=fetch or scoreboard(), clock=clock, sleep=clock.sleep)


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network or subprocess used")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)


# ------------------------------------------------------- no-network paths
def test_dry_run_inside_window_plans_without_network(no_network, capsys):
    rc = pd.run(base_args("--at", pd.iso(IN_WINDOW)))
    out = json.loads(capsys.readouterr().out)
    assert rc == pd.EXIT_OK
    assert out["mode"] == "dry-run" and out["dispatch"]["network_calls"] == 0
    assert out["target"]["window_open"] == "2026-09-24T22:45:00Z"
    assert out["target"]["dispatch_latest"] == "2026-09-24T23:40:00Z"
    assert "--execute" in out["dispatch"]["command"] and GAME in out["dispatch"]["command"]


@pytest.mark.parametrize("at,why", [
    ("2026-09-24T22:44:59Z", "before the dispatch window"),
    ("2026-09-24T23:40:01Z", "could land after kickoff"),
    ("2026-09-25T00:15:00Z", "retrospective"),
    ("2026-09-25T03:00:00Z", "retrospective"),
])
def test_dry_run_outside_window_refuses_without_network(no_network, capsys, at, why):
    assert pd.run(base_args("--at", at)) == pd.EXIT_REFUSED
    assert why in json.loads(capsys.readouterr().out)["decision"]


LATE = [KICK - dt.timedelta(minutes=20), KICK, KICK + dt.timedelta(hours=1)]


@pytest.mark.parametrize("now,mode", [(n, m) for n in [KICK - dt.timedelta(hours=3)] + LATE
                                      for m in (["--execute"], ["--fallback-after", "9"])]
                         + [(n, ["--check"]) for n in LATE])
def test_live_modes_refuse_on_timing_before_any_network(no_network, tmp_path, now, mode):
    rc = pd.run(base_args(*mode, receipt=tmp_path / "r"), clock=lambda: now)
    assert rc == pd.EXIT_REFUSED
    assert not (tmp_path / "r" / f"dispatch-{GAME}.lock").exists()


def test_check_before_window_is_read_only_preverification(env):
    clock, gh, receipt = env
    clock.t = KICK - dt.timedelta(hours=3)
    assert go(base_args("--check", receipt=receipt), clock, gh) == pd.EXIT_OK
    (receipt_file,) = receipt.glob("check-*.json")
    assert "before the dispatch window" in json.loads(receipt_file.read_text())["decision"]
    assert gh.dispatches == []


@pytest.mark.parametrize("args", [
    ["--season", "2026", "--week", "4", "--game", GAME, "--kickoff", "2026-09-25T00:15:00Z"],
    ["--season", "2026", "--week", "3", "--game", "ATL@GB", "--kickoff", "2026-09-25T00:15:00Z"],
    ["--season", "2026", "--week", "3", "--game", GAME, "--kickoff", "2026-09-24T20:15:00"],
])
def test_identity_refusals_without_network(no_network, args):
    assert pd.run(args) == pd.EXIT_REFUSED


def test_live_modes_need_full_sha_and_external_receipt_dir(no_network, tmp_path):
    common = base_args()
    assert pd.run(common + ["--execute", "--receipt-dir", str(tmp_path)]) == pd.EXIT_REFUSED
    assert pd.run(common + ["--execute", "--expect-sha", "abc", "--receipt-dir", str(tmp_path)]) \
        == pd.EXIT_REFUSED
    inside = pd.ROOT / "reports" / "dispatch"
    assert pd.run(common + ["--execute", "--expect-sha", SHA, "--receipt-dir", str(inside)]) \
        == pd.EXIT_REFUSED
    assert not inside.exists()
    assert pd.run(common + ["--execute", "--at", "2026-09-24T23:00:00Z",
                            "--expect-sha", SHA, "--receipt-dir", str(tmp_path)]) == pd.EXIT_REFUSED


# ------------------------------------------------------- readiness (check)
def test_check_ready_dispatches_nothing(env):
    clock, gh, receipt = env
    fetch = scoreboard()
    assert go(base_args("--check", receipt=receipt), clock, gh, fetch) == pd.EXIT_OK
    assert gh.dispatches == [] and len(fetch.calls) == 1
    assert not any(c[1:3] == ["workflow", "run"] for c in gh.calls)
    (receipt_file,) = receipt.glob("check-*.json")
    body = json.loads(receipt_file.read_text())
    assert all(c["ok"] for c in body["checks"])
    assert "closing resnap" in next(c for c in body["checks"] if c["name"] == "processed_state_guard")["detail"]


@pytest.mark.parametrize("breakage,failed", [
    (lambda gh: setattr(gh, "main_sha", "b" * 40), "remote_main_sha"),
    (lambda gh: setattr(gh, "ci_runs", []), "ci_green_on_sha"),
    (lambda gh: gh.ci_runs.append({"id": 2, "status": "completed", "conclusion": "failure"}), "ci_green_on_sha"),
    (lambda gh: setattr(gh, "wf_state", "disabled_manually"), "workflow_active"),
    (lambda gh: setattr(gh, "wf_text", "on:\n  push:\n"), "workflow_dispatch_t90"),
    (lambda gh: gh.runs["live-weekly.yml"].append(
        {"id": 7, "status": "queued", "event": "schedule", "created_at": "2026-09-24T22:00:00Z"}),
     "no_active_production_run"),
    (lambda gh: gh.runs.__setitem__("publication-ingest.yml", [
        {"id": 8, "status": "in_progress", "event": "workflow_run", "created_at": "2026-09-24T22:50:00Z"}]),
     "no_active_production_run"),
    (lambda gh: gh.runs["live-weekly.yml"].append(
        {"id": 9, "status": "completed", "conclusion": "success", "event": "workflow_dispatch",
         "created_at": "2026-09-24T22:46:00Z"}), "no_prior_dispatch_this_window"),
    (lambda gh: gh.set_state("state-101-1.tar.gz", t90_rows=2), "processed_state_guard"),
    (lambda gh: gh.pointer.__setitem__("sha256", "0" * 64), "processed_state_guard"),
])
def test_check_not_ready_blocks_dispatch(env, breakage, failed):
    clock, gh, receipt = env
    breakage(gh)
    assert go(base_args("--execute", receipt=receipt), clock, gh) == pd.EXIT_NOT_READY
    assert gh.dispatches == []
    assert not (receipt / f"dispatch-{GAME}.lock").exists()
    (receipt_file,) = receipt.glob("execute-*.json")
    bad = [c["name"] for c in json.loads(receipt_file.read_text())["checks"] if not c["ok"]]
    assert bad == [failed]


@pytest.mark.parametrize("fetch", [
    scoreboard(kickoff="2026-09-25T00:20Z"),
    scoreboard(status="STATUS_IN_PROGRESS"),
    scoreboard(away="NO"),
])
def test_official_kickoff_mismatch_is_not_ready(env, fetch):
    clock, gh, receipt = env
    assert go(base_args("--execute", receipt=receipt), clock, gh, fetch) == pd.EXIT_NOT_READY
    assert gh.dispatches == []


def test_espn_abbreviation_aliases(env):
    clock, gh, _ = env
    t = pd.Target(2026, 3, "2026_03_WAS_LA", KICK)
    off = pd.official_kickoff(scoreboard(away="WSH", home="LAR"), t)
    assert off["kickoff"] == "2026-09-25T00:15:00Z"


# ------------------------------------------------------- execute + readback
def test_execute_dispatches_once_and_reads_back_processed(env):
    clock, gh, receipt = env
    assert go(base_args("--execute", receipt=receipt), clock, gh) == pd.EXIT_OK
    assert len(gh.dispatches) == 1
    (receipt_file,) = receipt.glob("execute-*.json")
    body = json.loads(receipt_file.read_text())
    rb = body["readback"]
    assert body["decision"] == "dispatched and processed"
    assert rb["verdict"] == "processed" and rb["pointer_is_this_run"] and rb["gate_job_t90"]
    assert rb["state"]["t90_leans"] == 4 and rb["resnap_line"].startswith("[auto] closing resnap")
    # The same wrapper again (or a second operator) cannot dispatch a second time.
    assert go(base_args("--execute", receipt=receipt), clock, gh) == pd.EXIT_NOT_READY
    assert len(gh.dispatches) == 1


def test_lock_blocks_second_dispatch_even_if_remote_checks_pass(env):
    clock, gh, receipt = env
    gh.on_dispatch = lambda: None           # dispatch accepted, but no run ever appears
    assert go(base_args("--execute", receipt=receipt), clock, gh) == pd.EXIT_FAILED
    assert (receipt / f"dispatch-{GAME}.lock").exists()
    clock.t = IN_WINDOW
    assert go(base_args("--execute", receipt=receipt), clock, gh) == pd.EXIT_REFUSED
    assert len(gh.dispatches) == 1


def test_timing_rechecked_after_slow_remote_checks(env):
    clock, gh, receipt = env
    clock.t = KICK - dt.timedelta(minutes=36)
    real_api = gh.api

    def slow_api(path):
        if path == "branches/main":
            clock.sleep(120)                 # remote checks push us past kickoff-35m
        return real_api(path)
    gh.api = slow_api
    assert go(base_args("--execute", receipt=receipt), clock, gh) == pd.EXIT_REFUSED
    assert gh.dispatches == [] and not (receipt / f"dispatch-{GAME}.lock").exists()


@pytest.mark.parametrize("mutate", [
    lambda gh, rid: gh.logs.__setitem__(rid, "gate\tjob=t90\nrun\t[auto] no kickoffs within the T-90 window — no-op\n"),
    lambda gh, rid: gh.logs.__setitem__(rid, gh.logs[rid] + f"run\t[auto] t90 {GAME} FAILED: boom\n"),
    lambda gh, rid: gh.set_state("state-999-1.tar.gz", t90_rows=4),     # pointer is another run's
    lambda gh, rid: gh.run_infos[rid].__setitem__("head_sha", "c" * 40),
    lambda gh, rid: gh.run_infos[rid].__setitem__("conclusion", "failure"),
])
def test_readback_never_infers_success(env, mutate):
    clock, gh, receipt = env
    gh.successful_run()
    rid = gh.next_id - 1
    mutate(gh, rid)
    assert go(base_args("--readback", str(rid), receipt=receipt), clock, gh) == pd.EXIT_FAILED
    assert gh.dispatches == []


def test_readback_after_kickoff_is_allowed_and_read_only(env):
    clock, gh, receipt = env
    gh.successful_run()
    clock.t = KICK + dt.timedelta(hours=2)
    assert go(base_args("--readback", str(gh.next_id - 1), receipt=receipt), clock, gh) == pd.EXIT_OK
    assert gh.dispatches == []


# ------------------------------------------------------- fallback
def failed_run(gh, reached_run_step: bool, status="completed"):
    rid = gh.next_id
    gh.next_id += 1
    gh.runs["live-weekly.yml"].append({"id": rid, "status": status, "conclusion": "failure",
                                       "event": "workflow_dispatch", "created_at": "2026-09-24T22:46:00Z"})
    gh.run_infos[rid] = {"status": status, "conclusion": "failure", "event": "workflow_dispatch",
                         "head_sha": SHA, "run_attempt": 1, "html_url": "u"}
    step = ({"name": pd.RUN_STEP, "status": "completed", "conclusion": "failure"} if reached_run_step
            else {"name": pd.RUN_STEP, "status": "completed", "conclusion": "skipped"})
    gh.jobs[rid] = [{"steps": [{"name": "Restore checksummed production state", "status": "completed",
                                "conclusion": "success" if reached_run_step else "failure"}, step]}]
    gh.logs[rid] = "gate\tjob=t90\n" + ("run\t[auto] closing resnap: 1 game(s)\n" if reached_run_step else "")
    return rid


def test_fallback_refused_when_failed_run_may_have_pulled_odds(env):
    clock, gh, receipt = env
    rid = failed_run(gh, reached_run_step=True)
    assert go(base_args("--fallback-after", str(rid), receipt=receipt), clock, gh) == pd.EXIT_REFUSED
    assert gh.dispatches == []
    (receipt_file,) = receipt.glob("fallback-*.json")
    assert "pull them twice" in json.loads(receipt_file.read_text())["decision"]


def test_fallback_refused_while_earlier_run_still_running(env):
    clock, gh, receipt = env
    rid = failed_run(gh, reached_run_step=False, status="in_progress")
    assert go(base_args("--fallback-after", str(rid), receipt=receipt), clock, gh) == pd.EXIT_REFUSED
    assert gh.dispatches == []


def test_fallback_once_when_failure_preceded_odds_step(env):
    clock, gh, receipt = env
    rid = failed_run(gh, reached_run_step=False)
    # The normal execute path treats the failed dispatch as a duplicate ...
    assert go(base_args("--execute", receipt=receipt), clock, gh) == pd.EXIT_NOT_READY
    # ... the explicit fallback re-dispatches exactly once and reads back.
    assert go(base_args("--fallback-after", str(rid), receipt=receipt), clock, gh) == pd.EXIT_OK
    assert len(gh.dispatches) == 1
    assert go(base_args("--fallback-after", str(rid), receipt=receipt), clock, gh) != pd.EXIT_OK
    assert len(gh.dispatches) == 1


def test_fallback_refused_when_state_already_processed(env):
    clock, gh, receipt = env
    rid = failed_run(gh, reached_run_step=False)
    gh.set_state("state-102-1.tar.gz", t90_rows=1)
    assert go(base_args("--fallback-after", str(rid), receipt=receipt), clock, gh) == pd.EXIT_NOT_READY
    assert gh.dispatches == []


# ------------------------------------------------------- whole-slot dispatch
# One job_t90 run serves every game in the scheduler's slot (e.g. the eight 1 PM ET games of
# 2026 week 5), so the slot is done only when EVERY member game is read back.
GAME2 = "2026_03_DET_ARI"


def set_slot_state(gh, asset, t90_by_game, line_rows=3):
    n = len(list(gh.tmp.glob("slot-*")))                 # unique per call, apart from make_state's
    root = gh.tmp / f"slot-{n}" / "root"
    (root / "data").mkdir(parents=True)
    conn = sqlite3.connect(root / "data" / "nfl_props.db")
    conn.execute("CREATE TABLE leans (game_id TEXT, clock TEXT)")
    conn.execute("CREATE TABLE lines (game_id TEXT)")
    for g, n in t90_by_game.items():
        conn.executemany("INSERT INTO leans VALUES (?, ?)", [(g, "wed")] + [(g, "t90")] * n)
        conn.executemany("INSERT INTO lines VALUES (?)", [(g,)] * line_rows)
    conn.commit()
    conn.close()
    archive = gh.tmp / f"slot-{n}" / asset
    meta = state_store.pack(archive, root=root)
    gh.assets[asset] = archive
    gh.pointer = {"schema_version": 1, "asset": asset, "sha256": meta["sha256"]}


def _receipt(receipt, mode):
    (f,) = sorted(receipt.glob(f"{mode}-*.json"))[-1:]
    return json.loads(f.read_text())


def test_readback_is_not_processed_until_every_slot_game_is(env):
    clock, gh, receipt = env
    gh.successful_run()                                  # logs and processes the named game only
    rid = gh.next_id - 1
    set_slot_state(gh, f"state-{rid}-1.tar.gz", {GAME: 4, GAME2: 0})
    slot = ["--slot-games", f"{GAME},{GAME2}"]
    assert go(base_args("--readback", str(rid), *slot, receipt=receipt), clock, gh) == pd.EXIT_FAILED
    rb = _receipt(receipt, "readback")["readback"]
    assert rb["verdict"] == "not-processed" and rb["unprocessed_games"] == [GAME2]
    assert rb["games"][GAME] == {"processed_line": True, "failed_line": False, "voided": 0,
                                 "t90_leans": 4, "stored_lines": 3, "processed": True}
    assert rb["games"][GAME2]["processed_line"] is False and rb["games"][GAME2]["t90_leans"] == 0
    # The same run once the second game's line and leans exist: processed.
    gh.logs[rid] += f"run\t[auto] t90 {GAME2}: 1 voided\n"
    set_slot_state(gh, f"state-{rid}-1.tar.gz", {GAME: 4, GAME2: 2})
    clock.sleep(1)
    assert go(base_args("--readback", str(rid), *slot, receipt=receipt), clock, gh) == pd.EXIT_OK
    assert _receipt(receipt, "readback")["readback"]["games"][GAME2]["voided"] == 1


def test_execute_for_a_slot_names_the_unprocessed_games_in_its_decision(env):
    clock, gh, receipt = env
    def run_processing_only_the_named_game():
        gh.successful_run()
        set_slot_state(gh, f"state-{gh.next_id - 1}-1.tar.gz", {GAME: 4, GAME2: 0})
    gh.on_dispatch = run_processing_only_the_named_game
    code = go(base_args("--execute", "--slot-games", f"{GAME},{GAME2}", receipt=receipt), clock, gh)
    assert code == pd.EXIT_FAILED and len(gh.dispatches) == 1
    assert _receipt(receipt, "execute")["decision"].endswith(f"did NOT process {GAME2}")


def test_processed_guard_covers_the_slot_not_only_the_named_game(env):
    clock, gh, receipt = env
    slot = ["--slot-games", f"{GAME},{GAME2}"]
    # The named game was processed by another run; GAME2 was not: dispatching is still needed
    # (job_t90 skips and does not resnap processed games, so nothing is pulled twice).
    set_slot_state(gh, "state-101-1.tar.gz", {GAME: 2, GAME2: 0})
    assert go(base_args("--check", *slot, receipt=receipt), clock, gh) == pd.EXIT_OK
    # Every slot game processed: not ready, and the detail says so for the scheduler.
    set_slot_state(gh, "state-102-1.tar.gz", {GAME: 2, GAME2: 5})
    assert go(base_args("--execute", *slot, receipt=receipt), clock, gh) == pd.EXIT_NOT_READY
    assert gh.dispatches == []
    guard = next(c for c in _receipt(receipt, "execute")["checks"] if c["name"] == "processed_state_guard")
    assert not guard["ok"] and pd.ALREADY_PROCESSED in guard["detail"]


@pytest.mark.parametrize("bad", ["2026_04_DET_ARI", "2025_03_DET_ARI", "DET_ARI"])
def test_slot_games_from_another_week_are_refused(no_network, bad, capsys):
    assert pd.run(base_args("--slot-games", f"{GAME},{bad}")) == pd.EXIT_REFUSED
    assert "slot game" in capsys.readouterr().out


# ------------------------------------------------------- doubleheaders: distinct slots
# Kickoffs 41+ min apart are separate slots.  The first slot's own run used to refuse the
# second slot for its whole window (any workflow_dispatch since window-60 min counted), e.g. a
# 7:15 + 8:15 PM ET Monday doubleheader lost its second game.  A prior run now blocks only
# without evidence that it neither processed nor billed a slot game.
SETUP = dt.timedelta(minutes=15)                         # dispatch -> job_t90 clock
KA = dt.datetime(2026, 9, 28, 23, 15, tzinfo=UTC)        # Monday 7:15 PM ET
GA, GB, GC = "2026_03_AAA_BBB", "2026_03_CCC_DDD", "2026_03_EEE_FFF"
M = lambda n: dt.timedelta(minutes=n)  # noqa: E731


def dh_board(kicks):
    def fetch(url):
        return json.dumps({"events": [{
            "id": str(i), "date": pd.iso(k), "status": {"type": {"name": "STATUS_SCHEDULED"}},
            "competitions": [{"competitors": [
                {"homeAway": "away", "team": {"abbreviation": g.split("_")[2]}},
                {"homeAway": "home", "team": {"abbreviation": g.split("_")[3]}}]}]}
            for i, (g, k) in enumerate(kicks.items())]}).encode()
    return fetch


def prior_run(gh, rid, created, kicks, *, conclusion="success", status="completed", job="t90",
              lines=None, ended=None):
    """An earlier dispatch behaving like job_t90: at created+SETUP it processes every game due."""
    clock = created + SETUP
    due = [g for g, k in kicks.items() if k - M(90) <= clock < k] if lines is None else lines
    log = f"gate\tjob={job}\n" + "".join(
        f"run\t[auto] t90 {g}: 0 voided\n" if not g.endswith("!") else f"run\t[auto] t90 {g[:-1]} FAILED: x\n"
        for g in due)
    run = {"id": rid, "status": status, "conclusion": conclusion if status == "completed" else None,
           "event": "workflow_dispatch", "created_at": pd.iso(created),
           "updated_at": pd.iso(ended or clock + M(5 + 2 * len(due)))}
    gh.runs["live-weekly.yml"].append(run)
    if log is not False:
        gh.logs[rid] = log
    return [g for g in due if not g.endswith("!")], run


def check_slot(gh, receipt, clock, named, kicks, slot=None):
    args = ["--season", "2026", "--week", "3", "--game", named, "--kickoff", pd.iso(kicks[named]),
            "--slot-games", ",".join(slot or [named]), "--expect-sha", SHA,
            "--receipt-dir", str(receipt), "--check"]
    code = pd.run(args, runner=gh, fetch=dh_board(kicks), clock=clock, sleep=clock.sleep)
    return code, {c["name"]: c for c in _receipt(receipt, "check")["checks"]}


@pytest.mark.parametrize("sep,a_offset,b_done_by_a", [
    (41, 1, False), (60, 1, False), (90, 1, False), (100, 1, False),    # A dispatched as its window opens
    (41, 49, True), (60, 49, True), (90, 49, False), (100, 49, False),   # A's last-tick retry (A - 41 min)
    (64, 49, True), (65, 49, False),                                     # B due at A's clock: exact edge
])
def test_doubleheader_second_slot_dispatches_unless_the_first_run_processed_it(env, sep, a_offset, b_done_by_a):
    clock, gh, receipt = env
    kicks = {GA: KA, GB: KA + M(sep)}
    due, run = prior_run(gh, 600, KA - M(90) + M(a_offset), kicks)
    assert (GB in due) is b_done_by_a and GA in due
    set_slot_state(gh, "state-600-1.tar.gz", {g: 4 if g in due else 0 for g in kicks})
    clock.t = max(kicks[GB] - M(89), pd.parse_utc(run["updated_at"]) + M(1))
    assert clock.t < kicks[GB] - M(40)                       # still inside B's launch window
    code, checks = check_slot(gh, receipt, clock, GB, kicks)
    prior = checks["no_prior_dispatch_this_window"]
    if b_done_by_a:                                          # repeated slot: never dispatched again
        assert code == pd.EXIT_NOT_READY and not prior["ok"] and f"processed or attempted {GB}" in prior["detail"]
        assert pd.ALREADY_PROCESSED in checks["processed_state_guard"]["detail"]
    else:
        in_lookback = pd.parse_utc(run["created_at"]) >= kicks[GB] - M(150)
        assert code == pd.EXIT_OK and prior["ok"], prior
        assert ("run 600 distinct" if in_lookback else ": none") in prior["detail"], prior


def _ended(minutes_before_b_window):
    return KA + M(60) - M(90) - dt.timedelta(seconds=minutes_before_b_window)


@pytest.mark.parametrize("name,run_kw,ok,why", [
    ("first run still queued", dict(status="queued"), False, "may still process"),
    ("first run still running", dict(status="in_progress"), False, "may still process"),
    ("failed, ended 1 s before B's window", dict(conclusion="failure", lines=[f"{GA}!"], ended=_ended(1)), True, "distinct: completed"),
    ("failed, ended exactly at B's window", dict(conclusion="failure", lines=[f"{GA}!"], ended=_ended(0)), False, "no proof"),
    ("cancelled inside B's window", dict(conclusion="cancelled", lines=[], ended=_ended(-600)), False, "no proof"),
    ("success t90 inside B's window, names only A", dict(lines=[GA], ended=_ended(-600)), True, "successful job=t90"),
    ("success but job=wed inside B's window", dict(job="wed", lines=[], ended=_ended(-600)), False, "no proof"),
    ("log unreadable (unidentifiable)", dict(lines=[GA], ended=_ended(1)), False, "log unreadable"),
    ("failed on B itself (attempted)", dict(conclusion="failure", lines=[f"{GB}!"], ended=_ended(1)), False, f"attempted {GB}"),
])
def test_prior_run_evidence_table(env, name, run_kw, ok, why):
    clock, gh, receipt = env
    kicks = {GA: KA, GB: KA + M(60)}
    _, run = prior_run(gh, 601, KA - M(89), kicks, **run_kw)
    if name.startswith("log unreadable"):
        del gh.logs[601]
    clock.t = kicks[GB] - M(85)
    code, checks = check_slot(gh, receipt, clock, GB, kicks)
    prior = checks["no_prior_dispatch_this_window"]
    assert prior["ok"] is ok and why in prior["detail"], (name, prior["detail"])
    if ok:
        assert code == pd.EXIT_OK
    else:
        assert code == pd.EXIT_NOT_READY


@pytest.mark.parametrize("old_log,ok,why", [
    ([GB], False, f"processed or attempted {GB}"),          # processed at the old time: per-game guard
    ([f"{GB}!"], False, f"processed or attempted {GB}"),    # failed on B at the old time: parent decision
    ([], True, "distinct: completed"),                      # no-op at the old time
])
def test_shifted_kickoff_same_game_id(env, old_log, ok, why):
    clock, gh, receipt = env
    old_b, new_b = KA + M(60), KA + M(240)                  # B flexed three hours later
    prior_run(gh, 602, old_b - M(89), {GB: old_b}, lines=old_log,
              conclusion="failure" if old_log and old_log[0].endswith("!") else "success")
    set_slot_state(gh, "state-602-1.tar.gz", {GB: 4 if old_log == [GB] else 0})
    clock.t = new_b - M(85)
    gh.runs["live-weekly.yml"][-1]["created_at"] = pd.iso(new_b - M(150))   # inside the lookback
    code, checks = check_slot(gh, receipt, clock, GB, {GB: new_b})
    prior = checks["no_prior_dispatch_this_window"]
    assert prior["ok"] is ok and why in prior["detail"], prior["detail"]
    assert (code == pd.EXIT_OK) is ok
    if old_log == [GB]:
        assert pd.ALREADY_PROCESSED in checks["processed_state_guard"]["detail"]


@pytest.mark.parametrize("touch,board_has_gc,ok,why", [
    ([GA], True, True, "distinct: completed"),
    ([GA, GC], True, False, f"attempted {GC}"),             # any slot member counts, not only the named game
    ([GA], False, False, "no proof"),                       # slot member off the board: time rule disabled
])
def test_multi_game_slot_uses_every_member(env, touch, board_has_gc, ok, why):
    clock, gh, receipt = env
    kicks = {GA: KA, GB: KA + M(60), GC: KA + M(80)}        # B + C share one slot (20 min apart)
    prior_run(gh, 603, KA - M(89), kicks, lines=touch, conclusion="failure")
    clock.t = kicks[GC] - M(89)
    board = kicks if board_has_gc else {GA: KA, GB: kicks[GB]}
    args = ["--season", "2026", "--week", "3", "--game", GB, "--kickoff", pd.iso(kicks[GB]),
            "--slot-games", f"{GB},{GC}", "--expect-sha", SHA, "--receipt-dir", str(receipt), "--check"]
    code = pd.run(args, runner=gh, fetch=dh_board(board), clock=clock, sleep=clock.sleep)
    prior = next(c for c in _receipt(receipt, "check")["checks"] if c["name"] == "no_prior_dispatch_this_window")
    assert prior["ok"] is ok and why in prior["detail"], prior["detail"]
    assert (code == pd.EXIT_OK) is ok
