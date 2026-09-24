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
