#!/usr/bin/env python3
"""Bounded, operator-triggered T-90 dispatch for ONE named pregame game.

GitHub starts this repo's scheduled runs 0-5.6 h late (see
scripts/auto_weekly.py), so cron cannot be trusted to land a T-90 refresh in
[kickoff-90m, kickoff). The dependable trigger is a ``workflow_dispatch
job=t90`` sent inside that window. This wrapper sends exactly that dispatch,
for exactly one season/week/game/kickoff, and nothing else:

* default is a DRY RUN: arguments and timing are validated, the plan is
  printed, and no network call or subprocess of any kind is made;
* ``--check`` (allowed from any time before the window closes) performs the
  read-only remote readiness checks (remote main SHA,
  CI on that SHA, workflow dispatchability, no active production run, no prior
  dispatch in this window, the production state's processed-state guard, and
  the official kickoff) and dispatches nothing;
* ``--execute`` runs the same checks, takes a local one-shot lock, dispatches
  once, and reads the result back (run identity, gate job, per-game log line,
  published state pointer, processed-state rows);
* ``--readback RUN_ID`` repeats only the read-back;
* ``--fallback-after RUN_ID`` re-dispatches only when RUN_ID failed BEFORE its
  "Run weekly job" step started (so it cannot have requested odds) and the
  production state still shows the game unprocessed.

Timing refusals (before the window, too close to or after kickoff) are
decided before any network call. A run that could only land after kickoff is
never dispatched: nothing produced then may be called pregame.

The workflow itself remains the final duplicate guard: ``job_t90`` resnaps
and processes only games without ``leans clock='t90'`` rows, and the
``nfl-live-production`` concurrency group serializes runs. This wrapper adds
checks in front of it; it does not replace or loosen it.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UTC = dt.timezone.utc
REPO = "curtisdearing/fablesfable"
WORKFLOW = "live-weekly.yml"
CI_WORKFLOW = "ci.yml"
#: Workflows sharing the ``nfl-live-production`` concurrency group. A missing
#: one (404 on an older remote main) is treated as having no runs.
PRODUCTION_WORKFLOWS = ("live-weekly.yml", "publication-ingest.yml")
STATE_TAG = "model-state"
T90_DUE_MINUTES = 90            # scripts/auto_weekly.py: due in [T-90, T-0)
#: Latest dispatch = kickoff minus this. A dispatched run still installs
#: dependencies, restores state and ingests feeds (the observed full run is
#: ~15-25 min) before job_t90 reads the clock; later than this and the run
#: risks finding the game already kicked off (a no-op, not a pregame output).
DEFAULT_MIN_LEAD_MINUTES = 35
#: Earliest dispatch = window open minus this. 0 by default: an earlier run
#: relies on the bounded early-arrival wait in job_t90, which exists only if
#: the remote code under dispatch contains it.
DEFAULT_EARLY_MINUTES = 0
ACTIVE_STATUSES = {"queued", "in_progress", "waiting", "requested", "pending"}
RUN_STEP = "Run weekly job"
ESPN_SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
                   "?seasontype=2&week={week}&dates={season}")
USER_AGENT = "fablesfable-pregame-dispatch/1.0 (+https://github.com/curtisdearing/fablesfable)"
#: nflverse -> ESPN team abbreviations where they differ.
ESPN_ABBR = {"LA": "LAR", "WAS": "WSH"}
GAME_ID = re.compile(r"^(\d{4})_(\d{2})_([A-Z]{2,3})_([A-Z]{2,3})$")
STATE_ASSET = re.compile(r"^(state|pubstate)-([0-9]+)-([0-9]+)\.tar\.gz$")

EXIT_OK, EXIT_REFUSED, EXIT_NOT_READY, EXIT_FAILED = 0, 2, 3, 4

Runner = Callable[[list], "tuple[int, str, str]"]
Fetcher = Callable[[str], bytes]


class Refused(Exception):
    """A decision not to act (timing, identity, duplicate, readiness)."""


@dataclass(frozen=True)
class Target:
    season: int
    week: int
    game_id: str
    kickoff: dt.datetime
    early_minutes: int = DEFAULT_EARLY_MINUTES
    min_lead_minutes: int = DEFAULT_MIN_LEAD_MINUTES

    @property
    def away(self) -> str:
        return GAME_ID.match(self.game_id).group(3)

    @property
    def home(self) -> str:
        return GAME_ID.match(self.game_id).group(4)

    @property
    def window_open(self) -> dt.datetime:
        return self.kickoff - dt.timedelta(minutes=T90_DUE_MINUTES)

    @property
    def earliest(self) -> dt.datetime:
        return self.window_open - dt.timedelta(minutes=self.early_minutes)

    @property
    def latest(self) -> dt.datetime:
        return self.kickoff - dt.timedelta(minutes=self.min_lead_minutes)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    mode: str
    target: dict
    now_utc: str
    checks: list = field(default_factory=list)
    decision: str = ""
    dispatch: dict = field(default_factory=dict)
    readback: dict = field(default_factory=dict)


def iso(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"timestamp must carry an explicit offset or Z: {text!r}")
    return t.astimezone(UTC)


# --------------------------------------------------------------- local checks
def validate_target(t: Target) -> None:
    m = GAME_ID.match(t.game_id)
    if not m:
        raise Refused(f"game id {t.game_id!r} is not SEASON_WW_AWAY_HOME")
    if int(m.group(1)) != t.season or int(m.group(2)) != t.week:
        raise Refused(f"game id {t.game_id} does not belong to season {t.season} week {t.week}")
    if t.kickoff.tzinfo is None:
        raise Refused("kickoff must be timezone-aware")
    if not 0 <= t.early_minutes <= 35:
        raise Refused("--early-minutes must be within the job's 35-minute early-arrival wait")
    if not 15 <= t.min_lead_minutes < T90_DUE_MINUTES:
        raise Refused("--min-lead-minutes must be in [15, 90)")


def timing_refusal(t: Target, now: dt.datetime) -> str | None:
    """Why dispatching at ``now`` is refused, or None. No network involved."""
    if now >= t.kickoff:
        return (f"kickoff {iso(t.kickoff)} has passed; a run now would be retrospective, "
                f"not pregame")
    if now > t.latest:
        return (f"now {iso(now)} is later than {iso(t.latest)} (kickoff - "
                f"{t.min_lead_minutes} min); the run could land after kickoff")
    if now < t.earliest:
        return f"now {iso(now)} is before the dispatch window opens at {iso(t.earliest)}"
    return None


def activation_command(t: Target, expect_sha: str, receipt_dir: str) -> str:
    return (f"python scripts/pregame_dispatch.py --season {t.season} --week {t.week} "
            f"--game {t.game_id} --kickoff {iso(t.kickoff)} --expect-sha {expect_sha} "
            f"--receipt-dir {receipt_dir} --execute")


# --------------------------------------------------------------- remote access
def subprocess_runner(argv: list) -> tuple[int, str, str]:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    return done.returncode, done.stdout, done.stderr


def urllib_fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


class GitHub:
    def __init__(self, run: Runner, repo: str = REPO):
        self.run, self.repo = run, repo

    def _gh(self, *args: str, ok_404: bool = False):
        rc, out, err = self.run(["gh", *args])
        if rc != 0:
            if ok_404 and "404" in (err + out):
                return None
            raise RuntimeError(f"gh {' '.join(args)} failed (rc={rc}): {err.strip()[:300]}")
        return out

    def api(self, path: str, ok_404: bool = False):
        out = self._gh("api", f"repos/{self.repo}/{path}", ok_404=ok_404)
        return None if out is None else json.loads(out)

    def main_sha(self) -> str:
        return self.api("branches/main")["commit"]["sha"]

    def workflow(self, name: str, ok_404: bool = False):
        return self.api(f"actions/workflows/{name}", ok_404=ok_404)

    def workflow_runs(self, name: str, query: str = "per_page=30") -> list:
        body = self.api(f"actions/workflows/{name}/runs?{query}", ok_404=True)
        return [] if body is None else body.get("workflow_runs", [])

    def file_at(self, path: str, ref: str) -> str:
        body = self.api(f"contents/{path}?ref={ref}")
        return base64.b64decode(body["content"]).decode()

    def run_info(self, run_id: int) -> dict:
        return self.api(f"actions/runs/{run_id}")

    def run_jobs(self, run_id: int) -> list:
        return self.api(f"actions/runs/{run_id}/jobs").get("jobs", [])

    def run_log(self, run_id: int) -> str:
        return self._gh("run", "view", str(run_id), "-R", self.repo, "--log")

    def state_pointer(self) -> dict:
        body = self.api(f"releases/tags/{STATE_TAG}")["body"]
        return json.loads(body)

    def download_state(self, asset: str, dest: Path) -> Path:
        self._gh("release", "download", STATE_TAG, "-R", self.repo,
                 "--pattern", asset, "--dir", str(dest))
        return dest / asset

    def dispatch_t90(self) -> None:
        self._gh("workflow", "run", WORKFLOW, "-R", self.repo, "--ref", "main", "-f", "job=t90")


def state_evidence(gh: GitHub, game_id: str) -> dict:
    """Restore the CURRENT production state into a private temp dir (checksum
    verified, safe members only) and read the processed-state guard for the
    game. Never writes into this checkout."""
    from scripts import state_store
    pointer = gh.state_pointer()
    asset, sha = pointer.get("asset", ""), pointer.get("sha256", "")
    if not STATE_ASSET.match(asset) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RuntimeError(f"invalid production state pointer: {pointer}")
    with tempfile.TemporaryDirectory(prefix="pregame-state-") as tmp:
        tmp_path = Path(tmp)
        archive = gh.download_state(asset, tmp_path / "download")
        root = tmp_path / "root"
        state_store.restore(archive, sha, root=root)
        db = root / "data" / "nfl_props.db"
        if not db.exists():
            raise RuntimeError("production state has no data/nfl_props.db")
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            t90 = conn.execute("SELECT COUNT(*) FROM leans WHERE clock='t90' AND game_id=?",
                               (game_id,)).fetchone()[0]
            lines = conn.execute("SELECT COUNT(*) FROM lines WHERE game_id=?",
                                 (game_id,)).fetchone()[0]
        finally:
            conn.close()
    return {"asset": asset, "sha256": sha, "t90_leans": int(t90), "stored_lines": int(lines)}


def official_kickoff(fetch: Fetcher, t: Target) -> dict:
    body = json.loads(fetch(ESPN_SCOREBOARD.format(week=t.week, season=t.season)))
    want = {ESPN_ABBR.get(t.away, t.away): "away", ESPN_ABBR.get(t.home, t.home): "home"}
    for ev in body.get("events", []):
        comp = ev["competitions"][0]
        sides = {c["team"]["abbreviation"]: c["homeAway"] for c in comp["competitors"]}
        if sides == want:
            return {"espn_event": ev["id"], "kickoff": iso(parse_utc(ev["date"])),
                    "status": ev["status"]["type"]["name"]}
    raise RuntimeError(f"{t.away} at {t.home} not on the official week {t.week} scoreboard")


def dispatches_since(gh: GitHub, since: dt.datetime) -> list:
    return [r for r in gh.workflow_runs(WORKFLOW, "event=workflow_dispatch&per_page=30")
            if parse_utc(r["created_at"]) >= since]


def readiness(gh: GitHub, fetch: Fetcher, t: Target, expect_sha: str,
              now: dt.datetime, allow_prior_dispatch: int | None = None) -> list:
    checks: list[Check] = []

    def add(name, ok, detail):
        checks.append(Check(name, bool(ok), detail))

    sha = gh.main_sha()
    add("remote_main_sha", sha == expect_sha, f"remote main {sha}, expected {expect_sha}")

    ci = gh.workflow_runs(CI_WORKFLOW, f"head_sha={expect_sha}&per_page=10")
    ci_ok = [r for r in ci if r.get("status") == "completed" and r.get("conclusion") == "success"]
    ci_bad = [r for r in ci if r.get("status") == "completed" and r.get("conclusion") != "success"]
    add("ci_green_on_sha", ci_ok and not ci_bad,
        f"{len(ci_ok)} successful / {len(ci_bad)} unsuccessful completed Tests runs on {expect_sha[:12]}")

    wf = gh.workflow(WORKFLOW)
    add("workflow_active", wf.get("state") == "active", f"{WORKFLOW} state={wf.get('state')}")
    text = gh.file_at(f".github/workflows/{WORKFLOW}", expect_sha)
    add("workflow_dispatch_t90", "workflow_dispatch:" in text and re.search(r"options:.*\bt90\b", text),
        "workflow_dispatch with job option t90 present at the expected SHA")

    active = [(name, r["id"], r["status"]) for name in PRODUCTION_WORKFLOWS
              for r in gh.workflow_runs(name, "per_page=30") if r.get("status") in ACTIVE_STATUSES]
    add("no_active_production_run", not active,
        f"active runs in the production concurrency group: {active or 'none'}")

    prior = [r["id"] for r in dispatches_since(gh, t.earliest - dt.timedelta(minutes=60))
             if r["id"] != allow_prior_dispatch]
    add("no_prior_dispatch_this_window", not prior,
        f"workflow_dispatch runs since {iso(t.earliest - dt.timedelta(minutes=60))}: {prior or 'none'}")

    try:
        st = state_evidence(gh, t.game_id)
        add("processed_state_guard", st["t90_leans"] == 0,
            f"{st['asset']}: {st['t90_leans']} t90 lean rows, {st['stored_lines']} stored line rows "
            f"({'closing resnap, <=5 credits' if st['stored_lines'] else 'own one-game pull'})")
    except Exception as exc:  # noqa: BLE001 -- unreadable state is not-ready, not a crash
        add("processed_state_guard", False, f"could not read production state: {exc}")

    try:
        off = official_kickoff(fetch, t)
        add("official_kickoff", off["kickoff"] == iso(t.kickoff) and off["status"] == "STATUS_SCHEDULED",
            f"ESPN {off['espn_event']} kickoff {off['kickoff']} status {off['status']}; "
            f"requested {iso(t.kickoff)}")
    except Exception as exc:  # noqa: BLE001
        add("official_kickoff", False, f"official schedule unreadable: {exc}")
    return checks


# ------------------------------------------------------------------ read-back
def readback(gh: GitHub, run_id: int, t: Target, expect_sha: str | None) -> dict:
    """What a dispatched run actually did. Never infers success from the
    dispatch call; every field is read from GitHub or the published state."""
    info = gh.run_info(run_id)
    out = {"run_id": run_id, "status": info.get("status"), "conclusion": info.get("conclusion"),
           "event": info.get("event"), "head_sha": info.get("head_sha"),
           "run_attempt": info.get("run_attempt"), "html_url": info.get("html_url")}
    out["sha_matches"] = expect_sha is None or info.get("head_sha") == expect_sha
    if info.get("status") != "completed":
        out["verdict"] = "pending"
        return out
    steps = {s["name"]: s for j in gh.run_jobs(run_id) for s in j.get("steps", [])}
    step = steps.get(RUN_STEP)
    out["run_step"] = None if step is None else {k: step.get(k) for k in ("status", "conclusion")}
    log = gh.run_log(run_id) or ""
    out["gate_job_t90"] = bool(re.search(r"\bjob=t90\b", log))
    out["game_processed_line"] = f"[auto] t90 {t.game_id}:" in log
    out["game_failed_line"] = f"[auto] t90 {t.game_id} FAILED" in log
    out["noop_line"] = "no kickoffs within the T-90 window" in log
    resnap = re.search(r"\[auto\] closing resnap: .*", log)
    out["resnap_line"] = resnap.group(0)[:300] if resnap else None
    try:
        st = state_evidence(gh, t.game_id)
        out["state"] = st
        want = f"state-{run_id}-{info.get('run_attempt')}.tar.gz"
        out["pointer_is_this_run"] = st["asset"] == want
    except Exception as exc:  # noqa: BLE001
        out["state"] = {"error": str(exc)}
        out["pointer_is_this_run"] = False
    good = (info.get("conclusion") == "success" and out["sha_matches"] and out["gate_job_t90"]
            and out["game_processed_line"] and not out["game_failed_line"]
            and out["pointer_is_this_run"] and out["state"].get("t90_leans", 0) > 0)
    out["verdict"] = "processed" if good else "not-processed"
    return out


def fallback_permitted(rb: dict) -> tuple[bool, str]:
    """A failed run may be re-dispatched only if it provably never reached the
    step that can request odds. Otherwise the odds may already have been paid
    for while the state recording them was never published, so a second run
    would pull them again."""
    if rb.get("status") != "completed":
        return False, "the earlier run has not completed; it still holds the production queue"
    if rb.get("conclusion") == "success":
        return False, "the earlier run succeeded; nothing to fall back from"
    step = rb.get("run_step")
    if step is not None and step.get("conclusion") not in (None, "skipped"):
        return False, (f"the earlier run reached '{RUN_STEP}' ({step.get('conclusion')}); it may "
                       f"already have requested odds, so an automatic re-dispatch could pull them "
                       f"twice. Parent decision required.")
    return True, f"the earlier run failed before '{RUN_STEP}'; no odds were requested"


def wait_for_run(gh: GitHub, since: dt.datetime, poll: Callable[[float], None],
                 appear_timeout: float, clock: Callable[[], dt.datetime]) -> int | None:
    deadline = clock() + dt.timedelta(seconds=appear_timeout)
    while clock() <= deadline:
        runs = dispatches_since(gh, since - dt.timedelta(seconds=10))
        if runs:
            return min(runs, key=lambda r: r["created_at"])["id"]
        poll(10)
    return None


# ----------------------------------------------------------------------- main
def _lock(receipt_dir: Path, game_id: str) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / f"dispatch-{game_id}.lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.write(fd, f"{os.getpid()} {iso(dt.datetime.now(UTC))}\n".encode())
    os.close(fd)
    return path


def run(argv: list | None = None, *, runner: Runner | None = None, fetch: Fetcher | None = None,
        clock: Callable[[], dt.datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--game", required=True, help="nflverse game id, e.g. 2026_03_ATL_GB")
    ap.add_argument("--kickoff", required=True, help="official kickoff, ISO-8601 with Z/offset")
    ap.add_argument("--expect-sha", help="remote main SHA the parent released (check/execute)")
    ap.add_argument("--receipt-dir", help="receipt/lock directory OUTSIDE this checkout")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--early-minutes", type=int, default=DEFAULT_EARLY_MINUTES)
    ap.add_argument("--min-lead-minutes", type=int, default=DEFAULT_MIN_LEAD_MINUTES)
    ap.add_argument("--at", help="dry run only: evaluate timing at this instant instead of now")
    ap.add_argument("--wait-minutes", type=float, default=75,
                    help="execute: how long to wait for the run to finish before reading back")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only remote readiness")
    mode.add_argument("--execute", action="store_true", help="dispatch job=t90 once")
    mode.add_argument("--readback", type=int, metavar="RUN_ID")
    mode.add_argument("--fallback-after", type=int, metavar="RUN_ID")
    a = ap.parse_args(argv)
    clock = clock or (lambda: dt.datetime.now(UTC))
    name = ("check" if a.check else "execute" if a.execute else "readback" if a.readback
            else "fallback" if a.fallback_after else "dry-run")

    try:
        t = Target(a.season, a.week, a.game, parse_utc(a.kickoff), a.early_minutes,
                   a.min_lead_minutes)
        validate_target(t)
    except (Refused, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return EXIT_REFUSED
    if a.at and name != "dry-run":
        print("REFUSED: --at is a dry-run planning aid only")
        return EXIT_REFUSED
    now = parse_utc(a.at) if a.at else clock()
    rep = Report(name, {**asdict(t), "kickoff": iso(t.kickoff), "window_open": iso(t.window_open),
                        "dispatch_earliest": iso(t.earliest), "dispatch_latest": iso(t.latest)},
                 iso(now))

    def emit(code: int) -> int:
        text = json.dumps(asdict(rep), indent=2, default=str)
        print(text)
        if a.receipt_dir and name != "dry-run":
            d = Path(a.receipt_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}-{t.game_id}-{iso(clock()).replace(':', '')}.json").write_text(text + "\n")
        return code

    if name != "dry-run":
        if not a.expect_sha or not re.fullmatch(r"[0-9a-f]{40}", a.expect_sha):
            print("REFUSED: --expect-sha must be the full 40-hex remote SHA")
            return EXIT_REFUSED
        if not a.receipt_dir:
            print("REFUSED: --receipt-dir is required outside dry-run")
            return EXIT_REFUSED
        rd = Path(a.receipt_dir).resolve()
        if rd == ROOT or ROOT in rd.parents:
            print("REFUSED: --receipt-dir must be outside the repository checkout")
            return EXIT_REFUSED

    if name == "dry-run":
        why = timing_refusal(t, now)
        rep.decision = f"would refuse: {why}" if why else "would proceed to readiness checks"
        rep.dispatch = {"command": activation_command(t, a.expect_sha or "<RELEASED_MAIN_SHA>",
                                                      a.receipt_dir or "<RECEIPT_DIR>"),
                        "network_calls": 0}
        return emit(EXIT_REFUSED if why else EXIT_OK)

    gh = GitHub(runner or subprocess_runner, a.repo)
    fetch = fetch or urllib_fetch

    if name == "readback":
        rep.readback = readback(gh, a.readback, t, a.expect_sha)
        rep.decision = rep.readback["verdict"]
        return emit(EXIT_OK if rep.decision == "processed" else EXIT_FAILED)

    # check / execute / fallback all refuse on timing before touching the network;
    # --check alone may run before the window opens (a read-only pre-verification).
    why = timing_refusal(t, now)
    if why and not (name == "check" and now < t.earliest):
        rep.decision = f"refused: {why}"
        return emit(EXIT_REFUSED)

    allow = None
    if name == "fallback":
        prev = readback(gh, a.fallback_after, t, a.expect_sha)
        rep.readback = {"previous": prev}
        ok, why = fallback_permitted(prev)
        if not ok:
            rep.decision = f"refused fallback: {why}"
            return emit(EXIT_REFUSED)
        allow = a.fallback_after

    rep.checks = [asdict(c) for c in readiness(gh, fetch, t, a.expect_sha, now, allow)]
    failed = [c["name"] for c in rep.checks if not c["ok"]]
    if failed:
        rep.decision = f"not ready: {', '.join(failed)}"
        return emit(EXIT_NOT_READY)
    if name == "check":
        rep.decision = (f"remote ready (nothing dispatched); {why}" if why
                        else "ready (nothing dispatched)")
        return emit(EXIT_OK)

    # Re-check timing after the (slow) remote checks, then take the one-shot lock.
    now = clock()
    why = timing_refusal(t, now)
    if why:
        rep.decision = f"refused after checks: {why}"
        return emit(EXIT_REFUSED)
    try:
        lock = _lock(Path(a.receipt_dir) / ("fallback" if name == "fallback" else "."), t.game_id)
    except FileExistsError:
        rep.decision = "refused: this wrapper already dispatched for the game (lock exists)"
        return emit(EXIT_REFUSED)
    sent = clock()
    try:
        gh.dispatch_t90()
    except Exception as exc:  # noqa: BLE001
        rep.decision = f"dispatch call failed: {exc}"
        rep.dispatch = {"lock": str(lock), "sent_at": iso(sent), "accepted": False}
        return emit(EXIT_FAILED)
    rep.dispatch = {"lock": str(lock), "sent_at": iso(sent), "accepted": True,
                    "command": f"gh workflow run {WORKFLOW} -R {a.repo} --ref main -f job=t90"}
    run_id = wait_for_run(gh, sent, sleep, 180, clock)
    if run_id is None:
        rep.decision = "dispatched, but no run appeared within 180 s; do NOT re-dispatch, read back later"
        return emit(EXIT_FAILED)
    rep.dispatch["run_id"] = run_id
    deadline = clock() + dt.timedelta(minutes=a.wait_minutes)
    rb = readback(gh, run_id, t, a.expect_sha)
    while rb["verdict"] == "pending" and clock() < deadline:
        sleep(30)
        rb = readback(gh, run_id, t, a.expect_sha)
    rep.readback = rb
    rep.decision = {"processed": "dispatched and processed",
                    "pending": f"dispatched; run {run_id} still running, use --readback {run_id}"
                    }.get(rb["verdict"], f"dispatched; run {run_id} did NOT process the game")
    return emit(EXIT_OK if rb["verdict"] == "processed" else EXIT_FAILED)


if __name__ == "__main__":
    raise SystemExit(run())
