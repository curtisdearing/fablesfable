#!/usr/bin/env python3
"""Per-slot T-90 scheduler: runs ``pregame_dispatch.py --execute`` inside every kickoff window.

GitHub's crons land T-90 runs anywhere from on time to 5.6 h late, and
``pregame_dispatch.py`` deliberately installs no scheduler: someone awake on a machine with
an authenticated ``gh`` has to run it inside ``[kickoff - 90 min, kickoff - 35 min)``.  On
2026-10-05 nobody did for the Monday game (ATL@NO): no T-90 run, so an OUT tight end and two
non-starting QBs stayed on the Wednesday board.  This is that someone.

One tick (run every 5 minutes by launchd / cron, see ``docs/PREGAME_DISPATCH.md``):

1. Read the official ESPN scoreboard for the current and next regular-season week.
2. Group scheduled kickoffs into dispatch slots.  One ``job_t90`` run processes EVERY game
   kicking off within 90 minutes of its clock, and the wrapper refuses a second dispatch
   within an hour of a window, so kickoffs at most ``GROUP_SPAN_MINUTES`` apart share one
   dispatch, sent when the LAST of them enters its T-90 window (4:05 + 4:25 ET Sunday games:
   one run at 4:25 - 90 min, i.e. 4:05 - 70 min).
3. For a slot whose dispatch time has come and whose last safe launch has not passed, spawn
   ``pregame_dispatch.py --execute`` (detached; it waits for and reads back its own run) for
   the slot's earliest game, unless the wrapper's own one-shot lock already exists (dispatched)
   or an earlier attempt is still running.  A refused / not-ready attempt simply retries on
   the next tick while the slot is still open.
4. Notify (macOS notification when available) once per slot: dispatched-and-processed is
   silent; anything else -- dispatched but not processed, or the slot closing undispatched --
   is announced with the wrapper's last decision.

Every guard the wrapper applies (exact released SHA, green CI on it, no active production run,
no prior dispatch this window, processed-state guard, official kickoff) still applies; this
script decides only WHEN to ask.  Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pregame_dispatch as pd  # noqa: E402  (shares timing constants and team aliases)

UTC = dt.timezone.utc
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
#: Kickoffs at most this far apart share one dispatch.  The shared dispatch opens when the
#: latest game enters its T-90 window and must leave the earliest game at least two 5-minute
#: ticks before its last safe launch (asserted below).
GROUP_SPAN_MINUTES = 40
#: Stop launching this long before the wrapper's own latest dispatch: its readiness checks
#: (state download, API reads) take a few minutes and it re-checks timing afterwards.
LAUNCH_MARGIN_MINUTES = 5
NFLVERSE_ABBR = {espn: nv for nv, espn in pd.ESPN_ABBR.items()}   # WSH -> WAS, LAR -> LA

TICK_MINUTES = 5
assert GROUP_SPAN_MINUTES + 2 * TICK_MINUTES <= (
    pd.T90_DUE_MINUTES - pd.DEFAULT_MIN_LEAD_MINUTES - LAUNCH_MARGIN_MINUTES)


@dataclass
class Game:
    game_id: str
    season: int
    week: int
    kickoff: dt.datetime
    espn_event: str


@dataclass
class Slot:
    key: str                       # earliest game id: also the wrapper's lock name
    season: int
    week: int
    games: List[str]
    kickoffs: List[str]
    named_game: str                # the game the wrapper is invoked for (earliest kickoff)
    named_kickoff: dt.datetime
    dispatch_at: dt.datetime       # latest kickoff in the slot - 90 min
    last_launch: dt.datetime       # earliest kickoff - min lead - launch margin

    def phase(self, now: dt.datetime) -> str:
        if now < self.dispatch_at:
            return "future"
        if now < self.last_launch:
            return "due"
        return "closed"


def game_id(season: int, week: int, away: str, home: str) -> str:
    a, h = NFLVERSE_ABBR.get(away, away), NFLVERSE_ABBR.get(home, home)
    return f"{season}_{week:02d}_{a}_{h}"


def games_from_scoreboard(body: Dict) -> List[Game]:
    """Scheduled regular-season games on one ESPN week scoreboard."""
    out: List[Game] = []
    season = (body.get("season") or {}).get("year")
    stype = (body.get("season") or {}).get("type")
    week = (body.get("week") or {}).get("number")
    if stype != 2 or not season or not week:
        return out
    for ev in body.get("events") or []:
        if ((ev.get("status") or {}).get("type") or {}).get("name") != "STATUS_SCHEDULED":
            continue
        comp = (ev.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): (c.get("team") or {}).get("abbreviation")
                 for c in comp.get("competitors") or []}
        if not sides.get("away") or not sides.get("home"):
            continue
        out.append(Game(game_id(int(season), int(week), sides["away"], sides["home"]),
                        int(season), int(week), pd.parse_utc(ev["date"]), str(ev.get("id"))))
    return out


def plan_slots(games: Iterable[Game]) -> List[Slot]:
    """Group kickoffs into dispatch slots (greedy from the earliest, span <= GROUP_SPAN_MINUTES)."""
    games = sorted({g.game_id: g for g in games}.values(), key=lambda g: (g.kickoff, g.game_id))
    groups: List[List[Game]] = []
    for g in games:
        if groups and (g.kickoff - groups[-1][0].kickoff) <= dt.timedelta(minutes=GROUP_SPAN_MINUTES) \
                and g.week == groups[-1][0].week:
            groups[-1].append(g)
        else:
            groups.append([g])
    slots = []
    for grp in groups:
        first, last = grp[0], grp[-1]
        slots.append(Slot(
            key=first.game_id, season=first.season, week=first.week,
            games=[g.game_id for g in grp], kickoffs=[pd.iso(g.kickoff) for g in grp],
            named_game=first.game_id, named_kickoff=first.kickoff,
            dispatch_at=last.kickoff - dt.timedelta(minutes=pd.T90_DUE_MINUTES),
            last_launch=first.kickoff - dt.timedelta(
                minutes=pd.DEFAULT_MIN_LEAD_MINUTES + LAUNCH_MARGIN_MINUTES)))
    return slots


# --------------------------------------------------------------------------- #
# the tick

Fetch = Callable[[str], bytes]
Spawn = Callable[[List[str], Path], int]
Notify = Callable[[str, str], None]


def urllib_fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fablesfable-pregame-scheduler"})
    with urllib.request.urlopen(req, timeout=20) as r:   # noqa: S310 -- fixed https host
        return r.read()


def spawn_detached(cmd: List[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "ab")
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    return p.pid


def mac_notify(title: str, text: str) -> None:
    safe = lambda s: s.replace("\\", "\\\\").replace('"', "'")[:230]  # noqa: E731
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{safe(text)}" with title "{safe(title)}"'],
                       timeout=10, check=False, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        pass


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:   # a zombie child of an earlier tick is not alive
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        return bool(out) and not out.startswith("Z")
    except (OSError, subprocess.SubprocessError):
        return True


def last_decision(log: Optional[str]) -> Optional[str]:
    """The wrapper's final ``decision`` from its JSON report in a log, else its last line."""
    if not log or not os.path.isfile(log):
        return None
    text = Path(log).read_text(errors="replace")
    start = text.rfind("\n{")
    for chunk in ([text[start + 1:]] if start >= 0 else []) + ([text] if text.startswith("{") else []):
        try:
            return str(json.loads(chunk).get("decision"))
        except (ValueError, AttributeError):
            continue
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1][:300] if lines else None


@dataclass
class TickResult:
    now: str
    expect_sha: Optional[str]
    slots: List[Dict] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


def tick(*, ops_dir: Path, repo_dir: Path, expect_sha: Optional[str], now: dt.datetime,
         fetch: Fetch = urllib_fetch, spawn: Spawn = spawn_detached, notify: Notify = mac_notify,
         alive: Callable[[Optional[int]], bool] = pid_alive, python: str = sys.executable,
         dry_run: bool = False) -> TickResult:
    if dry_run:
        notify = lambda title, text: None  # noqa: E731 -- a plan announces nothing
    receipts, logs = ops_dir / "receipts", ops_dir / "logs"
    state_path = ops_dir / "state" / "scheduler.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Dict] = json.loads(state_path.read_text()) if state_path.is_file() else {}
    res = TickResult(pd.iso(now), expect_sha)

    body = json.loads(fetch(SCOREBOARD))
    games: List[Game] = games_from_scoreboard(body)
    season, week = (body.get("season") or {}).get("year"), (body.get("week") or {}).get("number")
    if season and week and (body.get("season") or {}).get("type") == 2:
        nxt = f"{SCOREBOARD}?seasontype=2&week={int(week) + 1}&dates={int(season)}"
        try:
            games += games_from_scoreboard(json.loads(fetch(nxt)))
        except Exception as exc:  # noqa: BLE001 -- the current week still schedules
            res.actions.append(f"next-week scoreboard unavailable ({type(exc).__name__})")

    for s in plan_slots(games):
        st = state.setdefault(s.key, {"games": s.games, "attempts": 0})
        phase = s.phase(now)
        lock = receipts / f"dispatch-{s.named_game}.lock"
        row = {"slot": s.key, "games": s.games, "dispatch_at": pd.iso(s.dispatch_at),
               "last_launch": pd.iso(s.last_launch), "phase": phase, "dispatched": lock.exists()}
        res.slots.append(row)
        running = alive(st.get("pid"))
        if lock.exists() and not running and not st.get("notified"):
            dec = last_decision(st.get("log")) or "dispatched (decision not recorded)"
            st["notified"], st["final"] = True, dec
            if dec != "dispatched and processed":
                notify("fablesfable T-90", f"{s.key}: {dec}")
            res.actions.append(f"{s.key}: finished -> {dec}")
            continue
        if phase != "due" or lock.exists() or running:
            if phase == "closed" and not lock.exists() and not running and not st.get("notified"):
                dec = last_decision(st.get("log")) or "never attempted (scheduler not running in window)"
                st["notified"], st["final"] = True, f"MISSED: {dec}"
                notify("fablesfable T-90 MISSED", f"{s.key} ({', '.join(s.games)}): {dec}")
                res.actions.append(f"{s.key}: missed -> {dec}")
            continue
        if not expect_sha:
            res.actions.append(f"{s.key}: due but remote main SHA unknown; not launched")
            continue
        cmd = [python, str(repo_dir / "scripts" / "pregame_dispatch.py"), "--execute",
               "--season", str(s.season), "--week", str(s.week), "--game", s.named_game,
               "--kickoff", pd.iso(s.named_kickoff), "--expect-sha", expect_sha,
               "--receipt-dir", str(receipts)]
        if dry_run:
            res.actions.append(f"{s.key}: would launch {' '.join(cmd)}")
            continue
        log = logs / f"dispatch-{s.named_game}-{pd.iso(now).replace(':', '')}.log"
        st.update(pid=spawn(cmd, log), log=str(log), attempts=st.get("attempts", 0) + 1,
                  last_attempt=pd.iso(now))
        res.actions.append(f"{s.key}: launched attempt {st['attempts']} (games {', '.join(s.games)})")

    if not dry_run:
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, state_path)
    return res


def sync_repo(repo_dir: Path) -> Optional[str]:
    """Fast-forward the scheduler's dedicated checkout to origin/main; return that SHA."""
    def git(*a):
        return subprocess.run(["git", "-C", str(repo_dir), *a], capture_output=True, text=True,
                              timeout=60, check=True).stdout.strip()
    try:
        git("fetch", "-q", "origin", "main")
        git("checkout", "-q", "--detach", "origin/main")
        return git("rev-parse", "origin/main")
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ops-dir", default=str(Path.home() / "fablesfable-ops"),
                    help="state/receipts/logs directory OUTSIDE the checkout")
    ap.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[1]),
                    help="the scheduler's own checkout (fast-forwarded to origin/main each tick)")
    ap.add_argument("--no-sync", action="store_true", help="do not fetch/advance the checkout")
    ap.add_argument("--dry-run", action="store_true", help="plan and print; launch nothing")
    a = ap.parse_args(argv)
    ops = Path(a.ops_dir).expanduser().resolve()
    repo = Path(a.repo_dir).expanduser().resolve()
    if ops == repo or repo in ops.parents:
        print("REFUSED: --ops-dir must be outside the checkout")
        return 2
    (ops / "state").mkdir(parents=True, exist_ok=True)
    with open(ops / "state" / "tick.lock", "w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another tick is running")
            return 0
        sha = None if a.no_sync else sync_repo(repo)
        if sha is None:
            sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "origin/main"],
                                 capture_output=True, text=True).stdout.strip() or None
        res = tick(ops_dir=ops, repo_dir=repo, expect_sha=sha, now=dt.datetime.now(UTC),
                   dry_run=a.dry_run)
    due = [r["slot"] for r in res.slots if r["phase"] == "due"]
    if res.actions or due:
        print(json.dumps(asdict(res), default=str))
    else:   # a quiet tick: one line, so the launchd log stays small
        nxt = min((r["dispatch_at"] for r in res.slots if r["phase"] == "future"), default=None)
        print(f"{res.now} idle; sha {str(sha)[:8]}; next dispatch {nxt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
