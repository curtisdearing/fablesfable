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

Each slot's plan is persisted, so a slot is accounted for even after its games leave the
scoreboard (kicked off while the Mac slept, or ESPN rolled to the next week): it ends as
processed, already processed, superseded (regrouped / postponed) or MISSED -- never silently
dropped.  Every tick writes ``state/heartbeat.json``; ``--health`` reads it (no network, no
writes) and exits 1 on a stale heartbeat, an unusable scoreboard or an unresolved slot.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
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
#: How a launched wrapper appears in ``ps`` (see the command built in :func:`tick`).
WRAPPER_CMD = re.compile(r"(^|\s)\S*scripts/pregame_dispatch\.py\s+--execute(\s|$)")

TICK_MINUTES = 5
#: --health calls the heartbeat stale after this long without a tick (three missed ticks).
STALE_AFTER_MINUTES = 3 * TICK_MINUTES
#: --health reports finished slots whose last kickoff is at most this old.
RECENT_DAYS = 8
#: A dispatched slot's run is read back (again) until this long after its last launch.
READBACK_HOURS = 3
#: ESPN statuses of a game that will not be played at its listed time.
NOT_PLAYED = {"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_CANCELLED", "STATUS_SUSPENDED",
              "STATUS_FORFEIT"}
#: Terminal slot outcomes that need nobody's attention.
OK_FINALS = ("dispatched and processed", "already processed", "superseded")  # prefixes
assert GROUP_SPAN_MINUTES + 2 * TICK_MINUTES <= (
    pd.T90_DUE_MINUTES - pd.DEFAULT_MIN_LEAD_MINUTES - LAUNCH_MARGIN_MINUTES)


@dataclass
class Game:
    game_id: str
    season: int
    week: int
    kickoff: dt.datetime
    espn_event: str
    status: str = "STATUS_SCHEDULED"


class ScoreboardInvalid(ValueError):
    """The payload is not a usable week scoreboard: it must never read as 'no games'."""


@dataclass
class Board:
    season: Optional[int]
    stype: Optional[int]
    week: Optional[int]
    games: List[Game]              # every regular-season event on the board, any status


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


def _int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _season_type(v) -> Optional[int]:
    """ESPN gives the season type as ``2`` at the root but ``{"type": 2, ...}`` under leagues[0]."""
    return _int(v.get("type", v.get("id")) if isinstance(v, dict) else v)


def parse_board(body) -> Board:
    """Validate one ESPN week scoreboard and return its regular-season games (any status).

    Season / type / week are read from the root, else ``leagues[0].season``, else each event's
    own ``season`` / ``week`` (ESPN has served each of these shapes).  Anything that cannot be
    resolved raises :class:`ScoreboardInvalid`; only a well-formed board with no events is an
    empty week.
    """
    if not isinstance(body, dict) or not isinstance(body.get("events"), list):
        raise ScoreboardInvalid("no events list")
    root = body.get("season") if isinstance(body.get("season"), dict) else {}
    leagues = body.get("leagues") if isinstance(body.get("leagues"), list) else []
    league = (leagues[0] or {}).get("season") or {} if leagues and isinstance(leagues[0], dict) else {}
    season = _int(root.get("year")) or _int(league.get("year"))
    stype = _season_type(root.get("type")) or _season_type(league.get("type"))
    week = _int((body.get("week") or {}).get("number") if isinstance(body.get("week"), dict) else None)
    if not body["events"] and (season is None or stype is None):
        raise ScoreboardInvalid("no events and no season metadata")
    games: List[Game] = []
    for ev in body["events"]:
        ev = ev if isinstance(ev, dict) else {}
        eid = ev.get("id")
        es = ev.get("season") if isinstance(ev.get("season"), dict) else {}
        e_season, e_type = _int(es.get("year")) or season, _season_type(es.get("type")) or stype
        e_week = _int((ev.get("week") or {}).get("number")) or week
        if e_season is None or e_type is None:
            raise ScoreboardInvalid(f"event {eid}: season/type not resolvable")
        if e_type != 2:
            continue
        comp = (ev.get("competitions") or [{}])[0] or {}
        sides = {c.get("homeAway"): (c.get("team") or {}).get("abbreviation")
                 for c in comp.get("competitors") or [] if isinstance(c, dict)}
        if e_week is None or not sides.get("away") or not sides.get("home") or not ev.get("date"):
            raise ScoreboardInvalid(f"event {eid}: week, teams or date missing")
        try:
            kickoff = pd.parse_utc(str(ev["date"]))
        except ValueError as exc:
            raise ScoreboardInvalid(f"event {eid}: bad date {ev.get('date')!r}") from exc
        status = ((ev.get("status") or {}).get("type") or {}).get("name") or "STATUS_UNKNOWN"
        games.append(Game(game_id(e_season, e_week, sides["away"], sides["home"]), e_season,
                          e_week, kickoff, str(eid), status))
    if games:   # board metadata missing at the root: take it from the events
        season = season or games[0].season
        stype = stype or 2
        week = week or max(g.week for g in games)
    return Board(season, stype, week, games)


def games_from_scoreboard(body: Dict) -> List[Game]:
    """Scheduled regular-season games on one ESPN week scoreboard (raises if invalid)."""
    return [g for g in parse_board(body).games if g.status == "STATUS_SCHEDULED"]


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
    try:   # a zombie child of an earlier tick is not alive; after a restart the PID may
        # belong to an unrelated program, which must not hold the slot "running" forever
        out = subprocess.run(["ps", "-o", "stat=,command=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return True
    stat, _, cmd = out.partition(" ")
    return bool(out) and not stat.startswith("Z") and bool(WRAPPER_CMD.search(cmd))


def last_report(log: Optional[str]) -> Optional[Dict]:
    """The wrapper's final JSON report in a log, if it wrote one."""
    if not log or not os.path.isfile(log):
        return None
    text = Path(log).read_text(errors="replace")
    start = text.rfind("\n{")
    for chunk in ([text[start + 1:]] if start >= 0 else []) + ([text] if text.startswith("{") else []):
        try:
            body = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(body, dict):
            return body
    return None


def last_decision(log: Optional[str]) -> Optional[str]:
    """The wrapper's final ``decision`` from its JSON report in a log, else its last line."""
    rep = last_report(log)
    if rep is not None:
        return str(rep.get("decision"))
    if not log or not os.path.isfile(log):
        return None
    lines = [ln for ln in Path(log).read_text(errors="replace").splitlines() if ln.strip()]
    return lines[-1][:300] if lines else None


def already_processed(log: Optional[str]) -> bool:
    """The wrapper found every slot game already processed: nothing is left to dispatch."""
    rep = last_report(log) or {}
    return any(c.get("name") == "processed_state_guard" and pd.ALREADY_PROCESSED in str(c.get("detail"))
               for c in rep.get("checks") or [] if isinstance(c, dict))


@dataclass
class TickResult:
    now: str
    expect_sha: Optional[str]
    slots: List[Dict] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _load_json(path: Path, res: Optional[TickResult] = None) -> Dict:
    if not path.is_file():
        return {}
    try:
        body = json.loads(path.read_text())
        return body if isinstance(body, dict) else {}
    except ValueError:
        if res is not None:   # keep the evidence, start clean rather than crash every tick
            aside = path.with_name(f"{path.name}.corrupt-{res.now.replace(':', '')}")
            os.replace(path, aside)
            res.errors.append(f"{path.name} unreadable; moved to {aside.name}")
        return {}


def _write_json(path: Path, body: Dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _finish(key: str, st: Dict, games: List[str], notify: Notify, res: TickResult,
            readback: Optional[Callable[[int], None]] = None,
            now: Optional[dt.datetime] = None) -> None:
    """The wrapper dispatched (its lock exists) and has exited: settle the slot from its report.

    A run still in progress when the wrapper's wait ended is read back again (read-only
    ``--readback RUN_ID``, same SHA and slot games) on later ticks, until READBACK_HOURS after
    the slot's last launch: the slot is not complete until every member game is read back."""
    rep = last_report(st.get("log")) or {}
    rb = rep.get("readback") if isinstance(rep.get("readback"), dict) else {}
    run_id = (rep.get("dispatch") or {}).get("run_id") or rb.get("run_id") or st.get("run_id")
    if rep.get("mode") == "readback" and rb.get("verdict") == "processed":
        dec = "dispatched and processed (read back after the wrapper's wait)"
    elif rep.get("mode") == "readback" and rb.get("verdict") == "not-processed":
        dec = (f"dispatched; run {run_id} did NOT process "
               f"{', '.join(rb.get('unprocessed_games') or []) or 'the slot'}")
    else:
        dec = last_decision(st.get("log")) or "dispatched (decision not recorded)"
    if rb.get("verdict") == "pending" and run_id and readback is not None and now is not None:
        if now - pd.parse_utc(st["last_launch"]) <= dt.timedelta(hours=READBACK_HOURS):
            st["run_id"] = run_id
            readback(int(run_id))
            st["readbacks"] = st.get("readbacks", 0) + 1
            res.actions.append(f"{key}: run {run_id} still running; read-back {st['readbacks']} launched")
            return
        dec = f"read-back still pending {READBACK_HOURS} h after last launch: use --readback {run_id}"
    st["notified"], st["final"] = True, dec
    if not dec.startswith("dispatched and processed"):
        notify("fablesfable T-90", f"{key} ({', '.join(games)}): {dec}")
    res.actions.append(f"{key}: finished -> {dec}")


def _missed(key: str, st: Dict, games: List[str], why: str, notify: Notify, res: TickResult) -> None:
    st["notified"], st["final"] = True, f"MISSED: {why}"
    notify("fablesfable T-90 MISSED", f"{key} ({', '.join(games)}): {why}")
    res.actions.append(f"{key}: missed -> {why}")


def _never_attempted(st: Dict, prev_tick: Optional[str], now: dt.datetime) -> str:
    if st.get("attempts"):
        return (f"{st['attempts']} attempt(s), last at {st.get('last_attempt')}; the wrapper left "
                f"no decision in {st.get('log')}")
    if st.get("window_ticks"):
        return f"never attempted: {st.get('skip') or 'no launch was possible'}"
    return (f"never attempted (scheduler not running in window): no tick between "
            f"{st.get('dispatch_at')} and {st.get('last_launch')}; previous tick "
            f"{prev_tick or 'not recorded'}, this tick {pd.iso(now)} -- host asleep/off or "
            f"agent not loaded")


def tick(*, ops_dir: Path, repo_dir: Path, expect_sha: Optional[str], now: dt.datetime,
         fetch: Fetch = urllib_fetch, spawn: Spawn = spawn_detached, notify: Notify = mac_notify,
         alive: Callable[[Optional[int]], bool] = pid_alive, python: str = sys.executable,
         dry_run: bool = False) -> TickResult:
    if dry_run:
        notify = lambda title, text: None  # noqa: E731 -- a plan announces nothing
    receipts, logs = ops_dir / "receipts", ops_dir / "logs"
    state_path = ops_dir / "state" / "scheduler.json"
    beat_path = ops_dir / "state" / "heartbeat.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    res = TickResult(pd.iso(now), expect_sha)
    state: Dict[str, Dict] = _load_json(state_path, None if dry_run else res)
    prev_tick = _load_json(beat_path).get("tick_at")

    # 1. Official schedule: current + next week, validated.  An unusable board is an error,
    #    never "no games"; persisted slots are still accounted for below.
    games: List[Game] = []
    boards: List[Dict] = []
    try:
        cur = parse_board(json.loads(fetch(SCOREBOARD)))
        games += cur.games
        boards.append({"week": cur.week, "events": len(cur.games)})
        if cur.season and cur.week and cur.stype == 2:
            nxt = f"{SCOREBOARD}?seasontype=2&week={int(cur.week) + 1}&dates={int(cur.season)}"
            try:
                nb = parse_board(json.loads(fetch(nxt)))
                games += nb.games
                boards.append({"week": nb.week, "events": len(nb.games)})
            except Exception as exc:  # noqa: BLE001 -- the current week still schedules
                res.actions.append(f"next-week scoreboard unavailable ({type(exc).__name__})")
                res.errors.append(f"next-week scoreboard unusable: {type(exc).__name__}: {exc}"[:300])
    except Exception as exc:  # noqa: BLE001 -- persisted slots are still accounted for
        res.errors.append(f"current scoreboard unusable: {type(exc).__name__}: {exc}"[:300])
        res.actions.append("current scoreboard unusable; nothing launched this tick")
    def readback_for(key: str, st: Dict) -> Optional[Callable[[int], None]]:
        sha = st.get("sha") or expect_sha
        if dry_run or not sha or not st.get("named_game"):
            return None

        def launch(run_id: int) -> None:
            cmd = [python, str(repo_dir / "scripts" / "pregame_dispatch.py"), "--readback", str(run_id),
                   "--season", str(st["season"]), "--week", str(st["week"]),
                   "--game", st["named_game"], "--kickoff", st["named_kickoff"], "--expect-sha", sha,
                   "--receipt-dir", str(receipts), "--slot-games", ",".join(st["games"])]
            log = logs / f"readback-{st['named_game']}-{res.now.replace(':', '')}.log"
            st.update(pid=spawn(cmd, log), log=str(log))
        return launch

    status = {g.game_id: g.status for g in games}
    fresh = plan_slots([g for g in games if g.status == "STATUS_SCHEDULED"])
    on_plan = {g for s in fresh for g in s.games}

    # 2. Slots on the current plan: launch when due, record what happened.
    for s in fresh:
        st = state.setdefault(s.key, {"games": s.games, "attempts": 0})
        if not st.get("final"):   # kickoffs may move (flex): keep the latest official plan
            st.update(games=s.games, kickoffs=s.kickoffs, season=s.season, week=s.week,
                      named_game=s.named_game, named_kickoff=pd.iso(s.named_kickoff),
                      dispatch_at=pd.iso(s.dispatch_at), last_launch=pd.iso(s.last_launch))
        phase = s.phase(now)
        lock = receipts / f"dispatch-{s.named_game}.lock"
        running = alive(st.get("pid"))
        res.slots.append({"slot": s.key, "games": s.games, "dispatch_at": pd.iso(s.dispatch_at),
                          "last_launch": pd.iso(s.last_launch), "phase": phase,
                          "dispatched": lock.exists(), "final": st.get("final")})
        if st.get("final") or running:
            continue
        if lock.exists():
            _finish(s.key, st, s.games, notify, res, readback_for(s.key, st), now)
            continue
        if already_processed(st.get("log")):
            st["notified"], st["final"] = True, "already processed: every slot game has t90 leans"
            res.actions.append(f"{s.key}: {st['final']}")
            continue
        if phase == "closed":
            _missed(s.key, st, s.games, last_decision(st.get("log")) or _never_attempted(st, prev_tick, now),
                    notify, res)
            continue
        if phase != "due":
            continue
        st["window_ticks"] = st.get("window_ticks", 0) + 1
        if not expect_sha:
            st["skip"] = "remote main SHA unknown"
            res.actions.append(f"{s.key}: due but remote main SHA unknown; not launched")
            continue
        cmd = [python, str(repo_dir / "scripts" / "pregame_dispatch.py"), "--execute",
               "--season", str(s.season), "--week", str(s.week), "--game", s.named_game,
               "--kickoff", pd.iso(s.named_kickoff), "--expect-sha", expect_sha,
               "--receipt-dir", str(receipts), "--slot-games", ",".join(s.games)]
        if dry_run:
            res.actions.append(f"{s.key}: would launch {' '.join(cmd)}")
            continue
        log = logs / f"dispatch-{s.named_game}-{pd.iso(now).replace(':', '')}.log"
        st.update(pid=spawn(cmd, log), log=str(log), attempts=st.get("attempts", 0) + 1,
                  last_attempt=pd.iso(now), sha=expect_sha)
        res.actions.append(f"{s.key}: launched attempt {st['attempts']} (games {', '.join(s.games)})")

    # 3. Persisted slots no longer on the plan (kicked off while the host slept, ESPN rolled
    #    the week, board unusable, kickoff moved or game postponed): settle each one explicitly.
    planned = {s.key for s in fresh}
    for key, st in state.items():
        if key in planned or st.get("final") or not st.get("last_launch"):
            continue
        members = list(st.get("games") or [])
        if alive(st.get("pid")):
            continue
        if (receipts / f"dispatch-{st.get('named_game', key)}.lock").exists():
            _finish(key, st, members, notify, res, readback_for(key, st), now)
        elif already_processed(st.get("log")):
            st["notified"], st["final"] = True, "already processed: every slot game has t90 leans"
            res.actions.append(f"{key}: {st['final']}")
        elif members and all(g in on_plan or status.get(g) in NOT_PLAYED for g in members):
            moved = [g for g in members if g in on_plan]
            off = [f"{g} {status[g]}" for g in members if g not in on_plan]
            st["notified"], st["final"] = True, ("superseded: " + "; ".join(
                ([f"regrouped (kickoff changed): {', '.join(moved)}"] if moved else []) + off))
            if off:
                notify("fablesfable T-90", f"{key}: {st['final']}")
            res.actions.append(f"{key}: {st['final']}")
        elif now >= pd.parse_utc(st["last_launch"]):
            seen = ", ".join(f"{g} {status.get(g, 'not on board')}" for g in members)
            why = last_decision(st.get("log")) or _never_attempted(st, prev_tick, now)
            _missed(key, st, members, f"{why} [now: {seen}]", notify, res)

    if not dry_run:
        _write_json(state_path, state)
        upcoming = sorted((st for st in state.values() if not st.get("final") and st.get("dispatch_at")),
                          key=lambda st: st["dispatch_at"])
        _write_json(beat_path, {
            "tick_at": res.now, "previous_tick_at": prev_tick, "expect_sha": expect_sha,
            "boards": boards, "errors": res.errors, "scheduled_games": len(on_plan),
            "next_slots": [{k: st.get(k) for k in ("named_game", "games", "dispatch_at",
                                                   "last_launch", "attempts")}
                           for st in upcoming[:3]]})
    return res


def health(ops_dir: Path, now: dt.datetime) -> Dict:
    """Read-only health: heartbeat age, last tick's board errors, unresolved slots, next slots."""
    state = _load_json(ops_dir / "state" / "scheduler.json")
    beat = _load_json(ops_dir / "state" / "heartbeat.json")
    problems: List[str] = []
    if not beat.get("tick_at"):
        problems.append("no heartbeat: the scheduler has not ticked with heartbeat support")
    else:
        age = (now - pd.parse_utc(beat["tick_at"])).total_seconds()
        if age < -60:
            problems.append(f"heartbeat {beat['tick_at']} is in the future: host clock skew")
        elif age > STALE_AFTER_MINUTES * 60:
            problems.append(f"stale heartbeat: last tick {beat['tick_at']} ({age / 60:.0f} min ago); "
                            f"host asleep/off or agent not running")
        problems += [f"last tick: {e}" for e in beat.get("errors") or []]
        if not beat.get("expect_sha"):
            problems.append("last tick could not resolve the remote main SHA")
    recent, upcoming = [], []
    for key, st in sorted(state.items(), key=lambda kv: kv[1].get("dispatch_at") or ""):
        if not st.get("last_launch"):
            continue
        row = {"slot": key, "games": st.get("games"), "dispatch_at": st.get("dispatch_at"),
               "last_launch": st.get("last_launch"), "attempts": st.get("attempts", 0),
               "final": st.get("final")}
        if st.get("final"):
            if now - pd.parse_utc(st["last_launch"]) <= dt.timedelta(days=RECENT_DAYS):
                recent.append(row)
                if not str(st["final"]).startswith(OK_FINALS):
                    problems.append(f"{key}: {st['final']}")
        elif now >= pd.parse_utc(st["last_launch"]):
            recent.append(row)
            locked = (ops_dir / "receipts" / f"dispatch-{st.get('named_game', key)}.lock").exists()
            if locked and now - pd.parse_utc(st["last_launch"]) <= dt.timedelta(hours=READBACK_HOURS):
                row["final"] = "dispatched; read-back pending"
            else:
                problems.append(f"{key}: window closed {st['last_launch']} with no recorded outcome yet")
        else:
            upcoming.append(row)
    return {"now": pd.iso(now), "healthy": not problems, "problems": problems,
            "last_tick": beat.get("tick_at"), "expect_sha": beat.get("expect_sha"),
            "next_slots": upcoming[:4], "recent_slots": recent,
            "limits": "ticks only while the Mac is awake and the user is logged in; a sleeping, "
                      "closed-lid or powered-off Mac misses windows (reported as MISSED on the next tick)"}


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
    ap.add_argument("--health", action="store_true",
                    help="read-only health from the last heartbeat and slot state; exit 1 if unhealthy")
    a = ap.parse_args(argv)
    ops = Path(a.ops_dir).expanduser().resolve()
    repo = Path(a.repo_dir).expanduser().resolve()
    if a.health:   # no lock, no sync, no network, no writes
        rep = health(ops, dt.datetime.now(UTC))
        print(json.dumps(rep, indent=2, default=str))
        return 0 if rep["healthy"] else 1
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
    due = [r["slot"] for r in res.slots if r["phase"] == "due" and not r["final"]]
    if res.actions or due or res.errors:
        print(json.dumps(asdict(res), default=str))
    else:   # a quiet tick: one line, so the launchd log stays small
        nxt = min((r["dispatch_at"] for r in res.slots if r["phase"] == "future"), default=None)
        print(f"{res.now} idle; sha {str(sha)[:8]}; next dispatch {nxt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
