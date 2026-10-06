"""pregame_scheduler: every kickoff slot gets exactly one wrapper dispatch inside its window.

Reproduced gap (2026-10-05, ATL@NO Monday night): no T-90 run was dispatched, so the
Wednesday board (OUT tight end, two non-starting QBs) was the only read of the game.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pregame_dispatch as pd  # noqa: E402
import pregame_scheduler as ps  # noqa: E402

UTC = dt.timezone.utc
SHA = "a" * 40


def _t(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _ev(eid, away, home, kickoff, status="STATUS_SCHEDULED"):
    return {"id": eid, "date": kickoff, "status": {"type": {"name": status}},
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"abbreviation": home}},
                {"homeAway": "away", "team": {"abbreviation": away}}]}]}


def _board(week, events, stype=2):
    return {"season": {"year": 2026, "type": stype}, "week": {"number": week}, "events": events}


WEEK5 = _board(5, [
    _ev("1", "TB", "DAL", "2026-10-09T00:15Z"),       # Thursday
    _ev("2", "PHI", "JAX", "2026-10-11T13:30Z"),      # London
    _ev("3", "CHI", "GB", "2026-10-11T17:00Z"),
    _ev("4", "WSH", "NYG", "2026-10-11T17:00Z"),
    _ev("5", "DEN", "LAC", "2026-10-11T20:05Z"),
    _ev("6", "SEA", "SF", "2026-10-11T20:25Z"),
    _ev("7", "BAL", "ATL", "2026-10-12T00:20Z"),      # Sunday night
    _ev("8", "BUF", "LAR", "2026-10-13T00:15Z"),      # Monday night
    _ev("9", "KC", "LV", "2026-10-11T17:00Z", status="STATUS_POSTPONED"),
])


def _fetch(boards):
    def f(url):
        if url == ps.SCOREBOARD:
            return json.dumps(boards["current"]).encode()
        for wk, b in boards.items():
            if wk != "current" and f"week={wk}&" in url:
                return json.dumps(b).encode()
        return json.dumps(_board(0, [])).encode()       # an empty week, as ESPN serves it
    return f


class Harness:
    def __init__(self, tmp_path, boards):
        self.ops = tmp_path / "ops"
        self.repo = tmp_path / "repo"
        self.fetch = _fetch(boards)
        self.spawned, self.notes, self.live = [], [], set()

    def spawn(self, cmd, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("")
        self.spawned.append((cmd, log))
        pid = 1000 + len(self.spawned)
        self.live.add(pid)
        return pid

    def finish(self, decision, dispatched):
        cmd, log = self.spawned[-1]
        self.live.clear()
        log.write_text("[gh] ...\n" + json.dumps({"mode": "execute", "decision": decision}, indent=2) + "\n")
        if dispatched:
            game = cmd[cmd.index("--game") + 1]
            (self.ops / "receipts").mkdir(parents=True, exist_ok=True)
            (self.ops / "receipts" / f"dispatch-{game}.lock").write_text("{}")

    def tick(self, now, sha=SHA):
        return ps.tick(ops_dir=self.ops, repo_dir=self.repo, expect_sha=sha, now=_t(now),
                       fetch=self.fetch, spawn=self.spawn, alive=lambda pid: pid in self.live,
                       notify=lambda title, text: self.notes.append((title, text)), python="py")


# --------------------------------------------------------------------------- planning

def test_slots_group_nearby_kickoffs_and_cover_every_day_of_the_week():
    slots = {s.key: s for s in ps.plan_slots(ps.games_from_scoreboard(WEEK5))}
    assert list(slots) == ["2026_05_TB_DAL", "2026_05_PHI_JAX", "2026_05_CHI_GB",
                           "2026_05_DEN_LAC", "2026_05_BAL_ATL", "2026_05_BUF_LA"]
    assert slots["2026_05_CHI_GB"].games == ["2026_05_CHI_GB", "2026_05_WAS_NYG"]
    late = slots["2026_05_DEN_LAC"]
    assert late.games == ["2026_05_DEN_LAC", "2026_05_SEA_SF"]
    assert late.dispatch_at == _t("2026-10-11T18:55:00Z")      # 4:25 game's window opens
    assert late.last_launch == _t("2026-10-11T19:25:00Z")      # 4:05 game - 35 - 5
    assert slots["2026_05_TB_DAL"].dispatch_at == _t("2026-10-08T22:45:00Z")


def test_espn_aliases_map_to_nflverse_ids_and_unscheduled_games_are_ignored():
    ids = [g.game_id for g in ps.games_from_scoreboard(WEEK5)]
    assert "2026_05_WAS_NYG" in ids and "2026_05_BUF_LA" in ids
    assert not any("KC_LV" in i for i in ids)
    assert ps.games_from_scoreboard(_board(1, [_ev("x", "A", "B", "2026-10-11T17:00Z")], stype=3)) == []


def test_kickoffs_further_apart_than_the_span_get_separate_dispatches():
    b = _board(5, [_ev("1", "A", "B", "2026-10-11T17:00Z"), _ev("2", "C", "D", "2026-10-11T17:41Z")])
    assert len(ps.plan_slots(ps.games_from_scoreboard(b))) == 2


@pytest.mark.parametrize("offset_min", [0, 10, 20, 30, 40])
def test_every_slot_leaves_at_least_two_ticks_and_the_wrapper_accepts_its_timing(offset_min, capsys):
    k0 = _t("2026-10-11T20:00:00Z")
    b = _board(5, [_ev("1", "AAA", "BBB", pd.iso(k0)),
                   _ev("2", "CCC", "DDD", pd.iso(k0 + dt.timedelta(minutes=offset_min)))])
    (slot,) = ps.plan_slots(ps.games_from_scoreboard(b))
    assert slot.last_launch - slot.dispatch_at >= dt.timedelta(minutes=2 * ps.TICK_MINUTES)
    for at in (slot.dispatch_at, slot.last_launch - dt.timedelta(minutes=1)):
        code = pd.run(["--season", "2026", "--week", "5", "--game", slot.named_game,
                       "--kickoff", pd.iso(slot.named_kickoff), "--at", pd.iso(at)])
        out = capsys.readouterr().out
        assert code == pd.EXIT_OK and "would proceed to readiness checks" in out, out


# --------------------------------------------------------------------------- ticking

def test_due_slot_launches_the_wrapper_once_with_the_exact_arguments(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    assert not h.tick("2026-10-08T22:40:00Z").actions                # before the window
    res = h.tick("2026-10-08T22:46:00Z")
    assert len(h.spawned) == 1 and "launched attempt 1" in res.actions[0]
    cmd, _ = h.spawned[0]
    assert cmd[:3] == ["py", str(h.repo / "scripts" / "pregame_dispatch.py"), "--execute"]
    assert cmd[cmd.index("--game") + 1] == "2026_05_TB_DAL"
    assert cmd[cmd.index("--kickoff") + 1] == "2026-10-09T00:15:00Z"
    assert cmd[cmd.index("--expect-sha") + 1] == SHA
    assert cmd[cmd.index("--receipt-dir") + 1] == str(h.ops / "receipts")
    h.tick("2026-10-08T22:51:00Z")                                    # child still running
    assert len(h.spawned) == 1
    h.finish("dispatched and processed", dispatched=True)
    res = h.tick("2026-10-08T22:56:00Z")
    assert len(h.spawned) == 1 and not h.notes
    assert any("finished -> dispatched and processed" in a for a in res.actions)
    h.tick("2026-10-08T23:01:00Z")
    assert len(h.spawned) == 1 and not h.notes


def test_refused_or_not_ready_attempt_retries_while_the_slot_is_open(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T22:46:00Z")
    h.finish("not ready: no_active_production_run", dispatched=False)
    h.tick("2026-10-08T22:51:00Z")
    assert len(h.spawned) == 2
    state = json.loads((h.ops / "state" / "scheduler.json").read_text())
    assert state["2026_05_TB_DAL"]["attempts"] == 2


def test_slot_closing_undispatched_is_announced_once(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T23:30:00Z")
    h.finish("not ready: ci_green_on_sha", dispatched=False)
    h.tick("2026-10-08T23:36:00Z")                                    # past last launch
    h.tick("2026-10-08T23:41:00Z")
    assert len(h.spawned) == 1
    assert len(h.notes) == 1 and "MISSED" in h.notes[0][0] and "ci_green_on_sha" in h.notes[0][1]


def test_scheduler_not_running_in_window_is_still_reported(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-09T00:00:00Z")
    assert not h.spawned
    assert h.notes and "never attempted" in h.notes[0][1]


def test_dispatched_but_not_processed_is_announced(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T22:46:00Z")
    h.finish("dispatched; run 99 did NOT process the game", dispatched=True)
    h.tick("2026-10-08T22:51:00Z")
    assert len(h.notes) == 1 and "did NOT process" in h.notes[0][1]


def test_unknown_released_sha_never_launches(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    res = h.tick("2026-10-08T22:46:00Z", sha=None)
    assert not h.spawned and "SHA unknown" in res.actions[0]


def test_next_week_scoreboard_failure_is_recorded_and_the_current_week_still_runs(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    real = h.fetch
    h.fetch = lambda url: real(url) if url == ps.SCOREBOARD else (_ for _ in ()).throw(OSError("x"))
    res = h.tick("2026-10-08T22:46:00Z")
    assert len(h.spawned) == 1 and any("next-week scoreboard unavailable" in a for a in res.actions)


def test_next_weeks_thursday_is_scheduled_from_the_previous_weeks_scoreboard(tmp_path):
    week4 = _board(4, [_ev("0", "ATL", "NO", "2026-10-06T00:15Z", status="STATUS_FINAL")])
    h = Harness(tmp_path, {"current": week4, 5: WEEK5})
    h.tick("2026-10-08T22:46:00Z")
    assert len(h.spawned) == 1 and "2026_05_TB_DAL" in h.spawned[0][0]


def test_ops_dir_inside_the_checkout_is_refused(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    assert ps.main(["--ops-dir", str(repo / "ops"), "--repo-dir", str(repo), "--no-sync",
                    "--dry-run"]) == 2


# --------------------------------------------------------------------------- feed shapes
# ESPN has served season/type/week at the root, under leagues[0] (type as an object there),
# and only on each event.  Observed 2026-10-06: leagues[0].season.type == {"id": "2", "type": 2,
# "name": "Regular Season", ...} and every event carries season {year, type} and week.

def _leagues_only(board):
    b = {k: v for k, v in board.items() if k != "season"}
    b["leagues"] = [{"season": {"year": 2026, "type": {"id": "2", "type": 2, "name": "Regular Season",
                                                       "abbreviation": "reg"}}}]
    return b


def _events_only(board):
    wk = board["week"]["number"]
    return {"events": [{**e, "season": {"year": 2026, "type": 2, "slug": "regular-season"},
                        "week": {"number": wk}} for e in board["events"]]}


@pytest.mark.parametrize("shape", [_leagues_only, _events_only])
def test_season_metadata_off_the_root_still_schedules_every_game(shape):
    ids = [g.game_id for g in ps.games_from_scoreboard(shape(WEEK5))]
    assert ids == [g.game_id for g in ps.games_from_scoreboard(WEEK5)] and len(ids) == 8


@pytest.mark.parametrize("shape", [_leagues_only, _events_only])
def test_next_weeks_thursday_is_found_when_the_current_board_has_no_root_season(tmp_path, shape):
    week4 = _board(4, [_ev("0", "ATL", "NO", "2026-10-06T00:15Z", status="STATUS_FINAL")])
    h = Harness(tmp_path, {"current": shape(week4), 5: WEEK5})
    h.tick("2026-10-08T22:46:00Z")
    assert len(h.spawned) == 1 and "2026_05_TB_DAL" in h.spawned[0][0]


@pytest.mark.parametrize("payload", [
    b'{"message": "rate limited"}',                               # no events list
    b"<html>maintenance</html>",                                  # not JSON
    json.dumps({"events": [_ev("1", "TB", "DAL", "2026-10-09T00:15Z")]}).encode(),   # no season anywhere
    json.dumps(_board(5, [{"id": "1", "date": "2026-10-09T00:15Z",
                           "status": {"type": {"name": "STATUS_SCHEDULED"}}}])).encode(),  # no teams
])
def test_an_unusable_scoreboard_is_an_error_never_an_empty_week(tmp_path, payload):
    h = Harness(tmp_path, {"current": WEEK5})
    h.fetch = lambda url: payload
    res = h.tick("2026-10-08T22:46:00Z")
    assert not h.spawned and res.errors and "current scoreboard unusable" in res.errors[0]
    beat = json.loads((h.ops / "state" / "heartbeat.json").read_text())
    assert beat["errors"] == res.errors
    rep = ps.health(h.ops, _t("2026-10-08T22:47:00Z"))
    assert not rep["healthy"] and any("scoreboard unusable" in p for p in rep["problems"])


# --------------------------------------------------------------------------- whole slot

def test_the_wrapper_is_told_every_game_in_the_slot(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-11T15:35:00Z")                     # 1 PM ET slot: CHI@GB + WSH@NYG
    (cmd, _), = h.spawned
    assert cmd[cmd.index("--game") + 1] == "2026_05_CHI_GB"
    assert cmd[cmd.index("--slot-games") + 1] == "2026_05_CHI_GB,2026_05_WAS_NYG"


def test_slot_already_processed_by_another_run_is_terminal_not_retried_or_missed(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T22:46:00Z")
    cmd, log = h.spawned[-1]
    h.live.clear()
    log.write_text(json.dumps({"decision": "not ready: processed_state_guard", "checks": [
        {"name": "processed_state_guard", "ok": False,
         "detail": "state-9-1.tar.gz: ...; ALREADY PROCESSED: every slot game has t90 lean rows"}]},
        indent=2) + "\n")
    for now in ("2026-10-08T22:51:00Z", "2026-10-08T23:40:00Z", "2026-10-09T03:00:00Z"):
        h.tick(now)
    assert len(h.spawned) == 1 and not h.notes
    st = json.loads((h.ops / "state" / "scheduler.json").read_text())["2026_05_TB_DAL"]
    assert st["final"].startswith("already processed")


# --------------------------------------------------------------------------- terminal status

def _with(board, **status_by_event):
    return {**board, "events": [{**e, "status": {"type": {"name": status_by_event.get(f"e{e['id']}",
                                                                                  e["status"]["type"]["name"])}}}
                                for e in board["events"]]}


@pytest.mark.parametrize("later_board", [
    lambda: _with(WEEK5, e1="STATUS_IN_PROGRESS"),          # host woke after kickoff
    lambda: _board(6, []),                                   # ESPN already rolled the week
])
def test_slot_that_passed_while_the_host_slept_is_reported_missed(tmp_path, later_board):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T20:00:00Z")                                     # Thursday, before the window
    h.fetch = _fetch({"current": later_board()})
    res = h.tick("2026-10-09T01:30:00Z")                               # first tick after kickoff
    assert not h.spawned
    (title, text), = [n for n in h.notes if "TB_DAL" in n[1]]
    assert "MISSED" in title and "never attempted" in text and "2026-10-08T20:00:00Z" in text
    assert any(a.startswith("2026_05_TB_DAL: missed") for a in res.actions)
    rep = ps.health(h.ops, _t("2026-10-09T01:31:00Z"))
    assert not rep["healthy"] and any(p.startswith("2026_05_TB_DAL: MISSED") for p in rep["problems"])


def test_dispatched_slot_settles_even_after_its_games_leave_the_board(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T22:46:00Z")
    h.fetch = _fetch({"current": _with(WEEK5, e1="STATUS_FINAL")})
    h.tick("2026-10-09T01:00:00Z")                 # wrapper still waiting on its run
    h.finish("dispatched; run 7 did NOT process 2026_05_TB_DAL", dispatched=True)
    h.tick("2026-10-09T03:30:00Z")
    assert [n for n in h.notes if "did NOT process" in n[1]]


def test_flexed_or_postponed_games_supersede_their_old_slot_without_a_false_miss(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-10T12:00:00Z")
    flexed = {**WEEK5, "events": [({**e, "date": "2026-10-11T20:25Z"} if e["id"] == "7" else e)
                                  for e in WEEK5["events"]]}
    h.fetch = _fetch({"current": _with(flexed, e8="STATUS_POSTPONED")})
    h.tick("2026-10-10T12:05:00Z")
    h.tick("2026-10-13T06:00:00Z")                 # long after the old SNF and MNF windows
    st = json.loads((h.ops / "state" / "scheduler.json").read_text())
    assert st["2026_05_BAL_ATL"]["final"].startswith("superseded: regrouped")
    assert "2026_05_BAL_ATL" in st["2026_05_DEN_LAC"]["games"]
    assert st["2026_05_BUF_LA"]["final"] == "superseded: 2026_05_BUF_LA STATUS_POSTPONED"
    missed = [n[1].split(" ")[0] for n in h.notes if "MISSED" in n[0]]
    assert "2026_05_BAL_ATL" not in missed and "2026_05_BUF_LA" not in missed
    assert "2026_05_DEN_LAC" in missed     # the regrouped slot itself had no tick in its window


def test_a_reused_pid_after_restart_does_not_count_as_the_running_wrapper():
    import os
    assert ps.pid_alive(os.getpid()) is False      # alive, but not the wrapper (even when this
    assert ps.pid_alive(None) is False             # pytest's argv names test_pregame_dispatch.py)
    assert ps.WRAPPER_CMD.search("/usr/bin/python3 /x/runner/scripts/pregame_dispatch.py --execute "
                                 "--season 2026")
    assert ps.WRAPPER_CMD.search("python3 /x/runner/scripts/pregame_dispatch.py --readback 77 --season 2026")
    assert not ps.WRAPPER_CMD.search("python -m pytest tests/test_pregame_dispatch.py --execute")
    assert not ps.WRAPPER_CMD.search("python3 /x/scripts/pregame_dispatch.py --check --season 2026")


def test_corrupt_state_is_set_aside_instead_of_crashing_every_tick(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    (h.ops / "state").mkdir(parents=True)
    (h.ops / "state" / "scheduler.json").write_text('{"2026_05_TB_DAL": {"attem')
    res = h.tick("2026-10-08T22:46:00Z")
    assert len(h.spawned) == 1 and any("unreadable" in e for e in res.errors)
    assert list((h.ops / "state").glob("scheduler.json.corrupt-*"))


# --------------------------------------------------------------------------- heartbeat / health

def test_heartbeat_health_and_stale_detection(tmp_path, capsys):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T20:00:00Z")
    rep = ps.health(h.ops, _t("2026-10-08T20:04:00Z"))
    assert rep["healthy"] and rep["last_tick"] == "2026-10-08T20:00:00Z"
    assert rep["next_slots"][0]["slot"] == "2026_05_TB_DAL"
    assert rep["next_slots"][0]["dispatch_at"] == "2026-10-08T22:45:00Z"
    stale = ps.health(h.ops, _t("2026-10-08T20:30:00Z"))
    assert not stale["healthy"] and stale["problems"][0].startswith("stale heartbeat")
    assert ps.health(tmp_path / "nowhere", _t("2026-10-08T20:00:00Z"))["healthy"] is False


def test_health_cli_is_read_only(tmp_path, capsys):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-09-30T20:00:00Z")                  # before the real clock: stale on any run date
    before = {p: p.read_bytes() for p in (h.ops / "state").iterdir()}
    code = ps.main(["--ops-dir", str(h.ops), "--health"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["problems"]
    assert {p: p.read_bytes() for p in (h.ops / "state").iterdir()} == before


# --------------------------------------------------------------------------- odd days and DST

ODD = _board(9, [
    _ev("1", "AAA", "BBB", "2026-10-29T23:15Z"),      # Thursday, still EDT
    _ev("2", "CCC", "DDD", "2026-10-31T20:30Z"),      # Saturday
    _ev("3", "EEE", "FFF", "2026-11-01T14:30Z"),      # Sunday 9:30 ET abroad, first EST day
    _ev("4", "GGG", "HHH", "2026-11-01T18:00Z"),      # Sunday 1:00 ET (EST)
    _ev("5", "III", "JJJ", "2026-11-01T21:05Z"),
    _ev("6", "KKK", "LLL", "2026-11-01T21:25Z"),
    _ev("7", "MMM", "NNN", "2026-11-03T01:15Z"),      # Monday night
    _ev("8", "OOO", "PPP", "2026-11-04T01:00Z"),      # Tuesday (weather move)
])


def test_every_odd_day_and_post_dst_slot_is_planned_in_utc_and_accepted_by_the_wrapper(capsys):
    slots = ps.plan_slots(ps.games_from_scoreboard(ODD))
    assert [s.games for s in slots] == [["2026_09_AAA_BBB"], ["2026_09_CCC_DDD"], ["2026_09_EEE_FFF"],
                                        ["2026_09_GGG_HHH"], ["2026_09_III_JJJ", "2026_09_KKK_LLL"],
                                        ["2026_09_MMM_NNN"], ["2026_09_OOO_PPP"]]
    for s in slots:
        assert s.dispatch_at == pd.parse_utc(s.kickoffs[-1]) - dt.timedelta(minutes=90)
        code = pd.run(["--season", "2026", "--week", "9", "--game", s.named_game,
                       "--kickoff", pd.iso(s.named_kickoff), "--at", pd.iso(s.dispatch_at)])
        assert code == pd.EXIT_OK, capsys.readouterr().out


def test_a_heartbeat_from_the_future_is_flagged_as_clock_skew(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T20:00:00Z")
    rep = ps.health(h.ops, _t("2026-10-08T19:00:00Z"))
    assert not rep["healthy"] and "clock skew" in rep["problems"][0]


def test_an_attempt_that_left_no_decision_is_not_reported_as_never_attempted(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T22:46:00Z")
    h.live.clear()                                 # wrapper died before writing its report
    h.tick("2026-10-08T23:40:00Z")
    (title, text), = h.notes
    assert "MISSED" in title and "1 attempt(s)" in text and "never attempted" not in text


def test_health_waits_for_a_dispatched_slots_read_back_but_not_forever(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T22:46:00Z")
    (h.ops / "receipts").mkdir(parents=True, exist_ok=True)
    (h.ops / "receipts" / "dispatch-2026_05_TB_DAL.lock").write_text("1 x\n")   # run in flight
    h.tick("2026-10-08T23:40:00Z")
    rep = ps.health(h.ops, _t("2026-10-08T23:41:00Z"))
    assert rep["healthy"], rep["problems"]
    assert rep["recent_slots"][0]["final"] == "dispatched; read-back pending"
    late = ps.health(h.ops, _t("2026-10-09T03:00:00Z"))
    assert any("no recorded outcome" in p for p in late["problems"])


def test_a_run_outlasting_the_wrappers_wait_is_read_back_later_for_the_whole_slot(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-11T15:31:00Z")                                     # 1 PM ET slot
    cmd, log = h.spawned[-1]
    h.live.clear()
    (h.ops / "receipts").mkdir(parents=True, exist_ok=True)
    (h.ops / "receipts" / "dispatch-2026_05_CHI_GB.lock").write_text("1 x\n")
    log.write_text(json.dumps({"mode": "execute", "decision": "dispatched; run 77 still running, "
                               "use --readback 77", "dispatch": {"run_id": 77},
                               "readback": {"run_id": 77, "verdict": "pending"}}) + "\n")
    res = h.tick("2026-10-11T16:50:00Z")
    rb_cmd, rb_log = h.spawned[-1]
    chi = lambda: [n for n in h.notes if n[1].startswith("2026_05_CHI_GB")]  # noqa: E731
    assert any("read-back 1 launched" in a for a in res.actions) and not chi()
    assert rb_cmd[rb_cmd.index("--readback") + 1] == "77" and "--execute" not in rb_cmd
    assert rb_cmd[rb_cmd.index("--expect-sha") + 1] == SHA
    assert rb_cmd[rb_cmd.index("--slot-games") + 1] == "2026_05_CHI_GB,2026_05_WAS_NYG"
    h.live.clear()
    rb_log.write_text(json.dumps({"mode": "readback", "decision": "not-processed", "readback": {
        "run_id": 77, "verdict": "not-processed", "unprocessed_games": ["2026_05_WAS_NYG"]}}) + "\n")
    h.tick("2026-10-11T16:55:00Z")
    (title, text), = chi()
    assert "did NOT process 2026_05_WAS_NYG" in text


def test_a_processed_late_read_back_settles_quietly_and_pending_is_bounded(tmp_path):
    h = Harness(tmp_path, {"current": WEEK5})
    h.tick("2026-10-08T22:46:00Z")
    _, log = h.spawned[-1]
    h.live.clear()
    (h.ops / "receipts").mkdir(parents=True, exist_ok=True)
    (h.ops / "receipts" / "dispatch-2026_05_TB_DAL.lock").write_text("1 x\n")
    pending = {"mode": "execute", "decision": "dispatched; run 5 still running",
               "dispatch": {"run_id": 5}, "readback": {"run_id": 5, "verdict": "pending"}}
    log.write_text(json.dumps(pending) + "\n")
    h.tick("2026-10-09T00:10:00Z")
    _, rb_log = h.spawned[-1]
    h.live.clear()
    rb_log.write_text(json.dumps({"mode": "readback", "decision": "processed",
                                  "readback": {"run_id": 5, "verdict": "processed"}}) + "\n")
    h.tick("2026-10-09T00:15:00Z")
    st = json.loads((h.ops / "state" / "scheduler.json").read_text())["2026_05_TB_DAL"]
    assert st["final"].startswith("dispatched and processed") and not h.notes
    assert ps.health(h.ops, _t("2026-10-09T00:16:00Z"))["healthy"]
    # Still pending 3 h after the last launch: settled as such, announced, unhealthy.
    h2 = Harness(tmp_path / "b", {"current": WEEK5})
    h2.tick("2026-10-08T22:46:00Z")
    _, log2 = h2.spawned[-1]
    h2.live.clear()
    (h2.ops / "receipts").mkdir(parents=True, exist_ok=True)
    (h2.ops / "receipts" / "dispatch-2026_05_TB_DAL.lock").write_text("1 x\n")
    log2.write_text(json.dumps(pending) + "\n")
    h2.tick("2026-10-09T02:40:00Z")
    assert h2.notes and "read-back still pending" in h2.notes[0][1]
