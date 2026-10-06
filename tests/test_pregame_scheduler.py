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
