"""The T-90 refresh only happens if the SCHEDULER fires while a game is due.

``scripts.auto_weekly.games_due_for_t90`` picks games in [T-90, T-0); this
test replays the workflow's cron strings -- evaluated in UTC, exactly as
GitHub Actions evaluates them -- against every 2026 regular-season kickoff
and asserts that at least one firing lands inside each game's due window.
That covers the Wednesday opener (2026-09-09), Black Friday, Christmas
Friday, the December Saturdays, and the 9:30 ET London kickoffs, across the
EDT->EST change on 2026-11-01.

The slate is read from historical/lines_extra.parquet when present (the
runner's ingest writes it); otherwise the embedded 2026 kickoff patterns
(every distinct weekday x kickoff-time pair on the published schedule, plus
the odd-day games by date) stand in.
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.auto_weekly as _aw  # noqa: E402

games_due_for_t90 = _aw.games_due_for_t90
T90_DUE_MINUTES = getattr(_aw, "T90_DUE_MINUTES", 90)   # absent before the review fix

ET = ZoneInfo("America/New_York")
WORKFLOW = ROOT / ".github" / "workflows" / "live-weekly.yml"

# (gameday, gametime) -- the 2026 REG slate's odd-day games by date plus one
# representative of every ordinary weekday x kickoff pair, in both DST regimes.
EMBEDDED_2026 = [
    ("2026-09-09", "20:20"),  # Wed opener
    ("2026-11-25", "20:00"),  # Wed (Thanksgiving eve)
    ("2026-11-26", "13:00"), ("2026-11-26", "16:30"), ("2026-11-26", "20:20"),  # Thanksgiving
    ("2026-11-27", "15:00"),  # Black Friday
    ("2026-12-19", "17:00"), ("2026-12-19", "20:20"),   # Saturdays
    ("2026-12-25", "13:00"), ("2026-12-25", "16:30"), ("2026-12-25", "20:15"),  # Xmas Friday
    ("2026-10-04", "09:30"), ("2026-11-15", "09:30"),   # London, EDT and EST
    ("2026-09-13", "13:00"), ("2026-09-13", "16:05"), ("2026-09-13", "16:25"),
    ("2026-09-13", "20:20"), ("2026-09-14", "20:15"), ("2026-09-10", "20:15"),
    ("2026-11-01", "13:00"), ("2026-11-01", "16:25"), ("2026-11-01", "20:20"),  # DST-change Sunday
    ("2026-12-06", "13:00"), ("2026-12-06", "16:05"), ("2026-12-06", "20:20"),
    ("2026-12-07", "20:15"), ("2026-12-03", "20:15"), ("2027-01-03", "13:00"),
    ("2026-10-19", "20:35"),  # late Monday window
]


def _slate():
    path = ROOT / "historical" / "lines_extra.parquet"
    if path.exists():
        s = pd.read_parquet(path)
        s = s[(s["season"] == 2026) & (s["game_type"] == "REG")]
        pairs = [(d, t or "13:00") for d, t in zip(s["gameday"], s["gametime"])]
        if pairs:
            return pairs
    return EMBEDDED_2026


def _crons():
    text = WORKFLOW.read_text()
    assert "timezone:" not in text, "on.schedule has no timezone field; crons must be UTC"
    block = text[text.index("  schedule:"):text.index("permissions:")]
    return re.findall(r'-\s*cron:\s*"([^"]+)"', block)


def _field_ok(spec: str, value: int) -> bool:
    if spec == "*":
        return True
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            if int(lo) <= value <= int(hi):
                return True
        elif int(part) == value:
            return True
    return False


def _cron_matches(cron: str, t_utc: dt.datetime) -> bool:
    minute, hour, dom, month, dow = cron.split()
    # cron: Sunday = 0; python: Monday = 0
    return (_field_ok(minute, t_utc.minute) and _field_ok(hour, t_utc.hour)
            and _field_ok(dom, t_utc.day) and _field_ok(month, t_utc.month)
            and _field_ok(dow, (t_utc.weekday() + 1) % 7))


def _job_for(cron: str) -> str:
    """Mirror the workflow's 'Select scheduled job' step."""
    text = WORKFLOW.read_text()
    sel = text[text.index("Select scheduled job"):text.index("Run weekly job")]
    for line in sel.splitlines():
        if f'"{cron}"' in line and "job=" not in line:
            nxt = sel.splitlines()[sel.splitlines().index(line) + 1]
            return re.search(r"job=(\w+)", nxt).group(1)
    return "t90"


def test_every_2026_kickoff_gets_a_t90_firing_inside_its_due_window():
    t90_crons = [c for c in _crons() if _job_for(c) == "t90"]
    assert t90_crons, "no T-90 crons in the workflow"
    missed = []
    for gameday, gametime in _slate():
        kickoff = dt.datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        slate = pd.DataFrame([{"game_id": f"{gameday}_{gametime}", "kickoff": kickoff}])
        hit = False
        t = kickoff - dt.timedelta(minutes=T90_DUE_MINUTES)
        while t < kickoff and not hit:
            if any(_cron_matches(c, t.astimezone(dt.timezone.utc)) for c in t90_crons):
                # the firing must also see the game as due (single source of truth)
                hit = not games_due_for_t90(slate, t).empty
            t += dt.timedelta(minutes=1)
        if not hit:
            missed.append(f"{gameday} {gametime} ET ({kickoff.strftime('%a')})")
    assert not missed, "no T-90 firing inside [T-90, T-0) for: " + ", ".join(missed)


def test_wed_and_tuesday_crons_map_to_their_jobs_and_t90_is_the_default():
    jobs = {c: _job_for(c) for c in _crons()}
    assert list(jobs.values()).count("wed") == 2      # EDT + EST variants
    assert list(jobs.values()).count("tuesday") == 2
    assert all(v == "t90" for c, v in jobs.items()
               if c not in [k for k, j in jobs.items() if j != "t90"])
    # a wed/tuesday cron must never fall in a game's due window
    for cron, job in jobs.items():
        if job == "t90":
            continue
        for gameday, gametime in _slate():
            kickoff = dt.datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
            t = kickoff - dt.timedelta(minutes=T90_DUE_MINUTES)
            while t < kickoff:
                assert not _cron_matches(cron, t.astimezone(dt.timezone.utc)), (cron, gameday, gametime)
                t += dt.timedelta(minutes=1)


def test_duplicate_t90_invocation_does_not_reprocess_or_resnap(monkeypatch, tmp_path):
    """A second firing inside the same window must skip games already
    processed AND must not spend odds credits on a second closing resnap."""
    import scripts.auto_weekly as aw
    from nflvalue import db as dbmod
    real_connect = dbmod.connect
    db_path = str(tmp_path / "t90.db")
    monkeypatch.setattr(dbmod, "connect", lambda p=None: real_connect(db_path))
    now = dt.datetime(2026, 9, 13, 11, 55, tzinfo=ET)
    slate = pd.DataFrame([{"game_id": "G1", "season": 2026, "week": 1,
                           "kickoff": now + dt.timedelta(minutes=65)}])
    monkeypatch.setattr(aw, "load_slate", lambda: slate)
    monkeypatch.setattr(aw, "now_et", lambda: now)
    monkeypatch.setattr(aw, "write_pipeline_heartbeat", lambda *a, **k: {})
    from nflvalue import config as cfgmod
    monkeypatch.setattr(cfgmod, "load_config", lambda: {"odds_api_key": "k", "discord_enabled": False})
    conn = real_connect(db_path)
    conn.execute("INSERT INTO lines VALUES ('t0','G1','b','receptions','p','P','over',4.5,1.9)")
    dbmod.upsert(conn, "leans", [{"season": 2026, "week": 1, "clock": "t90", "game_id": "G1",
                                  "player_id": "P", "market": "receptions", "status": "active"}],
                 ["season", "week", "clock", "game_id", "player_id", "market"])
    conn.commit()
    conn.close()
    resnaps, runs = [], []
    from nflvalue.sources import oddsapi_props as oap
    monkeypatch.setattr(oap, "resnap_lines", lambda cfg, emap, conn=None: resnaps.append(emap) or
                        {"pulled": list(emap), "rows_written": 0, "budget_remaining": 400})
    import pipeline_weekly as pwmod
    monkeypatch.setattr(pwmod, "build_event_map", lambda cfg, s: {g: f"e_{g}" for g in s.game_id})
    monkeypatch.setattr(pwmod, "run_t90", lambda *a, **k: runs.append(a) or {"voided": []})
    assert aw.job_t90() == 0
    assert runs == [] and resnaps == []
