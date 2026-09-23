"""The live-weekly gate script, executed for real with stubbed gh/date/sleep.

One-time early Wednesday run (2026-09-23): admitted only 09:00-10:25Z, only by the run that
atomically creates tag early-wed-2026-09-23; the regular Wednesday schedule that day skips
once the tag exists.  Every other event maps exactly as before."""
import datetime as dt
import os
import re
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WF = (ROOT / ".github" / "workflows" / "live-weekly.yml").read_text()

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")


def _script() -> str:
    gate = WF[WF.index("\n  gate:\n"):WF.index("\n  run:\n")]
    body = gate[gate.index("run: |") + len("run: |"):]
    return textwrap.dedent(body)


def _stub(path: Path, text: str):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run(tmp_path, event, now, sched="", input_job="", tag=False):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    if tag:
        (state / "tag").write_text("x")
    _stub(bindir / "gh", f"""#!/usr/bin/env python3
import sys, os
a = sys.argv[1:]; tag = os.path.join({str(state)!r}, "tag")
open(os.path.join({str(state)!r}, "calls"), "a").write(" ".join(a) + "\\n")
if a[:1] == ["api"] and "-X" not in a:
    sys.exit(0 if os.path.exists(tag) else 1)
if a[:3] == ["api", "-X", "POST"]:
    if os.path.exists(tag): sys.exit(1)
    open(tag, "w").write("claimed"); sys.exit(0)
sys.exit(2)
""")
    _stub(bindir / "date", f"""#!/usr/bin/env python3
import sys, datetime as dt, os
f = os.path.join({str(state)!r}, "now")
now = dt.datetime.fromisoformat(open(f).read().strip())
a = sys.argv[1:]
if "-d" in a:
    t = dt.datetime.fromisoformat(a[a.index("-d") + 1].replace("Z", "+00:00")); print(int(t.timestamp()))
elif "+%s" in a: print(int(now.timestamp()))
elif "+%F" in a: print(now.strftime("%Y-%m-%d"))
else: print(now.strftime("%Y-%m-%dT%H:%M:%SZ"))
""")
    _stub(bindir / "sleep", f"""#!/usr/bin/env python3
import sys, datetime as dt, os
f = os.path.join({str(state)!r}, "now")
now = dt.datetime.fromisoformat(open(f).read().strip()) + dt.timedelta(seconds=int(sys.argv[1]))
open(f, "w").write(now.isoformat()); open(os.path.join({str(state)!r}, "slept"), "w").write(sys.argv[1])
""")
    (state / "now").write_text(now.isoformat())
    out = tmp_path / "out"
    out.write_text("")
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "GH_TOKEN": "x", "EVENT": event,
           "SCHED": sched, "INPUT_JOB": input_job, "REPO": "o/r", "SHA": "abc", "GITHUB_OUTPUT": str(out)}
    p = subprocess.run(["bash", "-c", _script()], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    job = re.findall(r"job=(\S+)", out.read_text())[-1]
    return job, (state / "tag").exists(), p.stdout


U = dt.timezone.utc
D = lambda h, m=0, day=23: dt.datetime(2026, 9, day, h, m, tzinfo=U)  # noqa: E731


def test_primary_dispatch_in_window_claims_and_runs_wed(tmp_path):
    job, tagged, _ = _run(tmp_path, "workflow_dispatch", D(9, 3), input_job="wed-early")
    assert (job, tagged) == ("wed", True)


def test_second_trigger_after_claim_skips(tmp_path):
    job, _, out = _run(tmp_path, "schedule", D(9, 40), sched="35 9 23 9 *", tag=True)
    assert job == "skip" and "already claimed" in out


def test_regular_wednesday_run_skips_after_early_claim_but_runs_if_none(tmp_path):
    assert _run(tmp_path / "a", "schedule", D(18, 0), sched="17 14 * 9,10 3", tag=True)[0] == "skip"
    (tmp_path / "b").mkdir()
    assert _run(tmp_path / "b", "schedule", D(18, 0), sched="17 14 * 9,10 3")[0] == "wed"
    (tmp_path / "c").mkdir()   # another Wednesday: unaffected even if the tag exists
    assert _run(tmp_path / "c", "schedule", D(18, 0, day=30), sched="17 14 * 9,10 3", tag=True)[0] == "wed"


@pytest.fixture(autouse=True)
def _mk(tmp_path):
    for d in ("a",):
        (tmp_path / d).mkdir(exist_ok=True)


def test_early_backup_cron_waits_until_0900_then_claims(tmp_path):
    job, tagged, out = _run(tmp_path, "schedule", D(8, 30), sched="5 6 23 9 *")
    assert (job, tagged) == ("wed", True) and (tmp_path / "state" / "slept").read_text() == "1800"


def test_too_early_too_late_and_other_dates_skip_without_claiming(tmp_path):
    for i, (now, sched) in enumerate([(D(7, 0), "5 5 23 9 *"), (D(10, 30), "35 9 23 9 *"),
                                      (dt.datetime(2027, 9, 23, 9, 10, tzinfo=U), "5 9 23 9 *")]):
        d = tmp_path / f"s{i}"
        d.mkdir()
        job, tagged, _ = _run(d, "schedule", now, sched=sched)
        assert (job, tagged) == ("skip", False), now


def test_normal_events_map_as_before(tmp_path):
    cases = [("push", "", "", "deploy"), ("workflow_dispatch", "", "t90", "t90"),
             ("schedule", "23 14 * 9,10 2", "", "tuesday"), ("schedule", "15 23 * 9,10 0,1,3,4", "", "t90"),
             ("schedule", "17 15 * 11,12,1 3", "", "wed")]
    for i, (ev, sched, inp, want) in enumerate(cases):
        d = tmp_path / f"n{i}"
        d.mkdir()
        assert _run(d, ev, D(12, 0), sched=sched, input_job=inp)[0] == want, (ev, sched)
