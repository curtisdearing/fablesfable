"""Automatic ingest of website publication receipts into production state.

scripts/ingest_publication_receipts.sh is executed for real with a stubbed `gh` on PATH (no
network, no GitHub): trusted-run filtering, download, verification, idempotence, wrong-run and
failure handling. The workflow contract tests pin no-recursion, no model/odds/site work and the
shared state lock. Remote execution on GitHub is NOT exercised here."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue import issued_ledger as il  # noqa: E402
from scripts import publication_receipt as pr  # noqa: E402
from tests.test_issued_ledger import _db, _gen  # noqa: E402
from tests.test_publication_receipt import _live, _readback, _receipt_dir, _site  # noqa: E402

REPO = "curtisdearing/fablesfable"
WF = ROOT / ".github" / "workflows"
GH_STUB = r"""#!/bin/bash
# stub gh: serves $STUB/runs.json, $STUB/artifacts_<id> (a count) and $STUB/receipt_<id>/
echo "gh $*" >> "$STUB/calls.log"
if [[ "$1" == api && "$2" == */workflows/website.yml/runs* ]]; then cat "$STUB/runs.json"; exit 0; fi
if [[ "$1" == api && "$2" == */artifacts ]]; then id="${2%/artifacts}"; id="${id##*/}"; cat "$STUB/artifacts_$id" 2>/dev/null || echo 0; exit 0; fi
if [[ "$1" == run && "$2" == download ]]; then
  [[ -f "$STUB/fail_download" ]] && { echo "HTTP 500" >&2; exit 1; }
  id="$3"; dest="${@: -1}"; mkdir -p "$dest"; cp -R "$STUB/receipt_$id/." "$dest/"; exit 0
fi
echo "unexpected gh call: $*" >&2; exit 99
"""


def _run(**kw):
    r = {"id": 101, "repository": {"full_name": REPO}, "head_repository": {"full_name": REPO},
         "head_branch": "main", "path": ".github/workflows/website.yml", "event": "workflow_run",
         "status": "completed", "conclusion": "success"}
    r.update(kw)
    return r


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A production DB with one generated card, a verified receipt for website run 101, a stub gh."""
    monkeypatch.setattr("nflvalue.pick_cards.VALIDATED_MARKETS", frozenset({"pass_attempts"}))
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    conn.close()
    site = _site(tmp_path, cards)
    receipt, _ = pr.readback(site, "https://example.test", "101-1", attempts=1, sleep_s=0, fetch=_live(site),
                             clock=lambda: "2026-09-20T11:10:00Z", sleep=lambda s: None)
    stub = tmp_path / "stub"
    (stub / "bin").mkdir(parents=True)
    (stub / "bin" / "gh").write_text(GH_STUB)
    (stub / "bin" / "gh").chmod(0o755)
    src = _receipt_dir(tmp_path, site, receipt)
    os.rename(src, stub / "receipt_101")
    (stub / "artifacts_101").write_text("1\n")
    (stub / "runs.json").write_text(json.dumps({"workflow_runs": [
        _run(), _run(id=102, head_repository={"full_name": "someone/fork"}),
        _run(id=103, head_branch="feature"), _run(id=104, path=".github/workflows/other.yml"),
        _run(id=105, conclusion="failure"), _run(id=106, event="pull_request"),
        _run(id=107)]}))                                    # 107: trusted, no receipt (legacy/kept live)
    return {"tmp": tmp_path, "stub": stub, "db": tmp_path / "s.db"}


def _ingest(w, work="w", **env):
    e = {**os.environ, "PATH": f"{w['stub'] / 'bin'}:{os.environ['PATH']}", "STUB": str(w["stub"]),
         "REPO": REPO, "WORK": str(w["tmp"] / work), "DB": str(w["db"]), "PYTHON": sys.executable, **env}
    p = subprocess.run(["bash", str(ROOT / "scripts" / "ingest_publication_receipts.sh")], cwd=w["tmp"],
                       env=e, capture_output=True, text=True, timeout=120)
    summary = w["tmp"] / work / "summary.json"
    return p, (json.loads(summary.read_text()) if summary.exists() else None)


def _published(db):
    conn = dbmod.connect(str(db))
    try:
        return [e for r in il.load(conn) for e in r["events"] if e["stage"] == "published"]
    finally:
        conn.close()


def test_success_records_published_event_and_is_idempotent(world):
    p, s = _ingest(world, REQUIRE_RUN="101")
    assert p.returncode == 0, p.stderr
    assert s == {"written": 1, "recorded_runs": [101], "no_receipt": [107], "failed": []}
    ev = _published(world["db"])
    assert len(ev) == 1 and ev[0]["event_ts"] == "2026-09-20T11:10:00Z"
    assert ev[0]["recorded_at"] > "2026-09-23"                    # the real ingest clock, not backdated
    calls = (world["stub"] / "calls.log").read_text()
    assert "run download 101" in calls
    for untrusted in ("102", "103", "104", "105", "106"):         # never downloaded
        assert f"runs/{untrusted}/artifacts" not in calls and f"download {untrusted}" not in calls
    p2, s2 = _ingest(world, work="w2", REQUIRE_RUN="101")
    assert p2.returncode == 0 and s2["written"] == 0 and len(_published(world["db"])) == 1


def test_untrusted_triggering_run_is_refused_before_any_download(world):
    p, s = _ingest(world, REQUIRE_RUN="102")
    assert p.returncode == 7 and s is None
    assert "download" not in (world["stub"] / "calls.log").read_text()
    assert _published(world["db"]) == []


def test_receipt_for_another_run_fails_visibly_and_valid_ones_still_record(world):
    stub = world["stub"]
    (stub / "receipt_107").mkdir()
    for f in (stub / "receipt_101").iterdir():                    # run 107's artifact claims run 101
        (stub / "receipt_107" / f.name).write_bytes(f.read_bytes())
    (stub / "artifacts_107").write_text("1\n")
    p, s = _ingest(world)
    assert p.returncode == 5
    assert s["recorded_runs"] == [101] and s["failed"][0]["run_id"] == 107
    assert "is not the run '107'" in s["failed"][0]["reason"] and len(_published(world["db"])) == 1


def test_download_failure_fails_the_ingest_and_records_nothing(world):
    (world["stub"] / "fail_download").write_text("")
    p, s = _ingest(world)
    assert p.returncode != 0 and s is None and _published(world["db"]) == []


def test_missing_production_state_is_not_created_by_ingest(world):
    world["db"].unlink()
    p, s = _ingest(world, DB=str(world["tmp"] / "absent.db"))
    assert p.returncode == 1 and not (world["tmp"] / "absent.db").exists()


def test_trusted_run_filter_reasons():
    ok, rejected = pr.trusted_runs({"workflow_runs": [
        _run(), _run(id=2, repository={"full_name": "x/y"}), _run(id=3, status="in_progress"),
        _run(id=4, path=".github/workflows/website.yml@refs/heads/main")]}, REPO)
    assert ok == [4, 101] and dict(rejected) == {2: "run is not in this repository",
                                                 3: "not a successful completed run"}


# ---------------------------------------------------------------- workflow contracts --
def _text(name):
    return (WF / name).read_text()


def test_ingest_workflow_only_follows_website_and_holds_the_production_lock():
    wf = _text("publication-ingest.yml")
    on = wf[wf.index("\non:"):wf.index("\npermissions:")]
    assert 'workflows: ["Publish football website"]' in on and "types: [completed]" in on
    for trig in ("schedule", "push", "workflow_dispatch", "pull_request"):
        assert f"  {trig}:" not in on
    pre = wf[wf.index("  precheck:"):wf.index("  ingest:")]
    assert "concurrency" not in pre                               # outside the lock
    for guard in ("conclusion == 'success'", "head_branch == 'main'",
                  "head_repository.full_name == github.repository", "path == '.github/workflows/website.yml'"):
        assert guard in pre
    job = wf[wf.index("  ingest:"):]
    assert re.search(r"concurrency:\n      group: nfl-live-production\n      cancel-in-progress: false", job)
    assert "needs.precheck.outputs.proceed == 'true'" in job
    assert "scripts/ingest_publication_receipts.sh" in job and 'REQUIRE_RUN: ${{ github.event.workflow_run.id }}' in job
    assert "steps.ingest.outputs.written != '0'" in job and "steps.ingest.outputs.rc != '0'" in job
    assert "exit 1" in job.split("Fail visibly", 1)[1]
    assert "continue-on-error" not in wf and "always()" not in wf
    assert job.index("Restore checksummed production state") < job.index("Record verified website publications") \
        < job.index("Save production state")


def test_ingest_runs_no_model_odds_site_or_dispatch_and_cannot_loop():
    wf = _text("publication-ingest.yml")
    for forbidden in ("auto_weekly", "pipeline_weekly", "build_public_site", "prepare_pages", "deploy-pages",
                      "upload-pages-artifact", "ODDS_API_KEY", "secrets.", "workflow dispatch", "gh workflow run",
                      "repository_dispatch", "head_sha", "ref: ${{ github.event.workflow_run"):
        assert forbidden not in wf, forbidden
    site = _text("website.yml")
    assert re.findall(r"workflows: \[(.*?)\]", site) == ['"Live weekly model loop"']   # never this workflow
    assert "github.event.workflow_run.event != 'workflow_run'" in site
    live = _text("live-weekly.yml")
    assert "workflow_run" not in live.split("\npermissions:", 1)[0]   # model loop has no new trigger
    assert "'.github/workflows/publication-ingest.yml'" in live.split("paths-ignore:", 1)[1].split("workflow_dispatch", 1)[0]


def test_model_run_backfills_before_running_and_saves_with_its_state():
    live = _text("live-weekly.yml")
    i = live.index("Record verified website publications (backfill)")
    assert live.index("Restore checksummed production state") < i < live.index("Run weekly job") \
        < live.index("Publish successful production state")
    step = live[i:live.index("Select scheduled job")]
    assert "continue-on-error: true" in step and "::error::" in step
    assert "actions: read" in live.split("\njobs:", 1)[0]
