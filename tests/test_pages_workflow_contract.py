"""Contract tests for the GitHub Pages packaging slice of the live loop.

Two contracts live here, and they are deliberately in one file because they
only mean anything together:

1. **The workflow contract** (``.github/workflows/live-weekly.yml``) — which
   steps run, under which conditions, in which order.  These are static text
   assertions on purpose: the workflow is data that GitHub executes, so the
   only honest test is one that reads the same bytes GitHub will read.  No
   PyYAML — CI installs ``requirements.txt`` + ``pytest`` and nothing else,
   and a contract test that quietly ``importorskip``s in CI is not a test.

2. **The build contract** (``scripts/prepare_pages.py``) — what actually
   lands in ``_site``.  Exercised against a temporary fixture tree, fully
   offline, so the guarantees below hold without a network, a database, or a
   model.

The guarantee this file exists to protect: **a missing or stale weekly report
can never be published as the current one.**  ``drops/props_week_2023_9.html``
is committed to this repository as a sample; any "publish the newest file in
drops/" shortcut would ship a 2023 Week 9 document as this week's card.  The
season and week therefore come from the generated payload, and the resolved
document has to agree with it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "live-weekly.yml"
PREPARE_PAGES = ROOT / "scripts" / "prepare_pages.py"
HUB_FEED = ROOT / "scripts" / "build_hub_feed.py"

MODEL_JOBS = ("wed", "t90", "tuesday")


# =========================================================================== #
# A minimal, explicit reader for the workflow's step blocks.
#
# Steps are the 6-space-indented `- ` items under a job's `steps:`.  A block
# continues through blank lines and anything indented 8+.  That is the whole
# grammar this file needs, and being narrow keeps it honest: every assertion
# below is anchored to a step that must be *found by name*, and
# `test_the_step_reader_is_not_vacuous` fails if the reader stops finding
# them — so a malformed workflow can never turn these tests green by making
# the searches match nothing.
# =========================================================================== #
def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _step_blocks(text: str) -> list[str]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("      - "):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is not None:
            if line.strip() == "" or line.startswith("        "):
                current.append(line)
            else:
                blocks.append(current)
                current = None
    if current is not None:
        blocks.append(current)
    return ["\n".join(block) for block in blocks]


def _step(name: str) -> str:
    blocks = _step_blocks(_workflow_text())
    for block in blocks:
        if re.search(rf"^      - name: {re.escape(name)}\s*$", block, re.M):
            return block
    raise AssertionError(
        f"workflow step {name!r} not found; steps present: "
        + ", ".join(sorted(
            m.group(1) for b in blocks
            for m in [re.search(r"^      - name: (.+)$", b, re.M)] if m)))


def _if_condition(block: str) -> str:
    match = re.search(r"^        if:\s*(.+)$", block, re.M)
    return "" if match is None else match.group(1).strip()


# =========================================================================== #
# 1. The workflow contract
# =========================================================================== #
def test_the_step_reader_is_not_vacuous():
    """Guards every other assertion in this section.  If the reader stops
    resolving steps, the searches below would pass by matching nothing."""
    blocks = _step_blocks(_workflow_text())
    assert len(blocks) >= 12, f"only {len(blocks)} step blocks parsed out of the workflow"
    for name in ("Restore rebuildable seasonal feeds",
                 "Restore checksummed production state",
                 "Select scheduled job",
                 "Run weekly job",
                 "Publish successful production state",
                 "Save rebuildable seasonal feeds",
                 "Upload run evidence",
                 "Prepare dashboard for GitHub Pages"):
        assert _step(name)


@pytest.mark.parametrize("job", MODEL_JOBS)
def test_seasonal_feeds_are_saved_after_every_successful_model_job(job):
    """Only the Tuesday job used to save the rolling feed cache, so a
    Wednesday or T-90 refetch was thrown away and refetched next run.  Every
    job that actually rebuilds feeds must persist them."""
    condition = _if_condition(_step("Save rebuildable seasonal feeds"))
    assert condition, "the seasonal-feed save step lost its condition entirely"
    assert f'"{job}"' in condition or f"'{job}'" in condition, (
        f"the {job!r} job does not save the rebuildable seasonal feeds "
        f"(condition: {condition})")


def test_deploy_only_runs_never_overwrite_the_seasonal_feed_cache():
    """A push-triggered deploy runs no model and fetches no feeds.  Letting
    it save would replace a populated cache with an empty one."""
    condition = _if_condition(_step("Save rebuildable seasonal feeds"))
    assert "deploy" not in condition, (
        f"deploy-only runs would mutate the seasonal feed cache (condition: {condition})")
    assert "steps.select.outputs.job" in condition, (
        "the seasonal-feed save is no longer keyed to the selected job")


def test_seasonal_feed_save_is_still_success_only():
    """`always()` here would let a half-finished ingest become the cache the
    next run restores."""
    condition = _if_condition(_step("Save rebuildable seasonal feeds"))
    assert "always()" not in condition and "failure()" not in condition, (
        f"the seasonal-feed save can run after a failed step (condition: {condition})")


def test_state_publication_is_still_success_only():
    """No `if:` at all is the point: a failed earlier step must skip this."""
    block = _step("Publish successful production state")
    assert _if_condition(block) == "", (
        "the state publication step gained a condition; it must inherit "
        "step-level success-only semantics")


def test_state_publication_keeps_pointer_last_semantics():
    """Upload the versioned archive first, edit the release body last: a
    failed upload then leaves the previous pointer restorable."""
    block = _step("Publish successful production state")
    upload = block.index("gh release upload")
    pointer = block.index("state_store.py pointer")
    notes = block.index("gh release edit")
    assert upload < pointer < notes, (
        "the state pointer is no longer written last; a failed upload could "
        "strand the release body pointing at a missing asset")


def test_restore_still_verifies_the_state_checksum():
    block = _step("Restore checksummed production state")
    assert "--sha" in block and "sha256" in block
    assert "Invalid model-state release pointer" in block


def test_production_concurrency_group_is_unchanged():
    text = _workflow_text()
    assert "group: nfl-live-production" in text
    assert "cancel-in-progress: false" in text


def test_pages_are_built_by_the_testable_script():
    """Complex inline shell in a workflow is untestable by construction.  The
    build has to live in a script this suite can run offline."""
    block = _step("Prepare dashboard for GitHub Pages")
    assert "scripts/prepare_pages.py" in block, (
        "the Pages build is not routed through scripts/prepare_pages.py")
    assert PREPARE_PAGES.exists(), "scripts/prepare_pages.py is missing"
    assert "cp dashboard.html" not in block, (
        "the Pages build still copies the dashboard with inline shell")


def test_pages_artifact_is_still_the_site_directory():
    text = _workflow_text()
    assert "actions/upload-pages-artifact@v4" in text
    assert re.search(r"^          path: _site\s*$", text, re.M), (
        "the Pages artifact no longer uploads _site")


def test_run_evidence_includes_the_weekly_reports_and_drops():
    block = _step("Upload run evidence")
    assert "reports/" in block, "run evidence dropped the weekly reports"
    assert "drops/" in block, "run evidence does not include the weekly HTML drops"


def test_run_evidence_never_ships_secrets():
    """Evidence artifacts are downloadable by anyone with repo read access."""
    block = _step("Upload run evidence")
    paths = re.search(r"^          path: \|\n((?:            .+\n?)+)", block, re.M)
    assert paths, f"could not read the evidence path list from:\n{block}"
    listed = [line.strip() for line in paths.group(1).splitlines() if line.strip()]
    assert listed, "the evidence path list is empty"
    for entry in listed:
        assert "config.local.json" not in entry, f"evidence uploads secrets: {entry}"
        assert not entry.startswith("."), f"evidence uploads a dotfile path: {entry}"
        assert entry not in ("./", "/", "."), f"evidence uploads the whole workspace: {entry}"
        assert ".env" not in entry, f"evidence uploads an env file: {entry}"
    joined = "\n".join(listed)
    assert "secrets" not in joined.lower()


# =========================================================================== #
# 2. The build contract — scripts/prepare_pages.py against a fixture tree
# =========================================================================== #
STALE_MARKER = "STALE-2023-WEEK-9-SAMPLE"


def _drop_html(season: int, week: int, marker: str = "") -> str:
    """Shaped like nflvalue.document.render_drop: the season/week identity a
    reader sees is in the title and the h1."""
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>NFL Prop Leans — {season} Week {week}</title></head><body>"
            f"<h1>NFL Prop Leans — {season} Week {week}</h1>"
            f"<p>{marker}</p></body></html>")


def _tree(tmp_path: Path, *, season=2031, week=5, clock="wed", as_of="2031-09-10T12:00:00Z",
          drops=None, write_payload=True, dashboard=True, hub_feed=True) -> Path:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "drops").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    if dashboard:
        (root / "dashboard.html").write_text("<html><body>DASHBOARD</body></html>")
    if hub_feed:
        shutil.copy(HUB_FEED, root / "scripts" / "build_hub_feed.py")
    if write_payload:
        payload = {"season": season, "week": week, "clock": clock, "as_of": as_of,
                   "publish": True, "mode": "live", "games": []}
        (root / "data" / "weekly_props.json").write_text(json.dumps(payload))
    for name, html in (drops or {}).items():
        (root / "drops" / name).write_text(html)
    # A secret the packaging step must never carry into the published site.
    (root / "config.local.json").write_text(json.dumps({"odds_api_key": "SECRET-KEY-XYZ"}))
    return root


def _run(root: Path, *args, now="2031-09-11T00:00:00Z"):
    cmd = [sys.executable, str(PREPARE_PAGES), "--root", str(root)]
    if now is not None:
        cmd += ["--now", now]
    cmd += list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def _site_files(root: Path) -> set[str]:
    site = root / "_site"
    return {str(p.relative_to(site)) for p in site.rglob("*") if p.is_file()}


def test_prepare_pages_script_exists():
    assert PREPARE_PAGES.exists(), "scripts/prepare_pages.py has not been written"


def test_fixture_build_produces_index_hub_feed_and_both_report_paths(tmp_path):
    """The definition of done, executed offline."""
    root = _tree(tmp_path, drops={"props_week_2031_5.html": _drop_html(2031, 5, "CURRENT")})
    proc = _run(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    files = _site_files(root)
    for expected in ("index.html", os.path.join("api", "hub.json"),
                     os.path.join("reports", "latest.html"),
                     os.path.join("reports", "2031", "week-5.html")):
        assert expected in files, f"{expected} missing from _site (got {sorted(files)})"
    latest = (root / "_site" / "reports" / "latest.html").read_text()
    versioned = (root / "_site" / "reports" / "2031" / "week-5.html").read_text()
    assert "CURRENT" in latest and latest == versioned
    assert "DASHBOARD" in (root / "_site" / "index.html").read_text()
    feed = json.loads((root / "_site" / "api" / "hub.json").read_text())
    assert feed["project"] == "fablesfable"


def test_a_missing_report_never_republishes_a_previous_week(tmp_path):
    """The whole point.  A 2023 sample drop sits in the tree; the payload asks
    for 2031 week 5, whose document was never written."""
    root = _tree(tmp_path, drops={"props_week_2023_9.html": _drop_html(2023, 9, STALE_MARKER)})
    proc = _run(root)
    latest_path = root / "_site" / "reports" / "latest.html"
    assert latest_path.exists(), "a missing report must still degrade visibly, not vanish"
    latest = latest_path.read_text()
    assert STALE_MARKER not in latest, "a previous week was published as the current report"
    assert "2023 Week 9" not in latest
    assert not (root / "_site" / "reports" / "2023").exists(), (
        "a stale week was published to the versioned path")
    assert not (root / "_site" / "reports" / "2031").exists()
    assert re.search(r"no current", latest, re.I), (
        "the degraded page does not say plainly that no current report exists")
    assert "[pages]" in proc.stdout and re.search(
        r"missing|not published|no current", proc.stdout, re.I), (
        f"the missing report was not logged clearly: {proc.stdout}")


def test_a_stale_latest_html_from_a_previous_build_is_overwritten(tmp_path):
    """Rebuilding into a dirty _site must not leave last week's card standing."""
    root = _tree(tmp_path, drops={})
    site_reports = root / "_site" / "reports"
    site_reports.mkdir(parents=True)
    (site_reports / "latest.html").write_text(_drop_html(2023, 9, STALE_MARKER))
    _run(root)
    assert STALE_MARKER not in (site_reports / "latest.html").read_text(), (
        "a previous build's report survived as the current one")


def test_a_stale_payload_is_not_published_as_current(tmp_path):
    """The document exists and matches the payload, but the payload is from
    a month ago — a deploy-only run must not present it as this week."""
    root = _tree(tmp_path, as_of="2031-08-01T12:00:00Z",
                 drops={"props_week_2031_5.html": _drop_html(2031, 5, "OLD-BUT-MATCHING")})
    proc = _run(root, "--max-age-hours", "240")
    latest = (root / "_site" / "reports" / "latest.html").read_text()
    assert "OLD-BUT-MATCHING" not in latest
    assert not (root / "_site" / "reports" / "2031").exists()
    assert re.search(r"stale", proc.stdout, re.I), proc.stdout


def test_season_and_week_come_from_the_payload_not_the_newest_file(tmp_path):
    """`ls -t drops/ | head -1` is the shortcut this forbids."""
    root = _tree(tmp_path, drops={
        "props_week_2031_5.html": _drop_html(2031, 5, "PAYLOAD-WEEK"),
        "props_week_2099_1.html": _drop_html(2099, 1, "NEWEST-FILE"),
    })
    newest = root / "drops" / "props_week_2099_1.html"
    os.utime(newest, (2 ** 31 - 1, 2 ** 31 - 1))
    proc = _run(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    latest = (root / "_site" / "reports" / "latest.html").read_text()
    assert "PAYLOAD-WEEK" in latest and "NEWEST-FILE" not in latest
    assert (root / "_site" / "reports" / "2031" / "week-5.html").exists()
    assert not (root / "_site" / "reports" / "2099").exists()


def test_a_document_that_disagrees_with_the_payload_is_refused(tmp_path):
    """Filename alone is not identity: the rendered document has to name the
    same season and week the payload does."""
    root = _tree(tmp_path, drops={"props_week_2031_5.html": _drop_html(2030, 17, "WRONG-WEEK")})
    proc = _run(root)
    latest = (root / "_site" / "reports" / "latest.html").read_text()
    assert "WRONG-WEEK" not in latest
    assert not (root / "_site" / "reports" / "2031").exists()
    assert re.search(r"mismatch|disagree", proc.stdout, re.I), proc.stdout


def test_a_t90_payload_resolves_the_t90_document(tmp_path):
    root = _tree(tmp_path, clock="t90", drops={
        "props_week_2031_5.html": _drop_html(2031, 5, "WEDNESDAY"),
        "props_week_2031_5_t90.html": _drop_html(2031, 5, "T90-REFRESH"),
    })
    proc = _run(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "T90-REFRESH" in (root / "_site" / "reports" / "latest.html").read_text()


def test_a_missing_payload_degrades_instead_of_guessing(tmp_path):
    root = _tree(tmp_path, write_payload=False,
                 drops={"props_week_2023_9.html": _drop_html(2023, 9, STALE_MARKER)})
    proc = _run(root)
    latest = (root / "_site" / "reports" / "latest.html").read_text()
    assert STALE_MARKER not in latest
    assert (root / "_site" / "index.html").exists(), (
        "a missing report must not stop the dashboard from deploying")
    assert re.search(r"payload", proc.stdout, re.I), proc.stdout


def test_the_published_site_never_contains_secrets_or_config(tmp_path):
    root = _tree(tmp_path, drops={"props_week_2031_5.html": _drop_html(2031, 5, "CURRENT")})
    proc = _run(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    names = _site_files(root)
    assert "index.html" in names, "nothing was built, so this test would pass vacuously"
    for name in names:
        assert "config.local" not in name, f"_site contains {name}"
    for path in (root / "_site").rglob("*"):
        if path.is_file():
            assert "SECRET-KEY-XYZ" not in path.read_text(errors="ignore"), (
                f"a secret leaked into {path}")


def test_a_missing_dashboard_fails_the_build(tmp_path):
    """index.html is the product.  Silently deploying an empty site would be
    worse than a red run."""
    root = _tree(tmp_path, dashboard=False,
                 drops={"props_week_2031_5.html": _drop_html(2031, 5, "CURRENT")})
    proc = _run(root)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert re.search(r"dashboard", proc.stdout + proc.stderr, re.I)


def test_a_failing_hub_feed_fails_the_build(tmp_path):
    """Matches today's semantics: the feed builder is a plain `run:` step, so
    a crash there is a red run rather than a quietly incomplete deploy."""
    root = _tree(tmp_path, hub_feed=False,
                 drops={"props_week_2031_5.html": _drop_html(2031, 5, "CURRENT")})
    (root / "scripts" / "build_hub_feed.py").write_text(
        "import sys\nsys.stderr.write('hub feed exploded\\n')\nraise SystemExit(3)\n")
    proc = _run(root)
    assert proc.returncode != 0, proc.stdout + proc.stderr


def test_the_build_records_a_machine_readable_publication_decision(tmp_path):
    root = _tree(tmp_path, drops={"props_week_2031_5.html": _drop_html(2031, 5, "CURRENT")})
    _run(root)
    manifest = json.loads((root / "_site" / "reports" / "index.json").read_text())
    assert manifest["published"] is True
    assert manifest["season"] == 2031 and manifest["week"] == 5
    assert manifest["paths"]["versioned"] == "reports/2031/week-5.html"

    stale = _tree(tmp_path / "second", drops={})
    _run(stale)
    degraded = json.loads((stale / "_site" / "reports" / "index.json").read_text())
    assert degraded["published"] is False
    assert degraded["reason"]


def test_strict_report_mode_turns_a_missing_report_red(tmp_path):
    """Opt-in for a run that should not deploy without this week's card."""
    root = _tree(tmp_path, drops={})
    assert _run(root).returncode == 0
    assert _run(root, "--strict-report").returncode != 0


def test_prepare_pages_defaults_to_the_repository_root():
    """No --root means the real tree, so the workflow can call it bare."""
    source = PREPARE_PAGES.read_text(encoding="utf-8")
    assert "parents[1]" in source, (
        "prepare_pages does not derive the repository root from its own location")


def test_the_committed_sample_drop_is_still_the_trap_these_tests_assume():
    """If this sample is ever deleted, the regression these tests guard
    against stops being reachable from the repository itself — and this test
    is the reminder to re-point the guard rather than to relax it."""
    assert (ROOT / "drops" / "props_week_2023_9.html").exists(), (
        "the committed 2023 sample drop is gone; confirm no 'newest file in "
        "drops/' shortcut has been introduced anywhere in the Pages build")


def test_the_degraded_notice_does_not_leak_runner_filesystem_paths(tmp_path):
    """The notice is a public page.  Naming the missing document is useful;
    printing the runner's absolute layout to the internet is not."""
    root = _tree(tmp_path, drops={})
    _run(root)
    latest = (root / "_site" / "reports" / "latest.html").read_text()
    assert str(root) not in latest, "the notice page published an absolute path"
    assert "props_week_2031_5.html" in latest, (
        "the notice should still name the document it expected")
