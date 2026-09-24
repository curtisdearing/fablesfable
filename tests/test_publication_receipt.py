"""A `published` ledger event exists only after the LIVE Pages site was read back byte-identical
to the deployed artifact (scripts/publication_receipt.py, run by website.yml after deploy-pages).

Cards come from unit-test lean rows (not picks anyone was given); the box is the real ESPN final
fixture used by tests/test_issued_ledger.py. No network: the live site is a fake fetcher."""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue import issued_grading as ig  # noqa: E402
from nflvalue import issued_ledger as il  # noqa: E402
from nflvalue import pick_cards as pc  # noqa: E402
from scripts import publication_receipt as pr  # noqa: E402
from scripts import record_issued_pick as rip  # noqa: E402
from tests.test_issued_ledger import BOX_CAPTURED, BOX_FILE, T, Z, _db, _gen  # noqa: E402


@pytest.fixture
def actionable(monkeypatch):
    monkeypatch.setattr(pc, "VALIDATED_MARKETS", frozenset({"pass_attempts"}))


def _site(tmp_path, cards, published="11:05"):
    """The artifact build_public_site writes: api/hub.json + a manifest listing its sha256."""
    site = tmp_path / "published-site"
    (site / "api").mkdir(parents=True)
    raw = json.dumps({"season": 2026, "week": 2, "label": "fresh", "generated_at": Z(published),
                      "cards": json.loads(json.dumps(cards, default=str))}, indent=2).encode()
    (site / "api" / "hub.json").write_bytes(raw)
    (site / "publication.json").write_text(json.dumps(
        {"season": 2026, "week": 2, "label": "fresh", "published_at": Z(published),
         "files": {"api/hub.json": hashlib.sha256(raw).hexdigest()}}))
    return site


def _live(site, stale=0, broken=0):
    """Fake live site: `broken` failing fetches, then `stale` previous-deployment bodies, then the artifact."""
    def fetch(url):
        n = int(url.rsplit("-", 1)[1])                               # attempt number (cache-buster)
        body = (site / re.sub(r"^https://example\.test/|\?.*$", "", url)).read_bytes()
        if n <= broken:
            raise OSError("HTTP 503")
        return body + b" " if n <= broken + stale else body
    return fetch


def _readback(site, fetch, attempts=5):
    return pr.readback(site, "https://example.test", "777-1", attempts=attempts, sleep_s=0,
                       fetch=fetch, clock=lambda: Z("11:10"), sleep=lambda s: None)


def _receipt_dir(tmp_path, site, receipt):
    d = tmp_path / "receipt"
    d.mkdir()
    (d / "publication_receipt.json").write_text(json.dumps(receipt))
    shutil.copy(site / "publication.json", d / "publication.json")
    shutil.copy(site / "api" / "hub.json", d / "hub.json")
    return d


def test_verified_readback_is_recorded_as_published_and_grades_as_given(tmp_path, actionable):
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    site = _site(tmp_path, cards)
    receipt, why = _readback(site, _live(site, stale=2, broken=1))
    assert why is None and receipt["attempts"] == 4 and receipt["verified_at"] == Z("11:10")
    conn.close()
    d = _receipt_dir(tmp_path, site, receipt)
    assert pr.record(str(tmp_path / "s.db"), d, "777", recorded_at=Z("11:12")) == 1
    assert pr.record(str(tmp_path / "s.db"), d, "777", recorded_at=Z("11:20")) == 0     # idempotent
    conn = dbmod.connect(str(tmp_path / "s.db"))
    rec = il.load(conn)[0]
    pub = [e for e in rec["events"] if e["stage"] == "published"]
    assert len(il.load(conn)) == 1 and len(pub) == 1                # same record the run generated
    assert pub[0]["event_ts"] == Z("11:10") and pub[0]["recorded_at"] == Z("11:12")
    ev = json.loads(pub[0]["evidence_json"])
    assert ev["deploy_run_id"] == "777-1" and ev["clock_basis"] == "live pages readback verified_at"
    res = ig.grade(il.load(conn), ig.load_boxes([str(BOX_FILE)], BOX_CAPTURED))
    given = res["sections"]["recommendations_given"]["rows"]
    assert len(given) == 1 and given[0]["evidence_stage"] == "published" and given[0]["settlement"] == "win"


def test_live_site_never_matching_writes_no_receipt(tmp_path):
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    site = _site(tmp_path, cards)
    receipt, why = _readback(site, _live(site, stale=99))
    assert receipt is None and "differ from the deployed artifact" in why
    receipt, why = _readback(site, _live(site, broken=99))
    assert receipt is None and "fetch failed" in why
    out = tmp_path / "r" / "publication_receipt.json"
    rc = pr.main(["readback", "--site", str(site), "--base-url", "http://127.0.0.1:9", "--run-id", "1-1",
                  "--out", str(out), "--attempts", "1", "--sleep", "0"])
    assert rc == 6 and not out.exists()


def test_artifact_whose_manifest_disagrees_with_its_hub_is_refused(tmp_path):
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    site = _site(tmp_path, cards)
    (site / "api" / "hub.json").write_text("{}")
    assert _readback(site, _live(site))[0] is None


def test_mismatched_or_foreign_receipts_record_nothing(tmp_path, actionable):
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    site = _site(tmp_path, cards)
    receipt, _ = _readback(site, _live(site))
    conn.close()
    db = str(tmp_path / "s.db")
    d = _receipt_dir(tmp_path, site, receipt)
    assert pr.main(["record", "--db", db, "--dir", str(d), "--expect-run", "888"]) == 5   # other run's receipt
    (d / "publication_receipt.json").write_text(json.dumps({**receipt, "hub_sha256": "0" * 64}))
    assert pr.main(["record", "--db", db, "--dir", str(d), "--expect-run", "777"]) == 5   # different bytes
    (d / "publication_receipt.json").write_text(json.dumps({**receipt, "verified_at": "2026-09-20 11:10"}))
    assert pr.main(["record", "--db", db, "--dir", str(d), "--expect-run", "777"]) == 5   # unzoned clock
    (d / "publication_receipt.json").write_text(json.dumps(receipt))
    (d / "hub.json").write_text((d / "hub.json").read_text() + " ")                      # page not as deployed
    assert pr.main(["record", "--db", db, "--dir", str(d), "--expect-run", "777"]) == 5
    conn = dbmod.connect(db)
    with pytest.raises(ValueError, match="attestation"):
        il.record_publication(conn, str(site / "api" / "hub.json"), str(site / "publication.json"), None)
    assert {e["stage"] for r in il.load(conn) for e in r["events"]} == {"generated"}
    with pytest.raises(SystemExit):                                  # the manual CLI needs a receipt too
        rip.main(["--db", db, "published", "--hub", str(site / "api/hub.json"),
                  "--publication", str(site / "publication.json")])


def test_receipt_recorded_after_kickoff_is_retrospective_not_backdated(tmp_path, actionable):
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    site = _site(tmp_path, cards)
    receipt, _ = _readback(site, _live(site))
    conn.close()
    d = _receipt_dir(tmp_path, site, receipt)
    pr.record(str(tmp_path / "s.db"), d, "777", recorded_at="2026-09-20T21:00:00Z")   # kickoff 20:25Z
    conn = dbmod.connect(str(tmp_path / "s.db"))
    res = ig.grade(il.load(conn), ig.load_boxes([str(BOX_FILE)], BOX_CAPTURED))
    assert res["counts"]["recommendations_given"] == 0 and res["counts"]["generated_not_shown"] == 1


def test_website_workflow_reads_back_after_deploy_and_stays_out_of_model_state():
    wf = (ROOT / ".github/workflows/website.yml").read_text()
    deploy = wf.index("uses: actions/deploy-pages@v4")
    verify = wf.index("scripts/publication_receipt.py readback")
    keep = wf.index("name: publication-receipt")
    assert deploy < verify < keep
    step = wf[wf.index("- name: Verify the live site serves the deployed artifact"):keep]
    assert "steps.deployment.outputs.page_url" in step and "--site published-site" in step
    assert "cp published-site/api/hub.json" in step and "always()" not in wf[deploy:]
    assert "continue-on-error" not in wf[deploy:] and "if-no-files-found: error" in wf[keep:]
    assert "contents: read" in wf and "contents: write" not in wf
    for forbidden in ("state_store.py", "model-state", "nfl_props.db"):
        assert forbidden not in wf
    body = (ROOT / "scripts/publication_receipt.py").read_text()
    imports = {m.group(1) for m in re.finditer(r"^(?:import|from) (\w+)", body, re.M)}
    assert "nflvalue" not in imports                                 # readback is stdlib-only on the runner
    assert "@" not in pr.USER_AGENT
