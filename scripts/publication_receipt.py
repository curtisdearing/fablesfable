"""Attest that a deployed Pages site serves exactly the publication artifact, then record it.

``readback`` (stdlib only; runs in website.yml after actions/deploy-pages):
    python3 scripts/publication_receipt.py readback --site published-site \
        --base-url "$PAGE_URL" --run-id "$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT" --out receipt/publication_receipt.json

    Fetches ``publication.json`` and ``api/hub.json`` from the LIVE site (cache-busted, retried
    while the CDN may still serve the previous deployment) and writes a receipt only when both
    served bodies are byte-identical (sha256) to the artifact that was deployed and the manifest
    lists that hub sha256. Exit 6 and no receipt otherwise: an upload or a deploy step reporting
    success is not a publication.

``record`` (runs where production state is held, serialized with it):
    python scripts/publication_receipt.py record --db data/nfl_props.db --dir <receipt artifact dir> \
        --expect-run <website run id>

    Verifies the receipt against the hub.json/publication.json saved beside it and against the run
    it was downloaded from, then appends ``published`` events to the issued-pick ledger. The
    event's own clock is the receipt's live-readback clock; the ledger ``recorded_at`` is the real
    clock of this write (never backdated), so a receipt recorded after kickoff grades as
    retrospective. Exit 5 on any mismatch; re-recording the same receipt is a no-op.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "fablesfable.publication_receipt.v1"
USER_AGENT = "fablesfable-publication-readback"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def readback(site: Path, base_url: str, run_id: str, attempts: int = 20, sleep_s: float = 30.0,
             fetch=_fetch, clock=_now, sleep=time.sleep):
    """(receipt, None) when the live site serves the deployed bytes, else (None, last reason)."""
    manifest_raw = (site / "publication.json").read_bytes()
    hub_raw = (site / "api" / "hub.json").read_bytes()
    manifest = json.loads(manifest_raw)
    want = {"publication.json": _sha(manifest_raw), "api/hub.json": _sha(hub_raw)}
    if (manifest.get("files") or {}).get("api/hub.json") != want["api/hub.json"]:
        return None, "artifact manifest does not list the artifact's hub.json sha256"
    base = base_url.rstrip("/") + "/"
    reason = "not attempted"
    for n in range(1, attempts + 1):
        try:
            got = {name: _sha(fetch(f"{base}{name}?readback={run_id}-{n}")) for name in want}
        except Exception as exc:  # noqa: BLE001 -- network/HTTP errors are retried, then refused
            reason = f"attempt {n}: fetch failed ({type(exc).__name__})"
        else:
            if got == want:
                hub = json.loads(hub_raw)
                return {"schema": SCHEMA, "kind": "pages_readback", "run_id": run_id, "page_url": base,
                        "verified_at": clock(), "attempts": n, "manifest_sha256": want["publication.json"],
                        "hub_sha256": want["api/hub.json"], "season": hub.get("season"), "week": hub.get("week"),
                        "label": manifest.get("label"), "published_at": manifest.get("published_at")}, None
            reason = f"attempt {n}: live bytes differ from the deployed artifact"
        if n < attempts:
            sleep(sleep_s)
    return None, reason


def record(db: str, receipt_dir: Path, expect_run: str, recorded_at=None) -> int:
    """Append ``published`` events for one verified receipt; returns new events. Raises ValueError."""
    sys.path.insert(0, str(ROOT))
    from nflvalue import db as dbmod
    from nflvalue import issued_ledger as il
    receipt = json.loads((receipt_dir / "publication_receipt.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("kind") != "pages_readback":
        raise ValueError("not a pages readback receipt")
    if str(receipt.get("run_id", "")).split("-")[0] != str(expect_run):
        raise ValueError(f"receipt run {receipt.get('run_id')!r} is not the run {expect_run!r} it came from")
    conn = dbmod.connect(os.path.abspath(db))
    try:
        return il.record_publication(conn, str(receipt_dir / "hub.json"), str(receipt_dir / "publication.json"),
                                     attestation=receipt, recorded_at=recorded_at)
    finally:
        conn.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("readback")
    r.add_argument("--site", required=True)
    r.add_argument("--base-url", required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--attempts", type=int, default=20)
    r.add_argument("--sleep", type=float, default=30.0)
    c = sub.add_parser("record")
    c.add_argument("--db", required=True)
    c.add_argument("--dir", required=True)
    c.add_argument("--expect-run", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "readback":
        receipt, why = readback(Path(a.site), a.base_url, a.run_id, attempts=a.attempts, sleep_s=a.sleep)
        if receipt is None:
            print(f"[readback] NOT verified: {why}; no publication receipt written")
            return 6
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
        print(f"[readback] live site serves the deployed artifact (attempt {receipt['attempts']}): {out}")
        return 0
    try:
        n = record(a.db, Path(a.dir), a.expect_run)
    except (ValueError, OSError) as exc:
        print(f"[record] publication NOT recorded: {exc}")
        return 5
    print(f"[record] published events written: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
