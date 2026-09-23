"""What produced a stored lean: run id, code commit, forecast version, ranker artifact hash.

Recorded at write time from explicit sources, never inferred later from
timestamps. Outside GitHub Actions the run id is ``local:<pid>`` and the code
commit comes from ``git rev-parse HEAD`` (``unknown`` if unavailable).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _code_sha() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def file_sha256(path: str):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_provenance() -> Dict[str, str]:
    from .football_forecast import FORECAST_VERSION
    run = os.environ.get("GITHUB_RUN_ID")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    return {
        "run_id": f"gha:{run}-{attempt}" if run else os.environ.get("FF_RUN_ID", f"local:{os.getpid()}"),
        "code_sha": os.environ.get("FF_CODE_SHA") or _code_sha(),
        "forecast_version": FORECAST_VERSION,
        "ranker_sha256": file_sha256(os.path.join(ROOT, "data", "ml_ranker.joblib")),
    }
