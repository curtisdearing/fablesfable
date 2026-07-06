"""Tiny JSON-over-HTTP helper (standard library only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional


class HttpJsonError(RuntimeError):
    """Raised when a request cannot be completed / parsed after retries.

    A plain ``RuntimeError`` subclass so existing callers (espn, oddsapi,
    weather, sleeper) that already catch broad ``Exception`` keep degrading
    cleanly -- but callers that want to distinguish network failure from a
    programming bug can catch this specifically.
    """


def get_json(url: str, params: Optional[Dict] = None, timeout: float = 15.0,
             retries: int = 3, backoff: float = 0.5):
    """Fetch ``url`` (with optional query ``params``) and JSON-decode the body.

    BOUNDED retry-with-backoff: up to ``retries`` attempts, sleeping
    ``backoff * attempt`` seconds between them. Transient blips (URLError,
    timeout) and a non-JSON body (JSONDecodeError) are caught and retried; on
    final failure a typed :class:`HttpJsonError` is raised so callers can catch
    it. Timeouts are preserved. The success path is byte-for-byte unchanged.
    """
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-value/1.0"})
    attempts = max(1, retries)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(backoff * attempt)
    raise HttpJsonError(f"get_json failed after {attempts} attempts: {last_exc}") from last_exc
