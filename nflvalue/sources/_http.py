"""Tiny JSON-over-HTTP helper (standard library only).

User-Agent is load-bearing, not cosmetic.  ESPN's site/core APIs sit behind
Akamai, and on 2026-09-02 the first live-odds Wednesday run of the season
had every ``site.api.espn.com`` call answered ``403 Forbidden`` under the
original ``nfl-value/1.0`` string -- injuries (load-bearing, so the freshness
gate set ``publish=False`` on the whole board) and news alike.  Measured the
same day from two networks (the owner's Mac and the Cowork VM), across
``/injuries``, ``/news``, ``/scoreboard``, ``/summary`` and a core-API
event roster:

    nfl-value/1.0                                              403 403 403 403 200
    nfl-value/1.0 (+https://github.com/curtisdearing/fablesfable)  200 x5
    Python-urllib/3.11                                         200 x5
    Mozilla/5.0 ... Chrome/128 (a browser string from non-browser TLS)  403

The edge accepts a self-identifying agent that names where it comes from,
and rejects a short opaque token.  So the default names the repository, and
``NFLVALUE_HTTP_USER_AGENT`` lets an operator change it from the workflow
environment without a code change if the edge rules move again.  A failure
here is never silent: the caller's freshness gate reports the missing feed.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Dict, Optional

DEFAULT_USER_AGENT = "nfl-value/1.0 (+https://github.com/curtisdearing/fablesfable)"


def user_agent() -> str:
    """The agent string every feed request carries (env override wins)."""
    return (os.environ.get("NFLVALUE_HTTP_USER_AGENT") or "").strip() or DEFAULT_USER_AGENT


def get_json(url: str, params: Optional[Dict] = None, timeout: float = 15.0):
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
