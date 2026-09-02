"""The feed helper's User-Agent is a live dependency, not decoration.

On 2026-09-02 ESPN's Akamai edge answered 403 to every site-API call made
under the opaque ``nfl-value/1.0`` string -- the load-bearing injuries feed
went missing, the freshness gate set publish=False, and the first Wednesday
board of the season shipped under a NOT PUBLISHED banner.  A self-identifying
agent that names the repository was accepted on every endpoint.  These tests
pin the shape of that string and the operator override, so the fix cannot be
lost to a tidy-up.
"""

import json
import urllib.request

from nflvalue.sources import _http


def test_default_agent_identifies_the_project_and_where_it_comes_from(monkeypatch):
    monkeypatch.delenv("NFLVALUE_HTTP_USER_AGENT", raising=False)
    ua = _http.user_agent()
    assert ua.startswith("nfl-value/")
    assert "(+https://github.com/curtisdearing/fablesfable)" in ua
    assert ua == _http.DEFAULT_USER_AGENT


def test_operator_override_wins_and_blank_override_is_ignored(monkeypatch):
    monkeypatch.setenv("NFLVALUE_HTTP_USER_AGENT", "ops-agent/2.0 (+https://example.org)")
    assert _http.user_agent() == "ops-agent/2.0 (+https://example.org)"
    monkeypatch.setenv("NFLVALUE_HTTP_USER_AGENT", "   ")
    assert _http.user_agent() == _http.DEFAULT_USER_AGENT


def test_every_request_carries_the_agent(monkeypatch):
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"ok": True}).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")
        seen["url"] = req.full_url
        return _Resp()

    monkeypatch.setenv("NFLVALUE_HTTP_USER_AGENT", "probe/1 (+https://example.org)")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = _http.get_json("https://site.api.espn.com/x", params={"limit": 5})
    assert out == {"ok": True}
    assert seen["ua"] == "probe/1 (+https://example.org)"
    assert seen["url"] == "https://site.api.espn.com/x?limit=5"
