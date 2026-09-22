"""The metered odds pull must reconcile the credit ledger with the
provider's own accounting (x-requests-used) on the DEFAULT fetch path.

2026-09-22: production state's ledger read 165 credits used while The Odds
API reported 336 -- ``CreditBudget.spend`` trusts ``payload["_headers"]``,
but the default fetch (``_http.get_json``) returned the body only, so the
reconciliation never ran outside tests and the hard stop guarded a ledger
that under-counted real usage by 171 credits.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue.sources import _http  # noqa: E402
from nflvalue.sources import oddsapi_props as oap  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResp(io.BytesIO):
    def __init__(self, body: bytes, headers: dict):
        super().__init__(body)
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


def _cfg():
    return {"odds_api_key": "test", "regions": "us", "max_prop_games_per_run": 4,
            "odds_budget": {"monthly_credits": 500, "reserve": 50},
            "prop_markets_internal": ["receiving_yards", "receptions", "rushing_yards",
                                      "passing_yards", "anytime_td"]}


def _patch_urlopen(monkeypatch, used: int):
    payload = json.loads((FIXTURES / "oddsapi_event_props_synthetic.json").read_text())["payload"]
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.full_url)
        if req.full_url.split("?")[0].endswith("/events"):   # free quota preflight
            return _FakeResp(b"[]", {"x-requests-used": str(used - 5),
                                     "x-requests-remaining": str(505 - used),
                                     "x-requests-last": "0"})
        return _FakeResp(json.dumps(payload).encode(),
                         {"Content-Type": "application/json",
                          "x-requests-used": str(used), "x-requests-remaining": str(500 - used),
                          "x-requests-last": "5"})

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_default_fetch_reconciles_ledger_with_provider_usage(conn, monkeypatch):
    seen = _patch_urlopen(monkeypatch, used=341)
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    budget.spend(165)                      # the stale local count
    res = oap.pull_week_props(_cfg(), {"g1": "ev1"}, conn=conn, budget=budget)
    assert [u.split("?")[0].rsplit("/", 1)[-1] for u in seen] == ["events", "odds"]
    assert res["pulled"] == ["g1"]
    assert budget.used == 341              # provider's count, not 165 + 5
    row = dbmod.query_df(conn, "SELECT used, last_headers FROM api_credits WHERE month='2026-09'")
    assert float(row.iloc[0]["used"]) == 341
    assert json.loads(row.iloc[0]["last_headers"])["x-requests-remaining"] == "159"


def test_default_resnap_reconciles_ledger(conn, monkeypatch):
    _patch_urlopen(monkeypatch, used=346)
    oap.CreditBudget(conn, 500, 50).spend(170)
    oap.resnap_lines(_cfg(), {"g1": "ev1"}, conn=conn)
    assert oap.CreditBudget(conn, 500, 50).used == 346


def test_header_capture_never_leaks_into_list_payloads(monkeypatch):
    monkeypatch.setattr(_http.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(b"[]", {"x-requests-used": "1"}))
    assert _http.get_json_with_headers("https://example.invalid/x") == []
