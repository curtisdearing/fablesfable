from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from nflvalue.sources import team_intel

ROOT = Path(__file__).resolve().parents[1]


def _registry(*, feed_url=None):
    return {
        "schema_version": 1,
        "teams": [
            {
                "abbr": "BUF",
                "name": "Buffalo Bills",
                "aliases": ["Bills"],
                "sources": [
                    {
                        "id": "buf_official",
                        "name": "Buffalo Bills",
                        "domain": "buffalobills.com",
                        "source_class": "official_team",
                        "url": "https://www.buffalobills.com/news/",
                        "feed_url": feed_url,
                        "access": "free",
                    },
                    {
                        "id": "buf_buffalo_news",
                        "name": "The Buffalo News",
                        "domain": "buffalonews.com",
                        "source_class": "local_outlet",
                        "url": "https://buffalonews.com/sports/bills/",
                        "feed_url": None,
                        "access": "mixed; feed metadata only",
                    },
                ],
            }
        ],
    }


def _rss(*items: str) -> bytes:
    return ("<?xml version='1.0'?><rss version='2.0'><channel>" + "".join(items) + "</channel></rss>").encode()


def _item(title: str, url: str, date: str, description: str = "", source_url: str = "https://buffalonews.com") -> str:
    return (
        f"<item><title>{title}</title><link>{url}</link><pubDate>{date}</pubDate>"
        f"<description>{description}</description><source url='{source_url}'>The Buffalo News</source></item>"
    )


def test_parse_rss_binds_publisher_and_classifies_signal():
    request = team_intel.build_requests(_registry(), ["BUF"])[0]
    raw = _rss(
        _item(
            "Bills receiver limited at Thursday practice",
            "https://buffalonews.com/a",
            "Thu, 27 Aug 2026 15:00:00 GMT",
            "He worked with the first team but was listed with an ankle injury.",
        )
    )
    rows = team_intel.parse_feed(raw, request, fetched_at="2026-08-27T16:00:00Z")
    assert len(rows) == 1
    row = rows[0]
    assert row["team"] == "BUF"
    assert row["source_id"] == "buf_buffalo_news"
    assert row["source_class"] == "local_outlet"
    assert row["source_registry_match"] is True
    assert row["discovery_only"] is True
    assert row["url_kind"] == "aggregator_redirect"
    assert row["requires_corroboration"] is True
    assert {"availability", "role_usage"}.issubset(row["categories"])
    assert row["timestamp"] == "2026-08-27T15:00:00Z"
    assert row["performance_use"] == "context_only"


def test_parse_atom_and_clean_untrusted_markup():
    request = team_intel.build_requests(_registry(), ["BUF"])[0]
    raw = b"""<?xml version='1.0'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry><title>Bills announce practice squad transaction</title>
      <link href='https://www.buffalobills.com/news/a'/>
      <updated>2026-08-27T14:00:00Z</updated>
      <summary>&lt;b&gt;Player elevated&lt;/b&gt; before travel.</summary>
      <source url='https://www.buffalobills.com'>Buffalo Bills</source></entry>
    </feed>"""
    row = team_intel.parse_feed(raw, request, fetched_at="2026-08-27T16:00:00Z")[0]
    assert row["summary"] == "Player elevated before travel."
    assert row["source_id"] == "buf_official"
    assert {"transaction_roster", "travel_environment"}.issubset(row["categories"])


def test_collect_drops_future_stale_and_noise_items():
    raw = _rss(
        _item("Bills receiver ruled out", "https://buffalonews.com/current", "Thu, 27 Aug 2026 15:00:00 GMT"),
        _item("Bills receiver limited", "https://buffalonews.com/future", "Fri, 28 Aug 2026 15:00:00 GMT"),
        _item("Bills unveil new concessions", "https://buffalonews.com/noise", "Thu, 27 Aug 2026 15:30:00 GMT"),
        _item("Bills had an injury last month", "https://buffalonews.com/stale", "Mon, 17 Aug 2026 15:00:00 GMT"),
        _item(
            "Bills receiver injured at practice",
            "https://news.google.com/unregistered",
            "Thu, 27 Aug 2026 15:10:00 GMT",
            source_url="https://www.example-unregistered.com",
        ),
    )

    def fetcher(url: str, timeout: float) -> bytes:
        assert url.startswith(team_intel.GOOGLE_NEWS_RSS)
        assert timeout == 2.0
        return raw

    packet = team_intel.collect(
        _registry(), ["BUF"], as_of="2026-08-27T16:00:00Z", hours=48, timeout=2.0, fetcher=fetcher
    )
    assert [item["url"] for item in packet["items"]] == ["https://buffalonews.com/current"]
    assert packet["quality"]["future_dated_items_dropped"] == 1
    assert packet["quality"]["stale_items_dropped"] == 1
    assert packet["quality"]["unregistered_source_items_dropped"] == 1
    assert packet["quality"]["context_only"] is True
    assert packet["policy"]["projection_mutation_allowed"] is False
    assert packet["policy"]["public_redistribution_allowed"] is False


def test_google_query_is_team_and_domain_allowlisted():
    url = team_intel.build_google_news_url(_registry()["teams"][0])
    query = parse_qs(urlparse(url).query)["q"][0]
    assert '"Buffalo Bills"' in query
    assert "site:buffalobills.com" in query
    assert "site:buffalonews.com" in query
    assert "practice" in query


def test_direct_feed_precedes_discovery_and_wins_deduplication():
    registry = _registry(feed_url="https://www.buffalobills.com/rss/news")
    raw = _rss(
        _item(
            "Bills receiver limited at practice",
            "https://www.buffalobills.com/news/a",
            "Thu, 27 Aug 2026 15:00:00 GMT",
            source_url="https://www.buffalobills.com",
        )
    )
    packet = team_intel.collect(
        registry,
        ["BUF"],
        as_of="2026-08-27T16:00:00Z",
        fetcher=lambda _url, _timeout: raw,
    )
    assert len(packet["source_health"]) == 2
    assert len(packet["items"]) == 1
    assert packet["items"][0]["collection_method"] == "direct_rss"
    assert packet["items"][0]["discovery_only"] is False


def test_registry_rejects_unknown_team_and_duplicate_source():
    with pytest.raises(team_intel.TeamIntelSchemaError, match="unknown team"):
        team_intel.select_teams(team_intel.validate_registry(_registry()), ["XXX"])
    broken = _registry()
    broken["teams"][0]["sources"][1]["id"] = "buf_official"
    with pytest.raises(team_intel.TeamIntelSchemaError, match="duplicate source"):
        team_intel.validate_registry(broken)


def test_shipped_registry_has_exactly_32_canonical_teams_and_two_source_classes():
    registry = team_intel.load_registry(ROOT / "config" / "team_sources.json")
    assert len(registry["teams"]) == 32
    assert len({team["abbr"] for team in registry["teams"]}) == 32
    for team in registry["teams"]:
        assert any(source["source_class"] == "official_team" for source in team["sources"])
        assert any(source["source_class"] == "local_outlet" for source in team["sources"])


def test_synthesis_shape_keeps_only_citation_fields():
    packet = {
        "items": [
            {"team": "BUF", "text": "limited practice", "source": "team_intel:x", "timestamp": "2026-08-27T15:00:00Z"},
            {"team": "MIA", "text": "starter", "source": "team_intel:y", "timestamp": "2026-08-27T15:00:00Z"},
        ]
    }
    assert team_intel.synthesis_news(packet, ["BUF"]) == [
        {"text": "limited practice", "source": "team_intel:x", "timestamp": "2026-08-27T15:00:00Z"}
    ]
