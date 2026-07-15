import json

from nflvalue import config
from nflvalue.config import prop_markets_external, prop_markets_internal
from nflvalue.sources import oddsapi


def test_canonical_market_config_includes_receptions_and_deduplicates():
    cfg = {"prop_markets_internal": ["receptions", "receiving_yards", "receptions", "bogus"]}
    assert prop_markets_internal(cfg) == ["receptions", "receiving_yards"]
    assert prop_markets_external(cfg) == ["player_receptions", "player_reception_yds"]


def test_legacy_external_config_is_translated_at_boundary():
    cfg = {"prop_markets": ["player_receptions", "player_rush_yds"]}
    assert prop_markets_internal(cfg) == ["receptions", "rushing_yards"]


def test_legacy_file_overrides_canonical_defaults(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"prop_markets": ["player_receptions"]}))
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    loaded = config.load_config()
    assert prop_markets_external(loaded) == ["player_receptions"]


def test_legacy_odds_client_sends_canonical_markets(monkeypatch):
    captured = {}

    def fake_get_json(url, params):
        captured.update(params)
        return {"bookmakers": []}

    monkeypatch.setattr(oddsapi, "get_json", fake_get_json)
    oddsapi.fetch_event_props({"odds_api_key": "x", "regions": "us",
                               "prop_markets_internal": ["receptions", "anytime_td"]}, "event")
    assert captured["markets"] == "player_receptions,player_anytime_td"
