"""Paths and configuration loading."""

from __future__ import annotations

import json
import os
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

WEIGHTS_PATH = os.path.join(DATA_DIR, "weights.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
LATEST_PATH = os.path.join(DATA_DIR, "latest.json")
DASHBOARD_PATH = os.path.join(ROOT, "dashboard.html")
CONFIG_PATH = os.path.join(ROOT, "config.json")

DEFAULT_CONFIG: Dict = {
    "odds_api_key": "",                 # paste your free key from the-odds-api.com
    "regions": "us",
    "game_markets": ["h2h", "spreads", "totals"],
    # Canonical player-prop vocabulary is internal. API clients translate it
    # at their boundary so the weekly and legacy paths cannot silently drift.
    "prop_markets_internal": [
        "receiving_yards", "receptions", "rushing_yards",
        "passing_yards", "anytime_td",
    ],
    "fetch_props": True,
    "max_prop_games_per_run": 4,        # props cost more API credits; cap them
    "sharp_books": ["pinnacle"],
    "sharp_weight": 2.0,
    "ev_threshold": 0.03,               # only "recommend" bets with EV >= 3%
    "edge_shrinkage": 0.5,              # trust this fraction of the model's gap vs market
    "min_odds": 1.40,
    "max_odds": 6.00,
    "kelly_multiplier": 0.15,           # fractional Kelly for safety
    "max_stake_pct": 0.03,              # never stake more than 3% of bankroll
    "learning_rate": 0.06,
    "l2": 0.01,
    "bankroll_units": 100.0,
    "refresh_seconds": 90,              # dashboard auto-reload interval
}


DEFAULT_PROP_MARKETS_INTERNAL = tuple(DEFAULT_CONFIG["prop_markets_internal"])


def prop_markets_internal(cfg: Dict) -> list:
    """Return one validated canonical market list.

    ``prop_markets`` is accepted only as a backward-compatible external-key
    input. New configuration should use ``prop_markets_internal``.
    """
    from .sources.oddsapi_props import ODDS_TO_MARKET

    raw = cfg.get("prop_markets_internal")
    if raw is None and cfg.get("prop_markets"):
        raw = [ODDS_TO_MARKET[k] for k in cfg["prop_markets"] if k in ODDS_TO_MARKET]
    raw = raw or DEFAULT_PROP_MARKETS_INTERNAL
    valid = set(ODDS_TO_MARKET.values())
    return list(dict.fromkeys(str(m) for m in raw if str(m) in valid))


def prop_markets_external(cfg: Dict) -> list:
    """Translate canonical market names to The Odds API keys."""
    from .sources.oddsapi_props import MARKET_TO_ODDS

    return [MARKET_TO_ODDS[m] for m in prop_markets_internal(cfg)]


def load_config() -> Dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                file_cfg = json.load(f)
            # A legacy file should override the default market list instead of
            # being silently ignored because the canonical default is present.
            if "prop_markets" in file_cfg and "prop_markets_internal" not in file_cfg:
                cfg.pop("prop_markets_internal", None)
            cfg.update(file_cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"[config] could not parse config.json ({exc}); using defaults")
    # env var overrides the file, handy for scheduled runs
    if os.environ.get("ODDS_API_KEY"):
        cfg["odds_api_key"] = os.environ["ODDS_API_KEY"]
    return cfg


def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return default
    return default


def save_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
