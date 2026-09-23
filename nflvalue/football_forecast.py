"""Football-only forecast inputs: game margin from scores, dispersion from the mean.

The player forecast (mean, sd) must be built from football data alone.  Two
inputs in the live path did not meet that bar:

* ``candidates.enumerate_candidates`` tilted every player's pass/rush volume
  with ``game_script_multipliers(spread_line)`` -- the SPORTSBOOK spread.  The
  historical evaluator (``prop_backtest.py``) never applied that tilt, so it
  was both market-derived and unvalidated.  ``football_margins`` replaces it
  with a margin computed only from completed game scores strictly before the
  target game; ``margin_source`` selects which one the forecast uses.
* The SD was one pooled residual SD per market, so a backup QB and a 300-yard
  starter carried the same spread.  ``conditional_sd`` scales it with the
  projected mean (``sd = a * mean**b``), fitted once on a calibration window.

Offered lines and prices never enter here.  A threshold is applied only after
the forecast, to turn (mean, sd) into P(over); the price only after that, for
breakeven.  Market consensus is an evaluation comparator, not an input.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

FORECAST_VERSION = "ff-football-only-v1"

#: Allowed sources for the team margin that drives the pass/rush tilt.
#: "spread" is the sportsbook line (the pre-2026-09-22 behavior), kept only so
#: the evaluator can score it as a comparator.
MARGIN_SOURCES = ("football", "neutral", "spread")

#: Pre-declared, not tuned: games of history, and games of shrinkage toward 0.
MARGIN_WINDOW = 17
MARGIN_SHRINK_GAMES = 8.0

DISPERSION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "dispersion_v1.json")


def football_margins(schedules: pd.DataFrame, season: int, week: int) -> Dict[str, float]:
    """Team -> expected margin in its (season, week) game, from scores only.

    For each team: mean point differential over its last ``MARGIN_WINDOW``
    completed games whose gameday is strictly before the target game, shrunk by
    n/(n + ``MARGIN_SHRINK_GAMES``).  Margin = own net - opponent net, plus the
    league home-field edge measured over the same prior games (home team
    only).  Teams without completed history get 0.  The spread, total and
    moneyline columns are never read.
    """
    cols = ["season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"]
    s = schedules[[c for c in cols if c in schedules.columns]].copy()
    s["gameday"] = pd.to_datetime(s["gameday"])
    target = s[(s["season"] == season) & (s["week"] == week)]
    out: Dict[str, float] = {}
    for g in target.itertuples(index=False):
        prior = s[(s["gameday"] < g.gameday)
                  & s["home_score"].notna() & s["away_score"].notna()]
        if prior.empty:
            out[g.home_team] = out[g.away_team] = 0.0
            continue
        hfa = float((prior["home_score"] - prior["away_score"]).mean())
        long = pd.concat([
            pd.DataFrame({"team": prior["home_team"], "gameday": prior["gameday"],
                          "diff": prior["home_score"] - prior["away_score"]}),
            pd.DataFrame({"team": prior["away_team"], "gameday": prior["gameday"],
                          "diff": prior["away_score"] - prior["home_score"]}),
        ])

        def net(team: str) -> float:
            d = long[long["team"] == team].sort_values("gameday").tail(MARGIN_WINDOW)["diff"]
            n = len(d)
            return 0.0 if n == 0 else float(d.mean()) * n / (n + MARGIN_SHRINK_GAMES)

        m = net(g.home_team) - net(g.away_team) + hfa
        out[g.home_team] = round(m, 3)
        out[g.away_team] = round(-m, 3)
    return out


def load_dispersion(path: Optional[str] = None) -> Optional[dict]:
    path = path or DISPERSION_PATH
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def conditional_sd(market: str, mean: float, params: Optional[dict]) -> Optional[float]:
    """``a * mean**b`` for ``market``, floored at the fitted minimum.

    Returns None (caller keeps the incumbent pooled SD) when the market has no
    fitted parameters or the mean is missing/nonpositive.
    """
    if not params:
        return None
    p = params.get("markets", {}).get(market)
    if p is None or mean is None or not math.isfinite(mean) or mean <= 0:
        return None
    sd = p["a"] * mean ** p["b"]
    return float(max(sd, p["sd_floor"])) if math.isfinite(sd) else None


def fit_conditional_sd(mean: np.ndarray, actual: np.ndarray, bins: int = 10) -> dict:
    """Least-squares fit of log(bin SD) on log(bin mean) over mean deciles."""
    df = pd.DataFrame({"m": mean, "y": actual}).dropna()
    df = df[df["m"] > 0]
    df["bin"] = pd.qcut(df["m"], bins, labels=False, duplicates="drop")
    g = df.groupby("bin").apply(
        lambda x: pd.Series({"m": x["m"].mean(), "sd": (x["y"] - x["m"]).std(ddof=1),
                             "n": len(x)}))
    g = g[(g["sd"] > 0) & (g["n"] >= 30)]
    b, loga = np.polyfit(np.log(g["m"]), np.log(g["sd"]), 1)
    return {"a": float(math.exp(loga)), "b": float(b), "sd_floor": float(g["sd"].min()),
            "n_rows": int(len(df)), "n_bins": int(len(g))}


#: Chosen by analysis/football_only_protocol.json's margin rule on the 2025
#: test window (results: analysis/football_only_results.json).  "football"
#: runs in shadow via ``forecast_margin_shadow``; "spread" is never primary.
PRIMARY_MARGIN_SOURCE = "neutral"


def dispersion_fields(market: str, proj: dict, params: Optional[dict]) -> dict:
    """Mean-conditional SD for one projected row, primary or shadow per market.

    ``params['decisions'][market]`` is set by the frozen protocol's dispersion
    rule.  For a ``D1_primary`` market, ``sd``/``p_over``/``p_under`` are
    replaced and the pooled values are kept in ``sd_pooled``/``p_over_pooled``;
    otherwise the conditional values are only reported as shadow columns.
    """
    from .projection import p_over as _p_over
    out = {"dispersion_version": (params or {}).get("version"),
           "dispersion_role": None, "sd_conditional": None, "p_over_conditional": None,
           "sd_pooled": proj.get("sd"), "p_over_pooled": proj.get("p_over")}
    sd1 = conditional_sd(market, proj.get("mean"), params)
    if sd1 is None:
        return out
    out["sd_conditional"] = round(sd1, 3)
    line = proj.get("line")
    if line is not None and proj.get("mean") is not None:
        p1 = _p_over(float(proj["mean"]), sd1, float(line), proj["dist"])
        if math.isfinite(p1):
            out["p_over_conditional"] = round(p1, 4)
    primary = (params or {}).get("decisions", {}).get(market) == "D1_primary"
    out["dispersion_role"] = "primary" if primary else "shadow"
    if primary and out["p_over_conditional"] is not None:
        out["sd"] = out["sd_conditional"]
        out["p_over"] = out["p_over_conditional"]
        out["p_under"] = round(1.0 - out["p_over_conditional"], 4)
    return out
