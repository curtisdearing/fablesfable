"""Phase 6.4: weather-string parsing, fitted severity/multiplier, roof audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nflvalue import factors
from nflvalue.candidates import apply_weather_adjustment
from nflvalue.weather_study import (STADIUM_ORIENTATION_DEG, crosswind_headwind,
                                    parse_weather_string)


def test_parse_weather_string_variants():
    p = parse_weather_string("Mostly Cloudy Temp: 91° F, Humidity: 56%, Wind: SSE 8 mph")
    assert p["temp_f"] == 91 and p["wind_mph"] == 8 and p["wind_dir_deg"] == 157.5
    assert p["precip_flag"] == 0
    r = parse_weather_string("Rain showers Temp: 44° F, Humidity: 90%, Wind: W 15 mph")
    assert r["precip_flag"] == 1 and r["wind_dir_deg"] == 270
    dome = parse_weather_string("N/A (Indoors) Temp: ° F, Wind:  mph")
    assert np.isnan(dome["temp_f"]) and np.isnan(dome["wind_mph"])
    assert parse_weather_string(None)["precip_flag"] == 0


def test_crosswind_decomposition():
    # pure crosswind: wind 90° off the field axis
    c = crosswind_headwind(10, 90, 0)
    assert abs(c["cross_mph"] - 10) < 1e-9 and abs(c["along_mph"]) < 1e-9
    # pure along: same bearing
    a = crosswind_headwind(10, 20, 20)
    assert abs(a["along_mph"] - 10) < 1e-9 and abs(a["cross_mph"]) < 1e-9
    assert 0 <= STADIUM_ORIENTATION_DEG["GB"] < 360


def test_fitted_severity_shape():
    assert factors.weather_severity({"dome": True}) == 0.0
    assert factors.weather_severity(None) == 0.0
    typical = factors.weather_severity({"wind_mph": 8, "precip_mm": 0})
    assert typical == 0.0                       # centered at the typical day
    brutal = factors.weather_severity({"wind_mph": 20, "precip_mm": 3})
    assert brutal > 0.8                          # ~20mph + rain ~= 1.0
    windy = factors.weather_severity({"wind_mph": 18, "precip_mm": 0})
    rainy = factors.weather_severity({"wind_mph": 8, "precip_mm": 3})
    assert rainy > windy                         # precip is the dominant fitted term
    # cold deliberately does nothing (failed t>=2)
    assert factors.weather_severity({"wind_mph": 8, "precip_mm": 0, "temp_f": 5}) == typical


def test_pass_multiplier_centered_and_bounded():
    assert factors.weather_pass_multiplier(8, 0, True) == 1.0
    assert factors.weather_pass_multiplier(None, 0, True) == 1.0
    assert factors.weather_pass_multiplier(25, 5, False) == 1.0   # roof closed
    calm = factors.weather_pass_multiplier(2, 0, True)
    storm = factors.weather_pass_multiplier(22, 4, True)
    assert calm > 1.0 > storm
    assert 0.85 <= storm and calm <= 1.06


def test_apply_weather_adjustment_routes_markets():
    cands = pd.DataFrame([
        {"game_id": "g1", "market": "receiving_yards", "mean": 60.0, "sd": 20.0,
         "dist": "gamma", "line": 55.5, "p_over": 0.55, "p_under": 0.45},
        {"game_id": "g1", "market": "receptions", "mean": 5.0, "sd": 1.6,
         "dist": "negbinom", "line": 4.5, "p_over": 0.55, "p_under": 0.45},
        {"game_id": "g2", "market": "passing_yards", "mean": 240.0, "sd": 50.0,
         "dist": "normal", "line": 235.5, "p_over": 0.52, "p_under": 0.48},
    ])
    wx = {"g1": {"wind_mph": 22, "precip_mm": 4, "effective_outdoor": True},
          "g2": {"wind_mph": 8, "precip_mm": 0, "effective_outdoor": True}}
    out = apply_weather_adjustment(cands, wx)
    ry = out[out["market"] == "receiving_yards"].iloc[0]
    rc = out[out["market"] == "receptions"].iloc[0]
    py = out[out["market"] == "passing_yards"].iloc[0]
    assert ry["mean"] < 60.0 and ry["p_over"] < 0.55   # storm dampens yards
    assert rc["mean"] == 5.0                            # counts market untouched
    assert py["mean"] == 240.0                          # typical day = no-op
    assert "wx_pass_mult" in out.columns
