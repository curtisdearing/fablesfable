"""Open-Meteo weather (free, no API key). https://open-meteo.com"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from .. import factors as factmod
from ._http import get_json

BASE = "https://api.open-meteo.com/v1/forecast"


def forecast_for_game(home_team: str, commence_iso: str) -> Optional[Dict]:
    """Return {wind_mph, precip_mm, temp_f, dome} for the kickoff hour."""
    st = factmod.STADIUMS.get(home_team)
    if not st:
        return None
    if st.get("dome"):
        return {"dome": True}
    try:
        dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
        day = dt.strftime("%Y-%m-%d")
        data = get_json(BASE, {
            "latitude": st["lat"], "longitude": st["lon"],
            "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "precipitation_unit": "mm", "start_date": day, "end_date": day,
            "timezone": "UTC",
        })
        hours = data.get("hourly", {})
        times = hours.get("time", [])
        target = dt.strftime("%Y-%m-%dT%H:00")
        idx = times.index(target) if target in times else min(
            range(len(times)), key=lambda i: abs(int(times[i][11:13]) - dt.hour)) if times else None
        if idx is None:
            return {"dome": False}
        return {
            "dome": False,
            "temp_f": hours["temperature_2m"][idx],
            "precip_mm": hours["precipitation"][idx],
            "wind_mph": hours["wind_speed_10m"][idx],
            # Phase 6.4: direction (degrees) -- crosswind decomposition for the
            # FG model / game notes (passing measured direction-insensitive)
            "wind_dir_deg": (hours.get("wind_direction_10m") or [None] * len(times))[idx],
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[weather] lookup failed for {home_team}: {exc}")
        return {"dome": False}
