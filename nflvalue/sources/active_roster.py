"""Current-season active roster: WHO is on WHICH team, RIGHT NOW.

Live mode enumerates candidates by carry-forward history (``candidates.py``,
``roster_mode="carry_forward"``): a player's last played row is his seat on
the board. History is not roster membership -- a retired, released, or traded
player keeps a carry-forward row on his OLD team for as long as his history
exists. This module is the roster source that closes that gap.

Source: the nflverse ``weekly_rosters`` release asset for the season
(``roster_weekly_{season}.parquet``). It is the same table
``nflvalue.sources.rosters`` caches for positions, but read LIVE and with the
``status`` column kept, because roster status is exactly the volatile field
the position cache throws away. The asset is fetched directly rather than
through ``nflreadpy.load_rosters_weekly`` because that wrapper refuses the
new season until the Thursday after Labor Day (it raised "Season must be
between 2002 and 2025" on 2026-09-08, two days before Week 1), while the
2026 asset itself already carried all 32 Week-1 rosters.

Provenance: ``snapshot_at`` is the asset's own ``Last-Modified`` header --
the roster's timestamp, not ours. ``fetched_at`` is when we read it. The
freshness gate consumes ``snapshot_at``; a fetch clock would make a stale
roster look fresh.

nflverse status codes seen in the wild (2025 season, all weeks): ACT (active
53), DEV (practice squad), RES (reserve: IR/PUP/NFI/suspended), INA
(inactive), CUT, RET (retired), TRD/TRC (traded, row on the departing team),
EXE (exempt). Anything else is classified ``unknown`` and fails closed.

Standard library + pandas/pyarrow only; no nflreadpy dependency.
"""

from __future__ import annotations

import datetime as dt
import io
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from ..freshness import stamp_now

ASSET_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "weekly_rosters/roster_weekly_{season}.parquet")
SOURCE_NAME = "nflverse_weekly_rosters"

#: nflverse status code -> roster eligibility class (see prop_decision).
STATUS_CLASS: Dict[str, str] = {
    "ACT": "active",
    "DEV": "practice_squad",
    "RES": "reserve",
    "PUP": "reserve",
    "NON": "reserve",
    "SUS": "reserve",
    "EXE": "reserve",
    "INA": "inactive",
    "CUT": "released",
    "RET": "retired",
    "TRD": "traded_away",
    "TRC": "traded_away",
}

ROW_COLUMNS = ["player_id", "name", "team", "position", "status", "week"]


def _http_bytes(url: str, timeout: float = 30.0):
    """GET -> (bytes, headers). Split out so tests inject a recorded asset."""
    req = urllib.request.Request(url, headers={"User-Agent": "fablesfable/1.0 (roster gate)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), dict(resp.headers.items())


def _last_modified(headers: Dict[str, str]) -> Optional[str]:
    for k, v in (headers or {}).items():
        if k.lower() == "last-modified" and v:
            try:
                parsed = parsedate_to_datetime(v)
            except (TypeError, ValueError):
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def rows_from_frame(df: pd.DataFrame, week: Optional[int] = None) -> List[Dict]:
    """Normalize a weekly-roster frame into gate rows for ONE week.

    ``week=None`` takes the latest week present (the current roster). Rows
    without a gsis id cannot be matched to anything and are dropped -- they
    are counted by the caller as ``n_unidentified`` rather than silently lost.
    """
    if df is None or df.empty:
        return []
    cols = {c.lower(): c for c in df.columns}
    need = ["gsis_id", "team", "status", "week"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"weekly roster frame lacks columns {missing}")
    frame = df.rename(columns={cols[c]: c for c in cols})
    if week is None:
        week = int(pd.to_numeric(frame["week"], errors="coerce").max())
    frame = frame[pd.to_numeric(frame["week"], errors="coerce") == week]
    frame = frame.dropna(subset=["gsis_id"])
    name_col = "full_name" if "full_name" in frame.columns else None
    pos_col = "position" if "position" in frame.columns else None
    out: List[Dict] = []
    for r in frame.itertuples(index=False):
        d = r._asdict()
        out.append({
            "player_id": str(d["gsis_id"]),
            "name": str(d.get(name_col) or "") if name_col else "",
            "team": (str(d["team"]).upper() if d.get("team") is not None
                     and not pd.isna(d.get("team")) else None),
            "position": str(d.get(pos_col) or "") if pos_col else "",
            "status": (str(d["status"]).upper() if d.get("status") is not None
                       and not pd.isna(d.get("status")) else None),
            "week": int(week),
        })
    return out


def fetch_active_roster(season: int, week: Optional[int] = None,
                        http: Optional[Callable] = None) -> Dict:
    """Live roster snapshot for ``season`` -> gate payload.

    Returns::

        {"source", "url", "season", "week", "rows": [...], "n_rows",
         "n_unidentified", "snapshot_at", "fetched_at"}

    ``week`` selects a specific week's rows; default is the latest week the
    asset carries (the roster as nflverse currently publishes it).
    Raises on any transport/schema failure: the caller (``gather_live_feeds``)
    turns that into a missing load-bearing feed, never into an empty roster.
    """
    url = ASSET_URL.format(season=int(season))
    data, headers = (http or _http_bytes)(url)
    df = pd.read_parquet(io.BytesIO(data))
    if "season" in df.columns:
        seasons = set(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int))
        if seasons and seasons != {int(season)}:
            raise ValueError(f"roster asset for {season} carries seasons {sorted(seasons)}")
    rows = rows_from_frame(df, week=week)
    wk = rows[0]["week"] if rows else None
    n_unidentified = 0
    if rows:
        sub = df[pd.to_numeric(df["week"], errors="coerce") == wk]
        n_unidentified = int(sub["gsis_id"].isna().sum())
    return {
        "source": SOURCE_NAME, "url": url, "season": int(season), "week": wk,
        "rows": rows, "n_rows": len(rows), "n_unidentified": n_unidentified,
        "snapshot_at": _last_modified(headers), "fetched_at": stamp_now(),
    }
