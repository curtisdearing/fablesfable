"""FTN charting features (free nflverse subset, 2022->now, updated weekly).

The formations WORKAROUND: exact personnel/alignment data died with NGS
participation (2023) and full charting is paid (FTN Data API from ~$599 CSV /
custom API tiers; PFF+ $79.99/yr browsable, no API). The free FTN per-play
subset carries the formation-ADJACENT signals that matter and stays current:

  own_pa_rate       offense play-action rate     (walk-forward team rate)
  own_motion_rate   offense pre-snap motion rate
  opp_blitz_rate    DEFENSE sends 5+ rushers rate (aggression vs pass)
  opp_box_avg       defenders in the box faced    (run-defense commitment;
                    the "defense anticipates handoffs" dial, measured)

All strictly-before via AsOfLookup; pre-2022 rows are NaN (no history, never
current-week info). Cached per season at historical/ftn_{season}.parquet and
refreshed by ingest.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .advanced_features import AsOfLookup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "historical")
FEATURES = ["own_pa_rate", "own_motion_rate", "opp_blitz_rate", "opp_box_avg"]
FTN_SEASON_MIN = 2022


def cache_path(season: int) -> str:
    return os.path.join(HIST, f"ftn_{season}.parquet")


def refresh(season: int) -> int:
    import nflreadpy as nfl
    f = nfl.load_ftn_charting(seasons=[season]).to_pandas()
    keep = ["nflverse_game_id", "nflverse_play_id", "week", "is_play_action", "is_motion",
            "n_blitzers", "n_pass_rushers", "n_defense_box"]
    f = f[[c for c in keep if c in f.columns]].copy()
    f["season"] = season
    f.to_parquet(cache_path(season), index=False)
    return len(f)


def load_ftn() -> pd.DataFrame:
    frames = []
    for fn in sorted(os.listdir(HIST)) if os.path.isdir(HIST) else []:
        if fn.startswith("ftn_") and fn.endswith(".parquet"):
            frames.append(pd.read_parquet(os.path.join(HIST, fn)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_team_rates(ftn: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Join FTN plays to pbp for posteam/defteam, aggregate per team-week,
    cumulative (AsOf-consumed: value AT (s,w) includes w; lookups strictly-
    before)."""
    if ftn.empty:
        return pd.DataFrame(columns=["player_id", "season", "week"] + FEATURES)
    pb = pbp[["game_id", "play_id", "season", "week", "posteam", "defteam", "pass"]]
    if "nflverse_play_id" not in ftn.columns:
        return (pd.DataFrame(columns=["player_id", "season", "week",
                                      "own_pa_rate", "own_motion_rate"]),
                pd.DataFrame(columns=["player_id", "season", "week",
                                      "opp_blitz_rate", "opp_box_avg"]))
    ftn = ftn.drop(columns=[c for c in ("season", "week") if c in ftn.columns]).copy()
    ftn["nflverse_play_id"] = pd.to_numeric(ftn["nflverse_play_id"], errors="coerce")
    j = ftn.merge(pb, left_on=["nflverse_game_id", "nflverse_play_id"],
                  right_on=["game_id", "play_id"], how="inner")
    if j.empty:
        return pd.DataFrame(columns=["player_id", "season", "week"] + FEATURES)

    off = (j.groupby(["season", "week", "posteam"])
           .agg(pa=("is_play_action", "mean"), mo=("is_motion", "mean"),
                n=("is_play_action", "size")).reset_index()
           .sort_values(["posteam", "season", "week"]))
    dfn = j[j["pass"] == 1].copy()
    dfn["blitz"] = (dfn["n_pass_rushers"].fillna(4) >= 5).astype(float)
    dd = (dfn.groupby(["season", "week", "defteam"])
          .agg(blitz=("blitz", "mean"), box=("n_defense_box", "mean")).reset_index()
          .sort_values(["defteam", "season", "week"]))

    go = off.groupby("posteam")
    off["own_pa_rate"] = go["pa"].transform(lambda s: s.expanding(2).mean())
    off["own_motion_rate"] = go["mo"].transform(lambda s: s.expanding(2).mean())
    gd = dd.groupby("defteam")
    dd["opp_blitz_rate"] = gd["blitz"].transform(lambda s: s.expanding(2).mean())
    dd["opp_box_avg"] = gd["box"].transform(lambda s: s.expanding(2).mean())
    off = off.rename(columns={"posteam": "player_id"})[
        ["player_id", "season", "week", "own_pa_rate", "own_motion_rate"]]
    dd = dd.rename(columns={"defteam": "player_id"})[
        ["player_id", "season", "week", "opp_blitz_rate", "opp_box_avg"]]
    return off, dd


def _load_pbp_slim() -> pd.DataFrame:
    """Only the 7 columns the FTN join needs (full ext frame OOMs a 4GB box
    when other packs are resident)."""
    cols = ["game_id", "play_id", "season", "week", "posteam", "defteam",
            "pass", "season_type"]
    frames = [pd.read_parquet(os.path.join(HIST, "historical_pbp.parquet"), columns=cols)]
    for fn in sorted(os.listdir(HIST)):
        if fn.startswith("pbp_") and fn.endswith(".parquet"):
            frames.append(pd.read_parquet(os.path.join(HIST, fn), columns=cols))
    df = pd.concat(frames, ignore_index=True)
    return df[df["season_type"] == "REG"].reset_index(drop=True)


class FTNPack:
    def __init__(self, pbp: Optional[pd.DataFrame] = None):
        if pbp is None:
            pbp = _load_pbp_slim()
        ftn = load_ftn()
        built = build_team_rates(ftn, pbp)
        if isinstance(built, tuple):
            off, dd = built
            self.off = AsOfLookup(off, ["own_pa_rate", "own_motion_rate"])
            self.dfn = AsOfLookup(dd, ["opp_blitz_rate", "opp_box_avg"])
        else:
            self.off = AsOfLookup(pd.DataFrame(
                columns=["player_id", "season", "week", "own_pa_rate", "own_motion_rate"]),
                ["own_pa_rate", "own_motion_rate"])
            self.dfn = AsOfLookup(pd.DataFrame(
                columns=["player_id", "season", "week", "opp_blitz_rate", "opp_box_avg"]),
                ["opp_blitz_rate", "opp_box_avg"])

    def attach(self, cands: pd.DataFrame) -> pd.DataFrame:
        cands = cands.copy()
        o1, o2, d1, d2 = [], [], [], []
        for r in cands.itertuples(index=False):
            key = (int(r.season), int(r.week))
            o = self.off.get(r.team, *key)
            d = self.dfn.get(r.defteam, *key)
            o1.append(o[0]); o2.append(o[1]); d1.append(d[0]); d2.append(d[1])
        cands["own_pa_rate"], cands["own_motion_rate"] = o1, o2
        cands["opp_blitz_rate"], cands["opp_box_avg"] = d1, d2
        return cands


def attach_neutral(cands: pd.DataFrame) -> pd.DataFrame:
    cands = cands.copy()
    for f in FEATURES:
        cands[f] = np.nan
    return cands


def panel_items(lean: Dict) -> List[str]:
    items = []
    b = lean.get("opp_blitz_rate")
    if b is not None and not (isinstance(b, float) and np.isnan(b)) and b >= 0.30:
        items.append(f"opponent blitzes on {b:.0%} of dropbacks (aggressive front)")
    box = lean.get("opp_box_avg")
    if box is not None and not (isinstance(box, float) and np.isnan(box)) and box >= 6.6:
        items.append(f"opponent stacks the box ({box:.1f} avg defenders — run-committed)")
    return items
